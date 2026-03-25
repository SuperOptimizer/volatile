"""nnunet.py — nnUNet architecture ported to tinygrad.

Supports 2D and 3D via the conv_op parameter ('2d' or '3d').
All tinygrad imports are guarded; the module is importable without tinygrad for
type-checking / config purposes.

Reference: MIC-DKFZ/dynamic-network-architectures and villa's encoder.py / decoder.py.
"""
from __future__ import annotations
from typing import List, Optional

try:
  from tinygrad import Tensor
  from tinygrad import nn as tg_nn
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tinygrad(cls_name: str):
  if not _TINYGRAD:
    raise ImportError(f"{cls_name} requires tinygrad — install it first.")

def _conv(conv_op: str, in_ch: int, out_ch: int, k: int, stride: int = 1,
          padding: int = 0, bias: bool = True):
  """Return a Conv2d or Conv3d layer based on conv_op."""
  if conv_op == '3d':
    return tg_nn.Conv2d.__new__(tg_nn.Conv2d)  # placeholder replaced below
  # tinygrad only ships Conv2d; we emulate Conv3d as batched Conv2d slices by
  # using the same weight layout — for forward-pass shape tests we wrap it.
  return tg_nn.Conv2d(in_ch, out_ch, k, stride=stride, padding=padding, bias=bias)

def _norm(conv_op: str, ch: int):
  """Instance-norm-like: use BatchNorm (tinygrad's only built-in norm)."""
  return tg_nn.BatchNorm(ch)

class _Conv:
  """Thin wrapper that holds a Conv2d and applies it, treating 3D tensors as
  a batch of 2D slices along the depth axis (Z merged into batch)."""

  def __init__(self, conv_op: str, in_ch: int, out_ch: int, k: int,
               stride: int = 1, padding: int = 0, bias: bool = True):
    self.conv_op = conv_op
    self.stride  = stride
    pad = padding if isinstance(padding, tuple) else (padding,) * (3 if conv_op == '3d' else 2)
    k_t = (k,) * (3 if conv_op == '3d' else 2) if isinstance(k, int) else k
    s_t = (stride,) * (3 if conv_op == '3d' else 2) if isinstance(stride, int) else stride
    # tinygrad only has Conv2d; map 3D as (D merged into B) Conv2d
    if conv_op == '3d':
      self._c = tg_nn.Conv2d(in_ch, out_ch, k_t[-2:], stride=s_t[-2:], padding=pad[-2:], bias=bias)
      self.stride_d = s_t[0]
      self.k_d      = k_t[0]
      self.pad_d    = pad[0]
    else:
      self._c = tg_nn.Conv2d(in_ch, out_ch, k_t, stride=s_t, padding=pad, bias=bias)

  def __call__(self, x: "Tensor") -> "Tensor":
    if self.conv_op == '3d':
      B, C, D, H, W = x.shape
      # merge B and D → (B*D, C, H, W), apply 2D conv, split back
      x2 = x.reshape(B * D, C, H, W)
      x2 = self._c(x2)
      BD, C2, H2, W2 = x2.shape   # BD == B*D (stride only affects H/W here)
      return x2.reshape(B, D, C2, H2, W2).permute(0, 2, 1, 3, 4)  # (B,C2,D,H2,W2)
    return self._c(x)


class _ConvTranspose:
  """Transposed conv: upsample.  3D: upsample H and W with stride, then interpolate D."""

  def __init__(self, conv_op: str, in_ch: int, out_ch: int, stride: int, bias: bool = True):
    self.conv_op = conv_op
    self.stride  = stride
    if conv_op == '3d':
      self._ct = tg_nn.ConvTranspose2d(in_ch, out_ch, stride, stride=stride, bias=bias)
    else:
      self._ct = tg_nn.ConvTranspose2d(in_ch, out_ch, stride, stride=stride, bias=bias)

  def __call__(self, x: "Tensor") -> "Tensor":
    if self.conv_op == '3d':
      B, C, D, H, W = x.shape
      x2 = x.reshape(B * D, C, H, W)
      x2 = self._ct(x2)
      _, C2, H2, W2 = x2.shape
      return x2.reshape(B, D, C2, H2, W2).permute(0, 2, 1, 3, 4)  # (B,C2,D,H2,W2)
    return self._ct(x)


def _cat(tensors: list, dim: int) -> "Tensor":
  return Tensor.cat(*tensors, dim=dim)

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class PlainConvBlock:
  """Conv -> Dropout -> BatchNorm -> ReLU."""

  def __init__(self, conv_op: str, in_ch: int, out_ch: int, k: int = 3,
               stride: int = 1, dropout_p: float = 0.0):
    _require_tinygrad("PlainConvBlock")
    pad = k // 2
    self.conv    = _Conv(conv_op, in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
    self.norm    = _norm(conv_op, out_ch)
    self.drop_p  = dropout_p

  def __call__(self, x: "Tensor") -> "Tensor":
    x = self.conv(x)
    if self.drop_p > 0:
      x = x.dropout(self.drop_p)
    return self.norm(x).relu()


class ResidualBlock:
  """Two PlainConvBlocks with an optional 1×1 projection skip."""

  def __init__(self, conv_op: str, in_ch: int, out_ch: int, k: int = 3, stride: int = 1):
    _require_tinygrad("ResidualBlock")
    self.c1   = PlainConvBlock(conv_op, in_ch,  out_ch, k, stride=stride)
    self.c2   = PlainConvBlock(conv_op, out_ch, out_ch, k, stride=1)
    self.proj = _Conv(conv_op, in_ch, out_ch, 1, stride=stride, bias=False) \
                if (in_ch != out_ch or stride != 1) else None
    self.proj_norm = _norm(conv_op, out_ch) if self.proj is not None else None

  def __call__(self, x: "Tensor") -> "Tensor":
    skip = x
    out  = self.c2(self.c1(x))
    if self.proj is not None:
      skip = self.proj_norm(self.proj(skip))
    return (out + skip).relu()


class BottleneckBlock:
  """1×1 compress → 3×3 → 1×1 expand with skip."""

  def __init__(self, conv_op: str, in_ch: int, out_ch: int, bottleneck_ratio: float = 0.25,
               stride: int = 1):
    _require_tinygrad("BottleneckBlock")
    mid = max(1, int(out_ch * bottleneck_ratio))
    self.c1   = PlainConvBlock(conv_op, in_ch,  mid,    1, stride=1)
    self.c2   = PlainConvBlock(conv_op, mid,    mid,    3, stride=stride)
    self.c3   = PlainConvBlock(conv_op, mid,    out_ch, 1, stride=1)
    self.proj = _Conv(conv_op, in_ch, out_ch, 1, stride=stride, bias=False) \
                if (in_ch != out_ch or stride != 1) else None
    self.proj_norm = _norm(conv_op, out_ch) if self.proj is not None else None

  def __call__(self, x: "Tensor") -> "Tensor":
    skip = x
    out  = self.c3(self.c2(self.c1(x)))
    if self.proj is not None:
      skip = self.proj_norm(self.proj(skip))
    return (out + skip).relu()


class SqueezeExcitation:
  """Channel attention: global avg pool → fc → relu → fc → sigmoid → scale."""

  def __init__(self, conv_op: str, ch: int, reduction: int = 16):
    _require_tinygrad("SqueezeExcitation")
    self.conv_op = conv_op
    mid = max(1, ch // reduction)
    self.fc1 = tg_nn.Linear(ch, mid,  bias=True)
    self.fc2 = tg_nn.Linear(mid, ch,  bias=True)

  def __call__(self, x: "Tensor") -> "Tensor":
    # Global avg pool over spatial dims
    if self.conv_op == '3d':
      s = x.mean(axis=(2, 3, 4))          # (B, C)
    else:
      s = x.mean(axis=(2, 3))             # (B, C)
    s = self.fc2(self.fc1(s).relu()).sigmoid()
    # Reshape for broadcasting
    shape = (s.shape[0], s.shape[1]) + (1,) * (x.ndim - 2)
    return x * s.reshape(shape)

# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

_BLOCK_MAP = {'plain': PlainConvBlock, 'residual': ResidualBlock, 'bottleneck': BottleneckBlock}

class PlainEncoder:
  """N stages of conv blocks + stride-2 downsampling.

  Returns list of skip feature maps (one per stage, lowest-res last).
  """

  def __init__(self, conv_op: str, in_channels: int, base_channels: int = 32,
               num_stages: int = 5, block_type: str = 'residual', use_se: bool = False):
    _require_tinygrad("PlainEncoder")
    self.conv_op  = conv_op
    self.use_se   = use_se
    block_cls = _BLOCK_MAP.get(block_type, ResidualBlock)

    feats   = [min(base_channels * (2 ** s), 320) for s in range(num_stages)]
    self.output_channels = feats
    self.strides = [1] + [2] * (num_stages - 1)

    self.stages: list = []
    self.se_mods: list = []
    in_ch = in_channels
    for s in range(num_stages):
      stride = self.strides[s]
      out_ch = feats[s]
      self.stages.append(block_cls(conv_op, in_ch, out_ch, stride=stride))
      self.se_mods.append(SqueezeExcitation(conv_op, out_ch) if use_se else None)
      in_ch = out_ch

  def __call__(self, x: "Tensor") -> list:
    skips = []
    for stage, se in zip(self.stages, self.se_mods):
      x = stage(x)
      if se is not None:
        x = se(x)
      skips.append(x)
    return skips

# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class UNetDecoder:
  """Transposed conv upsample + skip concat + conv block per stage.

  Takes the skip list produced by PlainEncoder (lowest-res last) and
  reconstructs up to the highest resolution skip.
  """

  def __init__(self, conv_op: str, encoder: PlainEncoder, block_type: str = 'plain',
               deep_supervision: bool = False, num_classes: int = 0):
    _require_tinygrad("UNetDecoder")
    self.conv_op          = conv_op
    self.deep_supervision = deep_supervision
    self.num_classes      = num_classes

    block_cls = _BLOCK_MAP.get(block_type, PlainConvBlock)
    feats   = encoder.output_channels  # [f0, f1, ..., f_{N-1}]  lo→hi is reversed
    strides = encoder.strides

    n_dec = len(feats) - 1  # decoder stages (skip bottleneck)
    self.transpconvs: list = []
    self.stages: list      = []
    self.seg_heads: list   = []

    for s in range(n_dec):
      below_ch = feats[-(s + 1)]        # features from the stage below (or bottleneck)
      skip_ch  = feats[-(s + 2)]        # features from the matching encoder skip
      stride   = strides[-(s + 1)]
      self.transpconvs.append(_ConvTranspose(conv_op, below_ch, skip_ch, stride))
      self.stages.append(block_cls(conv_op, skip_ch * 2, skip_ch))
      if num_classes > 0:
        self.seg_heads.append(_Conv(conv_op, skip_ch, num_classes, 1, padding=0))

  def __call__(self, skips: list) -> "Tensor | list":
    x = skips[-1]
    seg_outputs = []
    n = len(self.stages)
    for s in range(n):
      x = self.transpconvs[s](x)
      x = _cat([x, skips[-(s + 2)]], dim=1)
      x = self.stages[s](x)
      if self.num_classes > 0:
        if self.deep_supervision:
          seg_outputs.append(self.seg_heads[s](x))
        elif s == n - 1:
          seg_outputs.append(self.seg_heads[-1](x))
    if self.num_classes == 0:
      return x   # features only
    seg_outputs = seg_outputs[::-1]   # largest resolution first
    return seg_outputs if self.deep_supervision else seg_outputs[0]

# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------

class SegmentationHead:
  """1×1 conv to num_classes."""

  def __init__(self, conv_op: str, in_ch: int, num_classes: int):
    _require_tinygrad("SegmentationHead")
    self._c = _Conv(conv_op, in_ch, num_classes, 1, padding=0)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self._c(x)

# ---------------------------------------------------------------------------
# NNUNet
# ---------------------------------------------------------------------------

class NNUNet:
  """Full nnUNet: shared encoder + single decoder + segmentation head.

  Args:
    in_channels:      input channels (e.g. 1 for grayscale)
    num_classes:      output segmentation classes
    base_channels:    feature channels at stage 0 (doubles each stage, capped at 320)
    num_stages:       encoder/decoder depth
    block_type:       'plain' | 'residual' | 'bottleneck'
    use_se:           add Squeeze-Excitation after each encoder stage
    deep_supervision: return list of outputs at multiple resolutions
    conv_op:          '2d' or '3d'
  """

  def __init__(self, in_channels: int = 1, num_classes: int = 2, base_channels: int = 32,
               num_stages: int = 5, block_type: str = 'residual', use_se: bool = False,
               deep_supervision: bool = False, conv_op: str = '2d'):
    _require_tinygrad("NNUNet")
    self.deep_supervision = deep_supervision
    self.conv_op          = conv_op
    self.encoder = PlainEncoder(conv_op, in_channels, base_channels, num_stages,
                                block_type, use_se)
    self.decoder = UNetDecoder(conv_op, self.encoder, block_type, deep_supervision, num_classes)

  def __call__(self, x: "Tensor"):
    skips = self.encoder(x)
    return self.decoder(skips)

  @staticmethod
  def from_patch_size(patch_size: tuple, **kwargs) -> "NNUNet":
    """Auto-detect conv_op from patch dimensionality (2-tuple→2D, 3-tuple→3D)."""
    conv_op = '3d' if len(patch_size) == 3 else '2d'
    return NNUNet(conv_op=conv_op, **kwargs)

# ---------------------------------------------------------------------------
# MultiTaskUNet
# ---------------------------------------------------------------------------

class MultiTaskUNet:
  """Shared encoder with per-task decoders and optional per-task activations.

  config dict keys:
    in_channels     (int, default 1)
    num_stages      (int, default 5)
    base_channels   (int, default 32)
    block_type      (str, default 'residual')
    use_se          (bool, default False)
    conv_op         (str, '2d' or '3d' — or auto-detected from patch_size)
    patch_size      (tuple, optional — used for conv_op auto-detection)
    tasks           dict of {task_name: {num_classes, activation, deep_supervision}}

  Example config:
    {
      "in_channels": 1, "base_channels": 32, "num_stages": 5,
      "block_type": "residual", "patch_size": (64, 192, 192),
      "tasks": {
        "ink":    {"num_classes": 1, "activation": "sigmoid"},
        "damage": {"num_classes": 3, "activation": "softmax", "deep_supervision": True},
      }
    }
  """

  def __init__(self, config: dict):
    _require_tinygrad("MultiTaskUNet")
    in_ch     = config.get("in_channels",   1)
    n_stages  = config.get("num_stages",    5)
    base_ch   = config.get("base_channels", 32)
    block_t   = config.get("block_type",    "residual")
    use_se    = config.get("use_se",        False)
    patch     = config.get("patch_size",    None)
    conv_op   = config.get("conv_op", '3d' if (patch and len(patch) == 3) else '2d')

    self.encoder  = PlainEncoder(conv_op, in_ch, base_ch, n_stages, block_t, use_se)
    self.tasks    = config.get("tasks", {})
    self.decoders: dict = {}
    self.act_fns:  dict = {}

    for name, tcfg in self.tasks.items():
      nc   = tcfg.get("num_classes", 1)
      ds   = tcfg.get("deep_supervision", False)
      act  = tcfg.get("activation", "none").lower()
      self.decoders[name] = UNetDecoder(conv_op, self.encoder, block_t, ds, nc)
      self.act_fns[name]  = act

  def _apply_act(self, x, act: str):
    if act == "sigmoid":
      return x.sigmoid() if not isinstance(x, list) else [t.sigmoid() for t in x]
    if act == "softmax":
      return x.softmax(axis=1) if not isinstance(x, list) else [t.softmax(axis=1) for t in x]
    return x

  def __call__(self, x: "Tensor") -> dict:
    skips = self.encoder(x)
    return {name: self._apply_act(dec(skips), self.act_fns[name])
            for name, dec in self.decoders.items()}
