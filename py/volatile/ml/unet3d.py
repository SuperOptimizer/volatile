"""
UNet3D — true 3-D U-Net (tinygrad port of villa's Vesuvius3dUnetModel).

Conv3d is not built into tinygrad, so we implement it via the unfold pattern:
  1. Extract patches with Tensor.unfold along each of (D, H, W)
  2. Reshape to (B, out_d*out_h*out_w, C_in*kD*kH*kW)  — an im2col matrix
  3. Multiply by the weight matrix  (C_in*kD*kH*kW, C_out)
  4. Reshape back to (B, C_out, out_d, out_h, out_w)

This is exact (no approximation) and avoids any external CUDA kernels.

Architecture (simplified from villa vesuvius_unet3d.py):
  Encoder: L levels, each Conv3dBlock (conv3×3×3 → BN → ReLU) × 2, then maxpool 2×2×2
  Bottleneck: Conv3dBlock × 2
  Decoder: L levels, upsample × 2 + skip-cat + Conv3dBlock × 2
  Head: Conv3d 1×1×1 → out_channels

Usage::

  model = UNet3D(in_channels=1, out_channels=4, base=32, levels=4)
  y = model(x)    # x: (B, 1, D, H, W) → y: (B, 4, D, H, W)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters
  _TG = True
except ImportError:
  _TG = False

# ---------------------------------------------------------------------------
# Conv3d via unfold-matmul
# ---------------------------------------------------------------------------

class Conv3d:
  """3-D convolution: weight (C_out, C_in, kD, kH, kW), stride 1, same padding."""

  def __init__(self, in_ch: int, out_ch: int, k: int = 3, stride: int = 1, padding: Optional[int] = None):
    if not _TG:
      raise ImportError("tinygrad is required for Conv3d")
    self.in_ch   = in_ch
    self.out_ch  = out_ch
    self.k       = k
    self.stride  = stride
    self.padding = padding if padding is not None else k // 2
    scale = (2.0 / (in_ch * k * k * k)) ** 0.5
    self.weight = Tensor.randn(out_ch, in_ch, k, k, k) * scale
    self.bias   = Tensor.zeros(out_ch)

  def __call__(self, x: "Tensor") -> "Tensor":
    """x: (B, C_in, D, H, W) → (B, C_out, D', H', W')"""
    B, C, D, H, W = x.shape
    k, s, p = self.k, self.stride, self.padding

    # Pad spatial dims symmetrically
    if p > 0:
      x = x.pad(((0,0),(0,0),(p,p),(p,p),(p,p)))

    _, _, Dp, Hp, Wp = x.shape

    out_d = (Dp - k) // s + 1
    out_h = (Hp - k) // s + 1
    out_w = (Wp - k) // s + 1

    # --- im2col via unfold along each spatial axis ---
    # unfold(dim, size, step) → appends a new trailing axis of length `size`
    col = x.unfold(2, k, s).unfold(3, k, s).unfold(4, k, s)
    # shape: (B, C, out_d, out_h, out_w, k, k, k)
    col = col.permute(0, 2, 3, 4, 1, 5, 6, 7)            # (B, od, oh, ow, C, k, k, k)
    col = col.reshape(B, out_d * out_h * out_w, C * k * k * k)  # im2col matrix

    # Weight: (C_out, C*k*k*k) → transpose to (C*k*k*k, C_out)
    w = self.weight.reshape(self.out_ch, C * k * k * k).T  # (fan_in, C_out)

    out = col @ w                                          # (B, out_d*out_h*out_w, C_out)
    out = out.reshape(B, out_d, out_h, out_w, self.out_ch)
    out = out.permute(0, 4, 1, 2, 3)                      # (B, C_out, out_d, out_h, out_w)
    return out + self.bias.reshape(1, self.out_ch, 1, 1, 1)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Conv3dBlock:
  """Conv3d(3×3×3) → InstanceNorm3d → LeakyReLU.  True 3-D, not 2.5-D."""

  def __init__(self, in_ch: int, out_ch: int):
    if not _TG:
      raise ImportError("tinygrad is required for Conv3dBlock")
    self.conv = Conv3d(in_ch, out_ch, k=3)
    self.norm = nn.InstanceNorm(out_ch)   # tinygrad InstanceNorm works across spatial dims

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.norm(self.conv(x)).leaky_relu()


class _EncLevel:
  """Two Conv3dBlocks (no downsampling — caller does pool)."""

  def __init__(self, in_ch: int, out_ch: int):
    self.b1 = Conv3dBlock(in_ch,  out_ch)
    self.b2 = Conv3dBlock(out_ch, out_ch)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.b2(self.b1(x))


class _DecLevel:
  """Upsample ×2 + skip-cat + two Conv3dBlocks."""

  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    self.b1 = Conv3dBlock(in_ch + skip_ch, out_ch)
    self.b2 = Conv3dBlock(out_ch, out_ch)

  def __call__(self, x: "Tensor", skip: "Tensor") -> "Tensor":
    # trilinear upsample × 2 to match skip shape
    d, h, w = skip.shape[2], skip.shape[3], skip.shape[4]
    x = x.interpolate((d, h, w), mode='linear')
    x = Tensor.cat(x, skip, dim=1)
    return self.b2(self.b1(x))


# ---------------------------------------------------------------------------
# UNet3D
# ---------------------------------------------------------------------------

class UNet3D:
  """True 3-D U-Net built with the custom Conv3d unfold implementation.

  Args:
    in_channels:  input channels (default 1)
    out_channels: output channels / classes (default 4)
    base:         channel count at the first level (default 32)
    levels:       number of encoder/decoder levels (default 4)
  """

  def __init__(self, in_channels: int = 1, out_channels: int = 4,
               base: int = 32, levels: int = 4):
    if not _TG:
      raise ImportError("tinygrad is required for UNet3D")

    self.levels = levels
    feats: List[int] = [base * (2 ** i) for i in range(levels)]   # e.g. [32,64,128,256]

    # Encoder
    self.enc: List[_EncLevel] = []
    ch = in_channels
    for f in feats:
      self.enc.append(_EncLevel(ch, f))
      ch = f

    # Bottleneck
    self.bottleneck = _EncLevel(ch, ch * 2)
    ch = ch * 2

    # Decoder (in reverse order)
    self.dec: List[_DecLevel] = []
    for f in reversed(feats):
      self.dec.append(_DecLevel(ch, f, f))
      ch = f

    # Output head: 1×1×1 conv
    self.head = Conv3d(ch, out_channels, k=1, padding=0)

  def __call__(self, x: "Tensor") -> "Tensor":
    """x: (B, in_channels, D, H, W) → (B, out_channels, D, H, W)"""
    # Encoder — collect skip connections
    skips: List["Tensor"] = []
    for enc_blk in self.enc:
      x = enc_blk(x)
      skips.append(x)
      # 3-D max pool ×2: reshape then reduce — tinygrad max_pool2d is 2-D only
      B, C, D, H, W = x.shape
      x = x.reshape(B, C, D//2, 2, H//2, 2, W//2, 2)
      x = x.max(axis=(3, 5, 7))               # (B, C, D//2, H//2, W//2)

    # Bottleneck
    x = self.bottleneck(x)

    # Decoder — use skip connections in reverse order
    for dec_blk, skip in zip(self.dec, reversed(skips)):
      x = dec_blk(x, skip)

    return self.head(x)

  def parameters(self) -> list:
    return get_parameters(self)
