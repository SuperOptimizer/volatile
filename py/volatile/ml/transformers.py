from __future__ import annotations

"""
Transformer architectures for 3-D volume understanding, ported from villa's
EVA / Primus / FlashRoPE / PoPE stack to pure tinygrad.

Reference implementations:
  - EVA / EVA02: He et al. "EVA: Exploring the Limits of Masked Visual Representation Learning" (2022)
  - FlashRoPE: EVA attention with Rotary Position Encoding backed by SDPA
  - PoPE: Polar-coordinate Positional Encoding (phase-bias attention variant)
  - Primus: "Primus: Enforcing Attention Usage for 3D Medical Image Segmentation" (2025)
  - Vesuvius3dViTModel: 3-D ViT backbone for scroll volume processing

All classes use 2-space indentation and accept / return tinygrad Tensors.
"""

import math
from typing import List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_tinygrad(cls_name: str) -> None:
  if not _TINYGRAD:
    raise ImportError(f"tinygrad is required for {cls_name}")


def _layer_norm(dim: int) -> "nn.LayerNorm":
  return nn.LayerNorm(dim)


# ---------------------------------------------------------------------------
# RoPE (Rotary Position Encoding) utilities
# ---------------------------------------------------------------------------

def build_rope_freqs(dim: int, max_seq: int = 4096, theta: float = 10000.0) -> np.ndarray:
  """
  Build 1-D RoPE frequency table.

  Returns (max_seq, dim) float32 — concatenated [sin, cos] over position indices.
  Actually returns (max_seq, dim//2, 2) arranged as freqs for apply_rope().
  """
  half = dim // 2
  freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
  t = np.arange(max_seq, dtype=np.float32)
  outer = np.outer(t, freqs)               # (max_seq, half)
  return np.stack([np.sin(outer), np.cos(outer)], axis=-1).astype(np.float32)  # (max_seq, half, 2)


def apply_rope_1d(x: "Tensor", freqs_np: np.ndarray) -> "Tensor":
  """
  Apply 1-D RoPE to x of shape (B, H, N, D).

  freqs_np: (N, D//2, 2) precomputed sin/cos table.
  Returns same shape as x.
  """
  B, H, N, D = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
  half = D // 2
  # split into even and odd halves
  x1 = x[:, :, :, :half]   # (B, H, N, half)
  x2 = x[:, :, :, half:]   # (B, H, N, half)
  sin_t = Tensor(freqs_np[:N, :, 0])   # (N, half)
  cos_t = Tensor(freqs_np[:N, :, 1])   # (N, half)
  # broadcast over B and H
  sin_t = sin_t.reshape(1, 1, N, half)
  cos_t = cos_t.reshape(1, 1, N, half)
  rx1 = x1 * cos_t - x2 * sin_t
  rx2 = x1 * sin_t + x2 * cos_t
  return rx1.cat(rx2, dim=-1)


def build_rope_nd_freqs(
  shape: Tuple[int, ...],
  head_dim: int,
  theta: float = 10000.0,
) -> np.ndarray:
  """
  Build N-D RoPE sin/cos table for a spatial grid.

  For D spatial dims, each dim gets head_dim // D // 2 frequency pairs.
  Returns (prod(shape), head_dim) float32 — concat of [sin_d0, cos_d0, sin_d1, cos_d1, ...]
  padded to head_dim with zeros/ones (identity rotation) if needed.

  This is the multi-axis variant used in EVA for 2-D and 3-D inputs.
  """
  ndim = len(shape)
  dim_per_axis = (head_dim // ndim) & ~1        # round down to even
  total_coded = dim_per_axis * ndim             # may be < head_dim

  axes = []
  for axis_len in shape:
    half = dim_per_axis // 2
    freqs = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) / max(1, half)))
    t = np.arange(axis_len, dtype=np.float64)
    outer = np.outer(t, freqs).astype(np.float32)  # (axis_len, half)
    sin_ = np.sin(outer)
    cos_ = np.cos(outer)
    axes.append(np.concatenate([sin_, cos_], axis=-1))  # (axis_len, dim_per_axis)

  # Build full grid by taking the Cartesian product of axes
  grids = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
  n_positions = int(np.prod(shape))
  rope_parts = []
  for axis_idx, (axis_emb, g) in enumerate(zip(axes, grids)):
    flat_idx = g.ravel().astype(np.int32)
    rope_parts.append(axis_emb[flat_idx])  # (n_positions, dim_per_axis)

  rope = np.concatenate(rope_parts, axis=-1)  # (n_positions, total_coded)

  if total_coded < head_dim:
    pad = head_dim - total_coded
    sin_part = rope[:, :total_coded // 2]
    cos_part = rope[:, total_coded // 2:]
    pad_sin = np.zeros((n_positions, pad // 2), dtype=np.float32)
    pad_cos = np.ones((n_positions, pad // 2), dtype=np.float32)
    rope = np.concatenate([sin_part, pad_sin, cos_part, pad_cos], axis=-1)  # (n_positions, head_dim)

  # Reshape to (n_positions, head_dim//2, 2) for apply_rope_nd
  half = head_dim // 2
  sin_full = rope[:, :half]
  cos_full = rope[:, half:]
  return np.stack([sin_full, cos_full], axis=-1).astype(np.float32)  # (n_positions, half, 2)


def apply_rope_nd(x: "Tensor", freqs_np: np.ndarray) -> "Tensor":
  """
  Apply N-D RoPE to x of shape (B, H, N, D).

  freqs_np: (N, D//2, 2) precomputed sin/cos.
  """
  N_x = x.shape[2]
  D = x.shape[3]
  half = D // 2
  sin_t = Tensor(freqs_np[:N_x, :, 0]).reshape(1, 1, N_x, half)
  cos_t = Tensor(freqs_np[:N_x, :, 1]).reshape(1, 1, N_x, half)
  x1 = x[:, :, :, :half]
  x2 = x[:, :, :, half:]
  return (x1 * cos_t - x2 * sin_t).cat(x1 * sin_t + x2 * cos_t, dim=-1)


# ---------------------------------------------------------------------------
# MultiHeadAttention
# ---------------------------------------------------------------------------

class MultiHeadAttention:
  """
  Standard scaled dot-product multi-head attention.

  Args:
    embed_dim:   total embedding dimension
    num_heads:   number of attention heads
    qkv_bias:    add bias to QKV projections
    attn_drop:   dropout probability on attention weights (ignored — tinygrad has no dropout)
    proj_drop:   dropout probability after output projection (ignored)
  """

  def __init__(self, embed_dim: int, num_heads: int, qkv_bias: bool = True,
               attn_drop: float = 0.0, proj_drop: float = 0.0):
    _require_tinygrad("MultiHeadAttention")
    assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    self.num_heads = num_heads
    self.head_dim = embed_dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
    self.proj = nn.Linear(embed_dim, embed_dim)

  def __call__(self, x: "Tensor", rope_freqs: Optional[np.ndarray] = None) -> "Tensor":
    B, N, C = x.shape[0], x.shape[1], x.shape[2]
    qkv = self.qkv(x)                                          # (B, N, 3C)
    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)                          # (3, B, H, N, D)
    q = qkv[0]                                                 # (B, H, N, D)
    k = qkv[1]
    v = qkv[2]

    if rope_freqs is not None:
      q = apply_rope_nd(q, rope_freqs)
      k = apply_rope_nd(k, rope_freqs)

    # Scaled dot-product attention
    attn = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, N, N)
    attn = attn.softmax(axis=-1)
    out = attn @ v                                             # (B, H, N, D)
    out = out.transpose(1, 2).reshape(B, N, C)                # (B, N, C)
    return self.proj(out)


# ---------------------------------------------------------------------------
# FlashRoPE — RoPE attention with optional layer-scale (gamma)
# ---------------------------------------------------------------------------

class FlashRoPEAttention:
  """
  EVA-style attention with N-D Rotary Position Encoding.

  Equivalent to villa's FlashRoPEAttention but using tinygrad's matmul
  instead of PyTorch SDPA.

  Args:
    dim:           input/output channels
    num_heads:     attention heads
    qkv_bias:      QKV bias
    scale_attn_inner: apply a learned LayerNorm after attention (EVA-scale-norm)
  """

  def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, scale_attn_inner: bool = False):
    _require_tinygrad("FlashRoPEAttention")
    assert dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    # Fused QKV — q gets bias, k no bias, v gets bias (EVA convention)
    self.qkv = nn.Linear(dim, dim * 3, bias=False)
    self.q_bias = Tensor.zeros(dim) if qkv_bias else None
    self.v_bias = Tensor.zeros(dim) if qkv_bias else None
    self.proj = nn.Linear(dim, dim)
    self.inner_norm = _layer_norm(dim) if scale_attn_inner else None

  def __call__(self, x: "Tensor", rope_freqs: Optional[np.ndarray] = None) -> "Tensor":
    B, N, C = x.shape[0], x.shape[1], x.shape[2]

    if self.q_bias is not None:
      k_bias = Tensor.zeros(C)
      qkv_bias = self.q_bias.cat(k_bias, dim=0).cat(self.v_bias, dim=0)
      qkv = x.linear(self.qkv.weight.T, qkv_bias)
    else:
      qkv = self.qkv(x)

    qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]

    if rope_freqs is not None:
      q = apply_rope_nd(q, rope_freqs)
      k = apply_rope_nd(k, rope_freqs)

    attn = (q @ k.transpose(-2, -1)) * self.scale
    attn = attn.softmax(axis=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)

    if self.inner_norm is not None:
      out = self.inner_norm(out)
    return self.proj(out)


class FlashRoPE:
  """
  Pre-computed N-D RoPE positional encoding module.

  Stores the sin/cos frequency table for a fixed spatial grid and provides
  `get_freqs()` for use in attention layers.

  Args:
    head_dim:    per-head dimension
    feat_shape:  spatial grid dimensions, e.g. (8, 8) or (4, 8, 8)
    theta:       RoPE base frequency
  """

  def __init__(self, head_dim: int, feat_shape: Tuple[int, ...], theta: float = 10000.0):
    _require_tinygrad("FlashRoPE")
    self.head_dim = head_dim
    self.feat_shape = feat_shape
    self._freqs: np.ndarray = build_rope_nd_freqs(feat_shape, head_dim, theta)

  def get_freqs(self) -> np.ndarray:
    """Return (n_positions, head_dim//2, 2) frequency table."""
    return self._freqs


# ---------------------------------------------------------------------------
# SwiGLU MLP (used inside EVA blocks)
# ---------------------------------------------------------------------------

class _SwiGLUMLP:
  """
  SwiGLU feed-forward network: x → Linear → SiLU(gate) * value → Linear.

  hidden_features is the intermediate gate dimension; actual hidden width
  is set so that the gating projection is 2 * hidden_features.
  """

  def __init__(self, in_features: int, hidden_features: int):
    _require_tinygrad("_SwiGLUMLP")
    self.fc1 = nn.Linear(in_features, hidden_features * 2)   # gate + value concatenated
    self.fc2 = nn.Linear(hidden_features, in_features)

  def __call__(self, x: "Tensor") -> "Tensor":
    gv = self.fc1(x)                                          # (B, N, 2*hidden)
    mid = gv.shape[-1] // 2
    gate = gv[:, :, :mid].silu()
    val  = gv[:, :, mid:]
    return self.fc2(gate * val)


class _StandardMLP:
  def __init__(self, in_features: int, hidden_features: int):
    _require_tinygrad("_StandardMLP")
    self.fc1 = nn.Linear(in_features, hidden_features)
    self.fc2 = nn.Linear(hidden_features, in_features)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.fc2(self.fc1(x).gelu())


# ---------------------------------------------------------------------------
# EVABlock — single transformer block (pre-norm, optional layer-scale)
# ---------------------------------------------------------------------------

class EVABlock:
  """
  EVA / EVA02 transformer block.

  Pre-norm with optional per-layer learnable scale (``init_values``).
  Uses FlashRoPEAttention internally and optionally SwiGLU MLP.

  Args:
    dim:            embedding dimension
    num_heads:      attention heads
    mlp_ratio:      hidden MLP width = dim * mlp_ratio (used for SwiGLU gate; halved internally)
    swiglu_mlp:     use SwiGLU activation (EVA02 default) vs GELU
    init_values:    layer-scale init value; None disables layer scale
    scale_attn_inner: apply LayerNorm inside attention after projection
  """

  def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 8.0 / 3.0,
    swiglu_mlp: bool = True,
    init_values: Optional[float] = None,
    scale_attn_inner: bool = False,
  ):
    _require_tinygrad("EVABlock")
    self.norm1 = _layer_norm(dim)
    self.attn = FlashRoPEAttention(dim, num_heads, scale_attn_inner=scale_attn_inner)
    self.gamma1 = Tensor([init_values] * dim) if init_values is not None else None

    self.norm2 = _layer_norm(dim)
    hidden = int(dim * mlp_ratio)
    if swiglu_mlp:
      # SwiGLU: gate width = hidden//2 so overall params ≈ GELU MLP with ratio mlp_ratio
      self.mlp: _SwiGLUMLP | _StandardMLP = _SwiGLUMLP(dim, max(1, hidden // 2))
    else:
      self.mlp = _StandardMLP(dim, hidden)
    self.gamma2 = Tensor([init_values] * dim) if init_values is not None else None

  def __call__(self, x: "Tensor", rope_freqs: Optional[np.ndarray] = None) -> "Tensor":
    attn_out = self.attn(self.norm1(x), rope_freqs=rope_freqs)
    if self.gamma1 is not None:
      attn_out = attn_out * self.gamma1
    x = x + attn_out

    mlp_out = self.mlp(self.norm2(x))
    if self.gamma2 is not None:
      mlp_out = mlp_out * self.gamma2
    return x + mlp_out


# ---------------------------------------------------------------------------
# PoPE embedding + block
# ---------------------------------------------------------------------------

class PopeEmbedding:
  """
  Polar-coordinate Positional Encoding (PoPE).

  Builds sinusoidal position embeddings for an N-D spatial grid using
  frequency bands, producing a (n_positions, 2 * n_bands * ndim) table
  of concatenated [sin, cos] values.

  This matches villa's PoPEEmbedding but is implemented purely with numpy
  and exposed via get_embed() for use in PopeBlock.

  Args:
    head_dim:    per-head attention dimension
    feat_shape:  spatial grid shape, e.g. (8, 8) or (4, 8, 8)
    temperature: RoPE base temperature (10000)
  """

  def __init__(self, head_dim: int, feat_shape: Tuple[int, ...], temperature: float = 10000.0):
    _require_tinygrad("PopeEmbedding")
    self.head_dim = head_dim
    self.feat_shape = feat_shape
    self._freqs = build_rope_nd_freqs(feat_shape, head_dim, theta=temperature)

  def get_embed(self) -> np.ndarray:
    """Return (n_positions, head_dim//2, 2) sin/cos table."""
    return self._freqs


class _PopeAttention:
  """
  PoPE attention: q/k magnitudes follow softplus; positions encoded via
  phase-shifted sinusoids and a learned per-head phase bias delta.

  The attention score is 2 * (q_cos · k_cos + q_sin · k_sin) / sqrt(D),
  which is the polar-coordinate dot-product.
  """

  def __init__(self, dim: int, num_heads: int, scale_attn_inner: bool = False):
    _require_tinygrad("_PopeAttention")
    assert dim % num_heads == 0
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.qkv = nn.Linear(dim, dim * 3, bias=True)
    self.proj = nn.Linear(dim, dim)
    self.inner_norm = _layer_norm(dim) if scale_attn_inner else None
    # Learned phase bias: (num_heads, head_dim) — clipped to [-2π, 0]
    self.phase_bias = Tensor.zeros(num_heads, self.head_dim)

  def __call__(self, x: "Tensor", rope_freqs: Optional[np.ndarray] = None) -> "Tensor":
    B, N, C = x.shape[0], x.shape[1], x.shape[2]
    H, D = self.num_heads, self.head_dim

    qkv = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]   # each (B, H, N, D)

    if rope_freqs is not None:
      # Build sin/cos from precomputed freqs
      half = D // 2
      sin_t = Tensor(rope_freqs[:N, :, 0]).reshape(1, 1, N, half)  # (1,1,N,half)
      cos_t = Tensor(rope_freqs[:N, :, 1]).reshape(1, 1, N, half)

      # softplus magnitudes
      mu_q = q.softplus()
      mu_k = k.softplus()

      # Phase bias: delta clipped to [-2π, 0]
      delta = self.phase_bias.reshape(1, H, 1, D)
      cos_delta = delta.cos()
      sin_delta = delta.sin()

      # Rotate cos/sin embeddings by delta
      cos_k = cos_t * cos_delta[:, :, :, :half] - sin_t * sin_delta[:, :, :, :half]
      sin_k = sin_t * cos_delta[:, :, :, :half] + cos_t * sin_delta[:, :, :, :half]

      q_cos = mu_q[:, :, :, :half] * cos_t
      q_sin = mu_q[:, :, :, :half] * sin_t
      k_cos = mu_k[:, :, :, :half] * cos_k
      k_sin = mu_k[:, :, :, :half] * sin_k

      q = q_cos.cat(q_sin, dim=-1)    # (B, H, N, D)
      k = k_cos.cat(k_sin, dim=-1)

      # PoPE scaling: sqrt(2) * Q to maintain unit expected attention magnitude
      q = q * math.sqrt(2.0)

    scale = D ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale
    attn = attn.softmax(axis=-1)
    out = (attn @ v).transpose(1, 2).reshape(B, N, C)
    if self.inner_norm is not None:
      out = self.inner_norm(out)
    return self.proj(out)


class PopeBlock:
  """
  Transformer block using PoPE positional encoding.

  Drop-in replacement for EVABlock but uses _PopeAttention instead of
  FlashRoPEAttention.
  """

  def __init__(
    self,
    dim: int,
    num_heads: int,
    mlp_ratio: float = 8.0 / 3.0,
    swiglu_mlp: bool = True,
    init_values: Optional[float] = None,
    scale_attn_inner: bool = False,
  ):
    _require_tinygrad("PopeBlock")
    self.norm1 = _layer_norm(dim)
    self.attn = _PopeAttention(dim, num_heads, scale_attn_inner=scale_attn_inner)
    self.gamma1 = Tensor([init_values] * dim) if init_values is not None else None

    self.norm2 = _layer_norm(dim)
    hidden = int(dim * mlp_ratio)
    self.mlp: _SwiGLUMLP | _StandardMLP = _SwiGLUMLP(dim, max(1, hidden // 2)) if swiglu_mlp else _StandardMLP(dim, hidden)
    self.gamma2 = Tensor([init_values] * dim) if init_values is not None else None

  def __call__(self, x: "Tensor", rope_freqs: Optional[np.ndarray] = None) -> "Tensor":
    attn_out = self.attn(self.norm1(x), rope_freqs=rope_freqs)
    if self.gamma1 is not None:
      attn_out = attn_out * self.gamma1
    x = x + attn_out

    mlp_out = self.mlp(self.norm2(x))
    if self.gamma2 is not None:
      mlp_out = mlp_out * self.gamma2
    return x + mlp_out


# ---------------------------------------------------------------------------
# VisionTransformer (ViT) backbone using EVABlocks
# ---------------------------------------------------------------------------

class VisionTransformer:
  """
  Vision Transformer backbone for 2-D or 3-D inputs.

  Accepts pre-embedded patch tokens (B, N, embed_dim) and applies a stack
  of EVABlocks with optional N-D RoPE position encoding.

  This is the tinygrad equivalent of villa's Eva class (encoder-only, no
  classification head or patch-embed — those live in the caller).

  Args:
    embed_dim:       token embedding dimension
    depth:           number of transformer blocks
    num_heads:       attention heads
    feat_shape:      spatial patch grid shape for RoPE, e.g. (8, 8) or (4, 8, 8)
    use_abs_pos_emb: add learned absolute position embedding
    use_rope:        add N-D rotary position encoding to Q/K
    mlp_ratio:       MLP hidden width multiplier
    swiglu_mlp:      use SwiGLU MLP (EVA02)
    init_values:     layer-scale init; None = no layer scale
    pos_emb_type:    'rope' or 'pope' — selects block type
    scale_attn_inner: inner LayerNorm in attention
  """

  def __init__(
    self,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    feat_shape: Optional[Tuple[int, ...]] = None,
    use_abs_pos_emb: bool = True,
    use_rope: bool = True,
    mlp_ratio: float = 8.0 / 3.0,
    swiglu_mlp: bool = True,
    init_values: Optional[float] = None,
    pos_emb_type: str = "rope",
    scale_attn_inner: bool = False,
  ):
    _require_tinygrad("VisionTransformer")
    self.embed_dim = embed_dim
    self.feat_shape = feat_shape

    n_patches = int(np.prod(feat_shape)) if feat_shape is not None else 0
    self.pos_embed = Tensor.zeros(1, n_patches, embed_dim) if use_abs_pos_emb and n_patches > 0 else None

    # Precompute RoPE freqs once
    self._rope_freqs: Optional[np.ndarray] = None
    if use_rope and feat_shape is not None:
      head_dim = embed_dim // num_heads
      self._rope_freqs = build_rope_nd_freqs(feat_shape, head_dim)

    use_pope = pos_emb_type.lower() == "pope"
    if use_pope:
      self.blocks = [
        PopeBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, swiglu_mlp=swiglu_mlp,
                  init_values=init_values, scale_attn_inner=scale_attn_inner)
        for _ in range(depth)
      ]
    else:
      self.blocks = [
        EVABlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, swiglu_mlp=swiglu_mlp,
                 init_values=init_values, scale_attn_inner=scale_attn_inner)
        for _ in range(depth)
      ]
    self.norm = _layer_norm(embed_dim)

  def __call__(self, x: "Tensor") -> "Tensor":
    """
    Args:
      x: (B, N, embed_dim) token sequence

    Returns:
      (B, N, embed_dim) normalised token sequence
    """
    if self.pos_embed is not None:
      x = x + self.pos_embed
    for blk in self.blocks:
      x = blk(x, rope_freqs=self._rope_freqs)
    return self.norm(x)


# ---------------------------------------------------------------------------
# 3-D patch embedding (tinygrad has no Conv3d — use linear projection)
# ---------------------------------------------------------------------------

class _PatchEmbed3d:
  """
  Non-overlapping 3-D patch embedding via linear projection.

  Equivalent to Conv3d(in_ch, embed_dim, kernel_size=p, stride=p) but
  implemented as a matrix multiply since tinygrad lacks Conv3d.

  Extracts non-overlapping (p, p, p) patches from (B, C, D, H, W) input and
  projects each patch's C*p^3 voxels to embed_dim.
  """

  def __init__(self, in_channels: int, embed_dim: int, patch_size: int):
    _require_tinygrad("_PatchEmbed3d")
    self.in_channels = in_channels
    self.embed_dim = embed_dim
    self.patch_size = patch_size
    self.proj = nn.Linear(in_channels * patch_size ** 3, embed_dim, bias=True)

  def __call__(self, x: "Tensor") -> "Tensor":
    """(B, C, D, H, W) → (B, embed_dim, fd, fh, fw)"""
    B, C, D, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]
    p = self.patch_size
    fd, fh, fw = D // p, H // p, W // p

    # Reshape into non-overlapping patches: (B, C, fd, p, fh, p, fw, p)
    x = x.reshape(B, C, fd, p, fh, p, fw, p)
    # → (B, fd, fh, fw, C, p, p, p) → (B, fd*fh*fw, C*p^3)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, fd * fh * fw, C * p * p * p)
    # Linear projection → (B, fd*fh*fw, embed_dim)
    x = self.proj(x)
    # Reshape to spatial format (B, embed_dim, fd, fh, fw)
    return x.permute(0, 2, 1).reshape(B, self.embed_dim, fd, fh, fw)


# ---------------------------------------------------------------------------
# Primus encoder / decoder (patch embedding + ViT + patch decoding)
# ---------------------------------------------------------------------------

class PrimusEncoder:
  """
  Primus patch encoder: Conv-based 8x downsampling → linear projection → ViT.

  Matches villa's PatchEmbed_deeper + Eva encoder sequence.  For simplicity
  the patch embedding uses a single Conv2d/3d (stride 8) rather than the
  three-level residual stack in the original (which requires nnU-Net's
  building blocks).

  Args:
    in_channels:  input image channels
    embed_dim:    output token dimension
    patch_size:   spatial downsampling factor (must equal 8 for Primus compatibility)
    input_shape:  full spatial input shape
    depth:        ViT encoder depth
    num_heads:    attention heads
    feat_shape:   spatial token grid (derived from input_shape / patch_size if None)
  """

  def __init__(
    self,
    in_channels: int = 1,
    embed_dim: int = 396,
    patch_size: int = 8,
    input_shape: Tuple[int, ...] = (64, 64),
    depth: int = 12,
    num_heads: int = 6,
    feat_shape: Optional[Tuple[int, ...]] = None,
    swiglu_mlp: bool = True,
    init_values: Optional[float] = 0.1,
    scale_attn_inner: bool = True,
    pos_emb_type: str = "rope",
  ):
    _require_tinygrad("PrimusEncoder")
    self.ndim = len(input_shape)
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    if feat_shape is None:
      feat_shape = tuple(s // patch_size for s in input_shape)
    self.feat_shape = feat_shape

    # Patch embedding: stride-p convolution (2-D) or linear projection (3-D, no Conv3d in tinygrad)
    if self.ndim == 2:
      self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    else:
      self.patch_embed = _PatchEmbed3d(in_channels, embed_dim, patch_size)

    self.vit = VisionTransformer(
      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
      feat_shape=feat_shape, use_abs_pos_emb=True, use_rope=True,
      swiglu_mlp=swiglu_mlp, init_values=init_values,
      scale_attn_inner=scale_attn_inner, pos_emb_type=pos_emb_type,
    )

  def __call__(self, x: "Tensor") -> "Tensor":
    """
    Args:
      x: (B, C, *spatial) image

    Returns:
      (B, n_patches, embed_dim) token sequence
    """
    z = self.patch_embed(x)         # (B, embed_dim, *feat_shape)
    B = z.shape[0]
    # Flatten spatial → sequence
    z = z.reshape(B, self.embed_dim, -1).permute(0, 2, 1)   # (B, N, embed_dim)
    return self.vit(z)                                        # (B, N, embed_dim)


class PrimusDecoder:
  """
  Primus patch decoder: linear projection → pixel-shuffle-style reshape → Conv upsampling.

  Reconstructs the spatial output at the original resolution.

  Args:
    embed_dim:    input token dimension (must match encoder)
    patch_size:   spatial upsampling factor (must equal 8 for Primus compatibility)
    out_channels: segmentation output classes
    input_shape:  full spatial resolution of the original input
  """

  def __init__(
    self,
    embed_dim: int = 396,
    patch_size: int = 8,
    out_channels: int = 2,
    input_shape: Tuple[int, ...] = (64, 64),
  ):
    _require_tinygrad("PrimusDecoder")
    self.ndim = len(input_shape)
    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.out_channels = out_channels
    self.feat_shape = tuple(s // patch_size for s in input_shape)

    # Project tokens to voxel-space representation then upsample
    voxels_per_patch = patch_size ** self.ndim
    self.proj = nn.Linear(embed_dim, out_channels * voxels_per_patch)

  def __call__(self, tokens: "Tensor") -> "Tensor":
    """
    Args:
      tokens: (B, N, embed_dim)

    Returns:
      (B, out_channels, *input_shape) spatial prediction
    """
    B = tokens.shape[0]
    ps = self.patch_size
    oc = self.out_channels

    # Project each token to out_channels * ps^ndim values
    x = self.proj(tokens)            # (B, N, oc * ps^ndim)

    if self.ndim == 2:
      fh, fw = self.feat_shape
      # Reshape: (B, fh, fw, oc, ps, ps) then permute to (B, oc, fh*ps, fw*ps)
      x = x.reshape(B, fh, fw, oc, ps, ps)
      x = x.permute(0, 3, 1, 4, 2, 5)   # (B, oc, fh, ps, fw, ps)
      x = x.reshape(B, oc, fh * ps, fw * ps)
    else:
      fd, fh, fw = self.feat_shape
      x = x.reshape(B, fd, fh, fw, oc, ps, ps, ps)
      x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)   # (B, oc, fd, ps, fh, ps, fw, ps)
      x = x.reshape(B, oc, fd * ps, fh * ps, fw * ps)
    return x


# ---------------------------------------------------------------------------
# Vesuvius3dViTModel — 3-D Vision Transformer for volume processing
# ---------------------------------------------------------------------------

class Vesuvius3dViTModel:
  """
  3-D Vision Transformer for scroll volume processing, modelled on villa's
  Primus architecture but exposed as a standalone feature extractor.

  Takes volumetric input (B, C, D, H, W), patchifies with a 3-D stride-p
  convolution, processes with a ViT encoder, and returns a flat feature
  vector or spatial token map depending on ``return_tokens``.

  Args:
    in_channels:   input image channels
    patch_size:    3-D patch size (cubic)
    embed_dim:     token embedding dimension
    depth:         transformer depth
    num_heads:     attention heads
    input_shape:   (D, H, W) spatial dimensions of the input volume;
                   required so that RoPE freqs and abs-pos-embed can be precomputed
    out_channels:  if > 0, a linear projection head maps pooled features to this dim
    return_tokens: if True return (B, N, embed_dim) tokens, else return (B, embed_dim)
    pos_emb_type:  'rope' (default) or 'pope'
  """

  def __init__(
    self,
    in_channels: int = 1,
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    input_shape: Tuple[int, int, int] = (64, 64, 64),
    out_channels: int = 0,
    return_tokens: bool = False,
    pos_emb_type: str = "rope",
  ):
    _require_tinygrad("Vesuvius3dViTModel")
    assert len(input_shape) == 3, "input_shape must be (D, H, W)"
    assert all(s % patch_size == 0 for s in input_shape), \
      f"All spatial dims {input_shape} must be divisible by patch_size {patch_size}"

    self.patch_size = patch_size
    self.embed_dim = embed_dim
    self.return_tokens = return_tokens
    self.feat_shape: Tuple[int, int, int] = tuple(s // patch_size for s in input_shape)  # type: ignore[assignment]

    self.patch_embed = _PatchEmbed3d(in_channels, embed_dim, patch_size)
    self.vit = VisionTransformer(
      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
      feat_shape=self.feat_shape, use_abs_pos_emb=True, use_rope=True,
      swiglu_mlp=True, pos_emb_type=pos_emb_type,
    )
    self.head = nn.Linear(embed_dim, out_channels) if out_channels > 0 else None

  def __call__(self, x: "Tensor") -> "Tensor":
    """
    Args:
      x: (B, C, D, H, W) float32 volume

    Returns:
      if return_tokens: (B, N, embed_dim) token sequence
      else:             (B, embed_dim) mean-pooled features
                        or (B, out_channels) if out_channels > 0
    """
    B = x.shape[0]
    z = self.patch_embed(x)                            # (B, embed_dim, fd, fh, fw)
    z = z.reshape(B, self.embed_dim, -1).permute(0, 2, 1)   # (B, N, embed_dim)
    z = self.vit(z)                                    # (B, N, embed_dim)

    if self.return_tokens:
      return z

    # Global average pool over tokens
    feats = z.mean(axis=1)                             # (B, embed_dim)
    if self.head is not None:
      feats = self.head(feats)
    return feats
