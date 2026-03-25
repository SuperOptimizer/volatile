"""
YoussefMAE — 3-D Masked Autoencoder for volume pre-training (tinygrad port).

Architecture (from villa/vesuvius neural_tracing/nets/youssef_mae.py):
  • Patch embedding: rearrange (B,C,D,H,W) → (B, N, patch_dim), project to `dim`
  • Encoder: Transformer on *visible* patches (random mask_ratio removed)
  • Decoder: full positional grid, masked positions filled with learnable mask token,
    light transformer → linear head → pixel reconstruction
  • Loss: MSE over masked patches only (normalised by patch mean/std, MAE-style)

Usage::

  model = YoussefMAE(encoder_dim=384, depth=6, heads=6, decoder_dim=256)
  loss, recon = model(x)              # x: (B, C, D, H, W)
  enc_tokens  = model.forward_encoder(x, mask)   # (B, N_vis, dim)
  recon_full  = model.forward_decoder(enc_tokens, mask)  # (B, N, patch_dim)
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn, dtypes
  from tinygrad.nn.state import get_parameters
  _TG = True
except ImportError:
  _TG = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_tg():
  if not _TG:
    raise ImportError("tinygrad is required for YoussefMAE")


# Simple multi-head self-attention (no flash; pure tinygrad ops)
class _Attention:
  def __init__(self, dim: int, heads: int):
    _check_tg()
    self.heads = heads
    self.dh    = dim // heads
    self.scale = self.dh ** -0.5
    self.norm  = nn.LayerNorm(dim)
    self.qkv   = nn.Linear(dim, dim * 3, bias=False)
    self.proj  = nn.Linear(dim, dim)

  def __call__(self, x: "Tensor") -> "Tensor":
    B, N, C = x.shape
    x = self.norm(x)
    qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.dh).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]                         # (B, H, N, dh)
    attn = (q @ k.transpose(-2, -1)) * self.scale             # (B, H, N, N)
    attn = attn.softmax(axis=-1)
    out  = (attn @ v).transpose(1, 2).reshape(B, N, C)        # (B, N, C)
    return self.proj(out)


class _FFN:
  def __init__(self, dim: int, hidden: int):
    _check_tg()
    self.norm = nn.LayerNorm(dim)
    self.fc1  = nn.Linear(dim, hidden)
    self.fc2  = nn.Linear(hidden, dim)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.fc2(self.fc1(self.norm(x)).gelu())


class _Block:
  def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
    _check_tg()
    self.attn = _Attention(dim, heads)
    self.ffn  = _FFN(dim, int(dim * mlp_ratio))

  def __call__(self, x: "Tensor") -> "Tensor":
    x = x + self.attn(x)
    x = x + self.ffn(x)
    return x


# ---------------------------------------------------------------------------
# YoussefMAE
# ---------------------------------------------------------------------------

class YoussefMAE:
  """Youssef's MAE variant for 3-D volume pre-training (tinygrad).

  Args:
    in_channels:   volume channels (default 1)
    image_size:    spatial extent (D=H=W assumed equal, default 64)
    patch_size:    cubic patch edge (default 16)
    encoder_dim:   encoder token dimension (default 384)
    depth:         encoder transformer depth (default 6)
    heads:         attention heads (default 6)
    decoder_dim:   decoder token dimension (default 256)
    decoder_depth: decoder transformer depth (default 4)
    mlp_ratio:     FFN hidden-to-dim ratio (default 4.0)
    mask_ratio:    fraction of patches masked during training (default 0.75)
  """

  def __init__(
    self,
    in_channels:   int   = 1,
    image_size:    int   = 64,
    patch_size:    int   = 16,
    encoder_dim:   int   = 384,
    depth:         int   = 6,
    heads:         int   = 6,
    decoder_dim:   int   = 256,
    decoder_depth: int   = 4,
    mlp_ratio:     float = 4.0,
    mask_ratio:    float = 0.75,
  ):
    _check_tg()
    assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
    self.in_ch      = in_channels
    self.image_size = image_size
    self.patch_size = patch_size
    self.mask_ratio = mask_ratio

    p = patch_size
    self.n_per_axis = image_size // p
    self.n_patches  = self.n_per_axis ** 3
    self.patch_dim  = in_channels * p * p * p

    # --- Encoder ---
    self.patch_norm   = nn.LayerNorm(self.patch_dim)
    self.patch_embed  = nn.Linear(self.patch_dim, encoder_dim)
    self.enc_pos_emb  = Tensor.randn(1, self.n_patches, encoder_dim) * 0.02
    self.enc_blocks   = [_Block(encoder_dim, heads, mlp_ratio) for _ in range(depth)]
    self.enc_norm     = nn.LayerNorm(encoder_dim)

    # --- Decoder ---
    self.mask_token   = Tensor.randn(1, 1, decoder_dim) * 0.02
    self.enc_to_dec   = nn.Linear(encoder_dim, decoder_dim)
    self.dec_pos_emb  = Tensor.randn(1, self.n_patches, decoder_dim) * 0.02
    self.dec_blocks   = [_Block(decoder_dim, max(1, decoder_dim // 64), mlp_ratio)
                         for _ in range(decoder_depth)]
    self.dec_norm     = nn.LayerNorm(decoder_dim)
    self.dec_head     = nn.Linear(decoder_dim, self.patch_dim)

  # ------------------------------------------------------------------
  # Patch utilities
  # ------------------------------------------------------------------

  def _patchify(self, x: "Tensor") -> "Tensor":
    """(B,C,D,H,W) → (B, N, patch_dim)."""
    B, C, D, H, W = x.shape
    p = self.patch_size
    n = self.n_per_axis
    # reshape: (B, C, n,p, n,p, n,p) → (B, n,n,n, C*p*p*p)
    x = x.reshape(B, C, n, p, n, p, n, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)        # (B, n,n,n, C,p,p,p)
    return x.reshape(B, n * n * n, C * p * p * p)  # (B, N, patch_dim)

  def _unpatchify(self, tokens: "Tensor") -> "Tensor":
    """(B, N, patch_dim) → (B, C, D, H, W)."""
    B = tokens.shape[0]
    p, n, C = self.patch_size, self.n_per_axis, self.in_ch
    x = tokens.reshape(B, n, n, n, C, p, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)        # (B, C, n,p, n,p, n,p)
    return x.reshape(B, C, n * p, n * p, n * p)

  @staticmethod
  def _random_mask(n_patches: int, mask_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (ids_keep, ids_mask) — numpy arrays of patch indices."""
    n_keep = max(1, int(n_patches * (1 - mask_ratio)))
    perm   = np.random.permutation(n_patches)
    return perm[:n_keep], perm[n_keep:]

  # ------------------------------------------------------------------
  # Encoder
  # ------------------------------------------------------------------

  def forward_encoder(self, x: "Tensor", ids_keep: "Optional[np.ndarray]" = None
                      ) -> Tuple["Tensor", "Optional[np.ndarray]"]:
    """Encode visible patches.  Returns (tokens, ids_keep)."""
    patches = self._patchify(x)                           # (B, N, patch_dim)
    patches = self.patch_embed(self.patch_norm(patches))  # (B, N, dim)
    patches = patches + self.enc_pos_emb                  # broadcast over B

    if ids_keep is not None:
      # gather visible subset — index along dim=1
      vis = patches[:, ids_keep, :]                       # (B, N_vis, dim)
    else:
      vis = patches

    for blk in self.enc_blocks:
      vis = blk(vis)
    return self.enc_norm(vis), ids_keep

  # ------------------------------------------------------------------
  # Decoder
  # ------------------------------------------------------------------

  def forward_decoder(self, enc_tokens: "Tensor", ids_keep: "Optional[np.ndarray]",
                      ids_mask: "Optional[np.ndarray]") -> "Tensor":
    """Reconstruct all patch positions.  Returns (B, N, patch_dim)."""
    B = enc_tokens.shape[0]
    N = self.n_patches
    D = self.dec_pos_emb.shape[-1]

    # Project encoder tokens to decoder dimension
    vis = self.enc_to_dec(enc_tokens)      # (B, N_vis, D)

    if ids_keep is None:
      # no masking — pass through directly
      tokens = vis
    else:
      # build full sequence: fill mask positions with learnable mask token
      mask_toks = self.mask_token.expand(B, len(ids_mask), D)
      # allocate and fill (use numpy indexing for positions)
      # We build a list of (idx, tensor) and scatter into a full buffer via cat+sort
      full = Tensor.zeros(B, N, D)  # placeholder
      # Overwrite rows — tinygrad supports gather-style indexing
      # Simpler: build full via concatenation with known sorted order
      all_ids   = np.concatenate([ids_keep, ids_mask])
      all_order = np.argsort(all_ids)
      all_toks  = Tensor.cat(vis, mask_toks, dim=1)    # (B, N, D)
      tokens    = all_toks[:, all_order, :]            # restore original order

    tokens = tokens + self.dec_pos_emb                  # add pos embed

    for blk in self.dec_blocks:
      tokens = blk(tokens)
    return self.dec_head(self.dec_norm(tokens))          # (B, N, patch_dim)

  # ------------------------------------------------------------------
  # Full forward (training)
  # ------------------------------------------------------------------

  def __call__(self, x: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """Returns (loss, reconstructed_volume).

    loss is MSE over masked patches only (patch-normalised as in original MAE).
    """
    ids_keep, ids_mask = self._random_mask(self.n_patches, self.mask_ratio)

    enc_tokens, _ = self.forward_encoder(x, ids_keep)
    pred = self.forward_decoder(enc_tokens, ids_keep, ids_mask)  # (B, N, patch_dim)

    # Target: patchified input, optionally normalised per-patch
    target = self._patchify(x)                          # (B, N, patch_dim)
    mean   = target.mean(axis=-1, keepdim=True)
    var    = ((target - mean) ** 2).mean(axis=-1, keepdim=True)
    target = (target - mean) / (var + 1e-6).sqrt()

    # Loss only on masked patches
    mask_pred   = pred[:, ids_mask, :]
    mask_target = target[:, ids_mask, :]
    loss = ((mask_pred - mask_target) ** 2).mean()

    recon = self._unpatchify(pred)
    return loss, recon

  def parameters(self) -> list:
    return get_parameters(self)
