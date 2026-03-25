from __future__ import annotations

"""
Masked Autoencoder (MAE) pre-training for tinygrad.

Approach:
  1. Divide the 2-D input (C, H, W) into non-overlapping patches.
  2. Randomly mask a fraction (default 75%) of patch positions.
  3. An encoder processes ONLY the visible patches → latent tokens.
  4. A decoder takes all positions (visible encoded + learnable mask tokens)
     and reconstructs the original pixel values at masked positions.
  5. MSE loss is computed only over masked patches.

After pre-training:
  - Discard the decoder.
  - The encoder weights initialise a downstream segmentation / classification model.

Architecture:
  The encoder and decoder are simple ConvNets (no transformers) to keep the
  implementation pure-Python + tinygrad without attention kernels.  The
  "patch masking" is implemented by zeroing out masked regions in the feature
  map rather than truly dropping tokens, which makes the conv architecture work
  without variable-length sequence handling.

References:
  He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters, get_state_dict
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

from .train import cosine_annealing_lr, _set_lr, clip_grad_norm


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def make_patch_grid(H: int, W: int, patch_h: int, patch_w: int) -> Tuple[int, int, int, int]:
  """
  Return (n_patches_h, n_patches_w, ph, pw).

  Silently clips H/W to be divisible by patch size.
  """
  nh = H // patch_h
  nw = W // patch_w
  return nh, nw, patch_h, patch_w


def random_mask(n_patches: int, mask_ratio: float, rng: np.random.Generator) -> np.ndarray:
  """
  Return a boolean mask array of shape (n_patches,).

  True = masked (to be reconstructed), False = visible.
  """
  n_masked = max(1, int(round(n_patches * mask_ratio)))
  idx = rng.permutation(n_patches)
  mask = np.zeros(n_patches, dtype=bool)
  mask[idx[:n_masked]] = True
  return mask


def patchify(x_np: np.ndarray, patch_h: int, patch_w: int) -> np.ndarray:
  """
  Divide (C, H, W) image into patches.

  Returns (n_patches, C * patch_h * patch_w) with raster scan order.
  """
  C, H, W = x_np.shape
  nh = H // patch_h
  nw = W // patch_w
  x = x_np[:, :nh * patch_h, :nw * patch_w]
  # (C, nh, ph, nw, pw) → (nh*nw, C*ph*pw)
  x = x.reshape(C, nh, patch_h, nw, patch_w)
  x = x.transpose(1, 3, 0, 2, 4)          # (nh, nw, C, ph, pw)
  return x.reshape(nh * nw, C * patch_h * patch_w)


def unpatchify(patches: np.ndarray, C: int, H: int, W: int, patch_h: int, patch_w: int) -> np.ndarray:
  """Inverse of patchify: (n_patches, C*ph*pw) → (C, H, W)."""
  nh = H // patch_h
  nw = W // patch_w
  x = patches.reshape(nh, nw, C, patch_h, patch_w)
  x = x.transpose(2, 0, 3, 1, 4)          # (C, nh, ph, nw, pw)
  return x.reshape(C, nh * patch_h, nw * patch_w)


# ---------------------------------------------------------------------------
# MAE Encoder
# ---------------------------------------------------------------------------

class MAEEncoder:
  """
  Convolutional encoder for Masked Autoencoder pre-training.

  Processes the masked input image (masked regions set to 0) through a stack
  of Conv+ReLU blocks, producing a latent feature map.

  Args:
    in_channels:   input channels (e.g. 1 for grayscale)
    latent_channels: output latent channels
    hidden_channels: intermediate channel width
    depth:         number of Conv+ReLU layers
  """

  def __init__(self, in_channels: int = 1, latent_channels: int = 64, hidden_channels: int = 64, depth: int = 4):
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    layers = []
    ch = in_channels
    for i in range(depth):
      out_ch = hidden_channels if i < depth - 1 else latent_channels
      layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
      ch = out_ch
    self._convs = layers

  def __call__(self, x: "Tensor") -> "Tensor":
    for i, conv in enumerate(self._convs):
      x = conv(x)
      if i < len(self._convs) - 1:
        x = x.relu()
    return x  # (B, latent_channels, H, W)


# ---------------------------------------------------------------------------
# MAE Decoder
# ---------------------------------------------------------------------------

class MAEDecoder:
  """
  Convolutional decoder for Masked Autoencoder pre-training.

  Takes the encoder latent map and reconstructs the original image pixels.
  The final layer outputs `in_channels` channels matching the original input.

  Args:
    latent_channels: input latent channels (must match encoder output)
    out_channels:    reconstruction output channels (= original in_channels)
    hidden_channels: intermediate channel width
    depth:           number of Conv+ReLU layers before final 1x1 projection
  """

  def __init__(self, latent_channels: int = 64, out_channels: int = 1, hidden_channels: int = 64, depth: int = 4):
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    layers = []
    ch = latent_channels
    for i in range(depth):
      out_ch = hidden_channels if i < depth - 1 else hidden_channels
      layers.append(nn.Conv2d(ch, out_ch, kernel_size=3, padding=1))
      ch = out_ch
    self._convs = layers
    self._proj = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

  def __call__(self, z: "Tensor") -> "Tensor":
    x = z
    for conv in self._convs:
      x = conv(x).relu()
    return self._proj(x)  # (B, out_channels, H, W)


# ---------------------------------------------------------------------------
# Full MAE model
# ---------------------------------------------------------------------------

class MaskedAutoencoder:
  """
  Masked Autoencoder combining encoder + decoder.

  Forward pass:
    1. Apply patch mask to input (zero out masked regions).
    2. Encode masked input → latent.
    3. Decode latent → reconstruction.
    4. Return reconstruction and the patch mask.

  The caller computes the MSE loss only over masked patch positions.

  Args:
    in_channels:     image input channels
    patch_h:         patch height in pixels
    patch_w:         patch width in pixels
    mask_ratio:      fraction of patches to mask (default 0.75)
    latent_channels: encoder/decoder latent width
    hidden_channels: conv intermediate width
    encoder_depth:   encoder conv layers
    decoder_depth:   decoder conv layers
    seed:            RNG seed for reproducible masking during tests
  """

  def __init__(
    self,
    in_channels: int = 1,
    patch_h: int = 16,
    patch_w: int = 16,
    mask_ratio: float = 0.75,
    latent_channels: int = 64,
    hidden_channels: int = 64,
    encoder_depth: int = 4,
    decoder_depth: int = 4,
    seed: Optional[int] = None,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    self.in_channels = in_channels
    self.patch_h = patch_h
    self.patch_w = patch_w
    self.mask_ratio = mask_ratio
    self._rng = np.random.default_rng(seed)

    self.encoder = MAEEncoder(in_channels, latent_channels, hidden_channels, encoder_depth)
    self.decoder = MAEDecoder(latent_channels, in_channels, hidden_channels, decoder_depth)

  def _build_mask_tensor(self, B: int, C: int, H: int, W: int) -> "Tensor":
    """Build a spatial mask (B, 1, H, W) — 0 = masked, 1 = visible."""
    nh, nw, ph, pw = make_patch_grid(H, W, self.patch_h, self.patch_w)
    n_patches = nh * nw
    mask_np = np.ones((B, 1, H, W), dtype=np.float32)
    for b in range(B):
      patch_mask = random_mask(n_patches, self.mask_ratio, self._rng)  # True = masked
      for pi in range(n_patches):
        if patch_mask[pi]:
          py = (pi // nw) * ph
          px = (pi % nw) * pw
          mask_np[b, 0, py:py + ph, px:px + pw] = 0.0
    return Tensor(mask_np)

  def __call__(self, x: "Tensor") -> Tuple["Tensor", "Tensor"]:
    """
    Forward pass.

    Returns:
      recon: (B, C, H, W) full reconstruction
      mask:  (B, 1, H, W) float — 0 = masked patch, 1 = visible patch
    """
    B, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
    mask = self._build_mask_tensor(B, C, H, W)
    masked_input = x * mask              # zero out masked patches
    latent = self.encoder(masked_input)
    recon = self.decoder(latent)
    return recon, mask


def mae_loss(recon: "Tensor", original: "Tensor", mask: "Tensor") -> "Tensor":
  """
  MSE reconstruction loss evaluated only on masked patch positions.

  Args:
    recon:    (B, C, H, W) model reconstruction
    original: (B, C, H, W) original un-masked input
    mask:     (B, 1, H, W) float — 0 = masked, 1 = visible

  Returns scalar Tensor.
  """
  inv_mask = (1.0 - mask)  # 1 where masked
  diff = (recon - original) * inv_mask
  n_masked = inv_mask.sum() + 1e-7
  return (diff * diff).sum() / n_masked


# ---------------------------------------------------------------------------
# MAE Pre-trainer
# ---------------------------------------------------------------------------

class MAEPretrainer:
  """
  Self-supervised Masked Autoencoder pre-training loop.

  Trains the encoder+decoder to reconstruct randomly masked image patches.
  After pre-training, the encoder weights can be transferred to a downstream
  segmentation model by copying the matching parameter tensors.

  Args:
    mae:            MaskedAutoencoder instance
    train_loader:   iterable of image_np arrays (no labels required);
                    each item is (B, C, H, W) or (C, H, W) float32
    val_loader:     optional validation iterable
    learning_rate:  Adam LR
    num_epochs:     total pre-training epochs
    grad_clip:      gradient clip norm (≤ 0 disables)
    verbose:        print progress
  """

  def __init__(
    self,
    mae: MaskedAutoencoder,
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    grad_clip: float = 1.0,
    verbose: bool = True,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad required for MAEPretrainer")
    self.mae = mae
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.grad_clip = grad_clip
    self.verbose = verbose

    enc_params = get_parameters(mae.encoder)
    dec_params = get_parameters(mae.decoder)
    self._params = enc_params + dec_params
    self.optimizer = nn.optim.Adam(self._params, lr=learning_rate)

  def _to_batch(self, item) -> np.ndarray:
    """Normalise an item from the loader to (B, C, H, W) float32."""
    if isinstance(item, dict):
      arr = np.asarray(item.get("image", item.get("img")), dtype=np.float32)
    elif isinstance(item, (list, tuple)):
      arr = np.asarray(item[0], dtype=np.float32)
    else:
      arr = np.asarray(item, dtype=np.float32)
    if arr.ndim == 2:          # (H, W)
      arr = arr[np.newaxis, np.newaxis]
    elif arr.ndim == 3:        # (C, H, W) or (B, H, W) — treat as (B, 1, H, W)
      arr = arr[:, np.newaxis] if arr.shape[0] > 1 else arr[np.newaxis]
    return arr  # (B, C, H, W)

  def train(self) -> Dict[str, List[float]]:
    """Run the MAE pre-training loop.  Returns history dict."""
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(self.num_epochs):
      lr = cosine_annealing_lr(self.learning_rate, epoch, self.num_epochs)
      _set_lr(self.optimizer, lr)

      epoch_losses = []
      Tensor.training = True

      for item in self.train_loader:
        x_np = self._to_batch(item)
        x = Tensor(x_np)
        recon, mask = self.mae(x)
        loss = mae_loss(recon, x, mask)

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip > 0:
          clip_grad_norm(self._params, self.grad_clip)
        self.optimizer.step()
        epoch_losses.append(float(loss.numpy()))

      Tensor.training = False

      mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
      history["train_loss"].append(mean_loss)
      history["lr"].append(lr)

      val_loss = None
      if self.val_loader is not None:
        val_loss = self._run_val_epoch()
        history["val_loss"].append(val_loss)

      if self.verbose:
        msg = f"[MAE] Epoch {epoch + 1}/{self.num_epochs}  loss={mean_loss:.4f}  lr={lr:.2e}"
        if val_loss is not None:
          msg += f"  val={val_loss:.4f}"
        print(msg)

    return history

  def _run_val_epoch(self) -> float:
    Tensor.training = False
    total, n = 0.0, 0
    for item in self.val_loader:
      x_np = self._to_batch(item)
      x = Tensor(x_np)
      recon, mask = self.mae(x)
      loss = mae_loss(recon, x, mask)
      total += float(loss.numpy())
      n += 1
    return total / max(1, n)

  def encoder_state_dict(self) -> Dict[str, np.ndarray]:
    """Return a copy of the trained encoder weights for downstream transfer."""
    return {k: v.numpy().copy() for k, v in get_state_dict(self.mae.encoder).items()}
