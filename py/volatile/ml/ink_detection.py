from __future__ import annotations

"""
3D ink detection pipeline for papyrus scroll analysis.

Ported from villa's TimeSformer / ResNet3D pipeline to tinygrad + volatile UNet.

The key insight: ink is detected by examining a stack of Z-slices (typically
26 layers) centred on the papyrus surface.  Each spatial position on the surface
gets a column of voxels through the depth axis, giving the model local 3-D context
to distinguish ink from vellum and noise.

Data layout:
  Volume:  (D, H, W) float32 — raw CT voxel intensities, typically clipped to [0, 200]
  Surface: (H, W) int   — per-pixel Z-index of the surface (from surface detection)
  Label:   (H, W) uint8 — binary ink ground truth (0 = no-ink, 1 = ink)

Inference tiling:
  The (H, W) surface is partitioned into overlapping tiles.  Each tile is fed
  to the model as a batch of (z_range, tile_h, tile_w) patches.  Predictions are
  blended back with Gaussian weighting to suppress tile boundary artefacts.
"""

import math
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict, safe_save, safe_load
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

from .model import UNet
from .loss import DiceCELoss
from .train import cosine_annealing_lr, _set_lr, clip_grad_norm, save_checkpoint


# ---------------------------------------------------------------------------
# Gaussian blend kernel
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(size: int, sigma: float = 1.0) -> np.ndarray:
  """Return a (size, size) Gaussian kernel normalised to max=1."""
  ax = np.arange(size) - size // 2
  xx, yy = np.meshgrid(ax, ax)
  k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2)).astype(np.float32)
  return k / k.max()


# ---------------------------------------------------------------------------
# Volume column extraction
# ---------------------------------------------------------------------------

def extract_surface_columns(
  volume: np.ndarray,
  surface: np.ndarray,
  z_range: int = 26,
  z_offset: int = 0,
) -> np.ndarray:
  """
  Extract a z_range-deep column of voxels centred on the surface at each (y, x).

  Args:
    volume:   (D, H, W) float32 volume
    surface:  (H, W) int array of per-pixel surface Z indices
    z_range:  number of Z slices to extract (centred on surface Z)
    z_offset: shift applied to all surface Z indices before extraction

  Returns:
    (z_range, H, W) float32 — the stacked depth columns
  """
  D, H, W = volume.shape
  half = z_range // 2
  out = np.zeros((z_range, H, W), dtype=np.float32)
  for iz in range(z_range):
    z_map = (surface + z_offset - half + iz).clip(0, D - 1).astype(np.int32)
    # Fancy-index each row independently using take_along_axis style
    # z_map: (H, W), volume: (D, H, W) → take D[z,h,w] for each (h,w)
    out[iz] = volume[z_map, np.arange(H)[:, None], np.arange(W)[None, :]]
  return out  # (z_range, H, W)


# ---------------------------------------------------------------------------
# Tiling helpers
# ---------------------------------------------------------------------------

def _tile_positions(H: int, W: int, tile_size: int, stride: int) -> List[Tuple[int, int, int, int]]:
  """Return list of (y1, x1, y2, x2) tile positions covering the full (H, W) area."""
  positions = []
  for y1 in range(0, max(1, H - tile_size + 1), stride):
    for x1 in range(0, max(1, W - tile_size + 1), stride):
      positions.append((y1, x1, min(y1 + tile_size, H), min(x1 + tile_size, W)))
  # Ensure last column/row is covered
  if H > tile_size and (H - tile_size) % stride != 0:
    for x1 in range(0, max(1, W - tile_size + 1), stride):
      y1 = H - tile_size
      positions.append((y1, x1, H, min(x1 + tile_size, W)))
  if W > tile_size and (W - tile_size) % stride != 0:
    for y1 in range(0, max(1, H - tile_size + 1), stride):
      x1 = W - tile_size
      positions.append((y1, x1, min(y1 + tile_size, H), W))
  # Corner
  if H > tile_size and W > tile_size:
    positions.append((H - tile_size, W - tile_size, H, W))
  return list(dict.fromkeys(positions))  # deduplicate preserving order


# ---------------------------------------------------------------------------
# Ink detection model (thin wrapper around UNet)
# ---------------------------------------------------------------------------

class InkUNet:
  """
  UNet adapted for ink detection: takes (B, z_range, H, W) input (z_range channels)
  and produces a single-channel logit map (B, 1, H, W).
  """

  def __init__(self, z_range: int = 26, base_channels: int = 32, num_levels: int = 4):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for InkUNet")
    self._unet = UNet(in_channels=z_range, out_channels=1, base_channels=base_channels, num_levels=num_levels)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self._unet(x)  # (B, 1, H, W) logits


# ---------------------------------------------------------------------------
# InkDataset
# ---------------------------------------------------------------------------

class InkDataset:
  """
  Dataset for ink detection training.

  Extracts (z_range, patch_size, patch_size) patches from a surface column
  stack, paired with binary ink labels.

  Args:
    volume_paths:  list of (D, H, W) numpy arrays or paths to .npy files
    label_paths:   list of (H, W) numpy arrays or paths to PNG/npy label files;
                   None entries are allowed (unlabelled volumes are skipped)
    surface_paths: list of (H, W) int surface arrays or paths; if None a flat
                   surface at D//2 is assumed for each volume
    z_range:       number of Z slices to extract per sample (channels)
    patch_size:    spatial patch size (square)
    stride:        stride between patches
    augment:       apply random horizontal/vertical flips during training
    clip_max:      clip voxel intensities to this maximum before normalising
    seed:          RNG seed for reproducible patch sampling
  """

  def __init__(
    self,
    volumes: List[np.ndarray],
    labels: List[Optional[np.ndarray]],
    surfaces: Optional[List[Optional[np.ndarray]]] = None,
    z_range: int = 26,
    patch_size: int = 64,
    stride: int = 64,
    augment: bool = True,
    clip_max: float = 200.0,
    seed: int = 0,
  ):
    self.z_range = z_range
    self.patch_size = patch_size
    self.stride = stride
    self.augment = augment
    self.clip_max = clip_max
    self._rng = np.random.default_rng(seed)

    self._samples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (col_stack, label, position)

    for vi, (vol, lbl) in enumerate(zip(volumes, labels)):
      if lbl is None:
        continue
      vol_np = np.asarray(vol, dtype=np.float32)
      lbl_np = np.asarray(lbl, dtype=np.float32)
      D, H, W = vol_np.shape

      surf_np: np.ndarray
      if surfaces is not None and surfaces[vi] is not None:
        surf_np = np.asarray(surfaces[vi], dtype=np.int32)
      else:
        surf_np = np.full((H, W), D // 2, dtype=np.int32)

      # Clip + normalise to [0, 1]
      col_stack = extract_surface_columns(
        np.clip(vol_np, 0.0, clip_max) / clip_max, surf_np, z_range
      )  # (z_range, H, W)

      for y1, x1, y2, x2 in _tile_positions(H, W, patch_size, stride):
        ph, pw = y2 - y1, x2 - x1
        if ph < patch_size or pw < patch_size:
          continue  # skip border tiles smaller than patch_size
        img_patch = col_stack[:, y1:y2, x1:x2]   # (z_range, patch_size, patch_size)
        lbl_patch = lbl_np[y1:y2, x1:x2]          # (patch_size, patch_size)
        self._samples.append((img_patch, lbl_patch))

  def __len__(self) -> int:
    return len(self._samples)

  def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
    img, lbl = self._samples[idx]
    if self.augment:
      if self._rng.random() > 0.5:
        img = img[:, ::-1, :].copy()
        lbl = lbl[::-1, :].copy()
      if self._rng.random() > 0.5:
        img = img[:, :, ::-1].copy()
        lbl = lbl[:, ::-1].copy()
    return img.astype(np.float32), lbl.astype(np.float32)

  def collate(self, items: List[Tuple[np.ndarray, np.ndarray]]) -> Dict[str, np.ndarray]:
    imgs = np.stack([it[0] for it in items], axis=0)   # (B, z_range, H, W)
    lbls = np.stack([it[1] for it in items], axis=0)   # (B, H, W)
    return {"image": imgs, "label": lbls}

  def batches(self, batch_size: int, shuffle: bool = True) -> Iterable[Dict[str, np.ndarray]]:
    """Yield batches of dicts with keys 'image' and 'label'."""
    idx = np.arange(len(self))
    if shuffle:
      self._rng.shuffle(idx)
    for start in range(0, len(idx), batch_size):
      batch_idx = idx[start:start + batch_size]
      if len(batch_idx) == 0:
        continue
      yield self.collate([self[i] for i in batch_idx])


# ---------------------------------------------------------------------------
# Ink BCE+Dice loss (pixel-level binary classification)
# ---------------------------------------------------------------------------

class _InkLoss:
  """Combined BCE + Dice loss for binary ink segmentation."""

  def __call__(self, logits: "Tensor", target: "Tensor") -> "Tensor":
    # logits: (B, 1, H, W), target: (B, H, W) float in {0, 1}
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    tgt = target.reshape(target.shape[0], 1, target.shape[1], target.shape[2])

    # Binary cross-entropy with logits: -[y log σ(x) + (1-y) log(1-σ(x))]
    bce = (logits.relu() - logits * tgt + (1.0 + (-logits.abs()).exp()).log()).mean()

    # Dice: 2 * TP / (2*TP + FP + FN) computed on sigmoid probabilities
    prob = logits.sigmoid()
    smooth = 1e-6
    inter = (prob * tgt).sum()
    dice_loss = 1.0 - (2.0 * inter + smooth) / (prob.sum() + tgt.sum() + smooth)

    return 0.5 * bce + 0.5 * dice_loss


# ---------------------------------------------------------------------------
# InkDetector
# ---------------------------------------------------------------------------

class InkDetector:
  """
  3-D ink detector for papyrus scrolls.

  Uses a UNet that takes a z_range-channel 2-D patch (one channel per Z-layer
  around the surface) and predicts per-pixel ink probability.

  Args:
    model_type:      architecture selector — currently only 'unet' is supported
                     ('resnet3d' and 'timesformer' are accepted aliases for UNet
                     to maintain API compatibility with villa's naming)
    checkpoint_path: optional path to a .safetensors checkpoint to load
    z_range:         number of Z-slices (channels) fed to the model
    base_channels:   UNet first-level channel count
    num_levels:      UNet encoder depth
  """

  def __init__(
    self,
    model_type: str = 'unet',
    checkpoint_path: Optional[str] = None,
    z_range: int = 26,
    base_channels: int = 32,
    num_levels: int = 4,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for InkDetector")

    self.model_type = model_type  # stored for reference; all use InkUNet internally
    self.z_range = z_range
    self._model = InkUNet(z_range=z_range, base_channels=base_channels, num_levels=num_levels)
    self._loss_fn = _InkLoss()
    self._params = get_parameters(self._model)

    if checkpoint_path is not None:
      self.load_checkpoint(checkpoint_path)

  # ------------------------------------------------------------------
  # Checkpoint I/O
  # ------------------------------------------------------------------

  def save_checkpoint(self, path: str) -> None:
    """Save model weights to a safetensors file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    safe_save(get_state_dict(self._model), path)

  def load_checkpoint(self, path: str) -> None:
    """Load model weights from a safetensors file."""
    sd = safe_load(path)
    load_state_dict(self._model, sd)

  # ------------------------------------------------------------------
  # Inference helpers
  # ------------------------------------------------------------------

  def _infer_patch(self, patch: np.ndarray) -> np.ndarray:
    """
    Run inference on a single (z_range, H, W) patch.

    Returns (H, W) float32 ink probabilities.
    """
    Tensor.training = False
    x = Tensor(patch[np.newaxis].astype(np.float32))  # (1, z_range, H, W)
    logits = self._model(x)                             # (1, 1, H, W)
    prob = logits.sigmoid().numpy()[0, 0]               # (H, W)
    return prob

  def predict_volume_slice(
    self,
    volume: np.ndarray,
    z: int,
    tile_size: int = 64,
    stride: int = 256,
    clip_max: float = 200.0,
  ) -> np.ndarray:
    """
    Run ink detection on a single Z-slice of the volume.

    Extracts a flat surface at depth z (i.e. the z_range channels centred on z),
    tiles the slice, runs inference, and blends with Gaussian weights.

    Args:
      volume:    (D, H, W) float32
      z:         Z index for the flat surface
      tile_size: spatial tile size
      stride:    tile stride (smaller → more overlap → smoother blending)
      clip_max:  voxel intensity clip before normalisation

    Returns:
      (H, W) float32 ink probability map in [0, 1].
    """
    D, H, W = volume.shape
    surface = np.full((H, W), z, dtype=np.int32)
    return self.predict_surface(volume, surface, z_range=self.z_range, stride=stride,
                                tile_size=tile_size, clip_max=clip_max)

  def predict_surface(
    self,
    volume: np.ndarray,
    surface: np.ndarray,
    z_range: Optional[int] = None,
    stride: int = 256,
    tile_size: int = 64,
    clip_max: float = 200.0,
  ) -> np.ndarray:
    """
    Run ink detection on a surface.

    For each surface point the model sees z_range depth slices centred on the
    surface normal.  The result is blended over overlapping tiles with Gaussian
    weighting to suppress tile-boundary artefacts.

    Args:
      volume:    (D, H, W) float32 CT volume
      surface:   (H, W) int array of per-pixel surface Z indices
      z_range:   depth channels (defaults to self.z_range)
      stride:    tile stride in pixels
      tile_size: spatial tile size
      clip_max:  intensity clip maximum before [0,1] normalisation

    Returns:
      (H, W) float32 ink probability map.
    """
    if z_range is None:
      z_range = self.z_range
    D, H, W = volume.shape

    # Build full surface column stack
    vol_norm = np.clip(volume, 0.0, clip_max).astype(np.float32) / clip_max
    col_stack = extract_surface_columns(vol_norm, surface, z_range)  # (z_range, H, W)

    # Gaussian blend kernel
    kernel = _gaussian_kernel_2d(tile_size, sigma=tile_size / 6.0)  # (tile_size, tile_size)

    pred_sum = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for y1, x1, y2, x2 in _tile_positions(H, W, tile_size, stride):
      ph, pw = y2 - y1, x2 - x1
      if ph != tile_size or pw != tile_size:
        continue  # skip under-sized border tiles
      patch = col_stack[:, y1:y2, x1:x2]  # (z_range, tile_size, tile_size)
      prob = self._infer_patch(patch)      # (tile_size, tile_size)
      pred_sum[y1:y2, x1:x2] += (prob * kernel).astype(np.float64)
      weight_sum[y1:y2, x1:x2] += kernel.astype(np.float64)

    # Normalise; leave areas without any tile prediction at 0
    out = np.where(weight_sum > 0, pred_sum / weight_sum, 0.0).astype(np.float32)
    return out

  # ------------------------------------------------------------------
  # Training loop
  # ------------------------------------------------------------------

  def train(
    self,
    dataset: InkDataset,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    grad_clip: float = 1.0,
    val_dataset: Optional[InkDataset] = None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 5,
    verbose: bool = True,
  ) -> Dict[str, List[float]]:
    """
    Train the ink detection model.

    Args:
      dataset:          InkDataset for training
      epochs:           number of training epochs
      lr:               Adam peak learning rate (cosine-annealed)
      batch_size:       training mini-batch size
      grad_clip:        global gradient clip norm (≤ 0 disables)
      val_dataset:      optional validation InkDataset
      checkpoint_dir:   directory to save epoch checkpoints
      checkpoint_every: checkpoint frequency in epochs
      verbose:          print per-epoch progress

    Returns:
      history dict with keys 'train_loss', 'val_loss', 'lr'.
    """
    optimizer = nn.optim.Adam(self._params, lr=lr)
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    if checkpoint_dir:
      os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
      epoch_lr = cosine_annealing_lr(lr, epoch, epochs)
      _set_lr(optimizer, epoch_lr)

      Tensor.training = True
      step_losses: List[float] = []

      for batch in dataset.batches(batch_size, shuffle=True):
        x_np = batch["image"]                              # (B, z_range, H, W)
        y_np = batch["label"]                              # (B, H, W)

        x = Tensor(x_np.astype(np.float32))
        y = Tensor(y_np.astype(np.float32))

        logits = self._model(x)                            # (B, 1, H, W)
        loss = self._loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
          clip_grad_norm(self._params, grad_clip)
        optimizer.step()

        step_losses.append(float(loss.numpy()))

      Tensor.training = False

      mean_loss = float(np.mean(step_losses)) if step_losses else 0.0
      history["train_loss"].append(mean_loss)
      history["lr"].append(epoch_lr)

      val_loss_val: Optional[float] = None
      if val_dataset is not None and len(val_dataset) > 0:
        val_loss_val = self._eval_epoch(val_dataset, batch_size)
        history["val_loss"].append(val_loss_val)

      if verbose:
        msg = f"[InkDetector] Epoch {epoch + 1}/{epochs}  loss={mean_loss:.4f}  lr={epoch_lr:.2e}"
        if val_loss_val is not None:
          msg += f"  val={val_loss_val:.4f}"
        print(msg)

      if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
        ckpt = os.path.join(checkpoint_dir, f"ink_epoch{epoch + 1:04d}.safetensors")
        self.save_checkpoint(ckpt)

    return history

  def _eval_epoch(self, dataset: InkDataset, batch_size: int) -> float:
    Tensor.training = False
    total, n = 0.0, 0
    for batch in dataset.batches(batch_size, shuffle=False):
      x = Tensor(batch["image"].astype(np.float32))
      y = Tensor(batch["label"].astype(np.float32))
      logits = self._model(x)
      loss = self._loss_fn(logits, y)
      total += float(loss.numpy())
      n += 1
    return total / max(1, n)
