from __future__ import annotations

"""
Auxiliary task framework for multi-task segmentation training.

Provides:
  - AuxHead: a lightweight prediction head attached to a shared encoder output
  - AuxTask: bundles head + loss function + weight schedule
  - AuxTaskTrainer: training loop that combines a primary segmentation model
    with one or more auxiliary heads (structure tensor, surface normals,
    distance transform, or arbitrary custom tasks)

Auxiliary loss annealing:
  The weight of each auxiliary task's loss can optionally decay over training
  (e.g. from 0.4 → 0.0 over the first half of training) so that auxiliary
  signal guides early learning without dominating the primary objective.

Auxiliary target generators (numpy, no tinygrad needed):
  - structure_tensor_targets:  compute eigenvalues of structure tensor
  - surface_normal_targets:    gradient-based approximate surface normals
  - distance_transform_targets: binary → distance transform (requires scipy)
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

from .train import cosine_annealing_lr, _set_lr, clip_grad_norm, save_checkpoint
from .loss import DiceCELoss


# ---------------------------------------------------------------------------
# Auxiliary target generators (numpy)
# ---------------------------------------------------------------------------

def structure_tensor_targets(image_np: np.ndarray, sigma: float = 1.0) -> np.ndarray:
  """
  Compute 2-D structure tensor eigenvalues for each pixel.

  Args:
    image_np: (H, W) float32 image
    sigma:    Gaussian smoothing sigma for gradient computation

  Returns:
    (2, H, W) float32 — channels are (lambda_max, lambda_min) per pixel,
    normalised to [0, 1].
  """
  H, W = image_np.shape[-2], image_np.shape[-1]
  if image_np.ndim == 3:
    gray = image_np[0]
  else:
    gray = image_np

  # Sobel gradients
  def _sobel_x(arr: np.ndarray) -> np.ndarray:
    k = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    return np.array([[np.convolve(arr[r], k[1], mode='same') for r in range(arr.shape[0])]])

  gx = np.gradient(gray.astype(np.float32), axis=1)
  gy = np.gradient(gray.astype(np.float32), axis=0)

  # Gaussian smooth the outer products
  def _gauss(arr: np.ndarray, s: float) -> np.ndarray:
    if s <= 0:
      return arr
    r = max(1, int(3 * s))
    k1d = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2).astype(np.float32)
    k1d /= k1d.sum()
    out = np.apply_along_axis(lambda row: np.convolve(row, k1d, mode='same'), axis=1, arr=arr)
    out = np.apply_along_axis(lambda col: np.convolve(col, k1d, mode='same'), axis=0, out)
    return out

  Jxx = _gauss(gx * gx, sigma)
  Jyy = _gauss(gy * gy, sigma)
  Jxy = _gauss(gx * gy, sigma)

  # Eigenvalues of 2x2 symmetric matrix [[Jxx, Jxy],[Jxy, Jyy]]
  trace = Jxx + Jyy
  det = Jxx * Jyy - Jxy * Jxy
  disc = np.sqrt(np.maximum(0.0, (trace / 2) ** 2 - det))
  lam_max = trace / 2 + disc
  lam_min = trace / 2 - disc

  # Normalise to [0, 1]
  def _norm(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-9:
      return np.zeros_like(arr)
    return ((arr - mn) / (mx - mn)).astype(np.float32)

  return np.stack([_norm(lam_max), _norm(lam_min)], axis=0)  # (2, H, W)


def surface_normal_targets(image_np: np.ndarray) -> np.ndarray:
  """
  Approximate surface normals from image gradients.

  Treats the image intensity as a height field z = f(x, y) and computes
  the normalised gradient (gx, gy, 1) as a proxy for surface normals.

  Args:
    image_np: (H, W) or (C, H, W) float32

  Returns:
    (3, H, W) float32 with values in [-1, 1] representing (nx, ny, nz).
  """
  if image_np.ndim == 3:
    gray = image_np[0]
  else:
    gray = image_np.astype(np.float32)

  gx = np.gradient(gray, axis=1)
  gy = np.gradient(gray, axis=0)
  gz = np.ones_like(gray)
  norm_len = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-9
  return np.stack([-gx / norm_len, -gy / norm_len, gz / norm_len], axis=0).astype(np.float32)  # (3, H, W)


def distance_transform_targets(label_np: np.ndarray, max_dist: float = 50.0) -> np.ndarray:
  """
  Compute normalised signed distance transform from a binary label map.

  Positive inside the foreground region, negative outside; clipped to
  [-max_dist, max_dist] and normalised to [0, 1].

  Requires scipy.  Falls back to a crude gradient-based approximation if
  scipy is not available.

  Args:
    label_np: (H, W) int or bool binary mask (1 = foreground)
    max_dist: clip distance (pixels)

  Returns:
    (1, H, W) float32 in [0, 1].
  """
  binary = (label_np > 0).astype(np.uint8)
  try:
    from scipy.ndimage import distance_transform_edt
    dist_fg = distance_transform_edt(binary).astype(np.float32)
    dist_bg = distance_transform_edt(1 - binary).astype(np.float32)
    sdt = dist_fg - dist_bg
  except ImportError:
    # Crude fallback: use gradient magnitude as proxy
    sdt = np.gradient(binary.astype(np.float32), axis=0) + np.gradient(binary.astype(np.float32), axis=1)

  sdt = np.clip(sdt, -max_dist, max_dist) / max_dist  # [-1, 1]
  sdt = (sdt + 1.0) / 2.0                              # [0, 1]
  return sdt[np.newaxis].astype(np.float32)             # (1, H, W)


# ---------------------------------------------------------------------------
# Auxiliary head
# ---------------------------------------------------------------------------

class AuxHead:
  """
  Lightweight convolutional auxiliary prediction head.

  Accepts a feature map from a shared encoder and produces a spatial
  prediction of shape (B, out_channels, H, W).

  Args:
    in_channels:  channels of the input feature map
    out_channels: output prediction channels
    hidden_channels: intermediate conv channels (default = in_channels)
  """

  def __init__(self, in_channels: int, out_channels: int, hidden_channels: Optional[int] = None):
    if not _TINYGRAD:
      raise ImportError("tinygrad required for AuxHead")
    hid = hidden_channels if hidden_channels is not None else in_channels
    self.conv1 = nn.Conv2d(in_channels, hid, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(hid, out_channels, kernel_size=1)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.conv2(self.conv1(x).relu())


# ---------------------------------------------------------------------------
# Weight annealing schedule
# ---------------------------------------------------------------------------

def aux_weight_schedule(
  epoch: int,
  total_epochs: int,
  initial_weight: float,
  final_weight: float,
  anneal_end_fraction: float = 0.5,
) -> float:
  """
  Cosine annealing of auxiliary loss weight from `initial_weight` → `final_weight`.

  Annealing occurs over the first `anneal_end_fraction * total_epochs` epochs;
  weight stays at `final_weight` thereafter.

  Args:
    epoch:               current epoch (0-based)
    total_epochs:        total training epochs
    initial_weight:      starting weight (e.g. 0.4)
    final_weight:        ending weight (e.g. 0.0)
    anneal_end_fraction: fraction of total_epochs over which annealing happens

  Returns scalar float weight.
  """
  anneal_end = int(math.ceil(anneal_end_fraction * total_epochs))
  if anneal_end <= 0 or epoch >= anneal_end:
    return final_weight
  frac = epoch / anneal_end
  cos_val = 0.5 * (1.0 + math.cos(math.pi * frac))
  return final_weight + cos_val * (initial_weight - final_weight)


# ---------------------------------------------------------------------------
# AuxTask descriptor
# ---------------------------------------------------------------------------

class AuxTask:
  """
  Bundles an auxiliary prediction head with its loss function and weight schedule.

  Args:
    name:              human-readable task name (e.g. "normals", "dist_transform")
    head:              AuxHead instance attached to the shared encoder output
    loss_fn:           callable (pred, target) → scalar Tensor
    initial_weight:    loss weight at epoch 0
    final_weight:      loss weight at end of annealing (default 0.0)
    anneal_end_fraction: fraction of total epochs for annealing (default 0.5)
  """

  def __init__(
    self,
    name: str,
    head: AuxHead,
    loss_fn: Callable,
    initial_weight: float = 0.3,
    final_weight: float = 0.0,
    anneal_end_fraction: float = 0.5,
  ):
    self.name = name
    self.head = head
    self.loss_fn = loss_fn
    self.initial_weight = initial_weight
    self.final_weight = final_weight
    self.anneal_end_fraction = anneal_end_fraction

  def weight(self, epoch: int, total_epochs: int) -> float:
    return aux_weight_schedule(epoch, total_epochs, self.initial_weight, self.final_weight, self.anneal_end_fraction)


# ---------------------------------------------------------------------------
# MSE auxiliary loss (useful for regression targets like normals/dist)
# ---------------------------------------------------------------------------

class MSELoss:
  """Element-wise mean squared error loss between pred and target Tensors."""

  def __call__(self, pred: "Tensor", target: "Tensor") -> "Tensor":
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    diff = pred - target
    return (diff * diff).mean()


# ---------------------------------------------------------------------------
# AuxTaskTrainer
# ---------------------------------------------------------------------------

class AuxTaskTrainer:
  """
  Multi-task trainer with a primary segmentation model and auxiliary heads.

  Each training batch:
    1. Run primary model forward: logits (B, primary_out_ch, H, W).
    2. Run each aux head forward on a specified feature layer or the input.
    3. Compute primary supervised loss + sum of weighted auxiliary losses.
    4. Backpropagate combined loss; update all parameters jointly.

  The auxiliary weight for each task follows `AuxTask.weight(epoch)` which
  can anneal from a high initial value down to 0 over the first half of training.

  Batch format:
    Each item from `train_loader` must be a dict (or 2-tuple) with:
      - "image":         (B, C, H, W) float32 primary input
      - "label":         (B, H, W) int32 segmentation target
      - task.name:       (B, task_out_ch, H, W) float32 auxiliary target,
                         for each AuxTask in `aux_tasks`

    If an auxiliary target key is absent, that task is silently skipped for
    that batch.

  Args:
    primary_model:  primary segmentation model; called as model(image) → logits
    aux_tasks:      list of AuxTask instances
    train_loader:   iterable of batch dicts (or 2-tuples)
    val_loader:     optional validation iterable
    primary_loss_fn: loss for the primary task (default DiceCELoss)
    learning_rate:  Adam LR
    num_epochs:     total epochs
    grad_clip:      gradient clip norm
    checkpoint_dir: checkpoint directory
    checkpoint_every: checkpoint frequency
    target_key:     key for primary segmentation target (default "label")
    feature_fn:     optional callable(image) → feature_map passed to aux heads
                    instead of the raw image; useful when heads attach to encoder output
    verbose:        print progress
  """

  def __init__(
    self,
    primary_model,
    aux_tasks: List[AuxTask],
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    primary_loss_fn: Optional[Callable] = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    grad_clip: float = 1.0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    target_key: str = "label",
    feature_fn: Optional[Callable] = None,
    verbose: bool = True,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad required for AuxTaskTrainer")

    self.primary_model = primary_model
    self.aux_tasks = aux_tasks
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.primary_loss_fn = primary_loss_fn if primary_loss_fn is not None else DiceCELoss(0.5, 0.5)
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.grad_clip = grad_clip
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_every = checkpoint_every
    self.target_key = target_key
    self.feature_fn = feature_fn
    self.verbose = verbose

    # Collect all parameters
    params = get_parameters(primary_model)
    for task in aux_tasks:
      params = params + get_parameters(task.head)
    self._params = params
    self.optimizer = nn.optim.Adam(params, lr=learning_rate)

    if checkpoint_dir:
      import os
      os.makedirs(checkpoint_dir, exist_ok=True)

  # ------------------------------------------------------------------
  # Batch extraction
  # ------------------------------------------------------------------

  def _extract_batch(self, batch) -> Dict[str, Any]:
    """Normalise batch to a flat dict."""
    if isinstance(batch, dict):
      return {k: np.asarray(v) if not isinstance(v, np.ndarray) else v for k, v in batch.items()}
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
      return {"image": np.asarray(batch[0], dtype=np.float32), self.target_key: np.asarray(batch[1], dtype=np.int32)}
    raise ValueError(f"unsupported batch type {type(batch)}")

  # ------------------------------------------------------------------
  # Main loop
  # ------------------------------------------------------------------

  def train(self) -> Dict[str, List[float]]:
    """Run the multi-task training loop. Returns history dict."""
    history: Dict[str, List[float]] = {"train_primary_loss": [], "lr": []}
    for task in self.aux_tasks:
      history[f"train_{task.name}_loss"] = []
      history[f"{task.name}_weight"] = []

    for epoch in range(self.num_epochs):
      lr = cosine_annealing_lr(self.learning_rate, epoch, self.num_epochs)
      _set_lr(self.optimizer, lr)

      primary_losses: List[float] = []
      aux_loss_sums: Dict[str, List[float]] = {t.name: [] for t in self.aux_tasks}
      Tensor.training = True

      for raw_batch in self.train_loader:
        batch = self._extract_batch(raw_batch)
        image = Tensor(batch["image"].astype(np.float32))
        label = Tensor(batch[self.target_key].astype(np.int32))

        # Primary forward
        primary_pred = self.primary_model(image)
        total_loss = self.primary_loss_fn(primary_pred, label)
        primary_losses.append(float(total_loss.numpy()))

        # Feature map for aux heads (shared encoder output or raw image)
        features = self.feature_fn(image) if self.feature_fn is not None else image

        # Auxiliary forwards
        for task in self.aux_tasks:
          w = task.weight(epoch, self.num_epochs)
          if w <= 0:
            continue
          tgt_key = task.name
          if tgt_key not in batch:
            continue
          tgt_np = batch[tgt_key].astype(np.float32)
          target = Tensor(tgt_np)
          aux_pred = task.head(features)
          # Resize target if spatial dims differ
          if aux_pred.shape[2] != target.shape[-2] or aux_pred.shape[3] != target.shape[-1]:
            target = target.interpolate((aux_pred.shape[2], aux_pred.shape[3]))
          aux_loss = task.loss_fn(aux_pred, target)
          total_loss = total_loss + w * aux_loss
          aux_loss_sums[task.name].append(float(aux_loss.numpy()))

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
          clip_grad_norm(self._params, self.grad_clip)
        self.optimizer.step()

      Tensor.training = False

      mean_primary = float(np.mean(primary_losses)) if primary_losses else 0.0
      history["train_primary_loss"].append(mean_primary)
      history["lr"].append(lr)
      for task in self.aux_tasks:
        w = task.weight(epoch, self.num_epochs)
        history[f"train_{task.name}_loss"].append(float(np.mean(aux_loss_sums[task.name])) if aux_loss_sums[task.name] else 0.0)
        history[f"{task.name}_weight"].append(w)

      if self.verbose:
        aux_str = "  ".join(
          f"{t.name}={np.mean(aux_loss_sums[t.name]):.4f}(w={t.weight(epoch, self.num_epochs):.3f})"
          for t in self.aux_tasks if aux_loss_sums[t.name]
        )
        msg = f"Epoch {epoch + 1}/{self.num_epochs}  primary={mean_primary:.4f}  {aux_str}  lr={lr:.2e}"
        print(msg)

      if self.checkpoint_dir and (epoch + 1) % self.checkpoint_every == 0:
        import os
        ckpt = os.path.join(self.checkpoint_dir, f"aux_epoch{epoch + 1:04d}.safetensors")
        save_checkpoint(self.primary_model, self.optimizer, epoch + 1, ckpt)

    return history

  def predict_aux(self, image_np: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Run all auxiliary heads on a single image batch.

    Args:
      image_np: (B, C, H, W) float32

    Returns dict of task_name → np.ndarray predictions.
    """
    Tensor.training = False
    image = Tensor(image_np.astype(np.float32))
    features = self.feature_fn(image) if self.feature_fn is not None else image
    out = {}
    for task in self.aux_tasks:
      out[task.name] = task.head(features).numpy()
    return out
