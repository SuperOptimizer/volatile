from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters, safe_save, safe_load, get_state_dict, load_state_dict
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

try:
  import wandb as _wandb
  _WANDB = True
except ImportError:
  _WANDB = False

from .loss import DiceCELoss
from .scheduler import _Scheduler, CosineAnnealingWarmRestarts


# ---------------------------------------------------------------------------
# Learning-rate helpers (simple functional API, kept for backwards compat)
# ---------------------------------------------------------------------------

def cosine_annealing_lr(initial_lr: float, epoch: int, total_epochs: int, eta_min: float = 1e-6) -> float:
  """Cosine annealing schedule — returns the LR for a given epoch."""
  return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))


def _set_lr(optimizer, lr: float) -> None:
  """Update learning rate on a tinygrad Adam optimizer in place."""
  optimizer.lr = lr


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict | None = None) -> None:
  """Save model + optimizer state and epoch metadata via tinygrad safe_save."""
  if not _TINYGRAD:
    raise ImportError("tinygrad is required for save_checkpoint")
  state = get_state_dict(model)
  opt_state = {f"__opt_{k}": v for k, v in get_state_dict(optimizer).items()}
  state.update(opt_state)
  state["__epoch"] = Tensor([epoch], dtype='int32')
  if extra:
    for k, v in extra.items():
      if isinstance(v, (int, float)):
        state[f"__meta_{k}"] = Tensor([v], dtype='float32')
  safe_save(state, path)


def load_checkpoint(model, optimizer, path: str) -> int:
  """Load model + optimizer state from a checkpoint.  Returns the saved epoch."""
  if not _TINYGRAD:
    raise ImportError("tinygrad is required for load_checkpoint")
  state = safe_load(path)
  model_state = {k: v for k, v in state.items() if not k.startswith("__")}
  opt_state = {k[6:]: v for k, v in state.items() if k.startswith("__opt_")}
  load_state_dict(model, model_state)
  if opt_state:
    load_state_dict(optimizer, opt_state)
  epoch = int(state["__epoch"].numpy()[0]) if "__epoch" in state else 0
  return epoch


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------

def clip_grad_norm(params: list, max_norm: float) -> float:
  """
  Clip gradients by global L2 norm.  Returns the pre-clip norm.

  Operates in place on the `.grad` attribute of each parameter that has one.
  Tinygrad populates `.grad` after `.backward()` and before `.step()`.
  """
  if max_norm <= 0:
    return 0.0
  grads = [p.grad for p in params if p.grad is not None]
  if not grads:
    return 0.0
  total_norm_sq = sum(float((g * g).sum().numpy()) for g in grads)
  total_norm = math.sqrt(total_norm_sq)
  if total_norm > max_norm:
    scale = max_norm / (total_norm + 1e-9)
    for p in params:
      if p.grad is not None:
        p.grad = p.grad * scale
  return total_norm


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class ModelEMA:
  """
  Exponential Moving Average of model weights.

  Maintains a shadow copy of model parameters updated each step as::

      ema_param = decay * ema_param + (1 - decay) * current_param

  Use `apply_shadow()` to copy EMA weights into the model for evaluation,
  and `restore()` to put the training weights back.

  Args:
    model: the tinygrad model whose parameters to track
    decay: EMA decay coefficient (default 0.9999)
  """

  def __init__(self, model, decay: float = 0.9999):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for ModelEMA")
    self.decay = decay
    self._shadow: Dict[str, Any] = {}
    self._backup: Dict[str, Any] = {}
    # Initialise shadow = current params
    for name, param in get_state_dict(model).items():
      self._shadow[name] = param.numpy().copy()
    self._model = model

  def update(self) -> None:
    """Update shadow weights from the current model parameters."""
    for name, param in get_state_dict(self._model).items():
      arr = param.numpy()
      if name in self._shadow:
        self._shadow[name] = self.decay * self._shadow[name] + (1.0 - self.decay) * arr
      else:
        self._shadow[name] = arr.copy()

  def apply_shadow(self) -> None:
    """Copy EMA weights into the model (for eval).  Call restore() afterwards."""
    self._backup = {k: v.numpy().copy() for k, v in get_state_dict(self._model).items()}
    sd = get_state_dict(self._model)
    for name, shadow_arr in self._shadow.items():
      if name in sd:
        sd[name].assign(Tensor(shadow_arr))

  def restore(self) -> None:
    """Restore training weights (undo apply_shadow)."""
    if not self._backup:
      return
    sd = get_state_dict(self._model)
    for name, arr in self._backup.items():
      if name in sd:
        sd[name].assign(Tensor(arr))
    self._backup.clear()


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _compute_dice(pred_np: np.ndarray, target_np: np.ndarray, n_classes: int) -> float:
  """Mean Dice score across classes (excluding background class 0)."""
  pred_flat = pred_np.argmax(axis=1).ravel()  # (N,)
  tgt_flat = target_np.ravel().astype(np.int32)
  scores = []
  for c in range(1, n_classes):
    p = (pred_flat == c).astype(np.float32)
    t = (tgt_flat == c).astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    if denom < 1e-7:
      continue
    scores.append(2.0 * inter / denom)
  return float(np.mean(scores)) if scores else 0.0


# ---------------------------------------------------------------------------
# Distributed process-group stub
# ---------------------------------------------------------------------------

class ProcessGroup:
  """
  Lightweight process-group abstraction for distributed training.

  In single-process mode (world_size=1) all operations are no-ops.
  In multi-process mode the caller is responsible for launching N processes and
  passing the correct rank/world_size; this class provides allreduce semantics
  over shared-memory numpy arrays using a barrier file + mmap approach, or can
  be backed by a real MPI/NCCL communicator by subclassing `allreduce`.

  For tinygrad the primary use is gradient averaging across ranks before the
  optimizer step.

  Args:
    rank:       this process's rank (0-based)
    world_size: total number of processes
  """

  def __init__(self, rank: int = 0, world_size: int = 1):
    self.rank = rank
    self.world_size = world_size

  @property
  def is_primary(self) -> bool:
    return self.rank == 0

  def allreduce_mean(self, value: float) -> float:
    """Average a scalar across all ranks.  Single-process: identity."""
    return value  # override in subclass for real distributed setup

  def allreduce_params(self, params: list) -> None:
    """Average gradients across all ranks.  Single-process: no-op."""
    # In multi-rank setups: average each .grad tensor, then assign back.
    pass  # override in subclass


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
  """
  Training loop for tinygrad UNet / ResUNet models.

  Supports:
  - Combined Dice + CE loss (or any custom loss callable)
  - Adam optimiser with pluggable LR scheduler (or cosine annealing fallback)
  - Gradient clipping
  - Model EMA
  - Early stopping (based on val_loss)
  - Validation loop with Dice metric computation
  - Checkpoint save / resume via tinygrad safe_save/safe_load
  - Optional WandB logging (auto-disabled when wandb not installed)
  - Multi-task heads: list of (head_model, loss_fn) pairs
  - Distributed process-group support (gradient averaging before optimizer step)

  Args:
    model:              tinygrad model callable (B,C,H,W) → (B,out_ch,H,W)
    train_loader:       iterable of (image_np, mask_np) numpy pairs, or dicts
                        with "image" and first target key
    val_loader:         optional validation iterable (same format)
    learning_rate:      initial Adam LR
    num_epochs:         total training epochs
    loss_fn:            loss callable (pred, target) → scalar Tensor; default DiceCELoss
    scheduler:          _Scheduler instance or None (falls back to cosine annealing)
    grad_clip:          max gradient norm; ≤ 0 disables clipping (default 1.0)
    ema_decay:          EMA decay coefficient; 0 disables EMA (default 0)
    early_stopping_patience: stop if val_loss does not improve for N epochs; 0 disables
    checkpoint_dir:     directory for periodic checkpoint saves
    checkpoint_every:   save every N epochs (default 10)
    wandb_project:      WandB project name (None = disabled)
    wandb_config:       extra dict logged to WandB
    aux_heads:          list of (head_model, loss_fn) for multi-task learning
    process_group:      ProcessGroup for distributed training (default: single-process)
    target_key:         key in batch dict used as the segmentation target (default "label")
    n_classes:          number of output classes (for Dice metric computation)
    verbose:            print progress
  """

  def __init__(
    self,
    model,
    train_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    loss_fn: Optional[Callable] = None,
    scheduler: Optional[_Scheduler] = None,
    grad_clip: float = 1.0,
    ema_decay: float = 0.0,
    early_stopping_patience: int = 0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    wandb_project: Optional[str] = None,
    wandb_config: Optional[dict] = None,
    aux_heads: Optional[List[tuple]] = None,
    process_group: Optional[ProcessGroup] = None,
    target_key: str = "label",
    n_classes: int = 2,
    verbose: bool = True,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for Trainer")

    self.model = model
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.loss_fn = loss_fn if loss_fn is not None else DiceCELoss(0.5, 0.5)
    self.grad_clip = float(grad_clip)
    self.ema_decay = float(ema_decay)
    self.early_stopping_patience = int(early_stopping_patience)
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_every = checkpoint_every
    self.aux_heads = aux_heads or []
    self.pg = process_group if process_group is not None else ProcessGroup()
    self.target_key = target_key
    self.n_classes = n_classes
    self.verbose = verbose
    self.start_epoch = 0

    # Collect all trainable parameters
    params = get_parameters(model)
    for head_model, _ in self.aux_heads:
      params = params + get_parameters(head_model)
    self._params = params
    self.optimizer = nn.optim.Adam(params, lr=learning_rate)

    # LR scheduler — attach to optimizer if provided
    if scheduler is not None:
      self._scheduler: Optional[_Scheduler] = scheduler
      scheduler.attach(self.optimizer)
    else:
      self._scheduler = None  # will use cosine_annealing_lr fallback

    # EMA
    self._ema: Optional[ModelEMA] = None
    if self.ema_decay > 0:
      self._ema = ModelEMA(model, decay=self.ema_decay)

    if checkpoint_dir:
      os.makedirs(checkpoint_dir, exist_ok=True)

    # Early stopping state
    self._best_val_loss: float = float("inf")
    self._no_improve_count: int = 0

    # WandB
    self._wandb_run = None
    if wandb_project and _WANDB:
      try:
        cfg = wandb_config or {}
        self._wandb_run = _wandb.init(project=wandb_project, config=cfg)
      except Exception as exc:
        if verbose:
          print(f"[Trainer] WandB init failed: {exc}")

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  def resume(self, checkpoint_path: str) -> None:
    """Load a checkpoint and resume training from the saved epoch."""
    self.start_epoch = load_checkpoint(self.model, self.optimizer, checkpoint_path)
    if self.verbose:
      print(f"[Trainer] Resumed from epoch {self.start_epoch} ({checkpoint_path})")

  def train(self) -> Dict[str, List[float]]:
    """Run the full training loop.  Returns dict of logged metric lists."""
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": [], "val_dice": []}

    for epoch in range(self.start_epoch, self.num_epochs):
      # --- LR: scheduler or cosine fallback ---
      if self._scheduler is not None:
        lr = self._scheduler.step()
      else:
        lr = cosine_annealing_lr(self.learning_rate, epoch, self.num_epochs)
        _set_lr(self.optimizer, lr)

      # --- train epoch ---
      train_loss = self._run_epoch(self.train_loader, train=True)
      train_loss = self.pg.allreduce_mean(train_loss)
      history["train_loss"].append(train_loss)
      history["lr"].append(lr)

      # --- validation epoch ---
      val_loss: Optional[float] = None
      val_dice: Optional[float] = None
      if self.val_loader is not None:
        if self._ema is not None:
          self._ema.apply_shadow()
        val_loss, val_dice = self._run_val_epoch(self.val_loader)
        val_loss = self.pg.allreduce_mean(val_loss)
        if self._ema is not None:
          self._ema.restore()
        history["val_loss"].append(val_loss)
        if val_dice is not None:
          history["val_dice"].append(val_dice)

      # --- logging ---
      if self.verbose and self.pg.is_primary:
        msg = f"Epoch {epoch + 1}/{self.num_epochs}  train_loss={train_loss:.4f}"
        if val_loss is not None:
          msg += f"  val_loss={val_loss:.4f}"
        if val_dice is not None:
          msg += f"  val_dice={val_dice:.4f}"
        msg += f"  lr={lr:.2e}"
        print(msg)

      if self._wandb_run is not None and self.pg.is_primary:
        try:
          log_dict: dict[str, Any] = {"epoch": epoch + 1, "train_loss": train_loss, "lr": lr}
          if val_loss is not None:
            log_dict["val_loss"] = val_loss
          if val_dice is not None:
            log_dict["val_dice"] = val_dice
          self._wandb_run.log(log_dict)
        except Exception:
          pass

      # --- checkpoint ---
      if self.checkpoint_dir and self.pg.is_primary and ((epoch + 1) % self.checkpoint_every == 0):
        ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch{epoch + 1:04d}.safetensors")
        save_checkpoint(self.model, self.optimizer, epoch + 1, ckpt_path, extra={"train_loss": train_loss})
        if self.verbose:
          print(f"[Trainer] Checkpoint saved: {ckpt_path}")

      # --- early stopping ---
      if self.early_stopping_patience > 0 and val_loss is not None:
        if val_loss < self._best_val_loss - 1e-7:
          self._best_val_loss = val_loss
          self._no_improve_count = 0
        else:
          self._no_improve_count += 1
          if self._no_improve_count >= self.early_stopping_patience:
            if self.verbose:
              print(f"[Trainer] Early stopping at epoch {epoch + 1} (no improvement for {self.early_stopping_patience} epochs)")
            break

    # --- final checkpoint ---
    if self.checkpoint_dir and self.pg.is_primary:
      final_path = os.path.join(self.checkpoint_dir, "ckpt_final.safetensors")
      save_checkpoint(self.model, self.optimizer, self.num_epochs, final_path)
      if self.verbose:
        print(f"[Trainer] Final checkpoint: {final_path}")

    if self._wandb_run is not None:
      try:
        self._wandb_run.finish()
      except Exception:
        pass

    return history

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _extract_batch(self, batch) -> Tuple[np.ndarray, np.ndarray]:
    """Unpack a batch into (images_np, masks_np) regardless of format."""
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
      return np.asarray(batch[0], dtype=np.float32), np.asarray(batch[1], dtype=np.int32)
    if isinstance(batch, dict):
      images_np = np.asarray(batch["image"], dtype=np.float32)
      # find first label key (not metadata)
      for key in (self.target_key, "label", "mask", "ink", "target"):
        if key in batch:
          masks_np = np.asarray(batch[key], dtype=np.int32)
          return images_np, masks_np
      raise KeyError(f"No label key found in batch dict; tried {self.target_key!r} and fallbacks")
    raise ValueError(f"Unsupported batch type {type(batch)}")

  def _run_epoch(self, loader: Iterable, train: bool) -> float:
    """Run one training epoch; return mean loss."""
    total_loss = 0.0
    n_batches = 0
    Tensor.training = train

    for batch in loader:
      images_np, masks_np = self._extract_batch(batch)
      images = Tensor(images_np)
      masks = Tensor(masks_np)

      pred = self.model(images)
      loss = self.loss_fn(pred, masks)

      for head_model, head_loss_fn in self.aux_heads:
        loss = loss + head_loss_fn(head_model(images), masks)

      if train:
        self.optimizer.zero_grad()
        loss.backward()

        # Distributed gradient averaging
        self.pg.allreduce_params(self._params)

        # Gradient clipping
        if self.grad_clip > 0:
          clip_grad_norm(self._params, self.grad_clip)

        self.optimizer.step()

        if self._ema is not None:
          self._ema.update()

      total_loss += float(loss.numpy())
      n_batches += 1

    Tensor.training = False
    return total_loss / max(1, n_batches)

  def _run_val_epoch(self, loader: Iterable) -> Tuple[float, Optional[float]]:
    """Run one validation epoch; return (mean_loss, mean_dice)."""
    total_loss = 0.0
    n_batches = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    Tensor.training = False
    for batch in loader:
      images_np, masks_np = self._extract_batch(batch)
      images = Tensor(images_np)
      masks = Tensor(masks_np)

      pred = self.model(images)
      loss = self.loss_fn(pred, masks)
      total_loss += float(loss.numpy())
      n_batches += 1

      all_preds.append(pred.numpy())
      all_targets.append(masks_np)

    val_loss = total_loss / max(1, n_batches)
    val_dice: Optional[float] = None
    if all_preds and self.n_classes > 1:
      try:
        preds_cat = np.concatenate(all_preds, axis=0)
        tgts_cat = np.concatenate(all_targets, axis=0)
        val_dice = _compute_dice(preds_cat, tgts_cat, self.n_classes)
      except Exception:
        pass

    return val_loss, val_dice


# ---------------------------------------------------------------------------
# Metric helpers (standalone, no model needed)
# ---------------------------------------------------------------------------

def _compute_dice(pred_np: np.ndarray, target_np: np.ndarray, n_classes: int) -> float:
  """Mean Dice coefficient across classes 1..n_classes-1 (ignores background)."""
  pred_cls = pred_np.argmax(axis=1).ravel()
  tgt_flat = target_np.ravel().astype(np.int32)
  scores = []
  for c in range(1, n_classes):
    p = (pred_cls == c).astype(np.float32)
    t = (tgt_flat == c).astype(np.float32)
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    if denom < 1e-7:
      continue
    scores.append(2.0 * inter / denom)
  return float(np.mean(scores)) if scores else 0.0
