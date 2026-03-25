from __future__ import annotations

import math
import os
import time
from typing import Any, Callable, Dict, Iterable, List

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


# ---------------------------------------------------------------------------
# Learning-rate schedulers
# ---------------------------------------------------------------------------

def cosine_annealing_lr(initial_lr: float, epoch: int, total_epochs: int, eta_min: float = 1e-6) -> float:
  """Cosine annealing schedule — returns the LR for a given epoch."""
  return eta_min + 0.5 * (initial_lr - eta_min) * (1.0 + math.cos(math.pi * epoch / max(1, total_epochs)))


def _set_lr(optimizer, lr: float) -> None:
  """Update learning rate on a tinygrad Adam optimizer in place."""
  optimizer.lr = lr  # tinygrad Adam stores lr as a plain attribute


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch: int, path: str, extra: dict | None = None) -> None:
  """Save model + optimizer state and epoch metadata via tinygrad safe_save."""
  if not _TINYGRAD:
    raise ImportError("tinygrad is required for save_checkpoint")
  state = get_state_dict(model)
  # Prefix optimizer params so they don't collide with model params
  opt_state = {f"__opt_{k}": v for k, v in get_state_dict(optimizer).items()}
  state.update(opt_state)
  # Epoch stored as a 1-element tensor so safe_save can handle it
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
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
  """
  Training loop for tinygrad UNet / ResUNet models.

  Supports:
  - Combined Dice + CE loss
  - Adam optimiser with cosine-annealing LR schedule
  - Checkpoint save / resume
  - Optional WandB logging (auto-disabled when wandb not installed)
  - Multi-task heads: pass a list of (head_model, loss_fn) pairs via `aux_heads`

  Args:
    model:          tinygrad model (callable, (B, C, H, W) → (B, out_ch, H, W))
    train_loader:   iterable of (image_np, mask_np) numpy arrays; shapes
                    (B, C, H, W) and (B, H, W) respectively
    val_loader:     optional validation iterable with same signature
    learning_rate:  initial Adam LR (default 1e-3)
    num_epochs:     total training epochs
    loss_fn:        loss callable (pred_tensor, target_tensor) → scalar Tensor;
                    defaults to DiceCELoss(0.5, 0.5)
    checkpoint_dir: directory for periodic checkpoints (None = no saving)
    checkpoint_every: save a checkpoint every N epochs (default 10)
    wandb_project:  WandB project name (None = disable WandB)
    wandb_config:   extra dict to log to WandB init
    aux_heads:      list of (head_model, loss_fn) for multi-task learning;
                    each head receives the *same* logits as the primary model
    verbose:        print progress (default True)
  """

  def __init__(
    self,
    model,
    train_loader: Iterable,
    val_loader: Iterable | None = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    loss_fn: Callable | None = None,
    checkpoint_dir: str | None = None,
    checkpoint_every: int = 10,
    wandb_project: str | None = None,
    wandb_config: dict | None = None,
    aux_heads: List[tuple] | None = None,
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
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_every = checkpoint_every
    self.aux_heads = aux_heads or []
    self.verbose = verbose
    self.start_epoch = 0

    # Collect all parameters: primary model + any aux heads
    params = get_parameters(model)
    for head_model, _ in self.aux_heads:
      params = params + get_parameters(head_model)
    self.optimizer = nn.optim.Adam(params, lr=learning_rate)

    if checkpoint_dir:
      os.makedirs(checkpoint_dir, exist_ok=True)

    # WandB init
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
    """Run the full training loop.  Returns a dict of logged metric lists."""
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(self.start_epoch, self.num_epochs):
      # LR schedule
      lr = cosine_annealing_lr(self.learning_rate, epoch, self.num_epochs)
      _set_lr(self.optimizer, lr)

      # ---- train epoch ----
      train_loss = self._run_epoch(self.train_loader, train=True)
      history["train_loss"].append(train_loss)
      history["lr"].append(lr)

      # ---- val epoch ----
      val_loss = None
      if self.val_loader is not None:
        val_loss = self._run_epoch(self.val_loader, train=False)
        history["val_loss"].append(val_loss)

      # ---- logging ----
      if self.verbose:
        msg = f"Epoch {epoch + 1}/{self.num_epochs}  train_loss={train_loss:.4f}"
        if val_loss is not None:
          msg += f"  val_loss={val_loss:.4f}"
        msg += f"  lr={lr:.2e}"
        print(msg)

      if self._wandb_run is not None:
        try:
          log_dict: dict[str, Any] = {"epoch": epoch + 1, "train_loss": train_loss, "lr": lr}
          if val_loss is not None:
            log_dict["val_loss"] = val_loss
          self._wandb_run.log(log_dict)
        except Exception:
          pass

      # ---- checkpoint ----
      if self.checkpoint_dir and ((epoch + 1) % self.checkpoint_every == 0):
        ckpt_path = os.path.join(self.checkpoint_dir, f"ckpt_epoch{epoch + 1:04d}.safetensors")
        save_checkpoint(self.model, self.optimizer, epoch + 1, ckpt_path, extra={"train_loss": train_loss})
        if self.verbose:
          print(f"[Trainer] Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    if self.checkpoint_dir:
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
  # Internal
  # ------------------------------------------------------------------

  def _run_epoch(self, loader: Iterable, train: bool) -> float:
    """Run one epoch over `loader`.  Returns mean loss as a Python float."""
    total_loss = 0.0
    n_batches = 0

    Tensor.training = train

    for batch in loader:
      if isinstance(batch, (list, tuple)) and len(batch) == 2:
        images_np, masks_np = batch
      else:
        raise ValueError("Loader must yield (image_np, mask_np) pairs")

      images = Tensor(np.asarray(images_np, dtype=np.float32))
      masks = Tensor(np.asarray(masks_np, dtype=np.int32))

      pred = self.model(images)
      loss = self.loss_fn(pred, masks)

      # Auxiliary heads receive same input; their losses are summed
      for head_model, head_loss_fn in self.aux_heads:
        aux_pred = head_model(images)
        loss = loss + head_loss_fn(aux_pred, masks)

      if train:
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      total_loss += float(loss.numpy())
      n_batches += 1

    Tensor.training = False
    return total_loss / max(1, n_batches)
