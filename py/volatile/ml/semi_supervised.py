from __future__ import annotations

"""
Mean Teacher semi-supervised trainer for tinygrad.

Two-stream training loop:
  - Labeled stream:   supervised loss (DiceCE by default)
  - Unlabeled stream: consistency loss between student and teacher (EMA) predictions

Teacher weights are an EMA of the student weights; the teacher never receives
gradient updates directly.  A sigmoid ramp-up schedule controls how much the
consistency loss contributes early in training (so the teacher has time to warm up).

Reference: Tarvainen & Valpola, "Mean teachers are better role models", NeurIPS 2017.
"""

import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
  from tinygrad import Tensor, nn
  from tinygrad.nn.state import get_parameters, get_state_dict, load_state_dict
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

from .loss import DiceCELoss
from .train import cosine_annealing_lr, _set_lr, clip_grad_norm, save_checkpoint, ModelEMA


# ---------------------------------------------------------------------------
# Ramp-up schedules
# ---------------------------------------------------------------------------

def sigmoid_rampup(current: float, rampup_length: float) -> float:
  """Sigmoid ramp-up from 0 → 1 over `rampup_length` steps."""
  if rampup_length <= 0:
    return 1.0
  current = max(0.0, min(current, rampup_length))
  phase = 1.0 - current / rampup_length
  return float(math.exp(-5.0 * phase * phase))


def linear_rampup(current: float, rampup_length: float) -> float:
  """Linear ramp-up from 0 → 1 over `rampup_length` steps."""
  if rampup_length <= 0:
    return 1.0
  return float(min(1.0, current / rampup_length))


# ---------------------------------------------------------------------------
# Two-stream batch builder
# ---------------------------------------------------------------------------

def build_two_stream_batches(
  labeled_items: List[Tuple[np.ndarray, np.ndarray]],
  unlabeled_items: List[np.ndarray],
  labeled_batch_size: int,
  unlabeled_batch_size: int,
  rng: np.random.Generator,
) -> List[Dict[str, np.ndarray]]:
  """
  Interleave labeled and unlabeled samples into mixed batches.

  Each batch contains `labeled_batch_size` labeled + `unlabeled_batch_size`
  unlabeled samples, concatenated along axis 0.

  Args:
    labeled_items:      list of (image_np, mask_np) pairs
    unlabeled_items:    list of image_np arrays (no mask needed)
    labeled_batch_size:   number of labeled samples per batch
    unlabeled_batch_size: number of unlabeled samples per batch
    rng:                numpy Generator for shuffling

  Yields:
    dict with keys "image" (B, C, H, W), "label" (B_labeled, H, W),
    "n_labeled" (int) indicating how many leading samples are labeled
  """
  labeled_idx = np.arange(len(labeled_items))
  unlabeled_idx = np.arange(len(unlabeled_items))
  rng.shuffle(labeled_idx)
  rng.shuffle(unlabeled_idx)

  n_lbl_batches = max(1, len(labeled_idx) // labeled_batch_size)
  n_unl_batches = max(1, len(unlabeled_idx) // unlabeled_batch_size)
  n_batches = min(n_lbl_batches, n_unl_batches)

  batches = []
  for b in range(n_batches):
    lbl_sl = labeled_idx[b * labeled_batch_size:(b + 1) * labeled_batch_size]
    unl_sl = unlabeled_idx[b * unlabeled_batch_size:(b + 1) * unlabeled_batch_size]

    lbl_imgs = np.stack([labeled_items[i][0] for i in lbl_sl], axis=0)
    lbl_masks = np.stack([labeled_items[i][1] for i in lbl_sl], axis=0)
    unl_imgs = np.stack([unlabeled_items[i] for i in unl_sl], axis=0)

    # ensure channel dim present: (B, C, *spatial)
    if lbl_imgs.ndim == 3:  # (B, H, W) → (B, 1, H, W)
      lbl_imgs = lbl_imgs[:, np.newaxis]
    if unl_imgs.ndim == 3:
      unl_imgs = unl_imgs[:, np.newaxis]

    images = np.concatenate([lbl_imgs, unl_imgs], axis=0).astype(np.float32)
    batches.append({"image": images, "label": lbl_masks.astype(np.int32), "n_labeled": len(lbl_sl)})

  return batches


# ---------------------------------------------------------------------------
# Mean Teacher Trainer
# ---------------------------------------------------------------------------

class MeanTeacherTrainer:
  """
  Mean Teacher semi-supervised training loop.

  Maintains a student model (trained with gradient descent) and a teacher model
  (EMA of student weights).  Each batch contains labeled + unlabeled samples:

  - Supervised loss is computed only on labeled samples.
  - Consistency loss (MSE between student and teacher softmax outputs) is
    computed on unlabeled samples and weighted by a sigmoid ramp-up schedule.

  Args:
    student:              tinygrad model (callable, (B,C,H,W) → (B,out_ch,H,W))
    labeled_loader:       iterable of (image_np, mask_np) pairs or dicts
    unlabeled_loader:     iterable of image_np arrays (no labels)
    val_loader:           optional validation iterable (same format as labeled)
    learning_rate:        Adam LR
    num_epochs:           total training epochs
    ema_decay:            teacher EMA decay (default 0.99)
    consistency_weight:   maximum consistency loss weight (default 0.1)
    consistency_rampup:   epochs over which consistency weight ramps to max (default 40)
    warmup_epochs:        epochs before consistency loss is applied (default 0)
    supervised_loss_fn:   loss callable for labeled data (default DiceCELoss)
    noise_scale:          std of additive noise applied to teacher input (default 0.1)
    grad_clip:            gradient clip norm; ≤ 0 disables (default 1.0)
    checkpoint_dir:       directory for checkpoint saves
    checkpoint_every:     checkpoint frequency in epochs
    verbose:              print progress
  """

  def __init__(
    self,
    student,
    labeled_loader: Iterable,
    unlabeled_loader: Iterable,
    val_loader: Optional[Iterable] = None,
    learning_rate: float = 1e-3,
    num_epochs: int = 100,
    ema_decay: float = 0.99,
    consistency_weight: float = 0.1,
    consistency_rampup: float = 40.0,
    warmup_epochs: int = 0,
    supervised_loss_fn: Optional[Callable] = None,
    noise_scale: float = 0.1,
    grad_clip: float = 1.0,
    checkpoint_dir: Optional[str] = None,
    checkpoint_every: int = 10,
    verbose: bool = True,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for MeanTeacherTrainer")

    self.student = student
    self.labeled_loader = labeled_loader
    self.unlabeled_loader = unlabeled_loader
    self.val_loader = val_loader
    self.learning_rate = learning_rate
    self.num_epochs = num_epochs
    self.ema_decay = ema_decay
    self.consistency_weight = consistency_weight
    self.consistency_rampup = consistency_rampup
    self.warmup_epochs = warmup_epochs
    self.sup_loss_fn = supervised_loss_fn if supervised_loss_fn is not None else DiceCELoss(0.5, 0.5)
    self.noise_scale = noise_scale
    self.grad_clip = grad_clip
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_every = checkpoint_every
    self.verbose = verbose

    self._params = get_parameters(student)
    self.optimizer = nn.optim.Adam(self._params, lr=learning_rate)

    # Teacher: separate model reference with EMA weights
    # We use ModelEMA which stores shadow weights; teacher forward pass uses apply_shadow()
    self._teacher_ema = ModelEMA(student, decay=ema_decay)

    self._global_step = 0

    if checkpoint_dir:
      import os
      os.makedirs(checkpoint_dir, exist_ok=True)

  # ------------------------------------------------------------------
  # Consistency weight schedule
  # ------------------------------------------------------------------

  def _current_consistency_weight(self, epoch: int) -> float:
    if epoch < self.warmup_epochs:
      return 0.0
    return self.consistency_weight * sigmoid_rampup(epoch - self.warmup_epochs, self.consistency_rampup)

  # ------------------------------------------------------------------
  # Per-step helpers
  # ------------------------------------------------------------------

  def _to_tensor(self, arr: np.ndarray) -> "Tensor":
    return Tensor(np.asarray(arr, dtype=np.float32))

  def _consistency_loss(self, student_logits: "Tensor", teacher_logits: "Tensor") -> "Tensor":
    """MSE between softmax probabilities (or sigmoid for single-channel)."""
    if student_logits.shape[1] == 1:
      s_prob = student_logits.sigmoid()
      t_prob = teacher_logits.sigmoid()
    else:
      s_prob = student_logits.softmax(axis=1)
      t_prob = teacher_logits.softmax(axis=1)
    diff = s_prob - t_prob
    return (diff * diff).mean()

  def _extract_image_label(self, batch) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if isinstance(batch, dict):
      img = np.asarray(batch["image"], dtype=np.float32)
      lbl = np.asarray(batch.get("label", batch.get("mask", batch.get("seg"))), dtype=np.int32) if any(
        k in batch for k in ("label", "mask", "seg")) else None
      return img, lbl
    if isinstance(batch, (list, tuple)):
      img = np.asarray(batch[0], dtype=np.float32)
      lbl = np.asarray(batch[1], dtype=np.int32) if len(batch) > 1 else None
      return img, lbl
    img = np.asarray(batch, dtype=np.float32)
    return img, None

  # ------------------------------------------------------------------
  # Main training loop
  # ------------------------------------------------------------------

  def train(self) -> Dict[str, List[float]]:
    """Run the full Mean Teacher training loop."""
    history: Dict[str, List[float]] = {
      "train_sup_loss": [], "train_cons_loss": [], "val_loss": [], "lr": [], "cons_weight": []
    }

    for epoch in range(self.num_epochs):
      lr = cosine_annealing_lr(self.learning_rate, epoch, self.num_epochs)
      _set_lr(self.optimizer, lr)
      cons_w = self._current_consistency_weight(epoch)

      sup_losses, cons_losses = [], []
      Tensor.training = True

      # Zip labeled and unlabeled loaders; stop at the shorter one
      for lbl_batch, unl_batch in zip(self.labeled_loader, self.unlabeled_loader):
        lbl_img_np, lbl_mask_np = self._extract_image_label(lbl_batch)
        unl_img_np, _ = self._extract_image_label(unl_batch)

        # Ensure (B, C, *spatial) shape
        if lbl_img_np.ndim == 3:
          lbl_img_np = lbl_img_np[:, np.newaxis]
        if unl_img_np.ndim == 3:
          unl_img_np = unl_img_np[:, np.newaxis]

        lbl_img = self._to_tensor(lbl_img_np)
        lbl_mask = Tensor(lbl_mask_np)

        # ---- consistency loss: get teacher output first (no grad needed) ----
        # Run teacher forward BEFORE student forward so that apply_shadow/restore
        # don't touch the tensors involved in the student's computation graph.
        teacher_out_np: Optional[np.ndarray] = None
        if cons_w > 0:
          noise = np.random.randn(*unl_img_np.shape).astype(np.float32) * self.noise_scale
          noisy_unl_np = unl_img_np + noise
          noisy_unl = self._to_tensor(noisy_unl_np)
          self._teacher_ema.apply_shadow()
          Tensor.training = False
          teacher_out_np = self.student(noisy_unl).numpy()
          Tensor.training = True
          self._teacher_ema.restore()

        # ---- supervised loss on labeled data ----
        student_lbl_out = self.student(lbl_img)
        sup_loss = self.sup_loss_fn(student_lbl_out, lbl_mask)

        # ---- consistency loss on unlabeled data ----
        cons_loss_val: "Tensor"
        if cons_w > 0 and teacher_out_np is not None:
          unl_img = self._to_tensor(unl_img_np)
          student_unl_out = self.student(unl_img)
          teacher_out = Tensor(teacher_out_np)
          cons_loss_val = self._consistency_loss(student_unl_out, teacher_out)
          total_loss = sup_loss + cons_w * cons_loss_val
        else:
          cons_loss_val = Tensor([0.0])
          total_loss = sup_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
          clip_grad_norm(self._params, self.grad_clip)
        self.optimizer.step()

        # Update teacher EMA after student step
        self._teacher_ema.update()
        self._global_step += 1

        sup_losses.append(float(sup_loss.numpy()))
        cons_losses.append(float(cons_loss_val.numpy()) if cons_w > 0 else 0.0)

      Tensor.training = False

      mean_sup = float(np.mean(sup_losses)) if sup_losses else 0.0
      mean_cons = float(np.mean(cons_losses)) if cons_losses else 0.0
      history["train_sup_loss"].append(mean_sup)
      history["train_cons_loss"].append(mean_cons)
      history["lr"].append(lr)
      history["cons_weight"].append(cons_w)

      # Validation
      val_loss = None
      if self.val_loader is not None:
        val_loss = self._run_val_epoch()
        history["val_loss"].append(val_loss)

      if self.verbose:
        msg = f"Epoch {epoch + 1}/{self.num_epochs}  sup={mean_sup:.4f}  cons={mean_cons:.4f}  cw={cons_w:.3f}  lr={lr:.2e}"
        if val_loss is not None:
          msg += f"  val={val_loss:.4f}"
        print(msg)

      if self.checkpoint_dir and (epoch + 1) % self.checkpoint_every == 0:
        import os
        ckpt = os.path.join(self.checkpoint_dir, f"mt_epoch{epoch + 1:04d}.safetensors")
        save_checkpoint(self.student, self.optimizer, epoch + 1, ckpt)

    return history

  def _run_val_epoch(self) -> float:
    Tensor.training = False
    total, n = 0.0, 0
    for batch in self.val_loader:
      img_np, mask_np = self._extract_image_label(batch)
      if img_np.ndim == 3:
        img_np = img_np[:, np.newaxis]
      if mask_np is None:
        continue
      pred = self.student(self._to_tensor(img_np))
      loss = self.sup_loss_fn(pred, Tensor(mask_np.astype(np.int32)))
      total += float(loss.numpy())
      n += 1
    return total / max(1, n)

  @property
  def teacher_weights(self) -> Dict[str, np.ndarray]:
    """Return a copy of the current teacher (EMA) weight dict."""
    return {k: v.copy() for k, v in self._teacher_ema._shadow.items()}
