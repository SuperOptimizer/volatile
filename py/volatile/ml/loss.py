from __future__ import annotations

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss:
  """
  Soft Dice loss for binary or multi-class segmentation.

  Expects:
    pred  — (B, C, H, W) logits or probabilities (sigmoid/softmax applied internally)
    target — (B, C, H, W) one-hot encoded float masks OR (B, H, W) integer class indices

  When `from_logits=True` (default) a sigmoid is applied for binary (C==1) problems
  and softmax for multi-class (C>1).
  """

  def __init__(self, smooth: float = 1.0, from_logits: bool = True):
    self.smooth = smooth
    self.from_logits = from_logits

  def __call__(self, pred: "Tensor", target: "Tensor") -> "Tensor":
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for DiceLoss")
    if self.from_logits:
      if pred.shape[1] == 1:
        pred = pred.sigmoid()
      else:
        pred = pred.softmax(axis=1)
    # bring target to one-hot float if it is class indices (B, H, W)
    if target.ndim == pred.ndim - 1:
      target = target.one_hot(pred.shape[1]).permute(0, 3, 1, 2).cast(pred.dtype)
    # flatten spatial dims: (B, C, N)
    B, C = pred.shape[0], pred.shape[1]
    p = pred.reshape(B, C, -1)
    t = target.reshape(B, C, -1)
    intersection = (p * t).sum(axis=2)  # (B, C)
    union = p.sum(axis=2) + t.sum(axis=2)  # (B, C)
    dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
    return (1.0 - dice_score).mean()


# ---------------------------------------------------------------------------
# Cross-Entropy Loss
# ---------------------------------------------------------------------------

class CrossEntropyLoss:
  """
  Cross-entropy loss.

  pred   — (B, C, H, W) logits
  target — (B, H, W) integer class indices
  """

  def __init__(self, weight: list | None = None):
    self.weight = weight  # per-class weight list, currently unused (tinygrad CE is unweighted)

  def __call__(self, pred: "Tensor", target: "Tensor") -> "Tensor":
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for CrossEntropyLoss")
    # tinygrad cross_entropy expects (B, C, ...) logits and (B, ...) int targets
    return pred.cross_entropy(target)


# ---------------------------------------------------------------------------
# Combined Dice + CE Loss
# ---------------------------------------------------------------------------

class DiceCELoss:
  """
  Weighted combination of Dice loss and cross-entropy loss.

  loss = dice_weight * DiceLoss(pred, target) + ce_weight * CrossEntropyLoss(pred, target)

  pred   — (B, C, H, W) logits
  target — (B, H, W) integer class indices  (DiceLoss handles one-hot conversion internally)
  """

  def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1.0):
    self.dice_weight = dice_weight
    self.ce_weight = ce_weight
    self.dice = DiceLoss(smooth=smooth, from_logits=True)
    self.ce = CrossEntropyLoss()

  def __call__(self, pred: "Tensor", target: "Tensor") -> "Tensor":
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for DiceCELoss")
    d = self.dice(pred, target)
    c = self.ce(pred, target)
    return self.dice_weight * d + self.ce_weight * c
