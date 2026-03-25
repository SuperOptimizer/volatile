from __future__ import annotations
"""Topology-aware loss functions for binary segmentation.

All losses accept tinygrad Tensors and return a scalar Tensor with gradients.
Non-differentiable parts (connected components, skeletonization) run on numpy
and feed back into the graph via a soft weighting mask multiplied against the
differentiable prediction, so gradients flow through the logits.

All losses expect:
  pred: (N, 1, H, W)  raw logits  (before sigmoid)
  gt:   (N, 1, H, W)  binary {0,1} float32

The differentiable bridge pattern used throughout:
  1. Detach pred, sigmoid → binary hard mask (numpy)
  2. Compute topology quantity on hard mask (numpy)
  3. Wrap that quantity as a constant Tensor weight
  4. Multiply weight × sigmoid(pred) so gradients flow
"""

import math
import numpy as np

try:
  from tinygrad.tensor import Tensor
  _HAS_TINYGRAD = True
except ImportError:
  _HAS_TINYGRAD = False
  Tensor = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Pure-numpy topology primitives
# ---------------------------------------------------------------------------

def _union_find_init(n: int) -> list[int]:
  return list(range(n))


def _uf_find(parent: list[int], x: int) -> int:
  while parent[x] != x:
    parent[x] = parent[parent[x]]   # path compression
    x = parent[x]
  return x


def _uf_union(parent: list[int], a: int, b: int) -> None:
  ra, rb = _uf_find(parent, a), _uf_find(parent, b)
  if ra != rb:
    parent[ra] = rb


def connected_components_2d(mask: np.ndarray) -> tuple[np.ndarray, int]:
  """Label 4-connected components in a 2-D binary mask.

  Returns (label_array, num_components) with labels starting at 1.
  """
  mask = np.asarray(mask, dtype=bool)
  h, w = mask.shape
  parent = _union_find_init(h * w)
  for i in range(h):
    for j in range(w):
      if not mask[i, j]:
        continue
      idx = i * w + j
      if i > 0 and mask[i - 1, j]:
        _uf_union(parent, idx, (i - 1) * w + j)
      if j > 0 and mask[i, j - 1]:
        _uf_union(parent, idx, i * w + (j - 1))
  labels = np.zeros((h, w), dtype=np.int32)
  root_to_label: dict[int, int] = {}
  count = 0
  for i in range(h):
    for j in range(w):
      if not mask[i, j]:
        continue
      r = _uf_find(parent, i * w + j)
      if r not in root_to_label:
        count += 1
        root_to_label[r] = count
      labels[i, j] = root_to_label[r]
  return labels, count


def euler_number_2d(mask: np.ndarray) -> float:
  """Euler number of a 2-D binary image using the 2×2 quad formula.

  For 4-connectivity: E = (Q1 - Q3 - 2·QD) / 4
  where Q1 = quads with exactly 1 fg pixel, Q3 = 3 fg pixels,
  QD = diagonal pairs (two fg on one diagonal, zero on the other).

  Euler number = #components - #holes  (4-connectivity).
  """
  b = np.asarray(mask, dtype=np.uint8)
  tl = b[:-1, :-1]; tr = b[:-1, 1:]
  bl = b[1:,  :-1]; br = b[1:,  1:]
  s = tl.astype(np.int32) + tr + bl + br
  q1  = int((s == 1).sum())
  q3  = int((s == 3).sum())
  qd  = int(((tl == 1) & (br == 1) & (tr == 0) & (bl == 0)).sum() +
            ((tl == 0) & (br == 0) & (tr == 1) & (bl == 1)).sum())
  return (q1 - q3 - 2 * qd) / 4.0


def betti_numbers_2d(mask: np.ndarray) -> tuple[int, int]:
  """Compute (b0, b1) = (components, holes) of a 2-D binary mask.

  Uses: b1 = b0 - euler_number  (Euler = b0 - b1 for 2-D planar topology).
  """
  _, b0 = connected_components_2d(mask)
  e = euler_number_2d(mask)
  b1 = max(0, int(round(b0 - e)))
  return b0, b1


def _zhang_suen_iter(mask: np.ndarray, odd: bool) -> np.ndarray:
  """One Zhang-Suen thinning sub-iteration. Returns pixels to remove."""
  b = mask.astype(np.uint8)
  h, w = b.shape

  # 8-neighbours in order: p2..p9 (N, NE, E, SE, S, SW, W, NW)
  p2 = np.roll(b, -1, 0); p3 = np.roll(np.roll(b, -1, 0), 1, 1)
  p4 = np.roll(b,  1, 1); p5 = np.roll(np.roll(b,  1, 0), 1, 1)
  p6 = np.roll(b,  1, 0); p7 = np.roll(np.roll(b,  1, 0), -1, 1)
  p8 = np.roll(b, -1, 1); p9 = np.roll(np.roll(b, -1, 0), -1, 1)

  neighbours = [p2, p3, p4, p5, p6, p7, p8, p9]
  N = sum(neighbours)   # number of fg neighbours (2..6 condition)

  # Transitions: count 0→1 transitions in the cyclic sequence p2..p9,p2
  seq = np.stack(neighbours + [p2], axis=0)          # (9, H, W)
  transitions = ((seq[:-1] == 0) & (seq[1:] == 1)).sum(axis=0)

  cond12 = (N >= 2) & (N <= 6) & (transitions == 1)
  if odd:
    cond34 = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
  else:
    cond34 = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)

  return (b == 1) & cond12 & cond34


def skeletonize_2d(mask: np.ndarray, max_iter: int = 200) -> np.ndarray:
  """Zhang-Suen thinning algorithm — returns boolean skeleton mask."""
  skel = np.asarray(mask, dtype=bool).copy()
  for _ in range(max_iter):
    remove1 = _zhang_suen_iter(skel, odd=True)
    skel[remove1] = False
    remove2 = _zhang_suen_iter(skel, odd=False)
    skel[remove2] = False
    if not remove1.any() and not remove2.any():
      break
  return skel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sigmoid_np(logits: np.ndarray) -> np.ndarray:
  return 1.0 / (1.0 + np.exp(-logits.astype(np.float64))).astype(np.float32)


def _hard_mask(pred_np: np.ndarray, threshold: float = 0.5) -> np.ndarray:
  """Sigmoid + threshold → binary bool array."""
  return _sigmoid_np(pred_np) >= threshold


def _soft_pred(pred: "Tensor") -> "Tensor":
  """Sigmoid of logits, clamped to (eps, 1-eps) for numerical safety."""
  eps = 1e-6
  return pred.sigmoid().clip(eps, 1.0 - eps)


def _dice(p: "Tensor", g: "Tensor") -> "Tensor":
  """Soft Dice loss between probability map p and binary target g, both (N,...)."""
  smooth = 1.0
  p_flat = p.reshape(p.shape[0], -1)
  g_flat = g.reshape(g.shape[0], -1)
  inter  = (p_flat * g_flat).sum(axis=1)
  denom  = p_flat.sum(axis=1) + g_flat.sum(axis=1)
  return (1.0 - (2.0 * inter + smooth) / (denom + smooth)).mean()


# ---------------------------------------------------------------------------
# BettiNumberLoss
# ---------------------------------------------------------------------------

class BettiNumberLoss:
  """Penalise incorrect Betti numbers (b0 = components, b1 = holes).

  Differentiable via a soft re-weighting: the topology error magnitude
  (|Δb0| + |Δb1|) is computed on the hard-thresholded prediction and used
  to scale the binary cross-entropy on every pixel so that gradients push
  all prediction probabilities toward the target.

  Args:
    weight_b0:  relative weight for component error (default 1.0)
    weight_b1:  relative weight for hole error (default 1.0)
    threshold:  sigmoid threshold for hard mask (default 0.5)
  """

  def __init__(self, weight_b0: float = 1.0, weight_b1: float = 1.0, threshold: float = 0.5):
    self.weight_b0 = float(weight_b0)
    self.weight_b1 = float(weight_b1)
    self.threshold = float(threshold)

  def __call__(self, pred: "Tensor", gt: "Tensor") -> "Tensor":
    """pred, gt: (N, 1, H, W)."""
    pred_np = pred.numpy()                    # (N, 1, H, W)
    gt_np   = gt.numpy()

    n = pred_np.shape[0]
    topo_errors = np.zeros(n, dtype=np.float32)

    for i in range(n):
      pmask = _hard_mask(pred_np[i, 0], self.threshold)
      gmask = gt_np[i, 0] >= 0.5

      pb0, pb1 = betti_numbers_2d(pmask) if pmask.any() else (0, 0)
      gb0, gb1 = betti_numbers_2d(gmask) if gmask.any() else (0, 0)

      err = self.weight_b0 * abs(pb0 - gb0) + self.weight_b1 * abs(pb1 - gb1)
      topo_errors[i] = float(err)

    # Scale BCE by topology error magnitude (minimum 1 so loss is always nonzero).
    scale = Tensor((topo_errors + 1.0).reshape(n, 1, 1, 1))
    sp = _soft_pred(pred)
    gt_t = gt
    bce = -(gt_t * sp.log() + (1.0 - gt_t) * (1.0 - sp).log())
    return (bce * scale).mean()


# ---------------------------------------------------------------------------
# SkeletonRecallLoss
# ---------------------------------------------------------------------------

class SkeletonRecallLoss:
  """Penalise predictions that fail to cover the ground-truth skeleton.

  Loss = 1 - (skeleton voxels covered by prediction) / (total skeleton voxels).

  The skeleton is extracted from the GT mask; coverage is measured by
  multiplying the prediction probability at each skeleton voxel.  This is
  differentiable: gradients push up the predicted probability at skeleton sites.

  Args:
    threshold:      sigmoid threshold for hard GT skeleton (default 0.5)
    skeleton_iters: max Zhang-Suen iterations (default 200)
  """

  def __init__(self, threshold: float = 0.5, skeleton_iters: int = 200):
    self.threshold      = float(threshold)
    self.skeleton_iters = int(skeleton_iters)

  def __call__(self, pred: "Tensor", gt: "Tensor") -> "Tensor":
    gt_np = gt.numpy()    # (N, 1, H, W)
    n     = gt_np.shape[0]
    sp    = _soft_pred(pred)   # (N, 1, H, W)

    batch_losses: list[Tensor] = []
    for i in range(n):
      gmask = gt_np[i, 0] >= self.threshold
      skel  = skeletonize_2d(gmask, max_iter=self.skeleton_iters)  # (H, W) bool

      total = int(skel.sum())
      if total == 0:
        # No skeleton — contribute zero loss without breaking the graph.
        batch_losses.append((sp[i:i+1] * 0.0).mean())
        continue

      skel_w = Tensor(skel.astype(np.float32).reshape(1, 1, *skel.shape))  # (1,1,H,W)
      # coverage = mean prediction probability at skeleton sites
      coverage = (sp[i:i+1] * skel_w).sum() / float(total)
      batch_losses.append(1.0 - coverage)

    return Tensor.stack(batch_losses).mean()


# ---------------------------------------------------------------------------
# CenterlineDiceLoss
# ---------------------------------------------------------------------------

class CenterlineDiceLoss:
  """Dice loss computed only on centerline voxels extracted from both masks.

  Skeletonizes both the hard prediction and the GT, then computes soft Dice
  between the prediction *probabilities* at skeleton locations and the GT
  skeleton mask.  This focuses gradient signal on the topologically critical
  centreline pixels.

  Args:
    threshold:      sigmoid threshold for hard masks (default 0.5)
    smooth:         Dice smoothing constant (default 1.0)
    skeleton_iters: max Zhang-Suen iterations (default 200)
  """

  def __init__(self, threshold: float = 0.5, smooth: float = 1.0, skeleton_iters: int = 200):
    self.threshold      = float(threshold)
    self.smooth         = float(smooth)
    self.skeleton_iters = int(skeleton_iters)

  def __call__(self, pred: "Tensor", gt: "Tensor") -> "Tensor":
    pred_np = pred.numpy()   # (N, 1, H, W)
    gt_np   = gt.numpy()
    n       = pred_np.shape[0]
    sp      = _soft_pred(pred)

    batch_losses: list[Tensor] = []
    for i in range(n):
      pmask  = _hard_mask(pred_np[i, 0], self.threshold)
      gmask  = gt_np[i, 0] >= self.threshold
      p_skel = skeletonize_2d(pmask, max_iter=self.skeleton_iters)
      g_skel = skeletonize_2d(gmask, max_iter=self.skeleton_iters)

      # Union of both skeletons — compute Dice over these voxels.
      union_skel = (p_skel | g_skel).astype(np.float32)
      if union_skel.sum() == 0:
        batch_losses.append((sp[i:i+1] * 0.0).mean())
        continue

      w = Tensor(union_skel.reshape(1, 1, *union_skel.shape))     # (1,1,H,W)
      g = Tensor(g_skel.astype(np.float32).reshape(1, 1, *g_skel.shape))

      p_w = sp[i:i+1] * w
      g_w = g * w
      inter = (p_w * g_w).sum()
      denom = p_w.sum() + g_w.sum()
      dice_loss = 1.0 - (2.0 * inter + self.smooth) / (denom + self.smooth)
      batch_losses.append(dice_loss)

    return Tensor.stack(batch_losses).mean()


# ---------------------------------------------------------------------------
# TopologicalLoss  (weighted combination)
# ---------------------------------------------------------------------------

class TopologicalLoss:
  """Weighted sum of BettiNumber + SkeletonRecall + CenterlineDice losses.

  Args:
    w_betti:     weight for BettiNumberLoss      (default 1.0)
    w_skeleton:  weight for SkeletonRecallLoss   (default 1.0)
    w_cl_dice:   weight for CenterlineDiceLoss   (default 1.0)
    betti_kwargs:    extra kwargs forwarded to BettiNumberLoss
    skeleton_kwargs: extra kwargs forwarded to SkeletonRecallLoss
    cl_dice_kwargs:  extra kwargs forwarded to CenterlineDiceLoss
  """

  def __init__(
    self,
    w_betti:    float = 1.0,
    w_skeleton: float = 1.0,
    w_cl_dice:  float = 1.0,
    betti_kwargs:    dict | None = None,
    skeleton_kwargs: dict | None = None,
    cl_dice_kwargs:  dict | None = None,
  ) -> None:
    self.w_betti    = float(w_betti)
    self.w_skeleton = float(w_skeleton)
    self.w_cl_dice  = float(w_cl_dice)
    self.betti_loss    = BettiNumberLoss   (**(betti_kwargs    or {}))
    self.skeleton_loss = SkeletonRecallLoss(**(skeleton_kwargs or {}))
    self.cl_dice_loss  = CenterlineDiceLoss(**(cl_dice_kwargs  or {}))

  def __call__(self, pred: "Tensor", gt: "Tensor") -> "Tensor":
    terms: list[Tensor] = []
    if self.w_betti > 0.0:
      terms.append(self.betti_loss(pred, gt)    * self.w_betti)
    if self.w_skeleton > 0.0:
      terms.append(self.skeleton_loss(pred, gt) * self.w_skeleton)
    if self.w_cl_dice > 0.0:
      terms.append(self.cl_dice_loss(pred, gt)  * self.w_cl_dice)
    if not terms:
      return Tensor(np.array(0.0, dtype=np.float32))
    return Tensor.stack(terms).sum()
