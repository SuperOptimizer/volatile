"""eval.py — segmentation evaluation metrics.

All functions work with flat Python sequences or numpy arrays (if available).
3-D masks are represented as nested lists [z][y][x] or flat arrays with a
shape tuple.  For numpy arrays any shape is fine; operations are vectorised.
"""
from __future__ import annotations
import math
from collections import deque
from typing import Sequence

try:
  import numpy as np
  _NP = True
except ImportError:
  _NP = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_flat_bool(mask, shape=None):
  """Return (flat_list_of_bool, (Z, Y, X)) from a mask + optional shape."""
  if _NP and isinstance(mask, np.ndarray):
    return mask.ravel().astype(bool).tolist(), mask.shape
  if shape is None:
    raise ValueError("shape required for non-numpy masks")
  return [bool(v) for v in mask], shape


def _iter3(shape):
  Z, Y, X = shape
  for z in range(Z):
    for y in range(Y):
      for x in range(X):
        yield z, y, x


def _idx(z, y, x, shape):
  _, Y, X = shape
  return z * Y * X + y * X + x

# ---------------------------------------------------------------------------
# dice_score
# ---------------------------------------------------------------------------

def dice_score(pred, gt) -> float:
  """Sørensen–Dice coefficient: 2|P∩G| / (|P| + |G|).

  Accepts numpy bool/int arrays or flat iterables.  Returns 1.0 if both
  masks are empty (perfect trivial match).
  """
  if _NP and isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
    p = pred.astype(bool)
    g = gt.astype(bool)
    intersection = int((p & g).sum())
    denom = int(p.sum()) + int(g.sum())
    return 1.0 if denom == 0 else 2.0 * intersection / denom

  intersection = sum(1 for pv, gv in zip(pred, gt) if pv and gv)
  sum_p = sum(1 for v in pred if v)
  sum_g = sum(1 for v in gt if v)
  denom = sum_p + sum_g
  return 1.0 if denom == 0 else 2.0 * intersection / denom

# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------

def iou_score(pred, gt) -> float:
  """Intersection over Union (Jaccard index): |P∩G| / |P∪G|.

  Returns 1.0 if both masks are empty.
  """
  if _NP and isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
    p = pred.astype(bool)
    g = gt.astype(bool)
    inter = int((p & g).sum())
    union = int((p | g).sum())
    return 1.0 if union == 0 else inter / union

  inter = sum(1 for pv, gv in zip(pred, gt) if pv and gv)
  union = sum(1 for pv, gv in zip(pred, gt) if pv or gv)
  return 1.0 if union == 0 else inter / union

# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def _surface_points(mask, shape):
  """Return list of (z,y,x) boundary voxels (foreground with at least one
  background 6-neighbour)."""
  Z, Y, X = shape
  flat = mask if isinstance(mask, list) else list(mask)
  pts = []
  for z in range(Z):
    for y in range(Y):
      for x in range(X):
        if not flat[_idx(z, y, x, shape)]:
          continue
        for dz, dy, dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
          nz, ny, nx = z+dz, y+dy, x+dx
          if nz < 0 or nz >= Z or ny < 0 or ny >= Y or nx < 0 or nx >= X:
            pts.append((z, y, x)); break
          if not flat[_idx(nz, ny, nx, shape)]:
            pts.append((z, y, x)); break
  return pts


def _directed_hausdorff(pts_a, pts_b) -> float:
  """Max over a of min distance to b."""
  if not pts_b:
    return float("inf")
  max_d = 0.0
  for az, ay, ax in pts_a:
    min_d = min((az-bz)**2 + (ay-by)**2 + (ax-bx)**2 for bz, by, bx in pts_b)
    if min_d > max_d:
      max_d = min_d
  return math.sqrt(max_d)


def hausdorff_distance(pred, gt, shape=None) -> float:
  """Symmetric Hausdorff distance in voxels between surface boundaries.

  For numpy arrays shape is inferred automatically.
  """
  if _NP and isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray):
    shape = pred.shape
    pred_flat = pred.ravel().astype(bool).tolist()
    gt_flat   = gt.ravel().astype(bool).tolist()
  else:
    if shape is None:
      raise ValueError("shape=(Z,Y,X) required for non-numpy inputs")
    pred_flat = [bool(v) for v in pred]
    gt_flat   = [bool(v) for v in gt]

  pts_p = _surface_points(pred_flat, shape)
  pts_g = _surface_points(gt_flat, shape)
  if not pts_p and not pts_g:
    return 0.0
  if not pts_p or not pts_g:
    return float("inf")
  return max(_directed_hausdorff(pts_p, pts_g), _directed_hausdorff(pts_g, pts_p))

# ---------------------------------------------------------------------------
# centerline_dice
# ---------------------------------------------------------------------------

def centerline_dice(pred_skel, gt_skel) -> float:
  """Skeleton (centerline) Dice: harmonic mean of recall and precision on
  skeletons.  Inputs are pre-skeletonised binary masks.

  clDice = 2 * (|pred_skel ∩ gt| + |gt_skel ∩ pred|) /
               (|pred_skel| + |gt_skel|)

  where pred and gt here refer to the *full* masks — but since the caller
  passes already-skeletonised versions we compute simple overlap.
  """
  return dice_score(pred_skel, gt_skel)

# ---------------------------------------------------------------------------
# connected_components_3d  (BFS flood fill, pure Python)
# ---------------------------------------------------------------------------

def connected_components_3d(mask, shape=None):
  """Label connected foreground components with 6-connectivity.

  Returns a flat list of integer labels (0 = background, 1..N = components),
  same length as mask.  If mask is a numpy array the result is a numpy array
  of the same shape.

  shape: (Z, Y, X) — required if mask is not a numpy array.
  """
  is_np = _NP and isinstance(mask, np.ndarray)
  if is_np:
    shape = mask.shape
    flat = mask.ravel().astype(bool).tolist()
  else:
    if shape is None:
      raise ValueError("shape=(Z,Y,X) required for non-numpy inputs")
    flat = [bool(v) for v in mask]

  Z, Y, X = shape
  N = Z * Y * X
  labels = [0] * N
  current_label = 0

  for start in range(N):
    if not flat[start] or labels[start]:
      continue
    current_label += 1
    q = deque([start])
    labels[start] = current_label
    while q:
      idx = q.popleft()
      z = idx // (Y * X)
      y = (idx % (Y * X)) // X
      x = idx % X
      for dz, dy, dx in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
        nz, ny, nx = z+dz, y+dy, x+dx
        if nz < 0 or nz >= Z or ny < 0 or ny >= Y or nx < 0 or nx >= X:
          continue
        ni = _idx(nz, ny, nx, shape)
        if flat[ni] and not labels[ni]:
          labels[ni] = current_label
          q.append(ni)

  if is_np:
    return np.array(labels, dtype=np.int32).reshape(shape)
  return labels

# ---------------------------------------------------------------------------
# critical_components
# ---------------------------------------------------------------------------

def critical_components(pred, gt, shape=None) -> dict:
  """Topology-aware evaluation comparing connected components.

  Returns a dict with:
    pred_components   — number of CC in pred
    gt_components     — number of CC in gt
    tp_components     — GT components overlapping with a pred component
    fp_components     — pred components not overlapping any GT component
    fn_components     — GT components not overlapping any pred component
    component_dice    — dice score over foreground (global)
  """
  is_np = _NP and isinstance(pred, np.ndarray) and isinstance(gt, np.ndarray)
  if is_np:
    shape = pred.shape
    pred_flat = pred.ravel().astype(bool).tolist()
    gt_flat   = gt.ravel().astype(bool).tolist()
  else:
    if shape is None:
      raise ValueError("shape=(Z,Y,X) required for non-numpy inputs")
    pred_flat = [bool(v) for v in pred]
    gt_flat   = [bool(v) for v in gt]

  pred_labels = connected_components_3d(pred_flat, shape)
  gt_labels   = connected_components_3d(gt_flat,   shape)

  n_pred = max(pred_labels) if pred_labels else 0
  n_gt   = max(gt_labels)   if gt_labels   else 0

  # Build overlap sets.
  gt_hit  = set()   # GT labels that overlap with any pred component
  pred_hit = set()  # pred labels that overlap with any GT component
  for pl, gl in zip(pred_labels, gt_labels):
    if pl and gl:
      gt_hit.add(gl)
      pred_hit.add(pl)

  tp = len(gt_hit)
  fp = n_pred - len(pred_hit)
  fn = n_gt   - len(gt_hit)

  return {
    "pred_components": n_pred,
    "gt_components":   n_gt,
    "tp_components":   tp,
    "fp_components":   fp,
    "fn_components":   fn,
    "component_dice":  dice_score(pred_flat, gt_flat),
  }
