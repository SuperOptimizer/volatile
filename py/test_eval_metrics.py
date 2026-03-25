"""test_eval_metrics.py — unit tests for eval.py and metrics.py using synthetic data.

Run with:  python -m pytest py/test_eval_metrics.py
       or: python py/test_eval_metrics.py
"""
from __future__ import annotations
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from volatile.eval import (
  dice_score, iou_score, hausdorff_distance,
  centerline_dice, connected_components_3d, critical_components,
)
from volatile.metrics import surface_metrics, segment_coverage

try:
  import numpy as np
  _NP = True
except ImportError:
  _NP = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box(Z, Y, X, z0, z1, y0, y1, x0, x1):
  """Flat bool list with a filled rectangular box."""
  out = [False] * (Z * Y * X)
  for z in range(z0, z1):
    for y in range(y0, y1):
      for x in range(x0, x1):
        out[z * Y * X + y * X + x] = True
  return out


SHAPE = (8, 8, 8)


# ---------------------------------------------------------------------------
# dice_score
# ---------------------------------------------------------------------------

def test_dice_perfect():
  a = [True, False, True]
  assert dice_score(a, a) == 1.0, "perfect overlap should be 1"

def test_dice_no_overlap():
  a = [True, False]
  b = [False, True]
  assert dice_score(a, b) == 0.0

def test_dice_empty():
  assert dice_score([], []) == 1.0

def test_dice_half_overlap():
  # pred=[T,T,F,F]  gt=[T,F,T,F]  intersection=1  sum=2+2=4  dice=0.5
  assert dice_score([True,True,False,False], [True,False,True,False]) == 0.5

def test_dice_numpy():
  if not _NP: return
  a = np.array([True, True, False])
  b = np.array([True, False, True])
  d = dice_score(a, b)
  assert abs(d - 0.5) < 1e-9

# ---------------------------------------------------------------------------
# iou_score
# ---------------------------------------------------------------------------

def test_iou_perfect():
  a = [True, True, False]
  assert iou_score(a, a) == 1.0

def test_iou_no_overlap():
  assert iou_score([True, False], [False, True]) == 0.0

def test_iou_empty():
  assert iou_score([], []) == 1.0

def test_iou_partial():
  # inter=1 union=3  -> 1/3
  a = [True, True, False]
  b = [True, False, True]
  assert abs(iou_score(a, b) - 1/3) < 1e-9

def test_iou_relationship_with_dice():
  # IoU = dice / (2 - dice)
  a = [True, True, False, True]
  b = [True, False, True, True]
  d = dice_score(a, b)
  expected_iou = d / (2.0 - d)
  assert abs(iou_score(a, b) - expected_iou) < 1e-9

# ---------------------------------------------------------------------------
# hausdorff_distance
# ---------------------------------------------------------------------------

def test_hausdorff_identical():
  shape = (4, 4, 4)
  a = _box(*shape, 1, 3, 1, 3, 1, 3)
  assert hausdorff_distance(a, a, shape) == 0.0

def test_hausdorff_disjoint():
  shape = (4, 4, 4)
  a = _box(*shape, 0, 1, 0, 1, 0, 1)   # single voxel at origin
  b = _box(*shape, 3, 4, 3, 4, 3, 4)   # single voxel far corner
  d = hausdorff_distance(a, b, shape)
  assert d == math.sqrt(3**2 + 3**2 + 3**2)

def test_hausdorff_empty_returns_inf():
  shape = (4, 4, 4)
  a = _box(*shape, 0, 1, 0, 1, 0, 1)
  empty = [False] * (4*4*4)
  assert hausdorff_distance(a, empty, shape) == float("inf")

def test_hausdorff_both_empty_returns_zero():
  shape = (4, 4, 4)
  empty = [False] * (4*4*4)
  assert hausdorff_distance(empty, empty, shape) == 0.0

def test_hausdorff_numpy():
  if not _NP: return
  a = np.zeros((4,4,4), dtype=bool)
  b = np.zeros((4,4,4), dtype=bool)
  a[0,0,0] = True
  b[0,0,3] = True
  d = hausdorff_distance(a, b)
  assert abs(d - 3.0) < 1e-6

# ---------------------------------------------------------------------------
# centerline_dice
# ---------------------------------------------------------------------------

def test_centerline_dice_delegates():
  skel = [True, False, True, True]
  assert centerline_dice(skel, skel) == 1.0
  assert centerline_dice(skel, [False]*4) == 0.0

# ---------------------------------------------------------------------------
# connected_components_3d
# ---------------------------------------------------------------------------

def test_cc_empty():
  shape = (2, 2, 2)
  mask = [False] * 8
  labels = connected_components_3d(mask, shape)
  assert all(l == 0 for l in labels)

def test_cc_single_component():
  shape = (2, 2, 2)
  mask = [True] * 8
  labels = connected_components_3d(mask, shape)
  assert max(labels) == 1
  assert all(l == 1 for l in labels)

def test_cc_two_components():
  # 1-D like: shape (1,1,6) with gap at index 3
  shape = (1, 1, 6)
  mask = [True, True, True, False, True, True]
  labels = connected_components_3d(mask, shape)
  assert max(labels) == 2
  assert labels[0] == labels[1] == labels[2]
  assert labels[3] == 0
  assert labels[4] == labels[5]
  assert labels[0] != labels[4]

def test_cc_numpy():
  if not _NP: return
  a = np.zeros((4,4,4), dtype=bool)
  a[0:2, 0:2, 0:2] = True
  a[2:4, 2:4, 2:4] = True
  out = connected_components_3d(a)
  assert isinstance(out, np.ndarray)
  assert out.shape == (4, 4, 4)
  assert len(set(out.ravel()) - {0}) == 2  # two components

# ---------------------------------------------------------------------------
# critical_components
# ---------------------------------------------------------------------------

def test_critical_components_perfect():
  shape = (4, 4, 4)
  a = _box(*shape, 0, 2, 0, 2, 0, 2)
  r = critical_components(a, a, shape)
  assert r["pred_components"] == r["gt_components"] == 1
  assert r["tp_components"] == 1
  assert r["fp_components"] == 0
  assert r["fn_components"] == 0
  assert r["component_dice"] == 1.0

def test_critical_components_split():
  # GT: one large box; pred: two smaller separate boxes => 1 GT, 2 pred
  shape = (4, 8, 4)
  gt   = _box(*shape, 0, 4, 0, 8, 0, 4)   # full
  pred = _box(*shape, 0, 4, 0, 3, 0, 4)   # left half
  # add right half with a gap
  for z in range(4):
    for y in range(5, 8):
      for x in range(4):
        pred[z * 8 * 4 + y * 4 + x] = True
  r = critical_components(pred, gt, shape)
  assert r["gt_components"] == 1
  assert r["pred_components"] == 2

# ---------------------------------------------------------------------------
# surface_metrics
# ---------------------------------------------------------------------------

def _flat_surface(rows, cols, scale=1.0):
  """Flat grid in z=0 plane; points are (x, y, 0)."""
  pts = [(c * scale, r * scale, 0.0) for r in range(rows) for c in range(cols)]
  return {"rows": rows, "cols": cols, "points": pts}

def test_surface_metrics_flat_area():
  surf = _flat_surface(5, 5, scale=1.0)
  m = surface_metrics(surf)
  # 4x4 quads each of area 1.0 = 16 total
  assert abs(m["area"] - 16.0) < 1e-4
  assert m["rows"] == 5
  assert m["cols"] == 5

def test_surface_metrics_curvature_flat():
  surf = _flat_surface(6, 6, scale=1.0)
  m = surface_metrics(surf)
  # Flat surface => curvature near 0
  assert m["mean_curvature"] < 1e-6

def test_surface_metrics_smoothness_flat():
  surf = _flat_surface(6, 6, scale=1.0)
  m = surface_metrics(surf)
  assert m["smoothness"] < 1e-6

def test_surface_metrics_bumpy():
  # Introduce a bump — curvature and smoothness should be > 0
  rows, cols = 6, 6
  pts = []
  for r in range(rows):
    for c in range(cols):
      z = 1.0 if (r == 3 and c == 3) else 0.0
      pts.append((float(c), float(r), z))
  surf = {"rows": rows, "cols": cols, "points": pts}
  m = surface_metrics(surf)
  assert m["mean_curvature"] > 0
  assert m["smoothness"] > 0

def test_surface_metrics_scale():
  # Area scales as scale^2
  surf1 = _flat_surface(5, 5, scale=1.0)
  surf2 = _flat_surface(5, 5, scale=2.0)
  m1 = surface_metrics(surf1)
  m2 = surface_metrics(surf2)
  assert abs(m2["area"] - m1["area"] * 4.0) < 1e-4

# ---------------------------------------------------------------------------
# segment_coverage
# ---------------------------------------------------------------------------

def test_coverage_all_valid():
  if not _NP: return
  vol = np.ones((4, 4, 4), dtype=float)
  pts = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
  assert segment_coverage(pts, vol) == 1.0

def test_coverage_none_valid():
  if not _NP: return
  vol = np.zeros((4, 4, 4), dtype=float)
  pts = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
  assert segment_coverage(pts, vol) == 0.0

def test_coverage_partial():
  if not _NP: return
  vol = np.zeros((4, 4, 4), dtype=float)
  vol[1, 1, 1] = 1.0  # only this voxel is valid
  pts = [(1.0, 1.0, 1.0), (2.0, 2.0, 2.0)]
  assert segment_coverage(pts, vol) == 0.5

def test_coverage_out_of_bounds():
  if not _NP: return
  vol = np.ones((4, 4, 4), dtype=float)
  pts = [(10.0, 10.0, 10.0)]  # outside volume
  assert segment_coverage(pts, vol) == 0.0

def test_coverage_empty_surface():
  if not _NP: return
  vol = np.ones((4, 4, 4), dtype=float)
  assert segment_coverage([], vol) == 1.0

def test_coverage_nan_treated_invalid():
  if not _NP: return
  vol = np.full((4, 4, 4), float("nan"))
  pts = [(1.0, 1.0, 1.0)]
  assert segment_coverage(pts, vol) == 0.0

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  passed = failed = 0
  for t in tests:
    try:
      t()
      print(f"  PASS  {t.__name__}")
      passed += 1
    except Exception as e:
      print(f"  FAIL  {t.__name__}: {e}")
      failed += 1
  print(f"\n{passed} passed, {failed} failed")
  sys.exit(0 if failed == 0 else 1)
