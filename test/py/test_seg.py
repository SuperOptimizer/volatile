from __future__ import annotations
import math
import os
import tempfile
import inspect

# ---------------------------------------------------------------------------
# QuadSurface basic API
# ---------------------------------------------------------------------------

def test_create_surface():
  from volatile.seg import QuadSurface
  surf = QuadSurface(rows=10, cols=10)
  assert surf.rows == 10
  assert surf.cols == 10

def test_set_get_vertex():
  from volatile.seg import QuadSurface
  surf = QuadSurface(rows=100, cols=100)
  surf.set(50, 50, (100.0, 200.0, 300.0))
  xyz = surf.get(50, 50)
  assert abs(xyz[0] - 100.0) < 1e-6
  assert abs(xyz[1] - 200.0) < 1e-6
  assert abs(xyz[2] - 300.0) < 1e-6

def test_default_vertices_zero():
  from volatile.seg import QuadSurface
  surf = QuadSurface(rows=5, cols=5)
  for r in range(5):
    for c in range(5):
      assert surf.get(r, c) == (0.0, 0.0, 0.0)

def test_out_of_bounds_raises():
  from volatile.seg import QuadSurface
  import pytest
  surf = QuadSurface(rows=10, cols=10)
  try:
    surf.get(10, 0)
    assert False, "expected IndexError"
  except IndexError:
    pass

# ---------------------------------------------------------------------------
# brush_apply / undo
# ---------------------------------------------------------------------------

def test_brush_apply_moves_center():
  from volatile.seg import QuadSurface, brush_apply
  surf = QuadSurface(rows=20, cols=20)
  # Give center vertex a non-degenerate normal by placing it above a flat plane
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))

  cx, cy = 10, 10
  before = surf.get(cx, cy)
  edit = brush_apply(surf, u=float(cx), v=float(cy), delta=5.0, radius=3.0, sigma=1.5)
  after = surf.get(cx, cy)

  # At least one coordinate should have changed
  changed = any(abs(after[i] - before[i]) > 1e-6 for i in range(3))
  assert changed, f"brush_apply did not move center vertex: before={before} after={after}"

def test_brush_apply_returns_edit():
  from volatile.seg import QuadSurface, brush_apply, SurfaceEdit
  surf = QuadSurface(rows=20, cols=20)
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))
  edit = brush_apply(surf, u=10.0, v=10.0, delta=2.0, radius=3.0, sigma=1.5)
  assert isinstance(edit, SurfaceEdit)

def test_undo_restores_vertices():
  from volatile.seg import QuadSurface, brush_apply
  surf = QuadSurface(rows=20, cols=20)
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))

  # Snapshot affected region before edit
  snap = {(r, c): surf.get(r, c) for r in range(20) for c in range(20)}

  edit = brush_apply(surf, u=10.0, v=10.0, delta=5.0, radius=4.0, sigma=2.0)
  edit.undo(surf)

  for (r, c), orig in snap.items():
    cur = surf.get(r, c)
    for i in range(3):
      assert abs(cur[i] - orig[i]) < 1e-5, f"undo failed at ({r},{c})[{i}]: {orig[i]} -> {cur[i]}"

def test_brush_does_not_affect_distant_vertices():
  from volatile.seg import QuadSurface, brush_apply
  surf = QuadSurface(rows=30, cols=30)
  for r in range(30):
    for c in range(30):
      surf.set(r, c, (float(c), float(r), 0.0))

  before_far = surf.get(0, 0)
  brush_apply(surf, u=29.0, v=29.0, delta=10.0, radius=3.0, sigma=1.5)
  after_far = surf.get(0, 0)
  assert before_far == after_far

# ---------------------------------------------------------------------------
# line_apply
# ---------------------------------------------------------------------------

def test_line_apply_returns_edit():
  from volatile.seg import QuadSurface, line_apply, SurfaceEdit
  surf = QuadSurface(rows=20, cols=20)
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))
  edit = line_apply(surf, 5.0, 5.0, 15.0, 15.0, delta=1.0, radius=2.0, sigma=1.0)
  assert isinstance(edit, SurfaceEdit)

def test_line_apply_undo():
  from volatile.seg import QuadSurface, line_apply
  surf = QuadSurface(rows=20, cols=20)
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))
  snap = {(r, c): surf.get(r, c) for r in range(20) for c in range(20)}
  edit = line_apply(surf, 2.0, 2.0, 18.0, 18.0, delta=3.0, radius=2.0, sigma=1.0)
  edit.undo(surf)
  for (r, c), orig in snap.items():
    cur = surf.get(r, c)
    for i in range(3):
      assert abs(cur[i] - orig[i]) < 1e-5

# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------

def test_save_load_roundtrip():
  from volatile.seg import QuadSurface
  surf = QuadSurface(rows=5, cols=5)
  surf.set(2, 3, (1.0, 2.0, 3.0))
  surf.set(0, 0, (-1.5, 0.5, 7.0))

  with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
    path = f.name
  try:
    surf.save(path)
    surf2 = QuadSurface.load(path)
    assert surf2.rows == 5
    assert surf2.cols == 5
    xyz = surf2.get(2, 3)
    assert abs(xyz[0] - 1.0) < 1e-6
    assert abs(xyz[1] - 2.0) < 1e-6
    assert abs(xyz[2] - 3.0) < 1e-6
    xyz0 = surf2.get(0, 0)
    assert abs(xyz0[0] - (-1.5)) < 1e-6
  finally:
    os.unlink(path)

# ---------------------------------------------------------------------------
# ink.detect_ink signature
# ---------------------------------------------------------------------------

def test_detect_ink_signature():
  from volatile.ink import detect_ink
  sig = inspect.signature(detect_ink)
  params = list(sig.parameters.keys())
  assert "volume" in params
  assert "surface" in params
  assert "model_path" in params

def test_detect_ink_returns_without_model():
  """detect_ink must return something even when no model/volume is available."""
  from volatile.seg import QuadSurface
  from volatile.ink import detect_ink
  surf = QuadSurface(rows=4, cols=4)
  result = detect_ink(None, surf, model_path=None)
  # Should be a 4×4 structure (list-of-lists or numpy array)
  assert result is not None
  if hasattr(result, "shape"):
    assert result.shape == (4, 4)
  else:
    assert len(result) == 4 and len(result[0]) == 4
