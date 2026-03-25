"""test_integration.py — end-to-end Python integration tests.

All tests guard C extension imports with pytest.importorskip / try-except so
they pass cleanly when the native extension has not been built.
"""
from __future__ import annotations

import json
import math
import os
import struct
import tempfile

import numpy as np
import pytest
import zarr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ome_zarr(path: str, data: np.ndarray, chunks=(4, 4, 4)) -> None:
  """Write a minimal OME-Zarr v2 group that vol_open can read."""
  root = zarr.open_group(path, mode="w", zarr_format=2)
  root.create_array("0", shape=data.shape, chunks=chunks, dtype=data.dtype)
  root["0"][:] = data
  attrs = {
    "multiscales": [{
      "version": "0.4",
      "axes": [{"name": ax, "type": "space"} for ax in ("z", "y", "x")],
      "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1, 1, 1]}]}],
    }]
  }
  with open(os.path.join(path, ".zattrs"), "w") as f:
    json.dump(attrs, f)


# ---------------------------------------------------------------------------
# 1. Create synthetic volume via zarr_tasks, read back via vol_open
# ---------------------------------------------------------------------------

def test_vol_open_reads_zarr_tasks_output(tmp_path):
  """threshold_zarr output is a valid zarr that vol_open can open."""
  _core = pytest.importorskip("volatile._core", reason="C extension not built")

  src = str(tmp_path / "src.zarr")
  thresh = str(tmp_path / "thresh.zarr")

  # Build a zarr_tasks-compatible plain array first
  from volatile.zarr_tasks import threshold_zarr
  arr = zarr.open_array(src, mode="w", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")
  arr[:] = np.arange(512, dtype=np.uint8).reshape(8, 8, 8)
  threshold_zarr(src, thresh, low=100, high=200, workers=1)

  # Convert the threshold output into an OME-Zarr that vol_open understands
  thresh_data = zarr.open_array(thresh, mode="r")[:]
  ome_path = str(tmp_path / "ome.zarr")
  _make_ome_zarr(ome_path, thresh_data, chunks=(4, 4, 4))

  vol = _core.vol_open(ome_path)
  try:
    assert _core.vol_num_levels(vol) >= 1
    shape = _core.vol_shape(vol, 0)
    assert shape == (8, 8, 8), f"expected (8,8,8), got {shape}"
  finally:
    _core.vol_free(vol)


def test_vol_open_sample(tmp_path):
  """vol_sample returns a float for a known voxel value."""
  _core = pytest.importorskip("volatile._core", reason="C extension not built")

  data = np.zeros((8, 8, 8), dtype=np.uint8)
  data[0, 0, 0] = 42
  ome_path = str(tmp_path / "vol.zarr")
  _make_ome_zarr(ome_path, data, chunks=(4, 4, 4))

  vol = _core.vol_open(ome_path)
  try:
    # vol_sample uses center-of-voxel coords: voxel [0,0,0] is at (z=0.5, y=0.5, x=0.5)
    val = _core.vol_sample(vol, 0, 0.5, 0.5, 0.5)
    assert isinstance(val, float)
    assert abs(val - 42.0) < 1.0
  finally:
    _core.vol_free(vol)


# ---------------------------------------------------------------------------
# 2. Structure tensor on synthetic data — verify 6-channel output shape
# ---------------------------------------------------------------------------

def test_structure_tensor_output_shape():
  """C binding: structure_tensor_3d(bytes, d, h, w, ds, ss) -> 6-channel bytes."""
  _core = pytest.importorskip("volatile._core", reason="C extension not built")
  if not hasattr(_core, "structure_tensor_3d"):
    pytest.skip("structure_tensor_3d not in this _core build (rebuild required)")
  d, h, w = 4, 8, 8
  data = np.random.rand(d, h, w).astype(np.float32)
  raw = data.tobytes()
  result_bytes = _core.structure_tensor_3d(raw, d, h, w, 1.0, 1.0)
  result = np.frombuffer(result_bytes, dtype=np.float32).reshape(d, h, w, 6)
  assert result.shape == (d, h, w, 6), f"unexpected shape {result.shape}"


def test_structure_tensor_numpy():
  """Python wrapper: structure_tensor_3d accepts ndarray and returns ndarray."""
  _core = pytest.importorskip("volatile._core", reason="C extension not built")
  if not hasattr(_core, "structure_tensor_3d"):
    pytest.skip("structure_tensor_3d not in this _core build (rebuild required)")
  from volatile.imgproc import structure_tensor_3d
  vol = np.random.rand(4, 8, 8).astype(np.float32)
  result = structure_tensor_3d(vol, 1.0, 1.0)
  assert result.shape == (4, 8, 8, 6)
  assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. QuadSurface: create, brush edit, undo, verify roundtrip
# ---------------------------------------------------------------------------

def test_surface_brush_undo_roundtrip():
  from volatile.seg import QuadSurface, brush_apply

  surf = QuadSurface(rows=20, cols=20)
  for r in range(20):
    for c in range(20):
      surf.set(r, c, (float(c), float(r), 0.0))

  # Snapshot
  snap = {(r, c): surf.get(r, c) for r in range(20) for c in range(20)}

  edit = brush_apply(surf, u=10.0, v=10.0, delta=5.0, radius=4.0, sigma=2.0)

  # Some vertex near center must have moved
  center = surf.get(10, 10)
  assert any(abs(center[i] - snap[(10, 10)][i]) > 1e-4 for i in range(3)), \
    "brush_apply did not displace center vertex"

  # Undo must restore every vertex
  edit.undo(surf)
  for (r, c), orig in snap.items():
    cur = surf.get(r, c)
    for i in range(3):
      assert abs(cur[i] - orig[i]) < 1e-5, f"undo mismatch at ({r},{c})[{i}]"


def test_surface_save_load(tmp_path):
  from volatile.seg import QuadSurface

  surf = QuadSurface(rows=10, cols=10)
  surf.set(3, 7, (1.5, -2.0, 99.0))
  path = str(tmp_path / "surf.json")
  surf.save(path)

  surf2 = QuadSurface.load(path)
  xyz = surf2.get(3, 7)
  assert abs(xyz[0] - 1.5) < 1e-6
  assert abs(xyz[1] - (-2.0)) < 1e-6
  assert abs(xyz[2] - 99.0) < 1e-6


# ---------------------------------------------------------------------------
# 4. tiled_infer with a tiny UNet on a 64×64 image — verify output shape
# ---------------------------------------------------------------------------

def test_tiled_infer_output_shape():
  pytest.importorskip("tinygrad", reason="tinygrad not installed")
  from volatile.ml.model import UNet
  from volatile.ml.infer import tiled_infer

  model = UNet(in_channels=1, out_channels=1, base_channels=4, num_levels=2)
  img = np.random.rand(64, 64).astype(np.float32)
  out = tiled_infer(model, img, tile_h=32, tile_w=32, overlap=8, batch_size=2)
  assert out.shape == (1, 64, 64), f"expected (1,64,64), got {out.shape}"
  assert out.dtype == np.float32


def test_tiled_infer_multichannel():
  """(C, H, W) input is treated as (Z=1, H, W) with C discarded — returns (Z, out_ch, H, W)."""
  pytest.importorskip("tinygrad", reason="tinygrad not installed")
  from volatile.ml.model import UNet
  from volatile.ml.infer import tiled_infer

  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  # 3-D input (Z, H, W) — tiled_infer treats dim-3 as Z slices with C=1
  img = np.random.rand(2, 64, 64).astype(np.float32)  # (Z=2, H, W)
  out = tiled_infer(model, img, tile_h=32, tile_w=32, overlap=8)
  assert out.shape == (2, 2, 64, 64), f"expected (2,2,64,64), got {out.shape}"


# ---------------------------------------------------------------------------
# 5. Eval metrics: dice and iou on synthetic binary masks
# ---------------------------------------------------------------------------

def test_dice_perfect():
  from volatile.eval import dice_score
  mask = np.ones((10, 10), dtype=np.uint8)
  assert abs(dice_score(mask, mask) - 1.0) < 1e-6


def test_dice_no_overlap():
  from volatile.eval import dice_score
  pred = np.zeros((10, 10), dtype=np.uint8)
  pred[:5, :] = 1
  gt = np.zeros((10, 10), dtype=np.uint8)
  gt[5:, :] = 1
  assert abs(dice_score(pred, gt)) < 1e-6


def test_iou_perfect():
  from volatile.eval import iou_score
  mask = np.ones((8, 8), dtype=np.uint8)
  assert abs(iou_score(mask, mask) - 1.0) < 1e-6


def test_iou_partial_overlap():
  from volatile.eval import iou_score
  pred = np.zeros((10, 10), dtype=np.uint8)
  gt   = np.zeros((10, 10), dtype=np.uint8)
  pred[:, :6] = 1   # 60 pixels
  gt[:, 4:]   = 1   # 60 pixels, overlap = 20 pixels
  # IoU = 20 / (60 + 60 - 20) = 20/100 = 0.2
  val = iou_score(pred, gt)
  assert abs(val - 0.2) < 0.01, f"expected ~0.2, got {val}"


def test_dice_and_iou_consistent():
  """For a given overlap, dice >= iou (by AM-GM relationship)."""
  from volatile.eval import dice_score, iou_score
  rng = np.random.default_rng(42)
  pred = (rng.random((20, 20)) > 0.4).astype(np.uint8)
  gt   = (rng.random((20, 20)) > 0.4).astype(np.uint8)
  d = dice_score(pred, gt)
  i = iou_score(pred, gt)
  assert d >= i - 1e-6, f"expected dice >= iou, got dice={d:.4f} iou={i:.4f}"


# ---------------------------------------------------------------------------
# 6. annot_store roundtrip (skip if module absent)
# ---------------------------------------------------------------------------

def test_annot_store_roundtrip(tmp_path):
  annot_store = pytest.importorskip("volatile.annot_store", reason="annot_store not yet implemented")

  # If the module exists, exercise a basic save/load cycle
  store_path = str(tmp_path / "annot.db")
  store = annot_store.AnnotStore(store_path)

  annot = {"surface_id": "surf_001", "label": "ink", "confidence": 0.9, "points": [[1, 2, 3], [4, 5, 6]]}
  store.save(annot)

  records = store.load(surface_id="surf_001")
  assert len(records) >= 1
  assert records[0]["label"] == "ink"


# ---------------------------------------------------------------------------
# 7. surface_metrics on a flat QuadSurface
# ---------------------------------------------------------------------------

def test_surface_metrics_flat():
  from volatile.seg import QuadSurface
  from volatile.metrics import surface_metrics

  surf = QuadSurface(rows=5, cols=5)
  for r in range(5):
    for c in range(5):
      surf.set(r, c, (float(c), float(r), 0.0))  # flat z=0 plane

  m = surface_metrics(surf)
  assert "area" in m and "mean_curvature" in m and "smoothness" in m
  assert m["area"] > 0, "flat surface should have positive area"
  assert m["smoothness"] >= 0


# ---------------------------------------------------------------------------
# 8. zarr_tasks pipeline: threshold -> recompress -> read back values
# ---------------------------------------------------------------------------

def test_pipeline_threshold_recompress(tmp_path):
  src    = str(tmp_path / "src.zarr")
  thresh = str(tmp_path / "thresh.zarr")
  final  = str(tmp_path / "final.zarr")

  arr = zarr.open_array(src, mode="w", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")
  data = np.zeros((8, 8, 8), dtype=np.uint8)
  data[2:6, 2:6, 2:6] = 150
  arr[:] = data

  from volatile.zarr_tasks import threshold_zarr, recompress_zarr
  threshold_zarr(src, thresh, low=100, high=200, workers=1)
  recompress_zarr(thresh, final, codec="zstd", level=1, workers=1)

  out = zarr.open_array(final, mode="r")[:]
  assert out[4, 4, 4] == 255, "inner cube voxel should be 255 after threshold"
  assert out[0, 0, 0] == 0,   "outer voxel should be 0 after threshold"
