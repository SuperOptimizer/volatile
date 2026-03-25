from __future__ import annotations
import os
import shutil
import tempfile

import numpy as np
import pytest
import zarr

# ---------------------------------------------------------------------------
# Fixture: temp directory cleaned up after each test
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp(tmp_path):
  return tmp_path


def _make_volume(path, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1", fill=None):
  """Create a synthetic zarr array at path."""
  arr = zarr.open_array(str(path), mode="w", shape=shape, chunks=chunks, dtype=dtype)
  data = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape) % 256
  if fill is not None:
    data[:] = fill
  arr[:] = data.astype(dtype)
  return arr


# ---------------------------------------------------------------------------
# threshold_zarr
# ---------------------------------------------------------------------------

def test_threshold_basic(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  _make_volume(src, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")

  from volatile.zarr_tasks import threshold_zarr
  threshold_zarr(str(src), str(dst), low=100, high=200, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  src_data = zarr.open_array(str(src), mode="r")[:].astype(np.float32)

  expected = np.where((src_data >= 100) & (src_data <= 200), np.uint8(255), np.uint8(0))
  np.testing.assert_array_equal(out, expected)


def test_threshold_all_below(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  _make_volume(src, shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1", fill=10)

  from volatile.zarr_tasks import threshold_zarr
  threshold_zarr(str(src), str(dst), low=100, high=200, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  assert out.max() == 0, "all voxels should be 0 (below range)"


def test_threshold_all_above(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  _make_volume(src, shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1", fill=250)

  from volatile.zarr_tasks import threshold_zarr
  threshold_zarr(str(src), str(dst), low=100, high=200, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  assert out.max() == 0, "all voxels should be 0 (above range)"


def test_threshold_all_inside(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  _make_volume(src, shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1", fill=150)

  from volatile.zarr_tasks import threshold_zarr
  threshold_zarr(str(src), str(dst), low=100, high=200, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  assert out.min() == 255, "all voxels should be 255 (inside range)"


# ---------------------------------------------------------------------------
# recompress_zarr
# ---------------------------------------------------------------------------

def test_recompress_data_preserved(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  arr = _make_volume(src, shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")
  original = arr[:]

  from volatile.zarr_tasks import recompress_zarr
  recompress_zarr(str(src), str(dst), codec="blosc", level=3, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  np.testing.assert_array_equal(out, original)


def test_recompress_zstd(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  arr = _make_volume(src, shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
  original = arr[:]

  from volatile.zarr_tasks import recompress_zarr
  recompress_zarr(str(src), str(dst), codec="zstd", level=1, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  np.testing.assert_array_equal(out, original)


# ---------------------------------------------------------------------------
# threshold + recompress roundtrip
# ---------------------------------------------------------------------------

def test_threshold_then_recompress_roundtrip(tmp):
  """Full pipeline: threshold a volume, then recompress the result."""
  src    = tmp / "src.zarr"
  thresh = tmp / "thresh.zarr"
  final  = tmp / "final.zarr"

  # Create source with known values
  arr = zarr.open_array(str(src), mode="w", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="u1")
  data = np.zeros((8, 8, 8), dtype=np.uint8)
  data[2:6, 2:6, 2:6] = 150  # inside [100,200]
  arr[:] = data

  from volatile.zarr_tasks import threshold_zarr, recompress_zarr
  threshold_zarr(str(src), str(thresh), low=100, high=200, workers=1)
  recompress_zarr(str(thresh), str(final), codec="blosc", level=5, workers=1)

  out = zarr.open_array(str(final), mode="r")[:]
  # Inner cube should be 255, outer should be 0
  assert out[4, 4, 4] == 255
  assert out[0, 0, 0] == 0
  assert out.shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# merge_zarr
# ---------------------------------------------------------------------------

def test_merge_max(tmp):
  src1 = tmp / "s1.zarr"
  src2 = tmp / "s2.zarr"
  dst  = tmp / "out.zarr"
  a1 = zarr.open_array(str(src1), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
  a2 = zarr.open_array(str(src2), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
  a1[:] = np.full((4, 4, 4), 100, dtype=np.uint8)
  a2[:] = np.full((4, 4, 4), 200, dtype=np.uint8)

  from volatile.zarr_tasks import merge_zarr
  merge_zarr(str(src1), str(src2), str(dst), op="max", workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  assert out.min() == 200 and out.max() == 200


# ---------------------------------------------------------------------------
# transpose_zarr
# ---------------------------------------------------------------------------

def test_transpose_shape(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  arr = zarr.open_array(str(src), mode="w", shape=(2, 4, 6), chunks=(2, 2, 2), dtype="u1")
  arr[:] = np.arange(48, dtype=np.uint8).reshape(2, 4, 6)

  from volatile.zarr_tasks import transpose_zarr
  transpose_zarr(str(src), str(dst), axes=(2, 1, 0), workers=1)

  out_arr = zarr.open_array(str(dst), mode="r")
  assert out_arr.shape == (6, 4, 2), f"expected (6,4,2) got {out_arr.shape}"


# ---------------------------------------------------------------------------
# remap_zarr
# ---------------------------------------------------------------------------

def test_remap_identity(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  arr = zarr.open_array(str(src), mode="w", shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")
  data = np.arange(64, dtype=np.uint8).reshape(4, 4, 4)
  arr[:] = data

  lut = list(range(256))  # identity
  from volatile.zarr_tasks import remap_zarr
  remap_zarr(str(src), str(dst), lut, workers=1)

  out = zarr.open_array(str(dst), mode="r")[:]
  np.testing.assert_array_equal(out, data)


def test_remap_invalid_lut(tmp):
  src = tmp / "src.zarr"
  _make_volume(src)
  lut = list(range(128))  # wrong size
  from volatile.zarr_tasks import remap_zarr
  with pytest.raises(ValueError, match="256"):
    remap_zarr(str(src), str(tmp / "dst.zarr"), lut)


# ---------------------------------------------------------------------------
# scale_zarr
# ---------------------------------------------------------------------------

def test_scale_output_shape(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  arr = zarr.open_array(str(src), mode="w", shape=(8, 8, 8), chunks=(4, 4, 4), dtype="f4")
  arr[:] = np.ones((8, 8, 8), dtype=np.float32)

  from volatile.zarr_tasks import scale_zarr
  scale_zarr(str(src), str(dst), factor=2, workers=1)

  out = zarr.open_array(str(dst), mode="r")
  assert out.shape == (4, 4, 4), f"expected (4,4,4) got {out.shape}"


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def test_cli_threshold(tmp):
  src = tmp / "src.zarr"
  dst = tmp / "dst.zarr"
  _make_volume(src, shape=(4, 4, 4), chunks=(2, 2, 2), dtype="u1")

  from volatile.zarr_tasks import main
  main(["threshold", str(src), str(dst), "--low", "50", "--high", "200"])
  assert (tmp / "dst.zarr").exists()
