"""test_tifxyz.py — unit tests for tifxyz.py (no tifffile dependency required).

Run with:  python -m pytest py/test_tifxyz.py
       or: python py/test_tifxyz.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from volatile.tifxyz import (
  read_tifxyz, write_tifxyz,
  tifxyz_to_mesh, tifxyz_to_mesh_fast,
  mesh_to_tifxyz, mesh_to_tifxyz_fast,
  _write_tiff_fallback, _read_tiff_fallback,
)
from volatile.seg import QuadSurface

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xyz(rows=4, cols=6) -> np.ndarray:
  """Return a deterministic (rows, cols, 3) float32 array."""
  arr = np.zeros((rows, cols, 3), dtype=np.float32)
  for r in range(rows):
    for c in range(cols):
      arr[r, c] = [float(r), float(c), float(r + c)]
  return arr


# ---------------------------------------------------------------------------
# Fallback TIFF roundtrip
# ---------------------------------------------------------------------------

class TestTiffFallback(unittest.TestCase):

  def test_write_read_roundtrip(self):
    xyz = _xyz(4, 6)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
      path = f.name
    try:
      _write_tiff_fallback(path, xyz)
      result = _read_tiff_fallback(path)  # returns (3, H, W)
      result_hwc = np.moveaxis(result, 0, -1)
      np.testing.assert_allclose(result_hwc, xyz, rtol=1e-5)
    finally:
      os.unlink(path)

  def test_write_read_square(self):
    xyz = np.random.rand(8, 8, 3).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
      path = f.name
    try:
      _write_tiff_fallback(path, xyz)
      result = np.moveaxis(_read_tiff_fallback(path), 0, -1)
      np.testing.assert_allclose(result, xyz, rtol=1e-5)
    finally:
      os.unlink(path)

  def test_invalid_magic_raises(self):
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
      f.write(b"IIx" + b"\x00" * 20)
      path = f.name
    try:
      with self.assertRaises((ValueError, Exception)):
        _read_tiff_fallback(path)
    finally:
      os.unlink(path)


# ---------------------------------------------------------------------------
# Public API: write_tifxyz / read_tifxyz (forces fallback path)
# ---------------------------------------------------------------------------

class TestWriteReadTifxyz(unittest.TestCase):

  def _roundtrip(self, xyz: np.ndarray) -> np.ndarray:
    import volatile.tifxyz as mod
    orig = mod._HAS_TIFFFILE
    mod._HAS_TIFFFILE = False
    try:
      with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        path = f.name
      write_tifxyz(path, xyz)
      result = read_tifxyz(path)
    finally:
      mod._HAS_TIFFFILE = orig
      os.unlink(path)
    return result

  def test_roundtrip_small(self):
    xyz = _xyz(4, 6)
    result = self._roundtrip(xyz)
    self.assertEqual(result.shape, xyz.shape)
    np.testing.assert_allclose(result, xyz, rtol=1e-5)

  def test_roundtrip_single_pixel(self):
    xyz = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    result = self._roundtrip(xyz)
    np.testing.assert_allclose(result, xyz, rtol=1e-5)

  def test_write_wrong_shape_raises(self):
    with self.assertRaises(ValueError):
      write_tifxyz("/tmp/bad.tif", np.zeros((4, 4), dtype=np.float32))

  def test_write_wrong_channels_raises(self):
    with self.assertRaises(ValueError):
      write_tifxyz("/tmp/bad.tif", np.zeros((4, 4, 2), dtype=np.float32))

  def test_read_channel_first_reordered(self):
    """write outputs (3,H,W); read must convert to (H,W,3)."""
    import volatile.tifxyz as mod
    orig = mod._HAS_TIFFFILE
    mod._HAS_TIFFFILE = False
    xyz = _xyz(3, 5)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
      path = f.name
    try:
      write_tifxyz(path, xyz)
      result = read_tifxyz(path)
      self.assertEqual(result.ndim, 3)
      self.assertEqual(result.shape[2], 3)
    finally:
      mod._HAS_TIFFFILE = orig
      os.unlink(path)


# ---------------------------------------------------------------------------
# tifxyz_to_mesh / mesh_to_tifxyz
# ---------------------------------------------------------------------------

class TestMeshConversion(unittest.TestCase):

  def test_tifxyz_to_mesh_shape(self):
    xyz = _xyz(4, 6)
    surf = tifxyz_to_mesh(xyz)
    self.assertEqual(surf.rows, 4)
    self.assertEqual(surf.cols, 6)

  def test_tifxyz_to_mesh_values(self):
    xyz = _xyz(3, 4)
    surf = tifxyz_to_mesh(xyz)
    for r in range(3):
      for c in range(4):
        pt = surf.get(r, c)
        self.assertAlmostEqual(pt[0], float(r))
        self.assertAlmostEqual(pt[1], float(c))
        self.assertAlmostEqual(pt[2], float(r + c))

  def test_mesh_to_tifxyz_shape(self):
    surf = QuadSurface(5, 7)
    xyz = mesh_to_tifxyz(surf)
    self.assertEqual(xyz.shape, (5, 7, 3))
    self.assertEqual(xyz.dtype, np.float32)

  def test_mesh_roundtrip(self):
    xyz = _xyz(4, 4)
    surf = tifxyz_to_mesh(xyz)
    xyz2 = mesh_to_tifxyz(surf)
    np.testing.assert_allclose(xyz2, xyz, rtol=1e-5)

  def test_tifxyz_to_mesh_wrong_shape_raises(self):
    with self.assertRaises(ValueError):
      tifxyz_to_mesh(np.zeros((4, 4), dtype=np.float32))

  def test_mesh_to_tifxyz_all_zeros(self):
    surf = QuadSurface(2, 3)
    xyz = mesh_to_tifxyz(surf)
    np.testing.assert_array_equal(xyz, np.zeros((2, 3, 3), dtype=np.float32))

  # Fast variants

  def test_fast_mesh_roundtrip(self):
    xyz = _xyz(4, 4)
    surf = tifxyz_to_mesh_fast(xyz)
    xyz2 = mesh_to_tifxyz_fast(surf)
    np.testing.assert_allclose(xyz2, xyz, rtol=1e-5)

  def test_fast_same_as_slow(self):
    xyz = _xyz(6, 8)
    s1 = tifxyz_to_mesh(xyz)
    s2 = tifxyz_to_mesh_fast(xyz)
    for r in range(6):
      for c in range(8):
        np.testing.assert_allclose(s1.get(r, c), s2.get(r, c), rtol=1e-5)

  def test_fast_to_tifxyz_same_as_slow(self):
    surf = QuadSurface(3, 4)
    for r in range(3):
      for c in range(4):
        surf.set(r, c, (float(r), float(c), 0.5))
    slow = mesh_to_tifxyz(surf)
    fast = mesh_to_tifxyz_fast(surf)
    np.testing.assert_allclose(fast, slow, rtol=1e-5)


if __name__ == "__main__":
  unittest.main()
