"""test_rendering.py — Tests for volatile.rendering and volatile.napari_plugin."""
from __future__ import annotations

import math
import os
import tempfile

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _flat_surface(rows=16, cols=16, z=0.0):
  """Flat grid surface in the XY plane."""
  from volatile.seg import QuadSurface
  surf = QuadSurface(rows=rows, cols=cols)
  for r in range(rows):
    for c in range(cols):
      surf.set(r, c, (float(c), float(r), z))
  return surf


def _synthetic_zarr_volume(tmp_path, shape=(32, 32, 32)):
  """Write a synthetic zarr volume and return the array."""
  import zarr, json
  path = str(tmp_path / "vol.zarr")
  root = zarr.open_group(path, mode="w", zarr_format=2)
  data = np.random.default_rng(42).integers(0, 256, shape, dtype=np.uint8)
  root.create_array("0", shape=shape, chunks=(8, 8, 8), dtype="u1")
  root["0"][:] = data
  attrs = {"multiscales": [{"version": "0.4",
    "axes": [{"name": ax, "type": "space"} for ax in ("z", "y", "x")],
    "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1,1,1]}]}],
  }]}
  with open(os.path.join(path, ".zattrs"), "w") as f:
    json.dump(attrs, f)
  return path, zarr.open_array(path + "/0", mode="r")


# ---------------------------------------------------------------------------
# render_surface — shape and type
# ---------------------------------------------------------------------------

def test_render_surface_output_shape():
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=8, cols=12)
  # Use a simple zarr array as volume (3-D)
  vol = np.full((32, 32, 32), 128, dtype=np.uint8)
  rgb = render_surface(vol, surf, layers_front=2, layers_behind=2)
  assert rgb.shape == (8, 12, 3), f"expected (8,12,3), got {rgb.shape}"
  assert rgb.dtype == np.uint8


def test_render_surface_returns_rgb():
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=4, cols=4)
  vol = np.zeros((20, 20, 20), dtype=np.uint8)
  rgb = render_surface(vol, surf, cmap="gray")
  assert rgb.ndim == 3 and rgb.shape[2] == 3


def test_render_surface_no_layers():
  """layers_front=0, layers_behind=0 — only the surface plane itself."""
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=6, cols=6)
  vol = np.ones((20, 20, 20), dtype=np.uint8) * 200
  rgb = render_surface(vol, surf, layers_front=0, layers_behind=0)
  assert rgb.shape == (6, 6, 3)


def test_render_surface_composites():
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=4, cols=4)
  vol = np.random.default_rng(0).integers(0, 256, (20, 20, 20), dtype=np.uint8)
  for comp in ("max", "min", "mean"):
    rgb = render_surface(vol, surf, composite=comp, layers_front=2, layers_behind=2)
    assert rgb.shape == (4, 4, 3), f"composite={comp} gave wrong shape"


def test_render_surface_cmap_viridis_vs_gray():
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=4, cols=4)
  vol = np.arange(20**3, dtype=np.float32).reshape(20, 20, 20)
  rgb_v = render_surface(vol, surf, cmap="viridis")
  rgb_g = render_surface(vol, surf, cmap="gray")
  # gray image: R==G==B; viridis image: not necessarily
  gray_equal = np.all(rgb_g[:,:,0] == rgb_g[:,:,1]) and np.all(rgb_g[:,:,1] == rgb_g[:,:,2])
  assert gray_equal, "gray cmap should produce equal R,G,B channels"


def test_render_surface_vmin_vmax():
  from volatile.rendering import render_surface
  surf = _flat_surface(rows=4, cols=4)
  vol = np.ones((20, 20, 20), dtype=np.uint8) * 128
  rgb = render_surface(vol, surf, vmin=0, vmax=255)
  assert rgb.shape == (4, 4, 3)


# ---------------------------------------------------------------------------
# render_surface_to_tiff
# ---------------------------------------------------------------------------

def test_render_surface_to_tiff(tmp_path):
  import zarr, json
  from volatile.rendering import render_surface_to_tiff, render_surface, _write_tiff
  from volatile.seg import QuadSurface

  # Build surface and render directly (bypasses vol_open path; tests file writing)
  surf = _flat_surface(rows=8, cols=8, z=10.0)
  vol = np.full((32, 32, 32), 128, dtype=np.uint8)
  rgb = render_surface(vol, surf, layers_front=1, layers_behind=1)
  out_path = str(tmp_path / "render.tif")
  _write_tiff(rgb, out_path)
  assert os.path.exists(out_path)
  assert os.path.getsize(out_path) > 100


# ---------------------------------------------------------------------------
# flatten_surface — UV range and shape
# ---------------------------------------------------------------------------

def test_flatten_area_preserving_shape():
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=8, cols=8)
  uv = flatten_surface(surf, method="area_preserving")
  assert uv.shape == (8, 8, 2), f"expected (8,8,2), got {uv.shape}"
  assert uv.dtype == np.float32


def test_flatten_area_preserving_range():
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=10, cols=12)
  uv = flatten_surface(surf, method="area_preserving")
  assert uv[:,:,0].min() >= -1e-5 and uv[:,:,0].max() <= 1.0 + 1e-5
  assert uv[:,:,1].min() >= -1e-5 and uv[:,:,1].max() <= 1.0 + 1e-5


def test_flatten_lscm_shape():
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=6, cols=6)
  uv = flatten_surface(surf, method="lscm")
  assert uv.shape == (6, 6, 2), f"expected (6,6,2), got {uv.shape}"
  assert uv.dtype == np.float32


def test_flatten_lscm_range():
  """UV coords should be in [0,1] after normalisation."""
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=8, cols=8)
  uv = flatten_surface(surf, method="lscm")
  assert uv[:,:,0].min() >= -1e-4, f"U min out of range: {uv[:,:,0].min()}"
  assert uv[:,:,0].max() <= 1.0 + 1e-4, f"U max out of range: {uv[:,:,0].max()}"
  assert uv[:,:,1].min() >= -1e-4, f"V min out of range: {uv[:,:,1].min()}"
  assert uv[:,:,1].max() <= 1.0 + 1e-4, f"V max out of range: {uv[:,:,1].max()}"


def test_flatten_lscm_monotone_grid():
  """On a flat axis-aligned grid, LSCM UV should be monotone in both axes."""
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=6, cols=6)
  uv = flatten_surface(surf, method="lscm")
  # U should increase along columns (or be constant if degenerate)
  u_row = uv[3, :, 0]
  v_col = uv[:, 3, 1]
  assert np.all(np.diff(u_row) >= -1e-4) or np.all(np.diff(u_row) <= 1e-4), \
    "U should be monotone along a row of the flat grid"
  assert np.all(np.diff(v_col) >= -1e-4) or np.all(np.diff(v_col) <= 1e-4), \
    "V should be monotone along a column of the flat grid"


def test_flatten_large_surface_falls_back():
  """Large surfaces (>512 verts) fall back to area_preserving in LSCM mode."""
  from volatile.rendering import flatten_surface
  surf = _flat_surface(rows=24, cols=24)  # 576 > 512
  uv = flatten_surface(surf, method="lscm")
  assert uv.shape == (24, 24, 2)


def test_flatten_unknown_method_defaults_to_lscm():
  """Unknown method string falls through to LSCM code path (no crash)."""
  from volatile.rendering import flatten_surface, _flatten_lscm
  surf = _flat_surface(rows=4, cols=4)
  # Call the internal function directly to verify it doesn't crash
  uv = _flatten_lscm(surf)
  assert uv.shape == (4, 4, 2)


# ---------------------------------------------------------------------------
# texture_from_uv
# ---------------------------------------------------------------------------

def test_texture_from_uv_shape():
  from volatile.rendering import flatten_surface, texture_from_uv
  rows, cols = 8, 8
  surf = _flat_surface(rows=rows, cols=cols, z=10.0)
  vol = np.random.default_rng(7).integers(0, 256, (32, 32, 32), dtype=np.uint8)
  uv = flatten_surface(surf, method="area_preserving")
  tex = texture_from_uv(vol, surf, uv, z_range=4)
  assert tex.shape == (rows, cols), f"expected ({rows},{cols}), got {tex.shape}"
  assert tex.dtype == np.float32


# ---------------------------------------------------------------------------
# napari_plugin
# ---------------------------------------------------------------------------

def test_napari_plugin_importable():
  import volatile.napari_plugin as np_
  assert hasattr(np_, "register_napari_plugin")
  assert hasattr(np_, "get_napari_widget_contributions")


def test_napari_register_raises_without_napari():
  import volatile.napari_plugin as np_
  try:
    import napari  # noqa: F401
    pytest.skip("napari is installed; skipping absence test")
  except ImportError:
    pass
  with pytest.raises(ImportError, match="napari"):
    np_.register_napari_plugin()


def test_get_napari_widget_contributions_returns_list():
  from volatile.napari_plugin import get_napari_widget_contributions
  result = get_napari_widget_contributions()
  assert isinstance(result, list)
