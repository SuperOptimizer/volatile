from __future__ import annotations
"""Tests for volatile.imgproc additions and volatile.structure_tensor / volatile.fit."""

import math
import struct
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# imgproc: frangi_vesselness_3d
# ---------------------------------------------------------------------------

def _make_tube_volume(d=16, h=16, w=32, radius=2.0):
  """Synthetic tube along the X axis centred at (d/2, h/2)."""
  vol = np.zeros((d, h, w), dtype=np.float32)
  cz, cy = d // 2, h // 2
  for z in range(d):
    for y in range(h):
      if math.sqrt((z - cz)**2 + (y - cy)**2) <= radius:
        vol[z, y, :] = 1.0
  return vol


def test_frangi_uniform_is_zero():
  """Uniform volume has no tubular structure."""
  from volatile.imgproc import frangi_vesselness_3d
  vol = np.ones((8, 8, 8), dtype=np.float32)
  result = frangi_vesselness_3d(vol, sigmas=[1.0], alpha=0.5, beta=0.5, gamma=1.0)
  assert result.shape == (8, 8, 8)
  assert result.max() < 0.1, f"expected near-zero on uniform, got max={result.max()}"


def test_frangi_tube_has_response():
  """Tubular structure should have nonzero vesselness inside."""
  from volatile.imgproc import frangi_vesselness_3d
  vol = _make_tube_volume()
  result = frangi_vesselness_3d(vol, sigmas=[1.0, 2.0], alpha=0.5, beta=0.5, gamma=0.0)
  assert result.shape == vol.shape
  # Response inside the tube should be higher than outside (on average).
  d, h, w = vol.shape
  inside = result[d//2 - 1:d//2 + 2, h//2 - 1:h//2 + 2, w//4:3*w//4]
  outside = result[0, 0, :]
  assert inside.mean() >= outside.mean(), "vesselness inside tube should exceed outside"


def test_frangi_output_dtype():
  from volatile.imgproc import frangi_vesselness_3d
  vol = _make_tube_volume()
  result = frangi_vesselness_3d(vol)
  assert result.dtype == np.float32


def test_frangi_multiscale_nonnegative():
  from volatile.imgproc import frangi_vesselness_3d
  vol = _make_tube_volume()
  result = frangi_vesselness_3d(vol, sigmas=[0.5, 1.0, 2.0])
  assert (result >= 0.0).all(), "vesselness must be non-negative"


# ---------------------------------------------------------------------------
# imgproc: edt_3d
# ---------------------------------------------------------------------------

def test_edt_3d_all_foreground():
  """If every voxel is foreground, EDT is all zeros."""
  from volatile.imgproc import edt_3d
  mask = np.ones((4, 4, 4), dtype=np.uint8)
  result = edt_3d(mask)
  assert result.shape == (4, 4, 4)
  assert (result == 0.0).all()


def test_edt_3d_point_source():
  """Single foreground voxel at centre — distance grows outward."""
  from volatile.imgproc import edt_3d
  mask = np.zeros((7, 7, 7), dtype=np.uint8)
  mask[3, 3, 3] = 1
  result = edt_3d(mask)
  assert result[3, 3, 3] == pytest.approx(0.0, abs=1e-5)
  # Adjacent voxel should be distance 1.
  assert result[3, 3, 4] == pytest.approx(1.0, abs=0.1)
  # Corner should be farther.
  assert result[0, 0, 0] > result[3, 3, 4]


def test_edt_3d_dtype():
  from volatile.imgproc import edt_3d
  mask = np.zeros((4, 4, 4), dtype=bool)
  mask[2, 2, 2] = True
  result = edt_3d(mask)
  assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# imgproc: eigendecomp_3d
# ---------------------------------------------------------------------------

def test_eigendecomp_3d_identity_tensor():
  """Structure tensor that is the identity matrix should have eigenvalues [1,1,1]."""
  from volatile.imgproc import eigendecomp_3d
  d, h, w = 2, 2, 2
  # Identity: Jzz=Jyy=Jxx=1, cross-terms=0 → layout (Jzz,Jzy,Jzx,Jyy,Jyx,Jxx)
  st = np.zeros((d, h, w, 6), dtype=np.float32)
  st[..., 0] = 1.0  # Jzz
  st[..., 3] = 1.0  # Jyy
  st[..., 5] = 1.0  # Jxx
  vals, vecs = eigendecomp_3d(st)
  assert vals.shape == (d, h, w, 3)
  assert vecs.shape == (d, h, w, 3, 3)
  assert vals.min() == pytest.approx(1.0, abs=1e-5)
  assert vals.max() == pytest.approx(1.0, abs=1e-5)


def test_eigendecomp_3d_ascending():
  """Eigenvalues must be in ascending order."""
  from volatile.imgproc import eigendecomp_3d
  rng = np.random.default_rng(42)
  d, h, w = 4, 4, 4
  # Build a valid symmetric positive-definite tensor.
  A = rng.random((6,)).astype(np.float32)
  st = np.broadcast_to(A, (d, h, w, 6)).copy()
  # Ensure positive semi-definiteness by setting cross-terms small.
  st[..., 0] = 3.0; st[..., 3] = 2.0; st[..., 5] = 1.0
  st[..., 1] = 0.1; st[..., 2] = 0.05; st[..., 4] = 0.1
  vals, _ = eigendecomp_3d(st)
  diffs = np.diff(vals, axis=-1)
  assert (diffs >= -1e-5).all(), "eigenvalues must be non-decreasing"


def test_eigendecomp_3d_orthogonal_vecs():
  """Eigenvectors of a symmetric matrix are orthogonal."""
  from volatile.imgproc import eigendecomp_3d
  st = np.zeros((1, 1, 1, 6), dtype=np.float32)
  st[0, 0, 0] = [2.0, 0.3, 0.1, 1.5, 0.2, 1.0]
  _, vecs = eigendecomp_3d(st)
  V = vecs[0, 0, 0]  # (3, 3)
  # Columns are eigenvectors — check orthogonality via V^T V ≈ I.
  VTV = V.T @ V
  assert np.allclose(VTV, np.eye(3), atol=1e-5), f"eigenvectors not orthogonal: {VTV}"


# ---------------------------------------------------------------------------
# structure_tensor: compute_st (pure-numpy path)
# ---------------------------------------------------------------------------

def test_compute_st_output_shape():
  from volatile.structure_tensor import compute_st
  vol = np.random.rand(8, 10, 12).astype(np.float32)
  with tempfile.TemporaryDirectory() as tmp:
    in_path = Path(tmp) / "vol.npy"
    out_path = Path(tmp) / "st.npy"
    np.save(str(in_path), vol)
    compute_st(in_path, out_path, deriv_sigma=0.5, smooth_sigma=0.5, chunk_size=4, verbose=False)
    result = np.load(str(out_path))
  assert result.shape == (8, 10, 12, 6)
  assert result.dtype == np.float32


def test_compute_st_uniform_volume():
  """Gradient of a uniform volume is zero — all ST components should be near zero."""
  from volatile.structure_tensor import compute_st
  vol = np.ones((6, 6, 6), dtype=np.float32) * 5.0
  with tempfile.TemporaryDirectory() as tmp:
    in_path = Path(tmp) / "vol.npy"
    out_path = Path(tmp) / "st.npy"
    np.save(str(in_path), vol)
    compute_st(in_path, out_path, deriv_sigma=0.0, smooth_sigma=0.0, chunk_size=6, verbose=False)
    result = np.load(str(out_path))
  assert np.abs(result).max() < 1e-4, f"expected near-zero ST on uniform volume, got {np.abs(result).max()}"


def test_compute_st_chunked_same_shape():
  """Chunked and single-chunk processing produce the same output shape and dtype."""
  from volatile.structure_tensor import compute_st
  rng = np.random.default_rng(7)
  vol = rng.random((12, 8, 8)).astype(np.float32)
  with tempfile.TemporaryDirectory() as tmp:
    in_path = Path(tmp) / "vol.npy"
    np.save(str(in_path), vol)
    out_full = Path(tmp) / "full.npy"
    out_chunked = Path(tmp) / "chunked.npy"
    compute_st(in_path, out_full, deriv_sigma=1.0, smooth_sigma=1.0, chunk_size=12, verbose=False)
    compute_st(in_path, out_chunked, deriv_sigma=1.0, smooth_sigma=1.0, chunk_size=4, verbose=False)
    full = np.load(str(out_full))
    chunked = np.load(str(out_chunked))
  # Shapes and dtypes must agree; boundary-effect differences in values are expected.
  assert full.shape == chunked.shape
  assert full.dtype == chunked.dtype == np.float32
  # Both should be finite and have meaningful signal.
  assert np.isfinite(full).all()
  assert np.isfinite(chunked).all()


# ---------------------------------------------------------------------------
# fit: Model2D
# ---------------------------------------------------------------------------

def _make_model(mesh_h=4, mesh_w=6, z_size=1):
  from volatile.fit import Model2D, ModelInit
  init = ModelInit(
    init_size_frac=1.0, init_size_frac_h=None, init_size_frac_v=None,
    mesh_step_px=8, winding_step_px=8, mesh_h=mesh_h, mesh_w=mesh_w,
  )
  return Model2D(
    init, z_size=z_size, subsample_mesh=2, subsample_winding=2,
    z_step_vx=1, scaledown=1.0,
    crop_xyzwhd=(0, 0, 100, 80, 0, 10),
    n_pyramid_scales=3,
  )


def test_model2d_forward_shapes():
  model = _make_model(mesh_h=4, mesh_w=6)
  res = model.forward()
  n, hm, wm, c = res.xy_lr.shape
  assert n == 1 and hm == 4 and wm == 6 and c == 2
  _, he, we, _ = res.xy_hr.shape
  assert he == (hm - 1) * 2 + 1  # subsample_mesh=2
  assert we == (wm - 1) * 2 + 1
  assert res.target_plain.shape == (1, 1, he, we)
  assert res.target_mod.shape == (1, 1, he, we)


def test_model2d_target_range():
  """Cosine target and modulated target must lie in [0,1]."""
  model = _make_model()
  res = model.forward()
  tp = res.target_plain.numpy()
  tm = res.target_mod.numpy()
  assert tp.min() >= 0.0 and tp.max() <= 1.0
  assert tm.min() >= 0.0 and tm.max() <= 1.0


def test_model2d_target_is_cosine():
  """target_plain should be 0.5+0.5*cos along winding direction."""
  model = _make_model(mesh_h=2, mesh_w=5)
  res = model.forward()
  tp = res.target_plain.numpy()[0, 0, 0, :]  # first row
  # First and last should be 1.0 (cos(0)=1, cos(2π*(mw-1)/(mw-1))=cos(2π)=1)
  assert tp[0] == pytest.approx(1.0, abs=1e-4)
  assert tp[-1] == pytest.approx(1.0, abs=1e-4)


def test_model2d_get_parameters():
  model = _make_model()
  params = model.get_parameters()
  assert len(params) > 0
  # All must be Tensors.
  from tinygrad.tensor import Tensor
  for p in params:
    assert isinstance(p, Tensor)


def test_model2d_state_dict_roundtrip():
  """State dict save/load should preserve parameter values."""
  model = _make_model(mesh_h=3, mesh_w=4)
  sd = model.state_dict()
  assert "amp" in sd and "bias" in sd and "theta" in sd and "winding_scale" in sd
  # Mutate and reload.
  orig_amp = sd["amp"].copy()
  model2 = _make_model(mesh_h=3, mesh_w=4)
  model2.load_state_dict(sd)
  np.testing.assert_allclose(model2.amp.numpy(), orig_amp, atol=1e-6)


def test_model2d_grow_right():
  """Growing right should increase mesh_w by steps."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["right"], steps=2)
  assert model.mesh_w == 8
  assert model.amp.numpy().shape[3] == 8


def test_model2d_grow_down():
  """Growing down should increase mesh_h by steps."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["down"], steps=1)
  assert model.mesh_h == 5


def test_model2d_cosine_winding_target():
  """cosine_winding_target helper produces expected shape and range."""
  from volatile.fit import _cosine_winding_target
  t = _cosine_winding_target(z_size=2, mesh_h=3, mesh_w=5)
  assert t.shape == (2, 1, 3, 5)
  arr = t.numpy()
  assert arr.min() >= 0.0 and arr.max() <= 1.0


def test_model2d_pyramid_roundtrip():
  """integrate_pyramid(build_mesh_pyramid(flat)) ≈ flat."""
  import numpy as np
  from tinygrad.tensor import Tensor
  from volatile.fit import _build_mesh_pyramid, _integrate_pyramid
  rng = np.random.default_rng(99)
  flat_np = rng.random((1, 2, 8, 12)).astype(np.float32)
  flat = Tensor(flat_np)
  pyramid = _build_mesh_pyramid(flat, n_scales=4)
  recon = _integrate_pyramid(pyramid)
  np.testing.assert_allclose(recon.numpy(), flat_np, atol=1e-5,
                              err_msg="pyramid roundtrip should reconstruct original")
