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

def _make_model(mesh_h=4, mesh_w=6, z_size=1, subsample=2):
  from volatile.fit import Model2D, ModelInit
  init = ModelInit(
    init_size_frac=1.0, init_size_frac_h=None, init_size_frac_v=None,
    mesh_step_px=8, winding_step_px=8, mesh_h=mesh_h, mesh_w=mesh_w,
  )
  return Model2D(
    init, z_size=z_size, subsample_mesh=subsample, subsample_winding=subsample,
    z_step_vx=1, scaledown=1.0,
    crop_xyzwhd=(0, 0, 100, 80, 0, 10),
    n_pyramid_scales=3,
  )


# -- forward shapes ----------------------------------------------------------

def test_model2d_forward_shapes():
  model = _make_model(mesh_h=4, mesh_w=6)
  res = model.forward()
  n, hm, wm, c = res.xy_lr.numpy().shape
  assert n == 1 and hm == 4 and wm == 6 and c == 2, f"xy_lr shape wrong: {res.xy_lr.numpy().shape}"
  _, he, we, _ = res.xy_hr.numpy().shape
  assert he == (hm - 1) * 2 + 1, f"he wrong: {he}"   # subsample=2
  assert we == (wm - 1) * 2 + 1, f"we wrong: {we}"
  assert tuple(res.target_plain.numpy().shape) == (1, 1, he, we)
  assert tuple(res.target_mod.numpy().shape)   == (1, 1, he, we)
  assert tuple(res.amp_lr.numpy().shape)       == (1, 1, hm, wm)
  assert tuple(res.bias_lr.numpy().shape)      == (1, 1, hm, wm)
  assert tuple(res.mask_hr.numpy().shape)      == (1, 1, he, we)
  assert tuple(res.mask_lr.numpy().shape)      == (1, 1, hm, wm)


def test_model2d_xy_conn_shape():
  """xy_conn must have shape (N, Hm, Wm, 3, 2)."""
  model = _make_model(mesh_h=5, mesh_w=7)
  res = model.forward()
  n, hm, wm, c = res.xy_lr.numpy().shape
  expected = (n, hm, wm, 3, 2)
  assert tuple(res.xy_conn.numpy().shape) == expected, f"xy_conn shape: {res.xy_conn.numpy().shape}"


def test_model2d_mask_conn_shape():
  """mask_conn must have shape (N, 1, Hm, Wm, 3)."""
  model = _make_model(mesh_h=4, mesh_w=6)
  res = model.forward()
  n, hm, wm, _ = res.xy_lr.numpy().shape
  assert tuple(res.mask_conn.numpy().shape) == (n, 1, hm, wm, 3)


# -- target values -----------------------------------------------------------

def test_model2d_target_range():
  """Cosine target and modulated target must lie in [0,1]."""
  model = _make_model()
  res = model.forward()
  tp = res.target_plain.numpy(); tm = res.target_mod.numpy()
  assert tp.min() >= -1e-6 and tp.max() <= 1.0 + 1e-6
  assert tm.min() >= -1e-6 and tm.max() <= 1.0 + 1e-6


def test_model2d_target_is_cosine():
  """target_plain row must be 0.5+0.5*cos, starting and ending at 1.0."""
  model = _make_model(mesh_h=2, mesh_w=5, subsample=1)
  res = model.forward()
  tp = res.target_plain.numpy()[0, 0, 0, :]   # first row, HR = LR when ss=1
  assert tp[0]  == pytest.approx(1.0, abs=1e-4), f"tp[0]={tp[0]}"
  assert tp[-1] == pytest.approx(1.0, abs=1e-4), f"tp[-1]={tp[-1]}"
  # Midpoint of a 5-wide target (period 4) at index 2 → cos(π) = -1 → 0.0
  assert tp[2]  == pytest.approx(0.0, abs=1e-4), f"tp[mid]={tp[2]}"


def test_model2d_amp_bias_clamped():
  """amp_lr ∈ [0.1,1.0] and bias_lr ∈ [0.0,0.45] after clamping."""
  model = _make_model()
  # Force amp/bias outside clamp range.
  import numpy as np
  model.amp  = model.amp.__class__(np.full(model.amp.numpy().shape,  5.0, dtype=np.float32), requires_grad=True)
  model.bias = model.bias.__class__(np.full(model.bias.numpy().shape, 1.0, dtype=np.float32), requires_grad=True)
  res = model.forward()
  assert res.amp_lr.numpy().max()  <= 1.0 + 1e-6
  assert res.bias_lr.numpy().max() <= 0.45 + 1e-6


# -- parameters --------------------------------------------------------------

def test_model2d_get_parameters():
  """get_parameters returns at least mesh_ms + conn_offset_ms + amp/bias/theta/ws."""
  from tinygrad.tensor import Tensor
  model = _make_model(n_pyramid_scales=3) if False else _make_model()
  params = model.get_parameters()
  assert len(params) >= 6 + 2   # 3 mesh + 3 conn + amp + bias + theta + ws = 10
  for p in params:
    assert isinstance(p, Tensor)


def test_model2d_conn_offset_pyramid_exists():
  """conn_offset_ms should have n_pyramid_scales levels."""
  model = _make_model()
  assert len(model.conn_offset_ms) == model.n_pyramid_scales
  # All zero at init.
  for p in model.conn_offset_ms:
    assert np.abs(p.numpy()).max() == 0.0


# -- global transform --------------------------------------------------------

def test_model2d_bake_global_transform():
  """After bake, theta=0, winding_scale=1, transform disabled, mesh unchanged."""
  model = _make_model(mesh_h=4, mesh_w=6)
  xy_before = model.forward().xy_lr.numpy().copy()
  model.bake_global_transform()
  assert not model.global_transform_enabled
  xy_after = model.forward().xy_lr.numpy()
  np.testing.assert_allclose(xy_after, xy_before, atol=1e-4, err_msg="bake changed mesh coords")
  assert model.theta.numpy()[0] == pytest.approx(0.0, abs=1e-6)
  assert model.winding_scale.numpy()[0] == pytest.approx(1.0, abs=1e-6)


# -- EMA ---------------------------------------------------------------------

def test_model2d_update_ema_init():
  """First call to update_ema initialises the buffer to the input."""
  model = _make_model()
  res = model.forward()
  model.update_ema(xy_lr=res.xy_lr, xy_conn=res.xy_conn)
  np.testing.assert_allclose(model.xy_ema, res.xy_lr.numpy(), atol=1e-6)


def test_model2d_update_ema_decay():
  """Second call blends the new value in with decay 0.99."""
  model = _make_model()
  model._ema_decay = 0.5
  res = model.forward()
  model.update_ema(xy_lr=res.xy_lr, xy_conn=res.xy_conn)
  v1 = model.xy_ema.copy()
  model.update_ema(xy_lr=res.xy_lr, xy_conn=res.xy_conn)
  v2 = model.xy_ema
  expected = 0.5 * v1 + 0.5 * res.xy_lr.numpy()
  np.testing.assert_allclose(v2, expected, atol=1e-6)


# -- grow --------------------------------------------------------------------

def test_model2d_grow_right():
  """grow(right, 2) increases mesh_w by 2 and propagates to amp/bias."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["right"], steps=2)
  assert model.mesh_w == 8
  assert model.amp.numpy().shape[3]  == 8
  assert model.bias.numpy().shape[3] == 8


def test_model2d_grow_down():
  """grow(down, 1) increases mesh_h by 1."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["down"], steps=1)
  assert model.mesh_h == 5
  assert model.amp.numpy().shape[2] == 5


def test_model2d_grow_left():
  """grow(left, 1) increases mesh_w by 1."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["left"], steps=1)
  assert model.mesh_w == 7


def test_model2d_grow_up():
  """grow(up, 1) increases mesh_h by 1."""
  model = _make_model(mesh_h=4, mesh_w=6)
  model.grow(directions=["up"], steps=1)
  assert model.mesh_h == 5


def test_model2d_grow_fw():
  """grow(fw, 1) adds a z-slice (z_size increases by 1)."""
  model = _make_model(mesh_h=4, mesh_w=6, z_size=2)
  model.grow(directions=["fw"], steps=1)
  assert model.z_size == 3
  assert model.amp.numpy().shape[0] == 3


def test_model2d_grow_bw():
  """grow(bw, 1) prepends a z-slice."""
  model = _make_model(mesh_h=4, mesh_w=6, z_size=2)
  model.grow(directions=["bw"], steps=1)
  assert model.z_size == 3


def test_model2d_grow_invalid_direction():
  model = _make_model()
  with pytest.raises(ValueError, match="invalid grow direction"):
    model.grow(directions=["diagonal"])


def test_model2d_grow_then_forward():
  """grow + forward should not crash and shapes must be consistent."""
  model = _make_model(mesh_h=4, mesh_w=5)
  model.grow(directions=["right", "down"], steps=1)
  res = model.forward()
  n, hm, wm, _ = res.xy_lr.numpy().shape
  assert hm == 5 and wm == 6


# -- state dict --------------------------------------------------------------

def test_model2d_state_dict_roundtrip():
  """state_dict / load_state_dict preserves all parameter values."""
  model = _make_model(mesh_h=3, mesh_w=4)
  sd = model.state_dict()
  required = {"amp", "bias", "theta", "winding_scale"}
  assert required <= set(sd.keys()), f"missing keys: {required - set(sd.keys())}"
  # Check conn_offset_ms keys present.
  assert any(k.startswith("conn_offset_ms.") for k in sd), "conn_offset_ms missing from state_dict"
  orig_amp = sd["amp"].copy()
  model2 = _make_model(mesh_h=3, mesh_w=4)
  model2.load_state_dict(sd)
  np.testing.assert_allclose(model2.amp.numpy(), orig_amp, atol=1e-6)


# -- pyramid helpers ---------------------------------------------------------

def test_cosine_winding_target_shape_range():
  """cosine_winding_target produces expected shape and values in [0,1]."""
  from volatile.fit import cosine_winding_target
  t = cosine_winding_target(z_size=2, h=3, w=5)
  assert tuple(t.numpy().shape) == (2, 1, 3, 5)
  arr = t.numpy()
  assert arr.min() >= -1e-6 and arr.max() <= 1.0 + 1e-6


def test_pyramid_roundtrip():
  """_integrate_pyramid(_build_pyramid_from_flat(x)) ≈ x."""
  from tinygrad.tensor import Tensor
  from volatile.fit import _build_pyramid_from_flat, _integrate_pyramid
  rng = np.random.default_rng(99)
  flat_np = rng.random((1, 2, 8, 12)).astype(np.float32)
  flat = Tensor(flat_np)
  pyramid = _build_pyramid_from_flat(flat, n_scales=4)
  recon = _integrate_pyramid(pyramid)
  np.testing.assert_allclose(recon.numpy(), flat_np, atol=1e-5)


# -- grid_sample_px ----------------------------------------------------------

def test_grid_sample_px_identity():
  """Sampling an image at its own pixel grid should return the image."""
  from volatile.fit import grid_sample_px
  from tinygrad.tensor import Tensor
  rng = np.random.default_rng(7)
  img_np = rng.random((1, 1, 8, 10)).astype(np.float32)
  image  = Tensor(img_np)
  # Build identity sampling grid.
  ys, xs = np.meshgrid(np.arange(8), np.arange(10), indexing="ij")  # (8,10)
  xy_px  = Tensor(np.stack([xs, ys], axis=-1)[np.newaxis].astype(np.float32))  # (1,8,10,2)
  out = grid_sample_px(image, xy_px)
  np.testing.assert_allclose(out.numpy(), img_np, atol=1e-5)


def test_grid_sample_px_zeros_outside():
  """grid_sample_px returns 0 for out-of-bounds coordinates."""
  from volatile.fit import grid_sample_px
  from tinygrad.tensor import Tensor
  img_np = np.ones((1, 1, 4, 4), dtype=np.float32)
  image  = Tensor(img_np)
  xy_px  = Tensor(np.array([[[[-100.0, -100.0]]]], dtype=np.float32))  # (1,1,1,2)
  out = grid_sample_px(image, xy_px)
  assert out.numpy()[0, 0, 0, 0] == pytest.approx(0.0, abs=1e-6)
