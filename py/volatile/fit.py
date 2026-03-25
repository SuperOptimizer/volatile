from __future__ import annotations
"""2-D surface fitting model — tinygrad port of villa's exps_2d_model/model.py.

The model represents a deformable mesh that maps papyrus-sheet winding
coordinates to 2-D image (x,y) positions.  A multi-scale residual pyramid
of mesh offsets enables fine-to-coarse optimisation.

Key design differences from the villa PyTorch version:
- Parameters are plain Tensors with requires_grad=True (no nn.Module/register_buffer).
- Pyramids are plain Python lists of Tensors, not nn.ParameterList.
- EMA buffers are plain numpy arrays updated outside autograd.
- grid_sample_px() is provided as a standalone function instead of being on FitData.
- No torch.device argument — tinygrad picks the device from the global backend.
"""

import math
from dataclasses import dataclass

import numpy as np

try:
  from tinygrad.tensor import Tensor as _Tensor
  _HAS_TINYGRAD = True
except ImportError:
  _HAS_TINYGRAD = False
  _Tensor = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Public type alias — avoids bare "Tensor" everywhere while keeping the guard
# ---------------------------------------------------------------------------

if _HAS_TINYGRAD:
  from tinygrad.tensor import Tensor
else:
  Tensor = None  # type: ignore[misc,assignment]


# ---------------------------------------------------------------------------
# Dataclasses  (mirror villa's ModelInit / ModelParams exactly)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelInit:
  init_size_frac: float
  init_size_frac_h: float | None
  init_size_frac_v: float | None
  mesh_step_px: int
  winding_step_px: int
  mesh_h: int
  mesh_w: int


@dataclass(frozen=True)
class ModelParams:
  mesh_step_px: int
  winding_step_px: int
  subsample_mesh: int
  subsample_winding: int
  z_step_vx: int
  scaledown: float
  crop_fullres_xyzwhd: tuple[int, int, int, int, int, int] | None = None
  data_margin_modelpx: tuple[float, float] = (0.0, 0.0)
  data_size_modelpx: tuple[int, int] = (0, 0)

  @property
  def crop_xyzwhd(self) -> tuple[int, int, int, int, int, int] | None:
    """3-D crop in model-pixel space (fullres / scaledown), rounded to int."""
    c = self.crop_fullres_xyzwhd
    if c is None:
      return None
    ds = float(self.scaledown) if float(self.scaledown) > 0.0 else 1.0
    x, y, w, h, z0, d = (int(v) for v in c)
    return (
      int(round(x / ds)), int(round(y / ds)),
      max(1, int(round(w / ds))), max(1, int(round(h / ds))),
      int(z0), int(d),
    )


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
  """Output of Model2D.forward()."""
  xy_lr: "Tensor"          # (N, Hm, Wm, 2) — mesh pixel coords at LR
  xy_hr: "Tensor"          # (N, He, We, 2) — mesh pixel coords at HR
  xy_conn: "Tensor"        # (N, Hm, Wm, 3, 2) — connection pts (left/self/right)
  target_plain: "Tensor"   # (N, 1, He, We) — raw cosine winding target
  target_mod: "Tensor"     # (N, 1, He, We) — modulated target (amp/bias applied)
  amp_lr: "Tensor"         # (N, 1, Hm, Wm) — clamped amplitude
  bias_lr: "Tensor"        # (N, 1, Hm, Wm) — clamped bias
  mask_hr: "Tensor"        # (N, 1, He, We) — validity mask at HR
  mask_lr: "Tensor"        # (N, 1, Hm, Wm) — validity mask at LR
  mask_conn: "Tensor"      # (N, 1, Hm, Wm, 3) — validity mask at conn pts
  params: ModelParams


# ---------------------------------------------------------------------------
# Pyramid helpers
# ---------------------------------------------------------------------------

def _upsample2_crop(src: "Tensor", h_t: int, w_t: int) -> "Tensor":
  """Bilinear 2× upsample then crop to (h_t, w_t).  src: (N, C, H, W)."""
  n, c, h, w = src.shape
  up = src.interpolate(size=(h * 2, w * 2), mode="linear")
  if up.shape[2] > h_t or up.shape[3] > w_t:
    up = up[:, :, :h_t, :w_t]
  if up.shape[2] < h_t or up.shape[3] < w_t:
    ph = h_t - up.shape[2]; pw = w_t - up.shape[3]
    up = up.pad(((0, 0), (0, 0), (0, ph), (0, pw)))
  return up


def _pyramid_shapes(h0: int, w0: int, n_scales: int) -> list[tuple[int, int]]:
  shapes: list[tuple[int, int]] = [(h0, w0)]
  for _ in range(1, max(1, n_scales)):
    gh, gw = shapes[-1]
    shapes.append((max(2, (gh + 1) // 2), max(2, (gw + 1) // 2)))
  return shapes


def _build_pyramid_from_flat(flat: "Tensor", n_scales: int) -> list["Tensor"]:
  """Construct a Laplacian-style residual pyramid from a fine-resolution tensor.

  flat: (N, C, H, W)
  Returns list[Tensor] of length n_scales, index 0 = finest residual,
  index -1 = coarsest (stored as-is). Reconstructing by summing from
  coarse to fine (with 2× upsampling) recovers flat exactly.
  """
  h0, w0 = int(flat.shape[2]), int(flat.shape[3])
  shapes = _pyramid_shapes(h0, w0, n_scales)

  # Downsample to each scale.
  targets: list[Tensor] = [flat]
  for gh, gw in shapes[1:]:
    targets.append(targets[-1].interpolate(size=(gh, gw), mode="linear"))

  # Build residuals coarse→fine.
  residuals: list[Tensor | None] = [None] * len(targets)
  recon = targets[-1]
  residuals[-1] = targets[-1]
  for i in range(len(targets) - 2, -1, -1):
    up = _upsample2_crop(recon, int(targets[i].shape[2]), int(targets[i].shape[3]))
    d = targets[i] - up
    residuals[i] = d
    recon = up + d

  return [Tensor(r.numpy(), requires_grad=True) for r in residuals]  # detach all


def _integrate_pyramid(pyramid: list["Tensor"]) -> "Tensor":
  """Reconstruct fine-resolution tensor from residual pyramid."""
  v = pyramid[-1]
  for d in reversed(pyramid[:-1]):
    v = _upsample2_crop(v, int(d.shape[2]), int(d.shape[3])) + d
  return v


def _zeros_pyramid(z_size: int, n_ch: int, h0: int, w0: int, n_scales: int) -> list["Tensor"]:
  """Build a pyramid initialised to zero."""
  shapes = _pyramid_shapes(h0, w0, n_scales)
  return [Tensor(np.zeros((z_size, n_ch, gh, gw), dtype=np.float32), requires_grad=True) for (gh, gw) in shapes]


# ---------------------------------------------------------------------------
# Validity mask
# ---------------------------------------------------------------------------

def xy_img_validity_mask(*, params: ModelParams, xy: "Tensor") -> "Tensor":
  """Float32 binary validity mask for model-pixel positions xy (..., 2).

  Returns Tensor of shape xy.shape[:-1].
  """
  dh, dw = params.data_size_modelpx
  if dh > 0 and dw > 0:
    h_lim = float(max(1, dh - 1)); w_lim = float(max(1, dw - 1))
  else:
    if params.crop_xyzwhd is None:
      raise ValueError("xy_img_validity_mask requires params.crop_xyzwhd or data_size_modelpx")
    _cx, _cy, cw, ch, _z0, _d = params.crop_xyzwhd
    h_lim = float(max(1, ch - 1)); w_lim = float(max(1, cw - 1))
  xy_np = xy.numpy()
  shape = xy_np.shape[:-1]
  flat = xy_np.reshape(-1, 2)
  inside = ((flat[:, 0] >= 0.0) & (flat[:, 0] <= w_lim) &
            (flat[:, 1] >= 0.0) & (flat[:, 1] <= h_lim)).astype(np.float32)
  return Tensor(inside.reshape(shape))


# ---------------------------------------------------------------------------
# Grid-sample helper (bilinear, zero padding)
# ---------------------------------------------------------------------------

def grid_sample_px(image: "Tensor", xy_px: "Tensor") -> "Tensor":
  """Bilinear image sampling at pixel-coordinate positions.

  image:  (N, C, H, W) float32
  xy_px:  (N, Hq, Wq, 2) in pixel units (x=col, y=row)

  Returns (N, C, Hq, Wq) sampled values, zero outside bounds.
  """
  n, c, h, w = int(image.shape[0]), int(image.shape[1]), int(image.shape[2]), int(image.shape[3])
  hq, wq = int(xy_px.shape[1]), int(xy_px.shape[2])

  img_np = image.numpy()       # (N, C, H, W)
  xy_np  = xy_px.numpy()       # (N, Hq, Wq, 2)

  out = np.zeros((n, c, hq, wq), dtype=np.float32)
  for bi in range(n):
    xs = xy_np[bi, :, :, 0]   # (Hq, Wq)
    ys = xy_np[bi, :, :, 1]
    # Integer floor coords; for pixels exactly on the right/bottom border x1==w
    # is still valid — we clamp and let wx1=0 handle it.
    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1; y1 = y0 + 1
    wx1 = (xs - x0.astype(np.float32)).astype(np.float32); wx0 = 1.0 - wx1
    wy1 = (ys - y0.astype(np.float32)).astype(np.float32); wy0 = 1.0 - wy1
    # A sample is valid if the floor corner is inside the image bounds.
    valid = (x0 >= 0) & (x0 < w) & (y0 >= 0) & (y0 < h)
    x0c = np.clip(x0, 0, w - 1); x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1); y1c = np.clip(y1, 0, h - 1)
    for ci in range(c):
      im = img_np[bi, ci]   # (H, W)
      v = (wx0 * wy0 * im[y0c, x0c] + wx1 * wy0 * im[y0c, x1c] +
           wx0 * wy1 * im[y1c, x0c] + wx1 * wy1 * im[y1c, x1c])
      out[bi, ci] = np.where(valid, v, 0.0)
  return Tensor(out)


# ---------------------------------------------------------------------------
# Cosine winding target
# ---------------------------------------------------------------------------

def cosine_winding_target(z_size: int, h: int, w: int, periods: int | None = None) -> "Tensor":
  """Return (z_size, 1, h, w) cosine winding target: 0.5 + 0.5*cos(2π*x/P).

  periods defaults to w-1 (one full cycle across the mesh width).
  """
  P = float(max(1, w - 1) if periods is None else periods)
  xs = np.linspace(0.0, P, w, dtype=np.float32)
  col = (0.5 + 0.5 * np.cos(2.0 * math.pi * xs / P)).reshape(1, 1, 1, w)
  return Tensor(np.broadcast_to(col, (z_size, 1, h, w)).copy())


# ---------------------------------------------------------------------------
# Model2D
# ---------------------------------------------------------------------------

class Model2D:
  """Deformable 2-D surface fitting model (tinygrad port of villa's Model2D).

  Trainable parameters
  --------------------
  mesh_ms       list[Tensor]  multi-scale mesh residual pyramid (N, 2, H, W) per scale
  conn_offset_ms list[Tensor] multi-scale connection-offset pyramid (N, 2, H, W) per scale
  amp           Tensor        per-voxel amplitude  (N, 1, Hm, Wm)
  bias          Tensor        per-voxel bias       (N, 1, Hm, Wm)
  theta         Tensor        global rotation scalar [1]
  winding_scale Tensor        global winding scale scalar [1]

  Non-trainable state
  -------------------
  xy_ema        np.ndarray    EMA of xy_lr (updated by update_ema)
  xy_conn_ema   np.ndarray    EMA of xy_conn (updated by update_ema)
  """

  # -- construction ----------------------------------------------------------

  def __init__(
    self,
    init: ModelInit,
    *,
    z_size: int = 1,
    subsample_mesh: int = 4,
    subsample_winding: int = 4,
    z_step_vx: int = 1,
    scaledown: float = 1.0,
    crop_xyzwhd: tuple[int, int, int, int, int, int] | None = None,
    data_margin_modelpx: tuple[float, float] | None = None,
    data_size_modelpx: tuple[int, int] | None = None,
    n_pyramid_scales: int = 5,
  ) -> None:
    if not _HAS_TINYGRAD:
      raise ImportError("tinygrad is required for Model2D")

    self.init = init
    self.z_size = max(1, int(z_size))
    self.mesh_h = max(2, int(init.mesh_h))
    self.mesh_w = max(2, int(init.mesh_w))
    self.n_pyramid_scales = max(1, int(n_pyramid_scales))
    self.global_transform_enabled: bool = True

    self.params = ModelParams(
      mesh_step_px=int(init.mesh_step_px),
      winding_step_px=int(init.winding_step_px),
      subsample_mesh=int(subsample_mesh),
      subsample_winding=int(subsample_winding),
      z_step_vx=max(1, int(z_step_vx)),
      scaledown=float(scaledown),
      crop_fullres_xyzwhd=None if crop_xyzwhd is None else tuple(int(v) for v in crop_xyzwhd),
      data_margin_modelpx=tuple(float(v) for v in data_margin_modelpx) if data_margin_modelpx else (0.0, 0.0),
      data_size_modelpx=tuple(int(v) for v in data_size_modelpx) if data_size_modelpx else (0, 0),
    )

    # Mesh pyramid: initialised to crop-centred grid (pixel units).
    flat = Tensor(self._build_base_grid_np(), requires_grad=False)
    self.mesh_ms: list[Tensor] = _build_pyramid_from_flat(flat, self.n_pyramid_scales)

    # Connection-offset pyramid: starts at zero (orthogonal connections at row offset 0).
    self.conn_offset_ms: list[Tensor] = _zeros_pyramid(
      self.z_size, 2, self.mesh_h, self.mesh_w, self.n_pyramid_scales
    )

    self.amp          = Tensor(np.ones ((self.z_size, 1, self.mesh_h, self.mesh_w), dtype=np.float32), requires_grad=True)
    self.bias         = Tensor(np.full ((self.z_size, 1, self.mesh_h, self.mesh_w), 0.5, dtype=np.float32), requires_grad=True)
    self.theta        = Tensor(np.array([0.0], dtype=np.float32), requires_grad=True)
    self.winding_scale= Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)

    # EMA buffers (numpy, no grad).
    self.xy_ema:      np.ndarray = np.empty((0,), dtype=np.float32)
    self.xy_conn_ema: np.ndarray = np.empty((0,), dtype=np.float32)
    self._ema_decay: float = 0.99

    # Transient intersection state (set during forward, consumed by update_conn_offsets).
    self._last_conn_t_l:   np.ndarray | None = None
    self._last_conn_t_r:   np.ndarray | None = None
    self._last_conn_seg_l: np.ndarray | None = None
    self._last_conn_seg_r: np.ndarray | None = None

  # -- parameter access ------------------------------------------------------

  def get_parameters(self) -> list["Tensor"]:
    """Return all trainable parameters as a flat list."""
    return (self.mesh_ms + self.conn_offset_ms +
            [self.amp, self.bias, self.theta, self.winding_scale])

  # -- mesh accessors --------------------------------------------------------

  def mesh_coarse(self) -> "Tensor":
    """Reconstruct fine-resolution mesh. Shape: (N, 2, Hm, Wm)."""
    return _integrate_pyramid(self.mesh_ms)

  def conn_offset_coarse(self) -> "Tensor":
    """Reconstruct fine-resolution connection offsets. Shape: (N, 2, Hm, Wm)."""
    return _integrate_pyramid(self.conn_offset_ms)

  def grid_uv(self) -> tuple["Tensor", "Tensor"]:
    """Return (u, v) pixel-unit mesh coords, each (N, 1, Hm, Wm)."""
    m = self.mesh_coarse()
    return m[:, 0:1], m[:, 1:2]

  # -- global transform ------------------------------------------------------

  def _apply_global_transform(self, u: "Tensor", v: "Tensor") -> tuple["Tensor", "Tensor"]:
    ws = self.winding_scale.reshape(1, 1, 1, 1)
    u  = ws * u
    c  = self.theta.cos().reshape(1, 1, 1, 1)
    s  = self.theta.sin().reshape(1, 1, 1, 1)
    xc = 0.5 * (u.min() + u.max())
    yc = 0.5 * (v.min() + v.max())
    x  = xc + c * (u - xc) - s * (v - yc)
    y  = yc + s * (u - xc) + c * (v - yc)
    return x, y

  def bake_global_transform(self) -> None:
    """Fold current global transform into mesh parameters and disable it."""
    u, v = self.grid_uv()
    x, y = self._apply_global_transform(u, v)
    flat = Tensor(np.concatenate([x.numpy(), y.numpy()], axis=1))
    self.mesh_ms = _build_pyramid_from_flat(flat, self.n_pyramid_scales)
    self.theta         = Tensor(np.array([0.0], dtype=np.float32), requires_grad=True)
    self.winding_scale = Tensor(np.array([1.0], dtype=np.float32), requires_grad=True)
    self.global_transform_enabled = False

  # -- grid helpers ----------------------------------------------------------

  def _grid_xy(self) -> "Tensor":
    """Mesh pixel coords with global transform applied. Shape: (N, Hm, Wm, 2)."""
    u, v = self.grid_uv()
    if self.global_transform_enabled:
      u, v = self._apply_global_transform(u, v)
    # (N, 2, Hm, Wm) → (N, Hm, Wm, 2)
    return Tensor(np.stack([u.numpy()[:, 0], v.numpy()[:, 0]], axis=-1))

  def _grid_xy_subsampled(self, *, xy_lr: "Tensor") -> "Tensor":
    """Upsample LR grid to HR using subsample_mesh × subsample_winding.

    Shape: (N, He, We, 2) where He = (Hm-1)*ss_h+1, We = (Wm-1)*ss_w+1.
    """
    ss_h = max(1, int(self.params.subsample_mesh))
    ss_w = max(1, int(self.params.subsample_winding))
    n, hm, wm, _ = xy_lr.shape
    he = max(2, (hm - 1) * ss_h + 1)
    we = max(2, (wm - 1) * ss_w + 1)
    # (N, Hm, Wm, 2) → (N, 2, Hm, Wm) → upsample → (N, 2, He, We) → (N, He, We, 2)
    xy_t  = Tensor(xy_lr.numpy().transpose(0, 3, 1, 2))
    xy_up = xy_t.interpolate(size=(he, we), mode="linear")
    return Tensor(xy_up.numpy().transpose(0, 2, 3, 1))

  # -- connection points -----------------------------------------------------

  def _xy_conn_px(self, *, xy_lr: "Tensor") -> "Tensor":
    """Per-mesh connection positions in pixel coords. Shape: (N, Hm, Wm, 3, 2).

    For each mesh point returns [left-conn, self, right-conn] where the left/
    right points are the intersection of the orthogonal-to-v ray with the
    neighbour column at the offset selected by conn_offset_ms.
    """
    xy_np = xy_lr.numpy()                     # (N, Hm, Wm, 2)
    n, hm, wm, _ = xy_np.shape
    xy_px = xy_np.transpose(0, 3, 1, 2)      # (N, 2, Hm, Wm)

    # V-direction via central / fwd / bwd differences.
    v_dir = np.zeros_like(xy_px)
    if hm >= 3:
      v_dir[:, :, 1:-1, :] = xy_px[:, :, 2:, :] - xy_px[:, :, :-2, :]
      v_dir[:, :, 0, :]    = xy_px[:, :, 1, :]  - xy_px[:, :, 0, :]
      v_dir[:, :, -1, :]   = xy_px[:, :, -1, :] - xy_px[:, :, -2, :]
    elif hm >= 2:
      v_dir[:, :, 0, :] = xy_px[:, :, 1, :] - xy_px[:, :, 0, :]
      v_dir[:, :, 1, :] = v_dir[:, :, 0, :]

    # Orthogonal direction: rotate 90° CCW → (-vy, vx).
    d_x = -v_dir[:, 1:2]   # (N, 1, Hm, Wm)
    d_y =  v_dir[:, 0:1]

    # Neighbor columns (edge-replicated).
    left_src  = np.concatenate([xy_px[:, :, :, 0:1], xy_px[:, :, :, :-1]], axis=3)
    right_src = np.concatenate([xy_px[:, :, :, 1:],  xy_px[:, :, :, -1:]], axis=3)

    off_np  = self.conn_offset_coarse().numpy()  # (N, 2, Hm, Wm)
    off_l   = off_np[:, 0]   # (N, Hm, Wm)
    off_r   = off_np[:, 1]
    base_i  = np.arange(hm, dtype=np.float32).reshape(1, hm, 1)  # (1, Hm, 1)

    def _intersect(src: np.ndarray, off: np.ndarray):
      idx   = base_i + off                                        # (N, Hm, Wm)
      seg_s = np.floor(idx).clip(0.0, float(hm - 2))
      seg_i = seg_s.astype(np.int32)                             # (N, Hm, Wm)

      idx0 = seg_i[:, np.newaxis].repeat(2, axis=1)              # (N, 2, Hm, Wm)
      idx1 = (seg_i + 1)[:, np.newaxis].repeat(2, axis=1)
      # Gather along the height axis (axis=2).
      A = np.take_along_axis(src, idx0, axis=2)
      B = np.take_along_axis(src, idx1, axis=2)

      e_x = B[:, 0:1] - A[:, 0:1]
      e_y = B[:, 1:2] - A[:, 1:2]
      g_x = xy_px[:, 0:1] - A[:, 0:1]
      g_y = xy_px[:, 1:2] - A[:, 1:2]

      g_cross_d = g_x * d_y - g_y * d_x
      e_cross_d = e_x * d_y - e_y * d_x

      eps   = 1e-8
      denom = np.where(np.abs(e_cross_d) < eps, np.full_like(e_cross_d, eps), e_cross_d)
      t     = (g_cross_d / denom)[:, 0]                          # (N, Hm, Wm)
      conn  = A + t[:, np.newaxis] * (B - A)                     # (N, 2, Hm, Wm)
      return conn, t, seg_s

    left_conn,  t_l, seg_l = _intersect(left_src,  off_l)
    right_conn, t_r, seg_r = _intersect(right_src, off_r)

    # Cache for update_conn_offsets (detached — numpy arrays).
    self._last_conn_t_l   = t_l
    self._last_conn_t_r   = t_r
    self._last_conn_seg_l = seg_l
    self._last_conn_seg_r = seg_r

    # Edge fallback: outermost left/right connections are synthetic.
    if wm > 1:
      v_in_l  = right_conn[:, :, :, 1] - xy_px[:, :, :, 1]
      v_in_r  = left_conn[:, :, :, -2] - xy_px[:, :, :, -2]
      left_conn  = left_conn.copy()
      right_conn = right_conn.copy()
      left_conn[:, :, :, 0]  = xy_px[:, :, :, 0]  - v_in_l
      right_conn[:, :, :, -1]= xy_px[:, :, :, -1] - v_in_r

    # Stack: (N, 2, Hm, Wm) × 3 → (N, 3, 2, Hm, Wm) → (N, Hm, Wm, 3, 2)
    conn_np = np.stack([left_conn, xy_px, right_conn], axis=1)  # (N, 3, 2, Hm, Wm)
    conn_np = conn_np.transpose(0, 3, 4, 1, 2)                  # (N, Hm, Wm, 3, 2)
    return Tensor(conn_np.astype(np.float32))

  def update_conn_offsets(self) -> None:
    """Update conn_offset_ms from the latest forward-pass intersection params."""
    if any(v is None for v in [self._last_conn_t_l, self._last_conn_t_r,
                                self._last_conn_seg_l, self._last_conn_seg_r]):
      return
    hm     = int(self._last_conn_t_l.shape[1])
    base_i = np.arange(hm, dtype=np.float32).reshape(1, hm, 1)
    new_l  = self._last_conn_seg_l + self._last_conn_t_l - base_i
    new_r  = self._last_conn_seg_r + self._last_conn_t_r - base_i

    cur_np = self.conn_offset_coarse().numpy()  # (N, 2, Hm, Wm)
    delta_l = new_l - cur_np[:, 0]
    delta_r = new_r - cur_np[:, 1]

    # Apply delta to finest level only (additive, no grad needed for this update).
    data = self.conn_offset_ms[0].numpy().copy()
    data[:, 0] += delta_l; data[:, 1] += delta_r
    self.conn_offset_ms[0] = Tensor(data, requires_grad=True)

    self._last_conn_t_l = self._last_conn_t_r = None
    self._last_conn_seg_l = self._last_conn_seg_r = None

  # -- EMA -------------------------------------------------------------------

  def update_ema(self, *, xy_lr: "Tensor", xy_conn: "Tensor") -> None:
    """Update EMA buffers (called after each optimizer step, no grad)."""
    d  = self._ema_decay
    x  = xy_lr.numpy().astype(np.float32)
    xc = xy_conn.numpy().astype(np.float32)
    if self.xy_ema.shape != x.shape:
      self.xy_ema = x.copy()
    else:
      self.xy_ema = d * self.xy_ema + (1.0 - d) * x
    if self.xy_conn_ema.shape != xc.shape:
      self.xy_conn_ema = xc.copy()
    else:
      self.xy_conn_ema = d * self.xy_conn_ema + (1.0 - d) * xc

  # -- forward ---------------------------------------------------------------

  def forward(self, image: "Tensor | None" = None) -> FitResult:
    """Run the model forward pass.

    image: optional (N, C, H, W) tensor.  When provided, data_s is sampled
           at xy_hr and attached to the returned FitResult as data_s; when
           None the result's data_s attribute is left as None.

    Returns FitResult with mesh coordinates, targets, and validity masks.
    """
    xy_lr   = self._grid_xy()                       # (N, Hm, Wm, 2)
    xy_hr   = self._grid_xy_subsampled(xy_lr=xy_lr) # (N, He, We, 2)
    xy_conn = self._xy_conn_px(xy_lr=xy_lr)         # (N, Hm, Wm, 3, 2)

    n, hm, wm, _ = xy_lr.numpy().shape
    _, he, we, _ = xy_hr.numpy().shape

    # Cosine winding target: 0.5 + 0.5*cos(2π*x / (wm-1)) tiled over z & h.
    periods = max(1, wm - 1)
    xs_np   = np.linspace(0.0, float(periods), we, dtype=np.float32)
    phase   = (2.0 * math.pi * xs_np / float(periods)).reshape(1, 1, 1, we)
    target_plain = Tensor(np.broadcast_to(0.5 + 0.5 * np.cos(phase), (n, 1, he, we)).copy())

    # Per-voxel amplitude / bias modulation.
    amp_lr  = self.amp.clip(0.1, 1.0)
    bias_lr = self.bias.clip(0.0, 0.45)
    amp_hr  = amp_lr.interpolate(size=(he, we), mode="linear")
    bias_hr = bias_lr.interpolate(size=(he, we), mode="linear")
    target_mod = (bias_hr + amp_hr * (target_plain - 0.5)).clip(0.0, 1.0)

    # Validity masks.
    mask_hr   = xy_img_validity_mask(params=self.params, xy=xy_hr).unsqueeze(1)
    mask_lr   = xy_img_validity_mask(params=self.params, xy=xy_lr).unsqueeze(1)
    # conn mask: sample validity at each of the 3 connection points.
    conn_np   = xy_conn.numpy()                           # (N, Hm, Wm, 3, 2)
    conn_flat = Tensor(conn_np.reshape(n, hm, wm * 3, 2))
    mask_flat = xy_img_validity_mask(params=self.params, xy=conn_flat)  # (N, Hm, Wm*3)
    mask_conn_np = mask_flat.numpy().reshape(n, 1, hm, wm, 3)
    # Edges are synthetic — zero out.
    mask_conn_np[:, :, :, 0,  0] = 0.0
    mask_conn_np[:, :, :, -1, 2] = 0.0
    mask_conn = Tensor(mask_conn_np)

    return FitResult(
      xy_lr=xy_lr, xy_hr=xy_hr, xy_conn=xy_conn,
      target_plain=target_plain, target_mod=target_mod,
      amp_lr=amp_lr, bias_lr=bias_lr,
      mask_hr=mask_hr, mask_lr=mask_lr, mask_conn=mask_conn,
      params=self.params,
    )

  # -- grow ------------------------------------------------------------------

  def grow(self, *, directions: list[str], steps: int = 1) -> None:
    """Expand mesh in one or more directions by `steps` grid cells.

    Supported directions: 'left', 'right', 'up', 'down', 'fw', 'bw'.
    'fw'/'bw' add a new z-slice (copies the outermost slice).
    Spatial directions use linear extrapolation for mesh_ms and edge-copy
    for conn_offset_ms, amp, and bias.
    """
    steps = max(0, int(steps))
    if steps == 0:
      return
    dirs = {str(d).strip().lower() for d in directions}
    bad  = dirs - {"left", "right", "up", "down", "fw", "bw"}
    if bad:
      raise ValueError(f"invalid grow direction(s): {sorted(bad)}")

    # -- Z directions (fw / bw): add z-slices --------------------------------
    for z_dir in ("bw", "fw"):
      if z_dir not in dirs:
        continue
      side = +1 if z_dir == "fw" else -1   # +1 = append at end, -1 = prepend
      for _ in range(steps):
        # mesh_ms and conn_offset_ms: copy edge z-slice.
        self.mesh_ms         = [_param_expand_z(p, side) for p in self.mesh_ms]
        self.conn_offset_ms  = [_param_expand_z(p, side) for p in self.conn_offset_ms]
        self.amp             = _param_expand_z(self.amp,  side)
        self.bias            = _param_expand_z(self.bias, side)
        self.z_size         += 1

    # -- XY directions -------------------------------------------------------
    xy_dirs = dirs - {"fw", "bw"}
    if not xy_dirs:
      return

    # (numpy axis, border-side for linear extrap, Δmesh_h, Δmesh_w)
    grow_specs: dict[str, tuple[int, int, int, int]] = {
      "up":    (2, 0,  +1, 0),
      "down":  (2, -1, +1, 0),
      "left":  (3, 0,  0,  +1),
      "right": (3, -1, 0,  +1),
    }
    order     = ["up", "down", "left", "right"]
    dirs_list = [d for d in order if d in xy_dirs]

    # Integrate pyramids to flat tensors for manipulation.
    flat_mesh = _integrate_pyramid(self.mesh_ms).numpy().copy()         # (N, 2, Hm, Wm)
    flat_conn = _integrate_pyramid(self.conn_offset_ms).numpy().copy()  # (N, 2, Hm, Wm)

    for _ in range(steps):
      for d in dirs_list:
        axis, side, dh, dw = grow_specs[d]
        flat_mesh  = _expand_linear_np(flat_mesh, axis=axis, side=side)
        flat_conn  = _expand_copy_edge_np(flat_conn, axis=axis, side=side)
        amp_np     = _expand_copy_edge_np(self.amp.numpy(),  axis=axis, side=side)
        bias_np    = _expand_copy_edge_np(self.bias.numpy(), axis=axis, side=side)
        self.amp   = Tensor(amp_np,  requires_grad=True)
        self.bias  = Tensor(bias_np, requires_grad=True)
        self.mesh_h += dh; self.mesh_w += dw

    self.mesh_ms        = _build_pyramid_from_flat(Tensor(flat_mesh), self.n_pyramid_scales)
    self.conn_offset_ms = _build_pyramid_from_flat(Tensor(flat_conn), self.n_pyramid_scales)

  # -- serialisation ---------------------------------------------------------

  def state_dict(self) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(self.mesh_ms):
      out[f"mesh_ms.{i}"] = p.numpy()
    for i, p in enumerate(self.conn_offset_ms):
      out[f"conn_offset_ms.{i}"] = p.numpy()
    out["amp"]           = self.amp.numpy()
    out["bias"]          = self.bias.numpy()
    out["theta"]         = self.theta.numpy()
    out["winding_scale"] = self.winding_scale.numpy()
    return out

  def load_state_dict(self, state: dict) -> None:
    def _load(key: str, current: "Tensor") -> "Tensor":
      if key in state:
        return Tensor(np.asarray(state[key], dtype=np.float32), requires_grad=True)
      return current
    self.mesh_ms        = [_load(f"mesh_ms.{i}",        p) for i, p in enumerate(self.mesh_ms)]
    self.conn_offset_ms = [_load(f"conn_offset_ms.{i}", p) for i, p in enumerate(self.conn_offset_ms)]
    self.amp            = _load("amp",           self.amp)
    self.bias           = _load("bias",          self.bias)
    self.theta          = _load("theta",         self.theta)
    self.winding_scale  = _load("winding_scale", self.winding_scale)

  # -- private ---------------------------------------------------------------

  def _build_base_grid_np(self) -> np.ndarray:
    """Build initial mesh as a centred grid in pixel space. Shape: (N, 2, Hm, Wm)."""
    c = self.params.crop_xyzwhd
    if c is not None:
      _cx, _cy, cw, ch, _z0, _d = c
      w = float(max(1, cw - 1)); h = float(max(1, ch - 1))
      mx, my = self.params.data_margin_modelpx
      xc = float(mx) + 0.5 * w; yc = float(my) + 0.5 * h
      fh = float(self.init.init_size_frac if self.init.init_size_frac_h is None else self.init.init_size_frac_h)
      fw = float(self.init.init_size_frac if self.init.init_size_frac_v is None else self.init.init_size_frac_v)
      u = np.linspace(xc - 0.5 * fw * w, xc + 0.5 * fw * w, self.mesh_w, dtype=np.float32)
      v = np.linspace(yc - 0.5 * fh * h, yc + 0.5 * fh * h, self.mesh_h, dtype=np.float32)
    else:
      u = np.linspace(0.0, float(self.mesh_w - 1), self.mesh_w, dtype=np.float32)
      v = np.linspace(0.0, float(self.mesh_h - 1), self.mesh_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)                                # (Hm, Wm) each
    grid = np.stack([uu, vv], axis=0)[np.newaxis]             # (1, 2, Hm, Wm)
    return np.broadcast_to(grid, (self.z_size, 2, self.mesh_h, self.mesh_w)).copy()


# ---------------------------------------------------------------------------
# Numpy grow helpers
# ---------------------------------------------------------------------------

def _expand_linear_np(arr: np.ndarray, axis: int, side: int) -> np.ndarray:
  """Linearly extrapolate one slice at `side` (0=front, -1=back) of `axis`."""
  n = arr.shape[axis]
  if n < 2:
    return _expand_copy_edge_np(arr, axis=axis, side=side)
  if side == 0:
    edge = np.take(arr, 0, axis=axis); nxt = np.take(arr, 1, axis=axis)
    new  = np.expand_dims(2.0 * edge - nxt, axis)
    return np.concatenate([new, arr], axis=axis)
  edge = np.take(arr, n - 1, axis=axis); nxt = np.take(arr, n - 2, axis=axis)
  new  = np.expand_dims(2.0 * edge - nxt, axis)
  return np.concatenate([arr, new], axis=axis)


def _expand_copy_edge_np(arr: np.ndarray, axis: int, side: int) -> np.ndarray:
  """Replicate the border slice at `side` (0=front, -1=back) of `axis`."""
  if side == 0:
    edge = np.expand_dims(np.take(arr, 0, axis=axis), axis)
    return np.concatenate([edge, arr], axis=axis)
  edge = np.expand_dims(np.take(arr, arr.shape[axis] - 1, axis=axis), axis)
  return np.concatenate([arr, edge], axis=axis)


def _param_expand_z(p: "Tensor", side: int) -> "Tensor":
  """Copy the outermost z-slice (axis=0) and append/prepend it."""
  arr = p.numpy()
  new = _expand_copy_edge_np(arr, axis=0, side=side if side == 0 else -1)
  return Tensor(new, requires_grad=True)
