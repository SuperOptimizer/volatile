from __future__ import annotations
"""2-D surface fitting model ported from villa's exps_2d_model/model.py.

Uses tinygrad for all tensor operations.  The model represents a deformable
mesh that maps papyrus-sheet winding coordinates to image (x,y) positions.
A multi-scale pyramid of mesh residuals allows fine-to-coarse optimisation.

Key design differences from the villa PyTorch version:
- Parameters stored as plain Tensor with `requires_grad=True` (no nn.Module).
- Pyramid is a plain list of Tensors, not nn.ParameterList.
- No EMA buffers (they're runtime-only in villa too).
- `get_parameters()` returns a flat list for optimiser consumption.
"""

import math
from dataclasses import dataclass, field
from typing import Sequence

from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_parameters


# ---------------------------------------------------------------------------
# Dataclasses (mirror villa's ModelInit / ModelParams)
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
# Helpers
# ---------------------------------------------------------------------------

def _upsample2_crop(src: Tensor, h_t: int, w_t: int) -> Tensor:
  """Bilinear upsample src by 2x then crop/pad to (h_t, w_t).

  src shape: (N, C, H, W)
  """
  n, c, h, w = src.shape
  up = src.interpolate(size=(h * 2, w * 2), mode="linear")
  # Crop to target size (bilinear upsample may be slightly larger).
  up_h, up_w = up.shape[2], up.shape[3]
  if up_h > h_t or up_w > w_t:
    up = up[:, :, :h_t, :w_t]
  # Pad if target is larger than upsample result (shouldn't happen normally).
  if up.shape[2] < h_t or up.shape[3] < w_t:
    pad_h = h_t - up.shape[2]
    pad_w = w_t - up.shape[3]
    up = up.pad(((0,0),(0,0),(0,pad_h),(0,pad_w)))
  return up


def _build_mesh_pyramid(flat: Tensor, n_scales: int) -> list[Tensor]:
  """Build a Laplacian-style residual pyramid from a flat (fine-resolution) mesh.

  Returns a list of Tensors [fine_residual, ..., coarse], where summing up
  the pyramid (with bilinear upsampling) reconstructs flat.

  flat shape: (N, 2, H, W)
  """
  h0, w0 = flat.shape[2], flat.shape[3]
  # Collect target shapes at each scale.
  shapes: list[tuple[int, int]] = [(h0, w0)]
  for _ in range(1, max(1, n_scales)):
    gh, gw = shapes[-1]
    shapes.append((max(2, (gh + 1) // 2), max(2, (gw + 1) // 2)))

  # Downsample targets.
  targets: list[Tensor] = [flat]
  for gh, gw in shapes[1:]:
    targets.append(targets[-1].interpolate(size=(gh, gw), mode="linear"))

  # Build residuals from coarsest up.
  residuals: list[Tensor | None] = [None] * len(targets)
  recon = targets[-1]
  residuals[-1] = targets[-1].detach() if hasattr(targets[-1], "detach") else targets[-1]
  for i in range(len(targets) - 2, -1, -1):
    up = _upsample2_crop(recon, int(targets[i].shape[2]), int(targets[i].shape[3]))
    d = targets[i] - up
    residuals[i] = d
    recon = up + d

  # Make each residual a leaf parameter.
  out = []
  for r in residuals:
    t = Tensor(r.numpy(), requires_grad=True)
    out.append(t)
  return out


def _integrate_pyramid(pyramid: list[Tensor]) -> Tensor:
  """Reconstruct fine-resolution mesh from a residual pyramid."""
  v = pyramid[-1]
  for d in reversed(pyramid[:-1]):
    v = _upsample2_crop(v, int(d.shape[2]), int(d.shape[3])) + d
  return v


def _cosine_winding_target(z_size: int, mesh_h: int, mesh_w: int) -> Tensor:
  """Generate a cosine winding target of shape (z_size, 1, mesh_h, mesh_w).

  The target is 0.5 + 0.5*cos(2π*x/periods) where periods = mesh_w - 1,
  matching villa's forward() target_plain computation.
  """
  import numpy as np
  periods = max(1, mesh_w - 1)
  xs = np.linspace(0.0, float(periods), mesh_w, dtype=np.float32)
  phase = 2.0 * math.pi * xs  # (mesh_w,)
  col = (0.5 + 0.5 * np.cos(phase)).astype(np.float32)  # (mesh_w,)
  target = np.tile(col[None, None, None, :], (z_size, 1, mesh_h, 1))
  return Tensor(target)


def xy_img_validity_mask(*, params: ModelParams, xy: Tensor) -> Tensor:
  """Binary validity mask for pixel positions xy.

  xy: (..., 2) in model pixel coordinates.
  Returns float32 Tensor of shape xy.shape[:-1].
  """
  import numpy as np
  dh, dw = params.data_size_modelpx
  if dh > 0 and dw > 0:
    h_lim = float(max(1, dh - 1))
    w_lim = float(max(1, dw - 1))
  else:
    if params.crop_xyzwhd is None:
      raise ValueError("xy_img_validity_mask requires params.crop_xyzwhd or data_size_modelpx")
    _cx, _cy, cw, ch, _z0, _d = params.crop_xyzwhd
    h_lim = float(max(1, ch - 1))
    w_lim = float(max(1, cw - 1))

  xy_np = xy.numpy()
  shape = xy_np.shape[:-1]
  flat = xy_np.reshape(-1, 2)
  x = flat[:, 0]; y = flat[:, 1]
  inside = ((x >= 0.0) & (x <= w_lim) & (y >= 0.0) & (y <= h_lim)).astype(np.float32)
  return Tensor(inside.reshape(shape))


# ---------------------------------------------------------------------------
# FitResult
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
  xy_lr: Tensor       # (N, Hm, Wm, 2)
  xy_hr: Tensor       # (N, He, We, 2)
  target_plain: Tensor  # (N, 1, He, We)
  target_mod: Tensor    # (N, 1, He, We)
  mask_hr: Tensor       # (N, 1, He, We)
  mask_lr: Tensor       # (N, 1, Hm, Wm)
  params: ModelParams


# ---------------------------------------------------------------------------
# Model2D
# ---------------------------------------------------------------------------

class Model2D:
  """Deformable 2-D surface fitting model (tinygrad port of villa's Model2D).

  Optimisable parameters:
  - mesh_ms: list[Tensor] — multi-scale mesh residual pyramid (2 channels, xy offsets)
  - amp: Tensor — per-mesh amplitude (z_size, 1, mesh_h, mesh_w)
  - bias: Tensor — per-mesh bias  (z_size, 1, mesh_h, mesh_w)
  - theta: Tensor — global rotation scalar
  - winding_scale: Tensor — global winding scale scalar

  Usage::

      model = Model2D(init, z_size=1, subsample_mesh=4, subsample_winding=4,
                      z_step_vx=1, scaledown=1.0, crop_xyzwhd=(0,0,100,100,0,10))
      result = model.forward()
      loss = ...
      loss.backward()
      # update parameters via optimiser
  """

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
    import numpy as np
    self.init = init
    self.z_size = max(1, int(z_size))
    self.mesh_h = max(2, int(init.mesh_h))
    self.mesh_w = max(2, int(init.mesh_w))
    self.n_pyramid_scales = max(1, int(n_pyramid_scales))
    self.global_transform_enabled = True

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

    # Initialise mesh as identity grid (pixel coordinates).
    xs = np.linspace(0.0, float(self.mesh_w - 1), self.mesh_w, dtype=np.float32)
    ys = np.linspace(0.0, float(self.mesh_h - 1), self.mesh_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (mesh_h, mesh_w)
    flat_xy = np.stack([grid_x, grid_y], axis=0)[None]  # (1, 2, mesh_h, mesh_w)
    flat_xy = np.broadcast_to(flat_xy, (self.z_size, 2, self.mesh_h, self.mesh_w)).copy()
    flat_t = Tensor(flat_xy)
    self.mesh_ms: list[Tensor] = _build_mesh_pyramid(flat_t, self.n_pyramid_scales)

    self.amp = Tensor(np.ones((self.z_size, 1, self.mesh_h, self.mesh_w), dtype=np.float32), requires_grad=True)
    self.bias = Tensor(np.full((self.z_size, 1, self.mesh_h, self.mesh_w), 0.5, dtype=np.float32), requires_grad=True)
    self.theta = Tensor([0.0], requires_grad=True)
    self.winding_scale = Tensor([1.0], requires_grad=True)

  # --- parameter access ---

  def get_parameters(self) -> list[Tensor]:
    """Return all trainable parameters as a flat list."""
    return self.mesh_ms + [self.amp, self.bias, self.theta, self.winding_scale]

  # --- forward ---

  def mesh_coarse(self) -> Tensor:
    """Reconstruct the full-resolution mesh from the pyramid. Shape: (N, 2, Hm, Wm)."""
    return _integrate_pyramid(self.mesh_ms)

  def _build_base_grid(self) -> Tensor:
    """Compute mesh pixel positions with global transform. Shape: (N, Hm, Wm, 2)."""
    mesh = self.mesh_coarse()  # (N, 2, Hm, Wm)
    u = mesh[:, 0:1]  # x
    v = mesh[:, 1:2]  # y
    if self.global_transform_enabled:
      u, v = self._apply_global_transform(u, v)
    # (N, 2, Hm, Wm) -> (N, Hm, Wm, 2)
    return Tensor.stack([u.squeeze(1), v.squeeze(1)], dim=-1)

  def _apply_global_transform(self, u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    ws = self.winding_scale.reshape(1, 1, 1, 1)
    u = ws * u
    c = self.theta.cos().reshape(1, 1, 1, 1)
    s = self.theta.sin().reshape(1, 1, 1, 1)
    xc = 0.5 * (u.min() + u.max())
    yc = 0.5 * (v.min() + v.max())
    x = xc + c * (u - xc) - s * (v - yc)
    y = yc + s * (u - xc) + c * (v - yc)
    return x, y

  def _grid_xy(self) -> Tensor:
    """Low-res mesh pixel positions. Shape: (N, Hm, Wm, 2)."""
    return self._build_base_grid()

  def _grid_xy_subsampled(self, xy_lr: Tensor) -> Tensor:
    """Upsample LR grid to HR via bilinear interpolation. Shape: (N, He, We, 2)."""
    ss = int(self.params.subsample_mesh)
    if ss <= 1:
      return xy_lr
    n, hm, wm, _ = xy_lr.shape
    he = (hm - 1) * ss + 1
    we = (wm - 1) * ss + 1
    # (N, Hm, Wm, 2) -> (N, 2, Hm, Wm) -> upsample -> (N, 2, He, We) -> (N, He, We, 2)
    xy_t = xy_lr.permute(0, 3, 1, 2)
    xy_up = xy_t.interpolate(size=(he, we), mode="linear")
    return xy_up.permute(0, 2, 3, 1)

  def forward(self) -> FitResult:
    """Run the model forward pass.

    Returns FitResult with mesh positions, targets, and validity masks.
    No data sampling is performed here — callers should sample their own
    data volumes using xy_hr as the sampling grid.
    """
    import numpy as np
    xy_lr = self._grid_xy()           # (N, Hm, Wm, 2)
    xy_hr = self._grid_xy_subsampled(xy_lr)  # (N, He, We, 2)

    n, hm, wm, _ = xy_lr.shape
    _, he, we, _ = xy_hr.shape

    # Cosine winding target.
    periods = max(1, wm - 1)
    xs = np.linspace(0.0, float(periods), we, dtype=np.float32)
    phase = (2.0 * math.pi * xs).reshape(1, 1, 1, we)
    target_plain_np = (0.5 + 0.5 * np.cos(phase)).astype(np.float32)
    target_plain_np = np.broadcast_to(target_plain_np, (n, 1, he, we)).copy()
    target_plain = Tensor(target_plain_np)

    # Amplitude / bias modulation.
    amp_lr = self.amp.clip(0.1, 1.0)
    bias_lr = self.bias.clip(0.0, 0.45)
    amp_hr = amp_lr.interpolate(size=(he, we), mode="linear")
    bias_hr = bias_lr.interpolate(size=(he, we), mode="linear")
    target_mod = (bias_hr + amp_hr * (target_plain - 0.5)).clip(0.0, 1.0)

    # Validity masks from crop bounds.
    mask_hr = xy_img_validity_mask(params=self.params, xy=xy_hr).unsqueeze(1)
    mask_lr = xy_img_validity_mask(params=self.params, xy=xy_lr).unsqueeze(1)

    return FitResult(
      xy_lr=xy_lr,
      xy_hr=xy_hr,
      target_plain=target_plain,
      target_mod=target_mod,
      mask_hr=mask_hr,
      mask_lr=mask_lr,
      params=self.params,
    )

  # --- grow ---

  def grow(self, *, directions: list[str], steps: int = 1) -> None:
    """Expand the mesh by `steps` voxels in the given directions.

    Supported directions: 'up', 'down', 'left', 'right'.
    """
    import numpy as np
    steps = max(0, int(steps))
    if steps == 0:
      return
    dirs = {str(d).strip().lower() for d in directions}
    bad = dirs - {"left", "right", "up", "down"}
    if bad:
      raise ValueError(f"invalid grow direction(s): {sorted(bad)}")

    grow_specs = {
      # (numpy axis in (N,2,H,W), side, d_mesh_h, d_mesh_w)
      "up":    (2, 0,  +1, 0),
      "down":  (2, -1, +1, 0),
      "left":  (3, 0,  0,  +1),
      "right": (3, -1, 0,  +1),
    }
    order = ["up", "down", "left", "right"]
    dirs_list = [d for d in order if d in dirs]

    flat = _integrate_pyramid(self.mesh_ms)

    for _ in range(steps):
      for d in dirs_list:
        axis, side, dh, dw = grow_specs[d]
        flat = _expand_linear_numpy(flat.numpy(), axis=axis, side=side)
        flat = Tensor(flat)
        self.mesh_h += dh
        self.mesh_w += dw

        amp_np = _expand_copy_edge_numpy(self.amp.numpy(), axis=axis, side=side)
        bias_np = _expand_copy_edge_numpy(self.bias.numpy(), axis=axis, side=side)
        self.amp = Tensor(amp_np, requires_grad=True)
        self.bias = Tensor(bias_np, requires_grad=True)

    self.mesh_ms = _build_mesh_pyramid(flat, self.n_pyramid_scales)

  # --- serialisation ---

  def state_dict(self) -> dict[str, "np.ndarray"]:
    out: dict[str, "np.ndarray"] = {}
    for i, p in enumerate(self.mesh_ms):
      out[f"mesh_ms.{i}"] = p.numpy()
    out["amp"] = self.amp.numpy()
    out["bias"] = self.bias.numpy()
    out["theta"] = self.theta.numpy()
    out["winding_scale"] = self.winding_scale.numpy()
    return out

  def load_state_dict(self, state: dict) -> None:
    import numpy as np
    for i in range(len(self.mesh_ms)):
      key = f"mesh_ms.{i}"
      if key in state:
        self.mesh_ms[i] = Tensor(np.asarray(state[key], dtype=np.float32), requires_grad=True)
    if "amp" in state:
      self.amp = Tensor(np.asarray(state["amp"], dtype=np.float32), requires_grad=True)
    if "bias" in state:
      self.bias = Tensor(np.asarray(state["bias"], dtype=np.float32), requires_grad=True)
    if "theta" in state:
      self.theta = Tensor(np.asarray(state["theta"], dtype=np.float32), requires_grad=True)
    if "winding_scale" in state:
      self.winding_scale = Tensor(np.asarray(state["winding_scale"], dtype=np.float32), requires_grad=True)


# ---------------------------------------------------------------------------
# Array manipulation helpers (numpy-side, used by grow)
# ---------------------------------------------------------------------------

def _expand_linear_numpy(arr: "np.ndarray", axis: int, side: int) -> "np.ndarray":
  """Linear extrapolation along `axis` at `side` (0=start, -1=end)."""
  import numpy as np
  if arr.shape[axis] < 2:
    return _expand_copy_edge_numpy(arr, axis=axis, side=side)
  if side == 0:
    edge = np.take(arr, 0, axis=axis)
    next_v = np.take(arr, 1, axis=axis)
    new = (2.0 * edge - next_v)
    return np.concatenate([np.expand_dims(new, axis), arr], axis=axis)
  edge = np.take(arr, arr.shape[axis] - 1, axis=axis)
  next_v = np.take(arr, arr.shape[axis] - 2, axis=axis)
  new = (2.0 * edge - next_v)
  return np.concatenate([arr, np.expand_dims(new, axis)], axis=axis)


def _expand_copy_edge_numpy(arr: "np.ndarray", axis: int, side: int) -> "np.ndarray":
  """Replicate the border slice along `axis` at `side` (0=start, -1=end)."""
  import numpy as np
  if side == 0:
    edge = np.take(arr, 0, axis=axis)
    return np.concatenate([np.expand_dims(edge, axis), arr], axis=axis)
  edge = np.take(arr, arr.shape[axis] - 1, axis=axis)
  return np.concatenate([arr, np.expand_dims(edge, axis)], axis=axis)
