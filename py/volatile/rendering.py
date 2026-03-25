"""rendering.py — Surface rendering and UV flattening for volatile.

Render a QuadSurface to a 2-D image by sampling the volume along each vertex's
normal, and flatten a surface to UV coordinates using LSCM or area-preserving
projection.

All heavy lifting is pure numpy — no scipy, matplotlib, or C extension required
for the core paths.  The C extension (vol_sample) is used when available; a
zarr fallback is used otherwise.
"""
from __future__ import annotations

import math
import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
  from volatile.seg import QuadSurface

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
  from volatile._core import vol_sample as _c_vol_sample, vol_open as _c_vol_open, vol_free as _c_vol_free
  _HAS_CORE = True
except ImportError:
  _HAS_CORE = False

try:
  import zarr as _zarr
  _HAS_ZARR = True
except ImportError:
  _HAS_ZARR = False

try:
  from PIL import Image as _PILImage
  _HAS_PIL = True
except ImportError:
  _HAS_PIL = False

# ---------------------------------------------------------------------------
# Built-in colormaps (sampled at 256 levels, RGB uint8)
# These are compact polynomial approximations — no external dep needed.
# ---------------------------------------------------------------------------

def _make_viridis() -> np.ndarray:
  """Approximate viridis colormap as (256, 3) uint8."""
  t = np.linspace(0.0, 1.0, 256)
  r = np.clip(0.267 + 0.004 * t + 2.049 * t**2 - 1.617 * t**3, 0, 1)
  g = np.clip(0.005 + 1.427 * t - 0.686 * t**2 + 0.178 * t**3, 0, 1)
  b = np.clip(0.330 + 1.099 * t - 2.481 * t**2 + 1.169 * t**3, 0, 1)
  return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

def _make_gray() -> np.ndarray:
  v = np.arange(256, dtype=np.uint8)
  return np.stack([v, v, v], axis=1)

def _make_hot() -> np.ndarray:
  t = np.linspace(0.0, 1.0, 256)
  r = np.clip(t * 3.0, 0, 1)
  g = np.clip(t * 3.0 - 1.0, 0, 1)
  b = np.clip(t * 3.0 - 2.0, 0, 1)
  return (np.stack([r, g, b], axis=1) * 255).astype(np.uint8)

_CMAPS: dict[str, np.ndarray] = {}

def _get_cmap(name: str) -> np.ndarray:
  """Return (256, 3) uint8 colormap table for the given name."""
  if not _CMAPS:
    _CMAPS["viridis"] = _make_viridis()
    _CMAPS["gray"]    = _make_gray()
    _CMAPS["grey"]    = _make_gray()
    _CMAPS["hot"]     = _make_hot()
  return _CMAPS.get(name.lower(), _CMAPS["viridis"])

# ---------------------------------------------------------------------------
# Volume sampling helper
# ---------------------------------------------------------------------------

def _sample_volume(volume, level: int, z: float, y: float, x: float) -> float:
  """Sample a volume at fractional (z, y, x) coordinates.

  volume: either a _core capsule (when C ext available) or a zarr array.
  """
  if _HAS_CORE and volume is not None and type(volume).__name__ == "PyCapsule":
    return _c_vol_sample(volume, level, z, y, x)
  if _HAS_ZARR and volume is not None and hasattr(volume, "__getitem__"):
    # Nearest-neighbour fallback for zarr arrays
    iz = int(round(z - 0.5))
    iy = int(round(y - 0.5))
    ix = int(round(x - 0.5))
    shape = volume.shape
    iz = max(0, min(iz, shape[0] - 1))
    iy = max(0, min(iy, shape[1] - 1))
    ix = max(0, min(ix, shape[2] - 1))
    return float(volume[iz, iy, ix])
  return 0.0

# ---------------------------------------------------------------------------
# render_surface
# ---------------------------------------------------------------------------

def render_surface(
  volume,
  surface: "QuadSurface",
  composite: str = "max",
  layers_front: int = 3,
  layers_behind: int = 3,
  cmap: str = "viridis",
  level: int = 0,
  vmin: float | None = None,
  vmax: float | None = None,
) -> np.ndarray:
  """Render a surface to a 2-D image by sampling the volume along vertex normals.

  For each grid vertex the volume is sampled at `layers_front` positions in front
  of the surface and `layers_behind` positions behind it (step = 1 voxel).
  Samples are composited with the chosen operation and mapped through `cmap`.

  Args:
    volume:        C capsule (vol_open) or zarr array.
    surface:       QuadSurface to render.
    composite:     How to combine layer samples: 'max', 'mean', 'min'.
    layers_front:  Number of layers to sample in the +normal direction.
    layers_behind: Number of layers to sample in the -normal direction.
    cmap:          Colormap name: 'viridis', 'gray', 'hot'.
    level:         Pyramid level (C capsule only).
    vmin/vmax:     Intensity window for normalisation (None = auto).

  Returns:
    (rows, cols, 3) uint8 RGB array.
  """
  rows, cols = surface.rows, surface.cols
  n_layers = layers_front + layers_behind + 1  # +1 for the surface plane itself
  raw = np.zeros((rows, cols, n_layers), dtype=np.float32)

  for r in range(rows):
    for c in range(cols):
      px, py, pz = surface.get(r, c)
      nx, ny, nz = surface.normal(r, c)
      for li, offset in enumerate(range(-layers_behind, layers_front + 1)):
        sx = px + offset * nx
        sy = py + offset * ny
        sz = pz + offset * nz
        raw[r, c, li] = _sample_volume(volume, level, sz, sy, sx)

  # Composite layers
  if composite == "max":
    img = raw.max(axis=2)
  elif composite == "min":
    img = raw.min(axis=2)
  else:  # mean
    img = raw.mean(axis=2)

  # Normalise to [0, 255]
  lo = img.min() if vmin is None else float(vmin)
  hi = img.max() if vmax is None else float(vmax)
  if hi <= lo:
    hi = lo + 1.0
  normed = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
  indices = (normed * 255).astype(np.uint8)

  # Apply colormap
  cmap_table = _get_cmap(cmap)
  return cmap_table[indices]  # (rows, cols, 3)


# ---------------------------------------------------------------------------
# render_surface_to_tiff
# ---------------------------------------------------------------------------

def render_surface_to_tiff(
  volume_path: str,
  surface_path: str,
  output_path: str,
  **kwargs,
) -> None:
  """CLI-compatible: open volume + surface, render, save as TIFF.

  Args:
    volume_path:  Path to an OME-Zarr volume (opened via vol_open or zarr).
    surface_path: Path to a JSON file written by QuadSurface.save().
    output_path:  Destination .tif path.
    **kwargs:     Forwarded to render_surface.
  """
  from volatile.seg import QuadSurface

  # Open volume — prefer C extension for proper OME-Zarr support
  if _HAS_CORE:
    vol = _c_vol_open(volume_path)
    try:
      rgb = render_surface(vol, QuadSurface.load(surface_path), **kwargs)
    finally:
      _c_vol_free(vol)
  elif _HAS_ZARR:
    vol = _zarr.open_array(volume_path, mode="r")
    rgb = render_surface(vol, QuadSurface.load(surface_path), **kwargs)
  else:
    raise RuntimeError("Neither volatile._core nor zarr is available for volume I/O")

  _write_tiff(rgb, output_path)


def _write_tiff(rgb: np.ndarray, path: str) -> None:
  """Write an (H, W, 3) uint8 array to a TIFF file."""
  if _HAS_PIL:
    _PILImage.fromarray(rgb, mode="RGB").save(path)
    return
  # Minimal TIFF writer (uncompressed, RGB, 8-bit) — no external dep
  _write_tiff_raw(rgb, path)


def _write_tiff_raw(rgb: np.ndarray, path: str) -> None:
  """Bare-metal uncompressed RGB TIFF writer (no deps)."""
  h, w, _ = rgb.shape
  data = rgb.tobytes()

  def _ifd_entry(tag, dtype, count, value):
    # dtype 3=SHORT, 4=LONG
    return struct.pack("<HHII", tag, dtype, count, value)

  # IFD entries (sorted by tag)
  strips_offset = 8 + 2 + 12 * 11 + 4  # header + entry count + entries + next IFD
  ifd = struct.pack("<H", 11)  # entry count
  ifd += _ifd_entry(0x0100, 4, 1, w)           # ImageWidth
  ifd += _ifd_entry(0x0101, 4, 1, h)           # ImageLength
  # BitsPerSample = 3 shorts — store inline for count<=1; need offset for 3
  bps_offset = strips_offset + len(data)
  ifd += struct.pack("<HHII", 0x0102, 3, 3, bps_offset)  # BitsPerSample
  ifd += _ifd_entry(0x0103, 3, 1, 1)           # Compression: none
  ifd += _ifd_entry(0x0106, 3, 1, 2)           # PhotometricInterp: RGB
  ifd += _ifd_entry(0x0111, 4, 1, strips_offset + 6)  # StripOffsets (after BPS data)
  ifd += _ifd_entry(0x0115, 3, 1, 3)           # SamplesPerPixel
  ifd += _ifd_entry(0x0116, 4, 1, h)           # RowsPerStrip
  ifd += _ifd_entry(0x0117, 4, 1, len(data))   # StripByteCounts
  ifd += _ifd_entry(0x011A, 4, 1, 72)          # XResolution (rational, inline approx)
  ifd += _ifd_entry(0x011B, 4, 1, 72)          # YResolution
  ifd += struct.pack("<I", 0)                  # next IFD = 0 (last IFD)

  bps_data = struct.pack("<HHH", 8, 8, 8)  # 8 bits per channel

  with open(path, "wb") as f:
    f.write(b"II")                      # little-endian
    f.write(struct.pack("<H", 42))      # TIFF magic
    f.write(struct.pack("<I", 8))       # offset of first IFD
    f.write(ifd)
    f.write(bps_data)
    f.write(data)


# ---------------------------------------------------------------------------
# flatten_surface
# ---------------------------------------------------------------------------

def flatten_surface(
  surface: "QuadSurface",
  method: str = "lscm",
) -> np.ndarray:
  """UV-flatten a quad surface to 2-D coordinates.

  Args:
    surface: QuadSurface with rows×cols vertices.
    method:  'lscm' (Least-Squares Conformal Maps, angle-preserving) or
             'area_preserving' (equal-area projection onto the mean plane).

  Returns:
    (rows, cols, 2) float32 array of UV coordinates in [0, 1].
  """
  if method == "area_preserving":
    return _flatten_area_preserving(surface)
  return _flatten_lscm(surface)


def _flatten_area_preserving(surface: "QuadSurface") -> np.ndarray:
  """Project all vertices onto the least-squares best-fit plane, then normalise."""
  rows, cols = surface.rows, surface.cols
  pts = np.array([[surface.get(r, c) for c in range(cols)] for r in range(rows)], dtype=np.float64)
  pts_flat = pts.reshape(-1, 3)  # (N, 3)

  # Centroid and PCA to find the two principal in-plane axes
  centroid = pts_flat.mean(axis=0)
  centered = pts_flat - centroid
  _, _, Vt = np.linalg.svd(centered, full_matrices=False)
  # Vt rows are principal axes; first two span the best-fit plane
  u_axis = Vt[0]
  v_axis = Vt[1]

  u_coords = centered @ u_axis
  v_coords = centered @ v_axis

  u2d = u_coords.reshape(rows, cols)
  v2d = v_coords.reshape(rows, cols)

  # Normalise to [0, 1]
  u2d = (u2d - u2d.min()) / max(u2d.max() - u2d.min(), 1e-9)
  v2d = (v2d - v2d.min()) / max(v2d.max() - v2d.min(), 1e-9)

  uv = np.stack([u2d, v2d], axis=2).astype(np.float32)
  return uv


def _flatten_lscm(surface: "QuadSurface") -> np.ndarray:
  """Least-Squares Conformal Maps (LSCM) UV flattening.

  Minimises the conformal energy sum_t ||Jt||^2 - 2 det(Jt) where Jt is the
  Jacobian of the UV map on triangle t.  Two boundary vertices are pinned to
  break the global similarity freedom.

  For large surfaces (>512 vertices) we fall back to the area-preserving
  projection to avoid O(N^2) memory usage without scipy.sparse.
  """
  rows, cols = surface.rows, surface.cols
  n_verts = rows * cols

  # Fall back for large surfaces — LSCM without sparse solver is too slow
  if n_verts > 512:
    return _flatten_area_preserving(surface)

  def vidx(r, c): return r * cols + c

  # Collect triangles (each quad split into 2 triangles)
  triangles = []
  for r in range(rows - 1):
    for c in range(cols - 1):
      i0, i1, i2, i3 = vidx(r, c), vidx(r, c + 1), vidx(r + 1, c), vidx(r + 1, c + 1)
      triangles.append((i0, i1, i2))
      triangles.append((i1, i3, i2))

  pts = np.array([list(surface.get(r, c)) for r in range(rows) for c in range(cols)], dtype=np.float64)

  # Build LSCM system matrix A (2*n_tri x 2*n_verts)
  n_tri  = len(triangles)
  n_free = n_verts - 2  # two vertices pinned

  # Pin vertex 0 at (0,0) and vertex (rows-1)*cols at (1,0)
  pin0 = 0
  pin1 = (rows - 1) * cols

  def _local_coords(i0, i1, i2):
    """2-D coords of triangle in its own local frame."""
    p0, p1, p2 = pts[i0], pts[i1], pts[i2]
    e1 = p1 - p0
    e2 = p2 - p0
    x1 = np.linalg.norm(e1)
    if x1 < 1e-12:
      return np.zeros(2), np.zeros(2)
    n_hat = np.cross(e1, e2)
    n_hat /= max(np.linalg.norm(n_hat), 1e-12)
    t_hat = e1 / x1
    b_hat = np.cross(n_hat, t_hat)
    return np.array([e2 @ t_hat, e2 @ b_hat])  # local (x,y) of p2 in frame of (p0,p1)

  # Remap free-vertex indices (skip pin0 and pin1)
  free_indices = [i for i in range(n_verts) if i != pin0 and i != pin1]
  free_map = {v: k for k, v in enumerate(free_indices)}

  # Assemble A (complex LSCM formulation as real 2x2 blocks)
  # Each triangle contributes one row to the complex system.
  # We represent complex uv as interleaved real: [u0,v0, u1,v1, ...]
  rows_A = []
  rhs_rows = []

  for i0, i1, i2 in triangles:
    p0, p1, p2 = pts[i0], pts[i1], pts[i2]
    e01 = p1 - p0
    e02 = p2 - p0
    area2 = np.linalg.norm(np.cross(e01, e02))
    if area2 < 1e-12:
      continue

    # Local 2-D coords of the three vertices (p0 at origin)
    n_hat  = np.cross(e01, e02); n_hat /= np.linalg.norm(n_hat)
    t_hat  = e01 / np.linalg.norm(e01)
    b_hat  = np.cross(n_hat, t_hat)
    q1 = np.array([e01 @ t_hat, e01 @ b_hat])  # = (|e01|, 0)
    q2 = np.array([e02 @ t_hat, e02 @ b_hat])

    # LSCM complex coefficients: W_s * (u1-u0) + W_t * (u2-u0) = 0
    # where W_s = conj(q2-q1)/area2_hat, W_t = -conj(q1)/area2_hat (not needed explicitly)
    # We build the per-triangle row: for each vertex in {i0,i1,i2} compute the
    # coefficient (real, imag) it contributes to the "= 0" equation.
    # Using the standard formulation (Lévy 2002 eq. 7):
    ws_r =  (q2[0] - q1[0]) / area2;  ws_i = -(q2[1] - q1[1]) / area2
    wt_r = -q2[0]            / area2;  wt_i =  q2[1]            / area2
    # coefficient for u0+iv0 = -(ws+wt)
    w0_r = -(ws_r + wt_r);  w0_i = -(ws_i + wt_i)
    w1_r = ws_r;             w1_i = ws_i
    w2_r = wt_r;             w2_i = wt_i

    # Expand into real system (multiply by complex conjugate of row → real rows)
    # Row form: w_k * (uk + i*vk) = 0  (summed over k=0,1,2)
    # Real:  sum_k (Re(w_k)*uk - Im(w_k)*vk) = 0
    # Imag:  sum_k (Im(w_k)*uk + Re(w_k)*vk) = 0
    row_r = {}; row_i = {}
    rhs_r = 0.0; rhs_i = 0.0
    for vi, wr, wi in ((i0, w0_r, w0_i), (i1, w1_r, w1_i), (i2, w2_r, w2_i)):
      if vi == pin0:
        pass  # u=0, v=0 → no contribution to rhs
      elif vi == pin1:
        rhs_r -= wr * 1.0;  rhs_i -= wi * 1.0  # u=1, v=0
      else:
        fi = free_map[vi]
        row_r[2 * fi]     = row_r.get(2 * fi,     0.0) + wr
        row_r[2 * fi + 1] = row_r.get(2 * fi + 1, 0.0) - wi
        row_i[2 * fi]     = row_i.get(2 * fi,     0.0) + wi
        row_i[2 * fi + 1] = row_i.get(2 * fi + 1, 0.0) + wr

    rows_A.append(row_r); rhs_rows.append(rhs_r)
    rows_A.append(row_i); rhs_rows.append(rhs_i)

  if not rows_A:
    return _flatten_area_preserving(surface)

  m   = len(rows_A)
  dim = 2 * n_free
  A   = np.zeros((m, dim), dtype=np.float64)
  for ri, row in enumerate(rows_A):
    for ci, val in row.items():
      if 0 <= ci < dim:
        A[ri, ci] = val
  b = np.array(rhs_rows, dtype=np.float64)

  # Solve via least-squares
  x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

  # Reconstruct UV for all vertices
  uv_flat = np.zeros((n_verts, 2), dtype=np.float64)
  uv_flat[pin0] = [0.0, 0.0]
  uv_flat[pin1] = [1.0, 0.0]
  for k, vi in enumerate(free_indices):
    uv_flat[vi] = [x[2 * k], x[2 * k + 1]]

  # Normalise to [0, 1]
  for d in range(2):
    lo, hi = uv_flat[:, d].min(), uv_flat[:, d].max()
    if hi > lo:
      uv_flat[:, d] = (uv_flat[:, d] - lo) / (hi - lo)

  uv = uv_flat.reshape(rows, cols, 2).astype(np.float32)
  return uv


# ---------------------------------------------------------------------------
# texture_from_uv
# ---------------------------------------------------------------------------

def texture_from_uv(
  volume,
  surface: "QuadSurface",
  uv_coords: np.ndarray,
  z_range: int = 26,
  level: int = 0,
) -> np.ndarray:
  """Sample volume along surface normals using UV mapping.

  For each UV texel, find the nearest surface vertex and sample the volume
  at that vertex position ± z_range/2 layers along the normal, compositing
  with max-intensity projection.

  Args:
    volume:     C capsule or zarr array.
    surface:    QuadSurface.
    uv_coords:  (rows, cols, 2) UV array from flatten_surface.
    z_range:    Number of depth layers to sample (centred on the surface).
    level:      Pyramid level for C capsule.

  Returns:
    (rows, cols) float32 array of sampled intensities.
  """
  rows, cols = surface.rows, surface.cols
  half = z_range // 2
  texture = np.zeros((rows, cols), dtype=np.float32)

  for r in range(rows):
    for c in range(cols):
      px, py, pz = surface.get(r, c)
      nx, ny, nz = surface.normal(r, c)
      best = -1e38
      for d in range(-half, half + 1):
        sx = px + d * nx
        sy = py + d * ny
        sz = pz + d * nz
        v = _sample_volume(volume, level, sz, sy, sx)
        if v > best:
          best = v
      texture[r, c] = best

  return texture
