"""metrics.py — per-volume and per-segment statistics.

All functions degrade gracefully when numpy is unavailable (pure Python path).
"""
from __future__ import annotations
import math
import struct
import os
from typing import Sequence

try:
  import numpy as np
  _NP = True
except ImportError:
  _NP = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _percentile_pure(data: list[float], p: float) -> float:
  """Compute percentile p in [0,100] from a sorted or unsorted list."""
  if not data:
    return float("nan")
  s = sorted(data)
  n = len(s)
  idx = p / 100.0 * (n - 1)
  lo = int(idx)
  hi = min(lo + 1, n - 1)
  frac = idx - lo
  return s[lo] * (1.0 - frac) + s[hi] * frac


def _read_zarr_chunk(path: str) -> list[float]:
  """Very minimal fallback: read raw float32 bytes from a single chunk file."""
  with open(path, "rb") as f:
    raw = f.read()
  n = len(raw) // 4
  return list(struct.unpack(f"{n}f", raw[:n * 4]))


# ---------------------------------------------------------------------------
# volume_stats
# ---------------------------------------------------------------------------

def volume_stats(vol_path: str, level: int = 0) -> dict:
  """Compute min, max, mean, std, and percentiles [1, 5, 25, 50, 75, 95, 99]
  for a volume at a given multiscale level.

  Tries to use the volatile._core C extension first; falls back to loading
  numpy zarr (zarr library) or scanning raw chunk files.

  Returns a dict with keys: min, max, mean, std, p1, p5, p25, p50, p75, p95, p99,
                             n_voxels, path, level.
  """
  data = None

  # --- attempt 1: C extension ---
  try:
    from volatile import vol_open, vol_free, vol_num_levels, vol_shape, vol_sample
    vol = vol_open(vol_path)
    shape = vol_shape(vol, level)
    Z, Y, X = shape[0], shape[1], shape[2]
    # For large volumes this is slow; only feasible for small ones via the C API.
    # In practice callers should use zarr/numpy for bulk stats.
    samples = []
    step = max(1, max(Z, Y, X) // 64)  # subsample to ≤64^3 for speed
    for z in range(0, Z, step):
      for y in range(0, Y, step):
        for x in range(0, X, step):
          samples.append(float(vol_sample(vol, level, float(z)/Z, float(y)/Y, float(x)/X)))
    vol_free(vol)
    data = samples
  except Exception:
    pass

  # --- attempt 2: zarr library ---
  if data is None:
    try:
      import zarr  # type: ignore
      store = zarr.open(vol_path, mode="r")
      # multiscale: level 0 is typically the root array or under "0", "1", …
      arr = store[str(level)] if str(level) in store else store
      if _NP:
        flat = np.asarray(arr).ravel().astype(float)
        data = flat
      else:
        data = [float(v) for v in arr.flat]
    except Exception:
      pass

  # --- attempt 3: scan raw chunk files ---
  if data is None:
    chunk_dir = os.path.join(vol_path, str(level)) if os.path.isdir(vol_path) else vol_path
    raw: list[float] = []
    if os.path.isdir(chunk_dir):
      for fname in os.listdir(chunk_dir):
        fpath = os.path.join(chunk_dir, fname)
        if os.path.isfile(fpath):
          try:
            raw.extend(_read_zarr_chunk(fpath))
          except Exception:
            pass
    if raw:
      data = raw

  if data is None:
    raise FileNotFoundError(f"Could not load volume data from {vol_path!r} level {level}")

  # --- compute stats ---
  if _NP and isinstance(data, np.ndarray):
    n = int(data.size)
    mn  = float(data.min())
    mx  = float(data.max())
    mu  = float(data.mean())
    std = float(data.std())
    pcts = {f"p{p}": float(np.percentile(data, p)) for p in (1, 5, 25, 50, 75, 95, 99)}
  else:
    lst = list(data)
    n   = len(lst)
    mn  = min(lst)
    mx  = max(lst)
    mu  = sum(lst) / n if n else float("nan")
    std = math.sqrt(sum((v - mu) ** 2 for v in lst) / n) if n else float("nan")
    pcts = {f"p{p}": _percentile_pure(lst, p) for p in (1, 5, 25, 50, 75, 95, 99)}

  return {"min": mn, "max": mx, "mean": mu, "std": std, "n_voxels": n,
          "path": vol_path, "level": level, **pcts}


# ---------------------------------------------------------------------------
# segment_coverage
# ---------------------------------------------------------------------------

def segment_coverage(surface, volume) -> float:
  """Fraction of surface vertices that fall within the valid (non-NaN, non-inf,
  non-zero-mask) region of a volume.

  surface: iterable of (z, y, x) world-coordinate vertices, OR a
           volatile.seg.QuadSurface-like object with `.points` attribute.
  volume:  3-D list/numpy array of float values, OR a dict with key "data"
           and optional "mask" (bool array, True=valid).

  Returns a float in [0, 1].  Returns 1.0 if the surface has no vertices.
  """
  # Unpack surface points.
  if hasattr(surface, "points"):
    pts = [(p[0], p[1], p[2]) if hasattr(p, "__iter__") else p for p in surface.points]
  else:
    pts = list(surface)

  if not pts:
    return 1.0

  # Unpack volume + mask.
  if isinstance(volume, dict):
    vol_data = volume["data"]
    mask = volume.get("mask", None)
  else:
    vol_data = volume
    mask = None

  if _NP and isinstance(vol_data, np.ndarray):
    shape = vol_data.shape
    def valid(z, y, x):
      iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
      if not (0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]):
        return False
      v = vol_data[iz, iy, ix]
      if mask is not None and not mask[iz, iy, ix]:
        return False
      return math.isfinite(float(v)) and float(v) != 0.0
  else:
    # Assume vol_data is a flat list with shape attribute or nested lists.
    shape = getattr(vol_data, "shape", None)
    if shape is None:
      raise ValueError("volume must be a numpy array or have a .shape attribute")
    Y, X = shape[1], shape[2]
    def valid(z, y, x):
      iz, iy, ix = int(round(z)), int(round(y)), int(round(x))
      if not (0 <= iz < shape[0] and 0 <= iy < Y and 0 <= ix < X):
        return False
      v = vol_data[iz * Y * X + iy * X + ix]
      if mask is not None and not mask[iz * Y * X + iy * X + ix]:
        return False
      return math.isfinite(float(v)) and float(v) != 0.0

  covered = sum(1 for z, y, x in pts if valid(z, y, x))
  return covered / len(pts)


# ---------------------------------------------------------------------------
# surface_metrics
# ---------------------------------------------------------------------------

def surface_metrics(surface) -> dict:
  """Compute area, average mean curvature, and smoothness for a quad surface.

  surface: a QuadSurface-like object with .rows, .cols, and callable .get(r,c)
           returning (x, y, z); OR a dict with keys "rows", "cols", "points"
           (flat list of (x,y,z) tuples in row-major order).

  Returns dict: area (float), mean_curvature (float), smoothness (float, lower=smoother).
  """
  # Normalise input.
  if isinstance(surface, dict):
    rows = surface["rows"]
    cols = surface["cols"]
    pts  = surface["points"]  # flat list of (x,y,z)
    def get(r, c): return pts[r * cols + c]
  else:
    rows = surface.rows
    cols = surface.cols
    def get(r, c):
      p = surface.get(r, c)
      return (p[0], p[1], p[2]) if hasattr(p, "__iter__") else (p.x, p.y, p.z)

  def sub(a, b): return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
  def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
  def norm(v): return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
  def dot(a, b): return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

  # --- area: sum of quad areas (each quad split into 2 triangles) ---
  total_area = 0.0
  for r in range(rows - 1):
    for c in range(cols - 1):
      p00, p10 = get(r, c),     get(r+1, c)
      p01, p11 = get(r, c+1),   get(r+1, c+1)
      # triangle 1: p00, p10, p01
      total_area += norm(cross(sub(p10, p00), sub(p01, p00))) * 0.5
      # triangle 2: p10, p11, p01
      total_area += norm(cross(sub(p11, p10), sub(p01, p10))) * 0.5

  # --- mean curvature (discrete): for each interior vertex, compute
  #     angle-weighted normal deviation from neighbours (umbrella operator) ---
  curvatures: list[float] = []
  for r in range(1, rows - 1):
    for c in range(1, cols - 1):
      p = get(r, c)
      nbrs = [get(r-1,c), get(r+1,c), get(r,c-1), get(r,c+1)]
      # Laplacian of position approximates 2*H*N; |L| / area_element ~ mean curvature
      lx = sum(n[0] for n in nbrs) / 4 - p[0]
      ly = sum(n[1] for n in nbrs) / 4 - p[1]
      lz = sum(n[2] for n in nbrs) / 4 - p[2]
      curvatures.append(math.sqrt(lx**2 + ly**2 + lz**2))

  mean_curvature = sum(curvatures) / len(curvatures) if curvatures else 0.0

  # --- smoothness: RMS of second-order differences (lower = smoother) ---
  second_diffs: list[float] = []
  for r in range(1, rows - 1):
    for c in range(1, cols - 1):
      p = get(r, c)
      # row direction
      pr = get(r, c+1); pl = get(r, c-1)
      d2x = (pr[0] - 2*p[0] + pl[0])**2 + (pr[1] - 2*p[1] + pl[1])**2 + (pr[2] - 2*p[2] + pl[2])**2
      # col direction
      pu = get(r-1, c); pd = get(r+1, c)
      d2y = (pu[0] - 2*p[0] + pd[0])**2 + (pu[1] - 2*p[1] + pd[1])**2 + (pu[2] - 2*p[2] + pd[2])**2
      second_diffs.append(d2x + d2y)

  smoothness = math.sqrt(sum(second_diffs) / len(second_diffs)) if second_diffs else 0.0

  return {"area": total_area, "mean_curvature": mean_curvature, "smoothness": smoothness,
          "rows": rows, "cols": cols}
