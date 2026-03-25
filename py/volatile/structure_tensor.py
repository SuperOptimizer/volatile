from __future__ import annotations
"""High-level structure tensor computation with chunked processing for large volumes."""

import math
import sys
from pathlib import Path
from typing import Iterable

try:
  import numpy as np
  _HAS_NUMPY = True
except ImportError:
  _HAS_NUMPY = False


def _chunk_ranges(total: int, chunk: int, overlap: int = 0) -> list[tuple[int, int, int, int]]:
  """Generate (read_start, read_end, write_start_offset, write_end_offset) tuples.

  Returns slices such that:
  - read range includes overlap on each side (clamped to [0, total))
  - write offset gives the valid (non-overlapping) region within the read slice
  """
  ranges = []
  pos = 0
  while pos < total:
    r_start = max(0, pos - overlap)
    r_end = min(total, pos + chunk + overlap)
    w_off_start = pos - r_start
    w_off_end = w_off_start + min(chunk, total - pos)
    ranges.append((r_start, r_end, w_off_start, w_off_end))
    pos += chunk
  return ranges


def compute_st(
  volume_path: str | Path,
  output_path: str | Path,
  deriv_sigma: float = 1.0,
  smooth_sigma: float = 3.0,
  chunk_size: int = 64,
  overlap: int = 0,
  verbose: bool = True,
) -> None:
  """Compute structure tensor over a large 3-D volume with chunked processing.

  Reads the input volume from *volume_path* (numpy .npy or any format readable
  via numpy.load / zarr.open), computes the 6-component structure tensor per
  voxel using the C core (or pure-numpy fallback), and writes the result to
  *output_path* as a numpy .npy file of shape (D, H, W, 6).

  The 6 components are stored in the order (Jzz, Jzy, Jzx, Jyy, Jyx, Jxx) to
  match volatile._core.structure_tensor_3d's output convention.

  Args:
    volume_path:  Path to the input volume (.npy, .npz, or zarr directory).
    output_path:  Path for the output .npy file.
    deriv_sigma:  Sigma for the derivative (pre-smoothing) kernel.
    smooth_sigma: Sigma for the tensor component smoothing kernel.
    chunk_size:   Depth of each processing chunk in voxels.
    overlap:      Extra voxels read on each side of a chunk for boundary accuracy.
    verbose:      Print progress to stderr.
  """
  if not _HAS_NUMPY:
    raise RuntimeError("numpy required for compute_st")

  volume_path = Path(volume_path)
  output_path = Path(output_path)

  # --- load volume ---
  vol = _load_volume(volume_path)
  if vol.ndim != 3:
    raise ValueError(f"expected 3-D volume, got shape {vol.shape}")
  vol = vol.astype(np.float32, copy=False)
  D, H, W = vol.shape

  if verbose:
    print(f"compute_st: volume {vol.shape}, deriv_sigma={deriv_sigma}, smooth_sigma={smooth_sigma}, "
          f"chunk={chunk_size}, overlap={overlap}", file=sys.stderr)

  out = np.zeros((D, H, W, 6), dtype=np.float32)

  # try C core first
  from volatile.imgproc import structure_tensor_3d as _st3d_py, _HAS_CORE

  ranges = _chunk_ranges(D, chunk_size, overlap)
  for i, (r_start, r_end, w_off_start, w_off_end) in enumerate(ranges):
    chunk_vol = vol[r_start:r_end]
    if verbose:
      pct = 100.0 * i / max(1, len(ranges) - 1)
      print(f"  chunk {i+1}/{len(ranges)} z=[{r_start},{r_end}) {pct:.0f}%", file=sys.stderr, end="\r")

    if _HAS_CORE:
      chunk_st = _st3d_py(chunk_vol, deriv_sigma, smooth_sigma)  # (cd, H, W, 6)
    else:
      chunk_st = _structure_tensor_numpy(chunk_vol, deriv_sigma, smooth_sigma)

    cd = r_end - r_start
    # Write only the non-overlapping region.
    src_z_start = w_off_start
    src_z_end = w_off_start + (w_off_end - w_off_start)
    dst_z_start = r_start + w_off_start
    dst_z_end = dst_z_start + (src_z_end - src_z_start)
    out[dst_z_start:dst_z_end] = chunk_st[src_z_start:src_z_end]

  if verbose:
    print(f"\ncompute_st: writing {output_path}", file=sys.stderr)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.save(str(output_path), out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_volume(path: Path) -> "np.ndarray":
  """Load a 3-D volume from .npy, .npz, or zarr."""
  suffix = path.suffix.lower()
  if suffix == ".npy":
    return np.load(str(path))
  if suffix == ".npz":
    data = np.load(str(path))
    key = list(data.keys())[0]
    return data[key]
  # Try zarr.
  try:
    import zarr
    store = zarr.open(str(path), mode="r")
    if hasattr(store, "shape"):
      return np.asarray(store)
    # group — try common dataset names
    for name in ("data", "volume", "0"):
      if name in store:
        return np.asarray(store[name])
    # fallback: first dataset
    key = next(iter(store))
    return np.asarray(store[key])
  except ImportError:
    pass
  raise ValueError(f"cannot load volume from {path}: unsupported format (install zarr for .zarr)")


def _gaussian_blur_1d_numpy(arr: "np.ndarray", sigma: float, axis: int) -> "np.ndarray":
  """Separable 1-D Gaussian blur along one axis using numpy."""
  radius = int(math.ceil(3.0 * sigma))
  x = np.arange(-radius, radius + 1, dtype=np.float32)
  kernel = np.exp(-x**2 / (2.0 * sigma * sigma))
  kernel /= kernel.sum()
  return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="same"), axis, arr)


def _gaussian_blur_numpy(arr: "np.ndarray", sigma: float) -> "np.ndarray":
  """3-D isotropic Gaussian blur via separable 1-D convolutions."""
  if sigma <= 0.0:
    return arr
  out = _gaussian_blur_1d_numpy(arr, sigma, 0)
  out = _gaussian_blur_1d_numpy(out, sigma, 1)
  return _gaussian_blur_1d_numpy(out, sigma, 2)


def _structure_tensor_numpy(vol: "np.ndarray", deriv_sigma: float, smooth_sigma: float) -> "np.ndarray":
  """Pure-numpy structure tensor fallback — Pavel Holoborodko-inspired but uses np.gradient."""
  smooth = _gaussian_blur_numpy(vol, deriv_sigma)
  gz = np.gradient(smooth, axis=0).astype(np.float32)
  gy = np.gradient(smooth, axis=1).astype(np.float32)
  gx = np.gradient(smooth, axis=2).astype(np.float32)

  Jzz = gz * gz; Jzy = gz * gy; Jzx = gz * gx
  Jyy = gy * gy; Jyx = gy * gx; Jxx = gx * gx

  components = [Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]
  if smooth_sigma > 0.0:
    components = [_gaussian_blur_numpy(c, smooth_sigma) for c in components]

  out = np.stack(components, axis=-1)  # (D, H, W, 6)
  return out
