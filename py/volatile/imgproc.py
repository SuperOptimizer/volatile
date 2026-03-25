from __future__ import annotations
import struct

try:
  import numpy as np
  _HAS_NUMPY = True
except ImportError:
  _HAS_NUMPY = False

try:
  from volatile._core import (
    gaussian_blur_2d as _c_gaussian_blur_2d,
    structure_tensor_3d as _c_structure_tensor_3d,
    histogram as _c_histogram,
    window_level as _c_window_level,
  )
  _HAS_CORE = True
except ImportError:
  _HAS_CORE = False


def _to_float32_bytes(data) -> bytes:
  """Convert ndarray or bytes-like to raw float32 bytes."""
  if _HAS_NUMPY and isinstance(data, np.ndarray):
    return data.astype(np.float32, copy=False).tobytes()
  if isinstance(data, (bytes, bytearray, memoryview)):
    return bytes(data)
  raise TypeError(f"expected ndarray or bytes-like, got {type(data)}")


def gaussian_blur_2d(data, sigma: float, *, height: int = 0, width: int = 0):
  """Gaussian blur a 2-D float array.

  Args:
    data:   2-D numpy float32 array, or raw float32 bytes with explicit height/width.
    sigma:  Gaussian sigma in pixels.
    height: Required when data is bytes.
    width:  Required when data is bytes.

  Returns:
    numpy float32 array of same shape, or bytes if numpy unavailable.
  """
  if not _HAS_CORE:
    raise RuntimeError("volatile._core C extension not available")

  if _HAS_NUMPY and isinstance(data, np.ndarray):
    if data.ndim != 2:
      raise ValueError(f"expected 2-D array, got {data.ndim}-D")
    h, w = data.shape
    raw = _to_float32_bytes(data)
  else:
    if height == 0 or width == 0:
      raise ValueError("height and width required when data is bytes")
    h, w = height, width
    raw = _to_float32_bytes(data)

  result_bytes = _c_gaussian_blur_2d(raw, h, w, float(sigma))

  if _HAS_NUMPY:
    return np.frombuffer(result_bytes, dtype=np.float32).reshape(h, w).copy()
  return result_bytes


def structure_tensor_3d(data, deriv_sigma: float, smooth_sigma: float, *,
                        depth: int = 0, height: int = 0, width: int = 0):
  """Compute 3-D structure tensor.

  Args:
    data:         3-D numpy float32 array, or raw float32 bytes with explicit shape.
    deriv_sigma:  Sigma for derivative kernel.
    smooth_sigma: Sigma for smoothing kernel.
    depth/height/width: Required when data is bytes.

  Returns:
    numpy float32 array of shape (depth, height, width, 6), or bytes if numpy unavailable.
  """
  if not _HAS_CORE:
    raise RuntimeError("volatile._core C extension not available")

  if _HAS_NUMPY and isinstance(data, np.ndarray):
    if data.ndim != 3:
      raise ValueError(f"expected 3-D array, got {data.ndim}-D")
    d, h, w = data.shape
    raw = _to_float32_bytes(data)
  else:
    if depth == 0 or height == 0 or width == 0:
      raise ValueError("depth, height and width required when data is bytes")
    d, h, w = depth, height, width
    raw = _to_float32_bytes(data)

  result_bytes = _c_structure_tensor_3d(raw, d, h, w, float(deriv_sigma), float(smooth_sigma))

  if _HAS_NUMPY:
    return np.frombuffer(result_bytes, dtype=np.float32).reshape(d, h, w, 6).copy()
  return result_bytes


def histogram(data, num_bins: int, *, num_elements: int = 0) -> dict:
  """Compute histogram of float32 data.

  Args:
    data:         1-D numpy float32 array, or raw float32 bytes with explicit num_elements.
    num_bins:     Number of histogram bins.
    num_elements: Required when data is bytes.

  Returns:
    dict with keys: bins (list[int]), min (float), max (float), mean (float).
  """
  if not _HAS_CORE:
    raise RuntimeError("volatile._core C extension not available")

  if _HAS_NUMPY and isinstance(data, np.ndarray):
    n = data.size
    raw = _to_float32_bytes(data)
  else:
    if num_elements == 0:
      raise ValueError("num_elements required when data is bytes")
    n = num_elements
    raw = _to_float32_bytes(data)

  return _c_histogram(raw, n, num_bins)


def window_level(data, window: float, level: float, *, num_elements: int = 0):
  """Apply window/level contrast mapping (float32 -> uint8).

  Args:
    data:         numpy float32 array, or raw float32 bytes with explicit num_elements.
    window:       Window width.
    level:        Window center (level).
    num_elements: Required when data is bytes.

  Returns:
    numpy uint8 array of same shape, or bytes if numpy unavailable.
  """
  if not _HAS_CORE:
    raise RuntimeError("volatile._core C extension not available")

  if _HAS_NUMPY and isinstance(data, np.ndarray):
    shape = data.shape
    n = data.size
    raw = _to_float32_bytes(data)
  else:
    if num_elements == 0:
      raise ValueError("num_elements required when data is bytes")
    shape = None
    n = num_elements
    raw = _to_float32_bytes(data)

  result_bytes = _c_window_level(raw, n, float(window), float(level))

  if _HAS_NUMPY:
    arr = np.frombuffer(result_bytes, dtype=np.uint8).copy()
    if shape is not None:
      arr = arr.reshape(shape)
    return arr
  return result_bytes


# ---------------------------------------------------------------------------
# Frangi vesselness filter (3-D)
# ---------------------------------------------------------------------------

def frangi_vesselness_3d(volume, sigmas=(1.0,), alpha: float = 0.5, beta: float = 0.5, gamma: float = 15.0):
  """Frangi multi-scale vesselness filter for 3-D volumes.

  Computes Hessian eigenvalues at each scale, applies the Frangi response
  function, and returns the maximum response across scales.

  The Hessian is computed from the structure tensor Jxx/Jxy/... components by
  using scipy.ndimage.gaussian_laplace for second derivatives, falling back to
  finite differences when scipy is unavailable.

  Args:
    volume: 3-D numpy float32 array (D, H, W).
    sigmas: iterable of Gaussian sigmas defining the scale range.
    alpha:  plate-vs-blob suppression weight (Ra term).
    beta:   blob-vs-plate suppression weight (Rb term).
    gamma:  background noise threshold (S term); if 0 uses automatic (half max S).

  Returns:
    numpy float32 array of shape (D, H, W) with vesselness in [0, 1].
  """
  if not _HAS_NUMPY:
    raise RuntimeError("numpy required for frangi_vesselness_3d")
  vol = np.asarray(volume, dtype=np.float32)
  if vol.ndim != 3:
    raise ValueError(f"expected 3-D volume, got {vol.ndim}-D")

  try:
    from scipy.ndimage import gaussian_filter
    _has_scipy = True
  except ImportError:
    _has_scipy = False

  best = np.zeros(vol.shape, dtype=np.float32)

  for sigma in sigmas:
    sigma = float(sigma)
    # Compute second-order partial derivatives via Gaussian smoothing + finite differences.
    if _has_scipy:
      smooth = gaussian_filter(vol, sigma=sigma)
    else:
      # pure-numpy fallback: very rough approximation (no proper sigma scaling)
      smooth = vol

    # Second derivatives using np.gradient twice.
    def _d2(arr, ax1, ax2):
      return np.gradient(np.gradient(arr, axis=ax1), axis=ax2)

    Hzz = _d2(smooth, 0, 0)
    Hzy = _d2(smooth, 0, 1)
    Hzx = _d2(smooth, 0, 2)
    Hyy = _d2(smooth, 1, 1)
    Hyx = _d2(smooth, 1, 2)
    Hxx = _d2(smooth, 2, 2)

    # Scale normalisation: multiply by sigma^2.
    scale = sigma * sigma
    Hzz *= scale; Hzy *= scale; Hzx *= scale
    Hyy *= scale; Hyx *= scale; Hxx *= scale

    # Assemble per-voxel 3×3 symmetric matrix and compute eigenvalues.
    d, h, w = vol.shape
    n = d * h * w
    mats = np.empty((n, 3, 3), dtype=np.float32)
    mats[:, 0, 0] = Hxx.ravel(); mats[:, 0, 1] = Hyx.ravel(); mats[:, 0, 2] = Hzx.ravel()
    mats[:, 1, 0] = Hyx.ravel(); mats[:, 1, 1] = Hyy.ravel(); mats[:, 1, 2] = Hzy.ravel()
    mats[:, 2, 0] = Hzx.ravel(); mats[:, 2, 1] = Hzy.ravel(); mats[:, 2, 2] = Hzz.ravel()

    eigvals = np.linalg.eigvalsh(mats)  # (n, 3), ascending order

    # Sort by absolute value so |λ1| ≤ |λ2| ≤ |λ3|.
    idx = np.argsort(np.abs(eigvals), axis=1)
    eigvals = np.take_along_axis(eigvals, idx, axis=1)

    L1 = np.abs(eigvals[:, 0])
    L2 = np.abs(eigvals[:, 1])
    L3 = eigvals[:, 2]      # signed — tubular structures have L3 < 0
    L3abs = np.abs(L3)

    S = np.sqrt(Hxx.ravel()**2 + Hyy.ravel()**2 + Hzz.ravel()**2 +
                2.0*(Hyx.ravel()**2 + Hzx.ravel()**2 + Hzy.ravel()**2))
    gamma_eff = float(gamma) if float(gamma) > 0.0 else (0.5 * float(S.max()) + 1e-10)

    eps = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
      Ra = np.where(L3abs > eps, L2 / (L3abs + eps), 0.0)
      Rb = np.where(L2 * L3abs > eps, L1 / np.sqrt(L2 * L3abs + eps), 0.0)

    vesselness = (
      (1.0 - np.exp(-0.5 * (Ra / alpha) ** 2))
      * np.exp(-0.5 * (Rb / beta) ** 2)
      * (1.0 - np.exp(-0.5 * (S / gamma_eff) ** 2))
    ).astype(np.float32)
    vesselness[L3 > 0] = 0.0  # suppress bright-on-dark structures

    best = np.maximum(best, vesselness.reshape(d, h, w))

  return best


# ---------------------------------------------------------------------------
# Euclidean distance transform (3-D)
# ---------------------------------------------------------------------------

def edt_3d(mask):
  """Euclidean distance transform of a 3-D binary mask.

  Returns the distance from each voxel to the nearest non-zero voxel in
  *mask* (i.e. the EDT of the background relative to the foreground).

  Args:
    mask: 3-D numpy array.  Non-zero voxels are treated as foreground.

  Returns:
    numpy float32 array of same shape.
  """
  if not _HAS_NUMPY:
    raise RuntimeError("numpy required for edt_3d")
  arr = np.asarray(mask)
  if arr.ndim != 3:
    raise ValueError(f"expected 3-D mask, got {arr.ndim}-D")

  try:
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(arr == 0).astype(np.float32)
  except ImportError:
    pass

  # Pure-numpy fallback: approximate via brute-force (slow, only for tiny arrays).
  fg = np.argwhere(arr != 0).astype(np.float32)
  if fg.size == 0:
    return np.full(arr.shape, float(max(arr.shape)), dtype=np.float32)
  coords = np.indices(arr.shape).reshape(3, -1).T.astype(np.float32)  # (N, 3)
  # Compute min distance for each voxel.
  out = np.empty(coords.shape[0], dtype=np.float32)
  chunk = 4096
  for i in range(0, coords.shape[0], chunk):
    diff = coords[i:i+chunk, None, :] - fg[None, :, :]  # (c, F, 3)
    out[i:i+chunk] = np.sqrt((diff**2).sum(axis=2).min(axis=1))
  return out.reshape(arr.shape)


# ---------------------------------------------------------------------------
# Eigendecomposition of per-voxel 3×3 symmetric structure tensor
# ---------------------------------------------------------------------------

def eigendecomp_3d(st_tensor):
  """Eigenvalues and eigenvectors of a per-voxel 3×3 symmetric structure tensor.

  Args:
    st_tensor: numpy array of shape (D, H, W, 6) — the 6 unique components of
               the symmetric 3×3 matrix stored as (Jzz, Jzy, Jzx, Jyy, Jyx, Jxx)
               matching the layout produced by structure_tensor_3d().

  Returns:
    eigenvalues:  float32 array of shape (D, H, W, 3), ascending order.
    eigenvectors: float32 array of shape (D, H, W, 3, 3), columns are eigenvectors.
  """
  if not _HAS_NUMPY:
    raise RuntimeError("numpy required for eigendecomp_3d")
  arr = np.asarray(st_tensor, dtype=np.float32)
  if arr.ndim != 4 or arr.shape[3] != 6:
    raise ValueError(f"expected shape (D, H, W, 6), got {arr.shape}")
  d, h, w, _ = arr.shape
  n = d * h * w

  # Unpack components: layout is (Jzz, Jzy, Jzx, Jyy, Jyx, Jxx).
  Jzz = arr[..., 0].ravel(); Jzy = arr[..., 1].ravel(); Jzx = arr[..., 2].ravel()
  Jyy = arr[..., 3].ravel(); Jyx = arr[..., 4].ravel(); Jxx = arr[..., 5].ravel()

  # Assemble symmetric matrices: ordered as (x,y,z) for standard orientation.
  mats = np.empty((n, 3, 3), dtype=np.float32)
  mats[:, 0, 0] = Jxx; mats[:, 0, 1] = Jyx; mats[:, 0, 2] = Jzx
  mats[:, 1, 0] = Jyx; mats[:, 1, 1] = Jyy; mats[:, 1, 2] = Jzy
  mats[:, 2, 0] = Jzx; mats[:, 2, 1] = Jzy; mats[:, 2, 2] = Jzz

  vals, vecs = np.linalg.eigh(mats)  # ascending eigenvalues, columns = eigenvectors
  return vals.reshape(d, h, w, 3), vecs.reshape(d, h, w, 3, 3)
