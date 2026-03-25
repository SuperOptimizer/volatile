from __future__ import annotations

import numpy as np
from typing import Iterator

try:
  import volatile
  _VOLATILE_CORE = True
except ImportError:
  _VOLATILE_CORE = False

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False


class ChunkDataset:
  """
  Iterates over random 2-D slices (XY planes) drawn from a zarr volume.

  Each item is a numpy float32 array of shape (1, tile_h, tile_w), normalised
  to [0, 1] based on the volume dtype range.  Labels are not provided here;
  this class is intended for unsupervised/self-supervised pre-processing or
  as a base for subclassing with label logic.

  Args:
    path:      path to a local .zarr volume (passed to vol_open)
    level:     resolution level to read from (default 0 = full resolution)
    tile_h:    tile height in voxels
    tile_w:    tile width in voxels
    n_samples: number of random tiles to yield per epoch
    seed:      RNG seed for reproducibility (None = random)
  """

  # dtype max values for normalisation
  _DTYPE_MAX = {0: 255.0, 1: 65535.0, 2: 1.0, 3: 1.0}  # U8, U16, F32, F64

  def __init__(
    self,
    path: str,
    level: int = 0,
    tile_h: int = 256,
    tile_w: int = 256,
    n_samples: int = 100,
    seed: int | None = None,
  ):
    if not _VOLATILE_CORE:
      raise ImportError("volatile C extension not available")

    self._vol = volatile.vol_open(path)
    self._level = level
    self._tile_h = tile_h
    self._tile_w = tile_w
    self._n_samples = n_samples
    self._rng = np.random.default_rng(seed)

    shape = volatile.vol_shape(self._vol, level)
    # shape is (Z, Y, X) for 3-D volumes
    if len(shape) < 3:
      raise ValueError(f"volume at level {level} has ndim={len(shape)}, need ≥3")
    self._depth, self._height, self._width = shape[-3], shape[-2], shape[-1]

    meta = volatile.core.vol_level_meta(self._vol, level) if hasattr(volatile, "core") else None
    self._dtype_max = 255.0  # safe default; refine if metadata accessible

  def __len__(self) -> int:
    return self._n_samples

  def __iter__(self) -> Iterator[np.ndarray]:
    for _ in range(self._n_samples):
      yield self._random_tile()

  def _random_tile(self) -> np.ndarray:
    """Sample a random (tile_h × tile_w) XY patch from a random Z slice."""
    z = int(self._rng.integers(0, self._depth))
    y0 = int(self._rng.integers(0, max(1, self._height - self._tile_h)))
    x0 = int(self._rng.integers(0, max(1, self._width - self._tile_w)))

    tile = np.zeros((self._tile_h, self._tile_w), dtype=np.float32)
    for iy in range(self._tile_h):
      for ix in range(self._tile_w):
        vy = min(y0 + iy, self._height - 1)
        vx = min(x0 + ix, self._width - 1)
        tile[iy, ix] = volatile.vol_sample(self._vol, self._level, float(z), float(vy), float(vx))

    tile /= self._dtype_max  # normalise to [0, 1]
    return tile[np.newaxis]  # (1, H, W)

  def as_tensor(self, tile: np.ndarray) -> "Tensor":
    """Wrap a numpy tile as a tinygrad Tensor with batch dim: (1, 1, H, W)."""
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for as_tensor")
    return Tensor(tile[np.newaxis].astype(np.float32))

  def close(self) -> None:
    """Release the underlying C volume handle."""
    if _VOLATILE_CORE and self._vol is not None:
      volatile.vol_free(self._vol)
      self._vol = None

  def __del__(self) -> None:
    self.close()

  def __enter__(self) -> "ChunkDataset":
    return self

  def __exit__(self, *_) -> None:
    self.close()
