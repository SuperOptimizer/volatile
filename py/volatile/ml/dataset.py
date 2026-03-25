from __future__ import annotations

"""
Sliding-window 3-D patch dataset for OME-Zarr volumes.

Loads image + multi-task label volumes, enumerates non-empty patches with a
configurable stride, and yields (image, targets) numpy pairs suitable for the
Trainer / DataLoader pattern.

The volatile C extension is used for data I/O when available; a pure-numpy
fallback (numpy arrays passed directly) is supported for tests and environments
without the C extension.
"""

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
  import volatile as _vol_ext
  _VOLATILE_CORE = True
except ImportError:
  _VOLATILE_CORE = False

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatchSpec:
  """Coordinates of one extractable patch inside a volume array."""
  vol_idx: int         # index into Dataset._volumes
  position: Tuple[int, ...]  # (z, y, x) or (y, x) top-left corner
  patch_size: Tuple[int, ...]


@dataclass
class _VolumeEntry:
  """One image + label set loaded into memory (or backed by a zarr array)."""
  name: str
  image: np.ndarray          # (Z, Y, X) or (Y, X) float32, already normalised
  labels: Dict[str, np.ndarray]  # target_name → (Z, Y, X) or (Y, X) int32/float32
  spatial_shape: Tuple[int, ...]  # (Z, Y, X) or (Y, X)


# ---------------------------------------------------------------------------
# Patch utilities (no external deps)
# ---------------------------------------------------------------------------

def _iter_positions_3d(
  spatial_shape: Tuple[int, int, int],
  patch_size: Tuple[int, int, int],
  stride: Tuple[int, int, int],
) -> List[Tuple[int, int, int]]:
  D, H, W = spatial_shape
  pd, ph, pw = patch_size
  sd, sh, sw = stride
  zs = list(range(0, max(1, D - pd + 1), sd))
  ys = list(range(0, max(1, H - ph + 1), sh))
  xs = list(range(0, max(1, W - pw + 1), sw))
  return [(z, y, x) for z in zs for y in ys for x in xs]


def _iter_positions_2d(
  spatial_shape: Tuple[int, int],
  patch_size: Tuple[int, int],
  stride: Tuple[int, int],
) -> List[Tuple[int, int]]:
  H, W = spatial_shape
  ph, pw = patch_size
  sh, sw = stride
  ys = list(range(0, max(1, H - ph + 1), sh))
  xs = list(range(0, max(1, W - pw + 1), sw))
  return [(y, x) for y in ys for x in xs]


def _extract_patch(arr: np.ndarray, position: Tuple[int, ...], patch_size: Tuple[int, ...]) -> np.ndarray:
  """Extract a patch from `arr` at `position`, padding with zeros if the volume boundary is hit."""
  slices = tuple(slice(p, p + s) for p, s in zip(position, patch_size))
  raw = arr[slices]
  if raw.shape == patch_size:
    return raw.copy()
  # Pad to exact patch_size
  out = np.zeros(patch_size, dtype=arr.dtype)
  dst = tuple(slice(0, s) for s in raw.shape)
  out[dst] = raw
  return out


def _patch_is_empty(patch: np.ndarray, empty_threshold: float = 0.0) -> bool:
  """Return True when the patch max value is at or below `empty_threshold`."""
  return float(patch.max()) <= empty_threshold


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PatchDataset:
  """
  Sliding-window patch dataset over one or more OME-Zarr (or numpy) volumes.

  Each item is a dict::

    {
      "image":  np.ndarray  (1, *patch_size) float32, normalised to [0, 1]
      "target_name": np.ndarray  (*patch_size) int32 or float32  (one per target)
      "padding_mask": np.ndarray  (1, *patch_size) float32  — 1 = real, 0 = padded
      "volume": str,
      "position": tuple[int, ...],
    }

  When no label volume is supplied for a target the corresponding array is all
  zeros (useful for inference-time datasets).

  Args:
    volumes:         list of dicts, each with keys:
                       "name"  : str (volume identifier)
                       "image" : np.ndarray (Z,Y,X) or (Y,X) float32
                       "labels": dict[str, np.ndarray] — may be empty
    patch_size:      (D, H, W) for 3-D or (H, W) for 2-D
    stride:          patch stride; defaults to patch_size (non-overlapping)
    target_names:    which label keys to include in each item; defaults to all
    skip_empty:      skip patches where image max ≤ empty_threshold (default True)
    empty_threshold: max-value cutoff for "empty" detection (default 0.0)
    z_partitions:    if > 1, only load every Nth z-slice (for Z-distributed training)
    z_partition_idx: which partition this worker owns (0-based)
    augment_fn:      optional callable(image, mask) → (image, mask); applied per patch
    seed:            RNG seed for any future stochastic operations
  """

  def __init__(
    self,
    volumes: List[Dict[str, Any]],
    patch_size: Tuple[int, ...],
    stride: Optional[Tuple[int, ...]] = None,
    target_names: Optional[List[str]] = None,
    skip_empty: bool = True,
    empty_threshold: float = 0.0,
    z_partitions: int = 1,
    z_partition_idx: int = 0,
    augment_fn=None,
    seed: Optional[int] = None,
  ):
    if not volumes:
      raise ValueError("volumes list is empty")

    self._patch_size = tuple(patch_size)
    self._ndim = len(self._patch_size)
    if self._ndim not in (2, 3):
      raise ValueError(f"patch_size must be 2-D or 3-D, got ndim={self._ndim}")

    self._stride = tuple(stride) if stride is not None else self._patch_size
    if len(self._stride) != self._ndim:
      raise ValueError(f"stride ndim {len(self._stride)} != patch_size ndim {self._ndim}")

    self._skip_empty = skip_empty
    self._empty_threshold = empty_threshold
    self._z_partitions = max(1, z_partitions)
    self._z_partition_idx = z_partition_idx % max(1, z_partitions)
    self._augment_fn = augment_fn
    self._rng = np.random.default_rng(seed)

    # Build internal volume list
    self._volumes: List[_VolumeEntry] = []
    all_target_names: set[str] = set()
    for v in volumes:
      labels = {k: np.asarray(arr) for k, arr in v.get("labels", {}).items()}
      all_target_names.update(labels.keys())
      img = np.asarray(v["image"], dtype=np.float32)
      if img.max() > 1.0 + 1e-6:
        img = img / float(img.max())  # normalise to [0, 1] if not already
      spatial = img.shape[-self._ndim:]
      self._volumes.append(_VolumeEntry(
        name=v["name"],
        image=img,
        labels=labels,
        spatial_shape=spatial,
      ))

    self._target_names: List[str] = list(target_names) if target_names is not None else sorted(all_target_names)

    # Build patch index
    self._patches: List[PatchSpec] = []
    self._build_patch_index()
    logger.debug("PatchDataset: %d volumes, %d patches", len(self._volumes), len(self._patches))

  # ------------------------------------------------------------------
  # Index building
  # ------------------------------------------------------------------

  def _build_patch_index(self) -> None:
    for vol_idx, vol in enumerate(self._volumes):
      positions = self._positions_for_volume(vol)
      for pos in positions:
        if self._skip_empty:
          patch_img = _extract_patch(vol.image, pos, self._patch_size)
          if _patch_is_empty(patch_img, self._empty_threshold):
            continue
        self._patches.append(PatchSpec(vol_idx=vol_idx, position=pos, patch_size=self._patch_size))

  def _positions_for_volume(self, vol: _VolumeEntry) -> List[Tuple[int, ...]]:
    shape = vol.spatial_shape
    if self._ndim == 3:
      positions = _iter_positions_3d(
        (shape[0], shape[1], shape[2]),
        (self._patch_size[0], self._patch_size[1], self._patch_size[2]),
        (self._stride[0], self._stride[1], self._stride[2]),
      )
      # Apply Z partitioning: keep only positions where z-index mod N == partition_idx
      if self._z_partitions > 1:
        positions = [p for p in positions if (p[0] // self._patch_size[0]) % self._z_partitions == self._z_partition_idx]
    else:
      positions = _iter_positions_2d(
        (shape[0], shape[1]),
        (self._patch_size[0], self._patch_size[1]),
        (self._stride[0], self._stride[1]),
      )
    return positions

  # ------------------------------------------------------------------
  # Dataset interface
  # ------------------------------------------------------------------

  def __len__(self) -> int:
    return len(self._patches)

  def __getitem__(self, idx: int) -> Dict[str, Any]:
    spec = self._patches[idx]
    vol = self._volumes[spec.vol_idx]

    img_patch = _extract_patch(vol.image, spec.position, self._patch_size).astype(np.float32)
    original_shape = img_patch.shape  # before any padding (already correct since _extract_patch pads)

    # Padding mask: 1 where data was real, 0 where we zero-padded
    raw_slices = tuple(slice(p, min(p + s, vol.spatial_shape[i])) for i, (p, s) in enumerate(zip(spec.position, self._patch_size)))
    raw_shape = tuple(sl.stop - sl.start for sl in raw_slices)
    padding_mask = np.zeros(self._patch_size, dtype=np.float32)
    dst = tuple(slice(0, s) for s in raw_shape)
    padding_mask[dst] = 1.0

    # Label patches
    label_patches: Dict[str, np.ndarray] = {}
    for tname in self._target_names:
      if tname in vol.labels:
        label_arr = vol.labels[tname]
        lp = _extract_patch(label_arr, spec.position, self._patch_size)
      else:
        lp = np.zeros(self._patch_size, dtype=np.int32)
      label_patches[tname] = lp

    # Augmentation (applied per-patch; only image + first label for simplicity)
    if self._augment_fn is not None:
      # Wrap as (C, *spatial) for augment API
      img_chw = img_patch[np.newaxis]  # (1, *spatial)
      first_key = self._target_names[0] if self._target_names else None
      aug_mask = label_patches[first_key] if first_key else None
      img_chw, aug_mask = self._augment_fn(img_chw, aug_mask)
      img_patch = img_chw[0]
      if first_key is not None and aug_mask is not None:
        label_patches[first_key] = aug_mask

    item: Dict[str, Any] = {
      "image": img_patch[np.newaxis].astype(np.float32),    # (1, *patch_size)
      "padding_mask": padding_mask[np.newaxis].astype(np.float32),  # (1, *patch_size)
      "volume": vol.name,
      "position": spec.position,
    }
    item.update(label_patches)
    return item

  def __iter__(self) -> Iterator[Dict[str, Any]]:
    for i in range(len(self)):
      yield self[i]

  # ------------------------------------------------------------------
  # Batch collation helper
  # ------------------------------------------------------------------

  @staticmethod
  def collate(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of dataset items into a batch dict with numpy arrays."""
    if not items:
      return {}
    batch: Dict[str, Any] = {}
    for key in items[0]:
      if isinstance(items[0][key], np.ndarray):
        batch[key] = np.stack([it[key] for it in items], axis=0)
      else:
        batch[key] = [it[key] for it in items]
    return batch

  # ------------------------------------------------------------------
  # Properties
  # ------------------------------------------------------------------

  @property
  def patch_size(self) -> Tuple[int, ...]:
    return self._patch_size

  @property
  def target_names(self) -> List[str]:
    return list(self._target_names)

  @property
  def n_volumes(self) -> int:
    return len(self._volumes)


# ---------------------------------------------------------------------------
# VolExtDataset — wraps the volatile C extension vol_open / vol_sample API
# ---------------------------------------------------------------------------

class VolExtDataset(PatchDataset):
  """
  PatchDataset that reads image data from the volatile C extension.

  Loads a single OME-Zarr volume via `volatile.vol_open`; label arrays must
  be supplied as pre-loaded numpy arrays (they are typically small compared to
  the image volume).

  Args:
    zarr_path:    path to the .zarr volume (passed to volatile.vol_open)
    level:        resolution level (default 0 = full resolution)
    label_arrays: dict of target_name → numpy label array (same spatial shape as image)
    patch_size:   (D, H, W) or (H, W)
    stride:       patch stride (defaults to patch_size)
    target_names: which label keys to expose
    skip_empty:   skip patches where image max == 0
    seed:         RNG seed
  """

  def __init__(
    self,
    zarr_path: str,
    level: int = 0,
    label_arrays: Optional[Dict[str, np.ndarray]] = None,
    patch_size: Tuple[int, ...] = (64, 64, 64),
    stride: Optional[Tuple[int, ...]] = None,
    target_names: Optional[List[str]] = None,
    skip_empty: bool = True,
    seed: Optional[int] = None,
  ):
    if not _VOLATILE_CORE:
      raise ImportError("volatile C extension required for VolExtDataset")

    self._vol_handle = _vol_ext.vol_open(zarr_path)
    shape = _vol_ext.vol_shape(self._vol_handle, level)
    ndim = len(patch_size)
    spatial = tuple(shape[-ndim:])

    # Read image volume into memory (or lazily on demand — full read here for simplicity)
    # For large volumes users should subclass and override __getitem__ to sample lazily.
    D, H, W = (spatial[0], spatial[1], spatial[2]) if ndim == 3 else (1, spatial[0], spatial[1])
    img_arr = np.zeros((D, H, W), dtype=np.float32)
    for z in range(D):
      for y in range(H):
        for x in range(W):
          img_arr[z, y, x] = _vol_ext.vol_sample(self._vol_handle, level, float(z), float(y), float(x))

    # Normalise
    mx = img_arr.max()
    if mx > 0:
      img_arr /= mx

    label_arrays = label_arrays or {}
    volumes = [{"name": Path(zarr_path).stem, "image": img_arr if ndim == 3 else img_arr[0], "labels": label_arrays}]

    super().__init__(volumes, patch_size=patch_size, stride=stride, target_names=target_names, skip_empty=skip_empty, seed=seed)

  def close(self) -> None:
    if _VOLATILE_CORE and self._vol_handle is not None:
      _vol_ext.vol_free(self._vol_handle)
      self._vol_handle = None

  def __del__(self) -> None:
    self.close()

  def __enter__(self) -> "VolExtDataset":
    return self

  def __exit__(self, *_) -> None:
    self.close()


# ---------------------------------------------------------------------------
# Simple DataLoader-style batcher
# ---------------------------------------------------------------------------

class SimpleBatcher:
  """
  Wraps a PatchDataset to yield batches of size `batch_size`.

  Optionally shuffles patch indices each epoch.  Yields collated dicts (numpy).

  Args:
    dataset:    PatchDataset (or any object supporting __len__ / __getitem__)
    batch_size: number of patches per batch
    shuffle:    shuffle order each call to __iter__
    drop_last:  drop incomplete final batch (default False)
    seed:       RNG seed for shuffle
  """

  def __init__(
    self,
    dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    seed: Optional[int] = None,
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.drop_last = drop_last
    self._rng = np.random.default_rng(seed)

  def __len__(self) -> int:
    n = len(self.dataset) // self.batch_size
    if not self.drop_last and len(self.dataset) % self.batch_size:
      n += 1
    return n

  def __iter__(self) -> Iterator[Dict[str, Any]]:
    indices = np.arange(len(self.dataset))
    if self.shuffle:
      self._rng.shuffle(indices)

    for start in range(0, len(indices), self.batch_size):
      batch_idx = indices[start:start + self.batch_size]
      if self.drop_last and len(batch_idx) < self.batch_size:
        break
      items = [self.dataset[int(i)] for i in batch_idx]
      yield PatchDataset.collate(items)
