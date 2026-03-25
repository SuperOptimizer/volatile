from __future__ import annotations

import math
import numpy as np

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False


def _linear_ramp(size: int) -> np.ndarray:
  """1-D linear ramp: 0 → 1 → 0 over `size` steps, for overlap blending."""
  half = size / 2.0
  ramp = np.minimum(np.arange(size) + 0.5, half) / half  # 0..1 rise
  ramp = np.minimum(ramp, (size - np.arange(size) - 0.5) / half)  # 1..0 fall
  return ramp.astype(np.float32)


def _blend_weights_2d(tile_h: int, tile_w: int) -> np.ndarray:
  """2-D blend weight map for a tile: outer product of two 1-D ramps."""
  wy = _linear_ramp(tile_h)
  wx = _linear_ramp(tile_w)
  return wy[:, np.newaxis] * wx[np.newaxis, :]  # (tile_h, tile_w)


def tiled_infer(
  model,
  volume_np: np.ndarray,
  tile_h: int = 256,
  tile_w: int = 256,
  overlap: int = 64,
  batch_size: int = 1,
) -> np.ndarray:
  """
  Sliding-window tiled inference over a 2-D or 3-D numpy volume.

  Runs `model` on overlapping tiles and blends results with linear ramp weights
  to suppress tile-boundary artefacts.

  Args:
    model:      callable (tinygrad model) that accepts a Tensor of shape
                (B, C, H, W) and returns (B, out_ch, H, W).
    volume_np:  input array.  Accepted shapes:
                  (H, W)        — single 2-D slice, treated as 1 channel
                  (C, H, W)     — 2-D with explicit channels
                  (Z, H, W)     — 3-D; inference runs slice-by-slice (C=1)
                  (Z, C, H, W)  — 3-D with explicit channels
    tile_h:     tile height
    tile_w:     tile width
    overlap:    overlap in pixels between adjacent tiles (must be < tile size)
    batch_size: number of tiles per model call (≥1)

  Returns:
    numpy float32 array.  Shape mirrors input with the channel dim replaced by
    the model's output channels:
      input (H, W)      → output (out_ch, H, W)
      input (C, H, W)   → output (out_ch, H, W)
      input (Z, H, W)   → output (Z, out_ch, H, W)
      input (Z, C, H, W)→ output (Z, out_ch, H, W)
  """
  if not _TINYGRAD:
    raise ImportError("tinygrad is required for tiled_infer")

  ndim = volume_np.ndim
  if ndim == 2:
    vol = volume_np[np.newaxis, np.newaxis]   # (1, 1, H, W)
    squeeze_z = True
  elif ndim == 3:
    # ambiguous: treat as (Z, H, W) with C=1
    vol = volume_np[:, np.newaxis]            # (Z, 1, H, W)
    squeeze_z = False
  elif ndim == 4:
    vol = volume_np                           # (Z, C, H, W)
    squeeze_z = False
  else:
    raise ValueError(f"unsupported volume ndim={ndim}, expected 2-4")

  n_slices, in_ch, H, W = vol.shape

  stride_h = max(1, tile_h - overlap)
  stride_w = max(1, tile_w - overlap)
  weights_2d = _blend_weights_2d(tile_h, tile_w)  # (tile_h, tile_w)

  # discover out_channels from a dummy forward pass
  dummy = Tensor(np.zeros((1, in_ch, tile_h, tile_w), dtype=np.float32))
  dummy_out = model(dummy)
  out_ch = dummy_out.shape[1]
  del dummy, dummy_out

  # accumulator for all slices
  output = np.zeros((n_slices, out_ch, H, W), dtype=np.float32)
  weight_acc = np.zeros((n_slices, 1, H, W), dtype=np.float32)

  # collect tile positions
  ys = list(range(0, H - tile_h + 1, stride_h))
  if not ys or ys[-1] + tile_h < H:
    ys.append(max(0, H - tile_h))
  xs = list(range(0, W - tile_w + 1, stride_w))
  if not xs or xs[-1] + tile_w < W:
    xs.append(max(0, W - tile_w))

  # process slice by slice
  for zi in range(n_slices):
    slice_np = vol[zi].astype(np.float32)  # (C, H, W)

    # batch tiles across (y, x) positions
    tile_coords: list[tuple[int, int]] = [(y, x) for y in ys for x in xs]
    n_tiles = len(tile_coords)

    for batch_start in range(0, n_tiles, batch_size):
      batch_coords = tile_coords[batch_start : batch_start + batch_size]
      batch_arr = np.stack(
        [slice_np[:, y : y + tile_h, x : x + tile_w] for y, x in batch_coords],
        axis=0,
      ).astype(np.float32)  # (B, C, tile_h, tile_w)

      batch_tensor = Tensor(batch_arr)
      pred = model(batch_tensor)  # (B, out_ch, tile_h, tile_w)
      pred_np = pred.numpy()

      for i, (y, x) in enumerate(batch_coords):
        # actual tile extents may be smaller than tile_h/w at volume edges
        actual_h = pred_np.shape[2]
        actual_w = pred_np.shape[3]
        w2d = weights_2d[:actual_h, :actual_w]
        output[zi, :, y : y + actual_h, x : x + actual_w] += pred_np[i] * w2d
        weight_acc[zi, :, y : y + actual_h, x : x + actual_w] += w2d

  # normalise by accumulated weights (avoid div-by-zero for uncovered edges)
  weight_acc = np.maximum(weight_acc, 1e-6)
  output /= weight_acc

  if squeeze_z:
    return output[0]   # (out_ch, H, W)
  return output        # (Z, out_ch, H, W)
