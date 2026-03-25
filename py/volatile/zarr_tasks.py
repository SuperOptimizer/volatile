"""zarr_tasks.py — Chunk-level operations on zarr volumes, parallelised via multiprocessing.

Each function opens the input(s) with the zarr Python library, farms out
per-chunk work via multiprocessing.Pool, and writes results to a new zarr
array.  The volatile C extension is used when available (vol_sample for
sampling), but all I/O falls back to the pure-Python zarr library so the
module works without a built extension.

CLI: python -m volatile.zarr_tasks <command> [options]
"""
from __future__ import annotations

import argparse
import itertools
import multiprocessing
import os
import sys
from typing import Sequence

import numpy as np
import zarr
from zarr.codecs import BloscCodec, BytesCodec, GzipCodec, ZstdCodec

try:
  from tqdm import tqdm as _tqdm
  _HAS_TQDM = True
except ImportError:
  _HAS_TQDM = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_coords(shape: tuple, chunks: tuple) -> list[tuple]:
  """Return all chunk grid coordinate tuples for a given shape / chunk shape."""
  grid = [range(int(np.ceil(s / c))) for s, c in zip(shape, chunks)]
  return list(itertools.product(*grid))


def _chunk_slices(coord: tuple, chunks: tuple, shape: tuple) -> tuple[slice, ...]:
  """Convert a chunk grid coordinate to a tuple of slice objects."""
  return tuple(slice(ci * ch, min((ci + 1) * ch, s)) for ci, ch, s in zip(coord, chunks, shape))


def _codec_for(codec_name: str, level: int):
  """Return a zarr 3 codec list for the given codec name and level."""
  name = codec_name.lower()
  if name == "blosc":
    return [BytesCodec(), BloscCodec(cname="zstd", clevel=level)]
  if name == "zstd":
    return [BytesCodec(), ZstdCodec(level=level)]
  if name == "gzip":
    return [BytesCodec(), GzipCodec(level=level)]
  if name in ("none", ""):
    return [BytesCodec()]
  return [BytesCodec(), BloscCodec(cname="zstd", clevel=level)]


def _progress(iterable, total: int, desc: str):
  if _HAS_TQDM:
    return _tqdm(iterable, total=total, desc=desc, unit="chunk")
  # Simple counter fallback
  class _Counter:
    def __init__(self, it, total, desc):
      self._it = iter(it)
      self._n = 0
      self._total = total
      self._desc = desc
    def __iter__(self):
      for item in self._it:
        self._n += 1
        if self._n % max(1, self._total // 20) == 0 or self._n == self._total:
          print(f"\r{self._desc}: {self._n}/{self._total}", end="", flush=True)
        yield item
      print()
  return _Counter(iterable, total, desc)


# ---------------------------------------------------------------------------
# Worker functions (module-level so they are picklable)
# ---------------------------------------------------------------------------

def _worker_threshold(args):
  src_path, dst_path, coord, low, high = args
  src = zarr.open_array(src_path, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices = _chunk_slices(coord, dst.chunks, dst.shape)
  chunk = src[slices].astype(np.float32)
  out = np.where((chunk >= low) & (chunk <= high), np.uint8(255), np.uint8(0))
  dst[slices] = out.astype(dst.dtype)


def _worker_scale(args):
  src_path, dst_path, coord, factor = args
  src = zarr.open_array(src_path, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices_dst = _chunk_slices(coord, dst.chunks, dst.shape)
  # Map dst chunk back to src region
  slices_src = tuple(slice(s.start * factor, min(s.stop * factor, src.shape[i])) for i, s in enumerate(slices_dst))
  chunk_src = src[slices_src].astype(np.float32)
  # Mean pool — handle non-divisible edges with reshape + mean
  out_shape = tuple(s.stop - s.start for s in slices_dst)
  pooled = np.zeros(out_shape, dtype=np.float32)
  for idx in itertools.product(*[range(f) for f in [factor] * src.ndim]):
    sub = tuple(slice(i, chunk_src.shape[d], factor) for d, i in enumerate(idx))
    sliced = chunk_src[sub]
    # Crop to pooled shape if needed (edge chunks)
    crop = tuple(slice(0, pooled.shape[d]) for d in range(pooled.ndim))
    pooled[crop] += sliced[crop]
  pooled /= factor ** src.ndim
  dst[slices_dst] = pooled.astype(dst.dtype)


def _worker_remap(args):
  src_path, dst_path, coord, lut = args
  lut_arr = np.asarray(lut, dtype=np.uint8)
  src = zarr.open_array(src_path, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices = _chunk_slices(coord, dst.chunks, dst.shape)
  chunk = src[slices].astype(np.uint8)
  dst[slices] = lut_arr[chunk]


def _worker_merge(args):
  path1, path2, dst_path, coord, op = args
  a1 = zarr.open_array(path1, mode="r")
  a2 = zarr.open_array(path2, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices = _chunk_slices(coord, dst.chunks, dst.shape)
  c1 = a1[slices].astype(np.float32)
  c2 = a2[slices].astype(np.float32)
  if op == "max":
    out = np.maximum(c1, c2)
  elif op == "min":
    out = np.minimum(c1, c2)
  elif op == "add":
    out = np.clip(c1 + c2, 0, np.iinfo(dst.dtype).max if np.issubdtype(dst.dtype, np.integer) else 1e38)
  elif op == "mean":
    out = (c1 + c2) * 0.5
  else:
    raise ValueError(f"unknown merge op: {op!r}")
  dst[slices] = out.astype(dst.dtype)


def _worker_recompress(args):
  src_path, dst_path, coord = args
  src = zarr.open_array(src_path, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices = _chunk_slices(coord, dst.chunks, dst.shape)
  dst[slices] = src[slices]


def _worker_transpose(args):
  src_path, dst_path, coord, axes = args
  src = zarr.open_array(src_path, mode="r")
  dst = zarr.open_array(dst_path, mode="r+")
  slices_dst = _chunk_slices(coord, dst.chunks, dst.shape)
  # Map dst slices back to src order
  inv_axes = [0] * len(axes)
  for i, a in enumerate(axes):
    inv_axes[a] = i
  slices_src = tuple(slices_dst[inv_axes[i]] for i in range(len(axes)))
  chunk_src = src[slices_src]
  dst[slices_dst] = np.transpose(chunk_src, axes)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _open_src(path: str) -> zarr.Array:
  return zarr.open_array(path, mode="r")


def _create_dst(path: str, src: zarr.Array, *, dtype=None, shape=None, chunks=None, codec: str = "blosc", level: int = 5) -> zarr.Array:
  return zarr.open_array(
    path, mode="w",
    shape=shape or src.shape,
    chunks=chunks or src.chunks,
    dtype=dtype or src.dtype,
    codecs=_codec_for(codec, level),
  )


def scale_zarr(input_path: str, output_path: str, factor: int = 2, *, workers: int | None = None) -> None:
  """Downsample a zarr volume by integer factor using mean pooling."""
  src = _open_src(input_path)
  out_shape = tuple(max(1, int(np.ceil(s / factor))) for s in src.shape)
  out_chunks = tuple(min(c, os_) for c, os_ in zip(src.chunks, out_shape))
  dst = _create_dst(output_path, src, shape=out_shape, chunks=out_chunks)
  coords = _chunk_coords(out_shape, dst.chunks)
  task_args = [(input_path, output_path, c, factor) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_scale, task_args), len(task_args), "scale"):
      pass


def threshold_zarr(input_path: str, output_path: str, low: float = 0, high: float = 255, *, workers: int | None = None) -> None:
  """Binarise a zarr volume: voxels in [low, high] -> 255, else -> 0."""
  src = _open_src(input_path)
  dst = _create_dst(output_path, src, dtype=np.uint8)
  coords = _chunk_coords(dst.shape, dst.chunks)
  task_args = [(input_path, output_path, c, low, high) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_threshold, task_args), len(task_args), "threshold"):
      pass


def remap_zarr(input_path: str, output_path: str, lut: Sequence[int], *, workers: int | None = None) -> None:
  """Apply a 256-entry lookup table (uint8->uint8) to every voxel."""
  if len(lut) != 256:
    raise ValueError(f"lut must have 256 entries, got {len(lut)}")
  src = _open_src(input_path)
  dst = _create_dst(output_path, src, dtype=np.uint8)
  coords = _chunk_coords(dst.shape, dst.chunks)
  lut_list = list(lut)  # must be picklable
  task_args = [(input_path, output_path, c, lut_list) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_remap, task_args), len(task_args), "remap"):
      pass


def merge_zarr(path1: str, path2: str, output_path: str, op: str = "max", *, workers: int | None = None) -> None:
  """Combine two volumes element-wise. op: 'max', 'min', 'add', 'mean'."""
  src1 = _open_src(path1)
  src2 = _open_src(path2)
  if src1.shape != src2.shape:
    raise ValueError(f"shapes must match: {src1.shape} vs {src2.shape}")
  dst = _create_dst(output_path, src1)
  coords = _chunk_coords(dst.shape, dst.chunks)
  task_args = [(path1, path2, output_path, c, op) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_merge, task_args), len(task_args), f"merge({op})"):
      pass


def recompress_zarr(input_path: str, output_path: str, codec: str = "blosc", level: int = 5, *, workers: int | None = None) -> None:
  """Copy a zarr volume, re-encoding with the given codec and compression level."""
  src = _open_src(input_path)
  dst = _create_dst(output_path, src, codec=codec, level=level)
  coords = _chunk_coords(dst.shape, dst.chunks)
  task_args = [(input_path, output_path, c) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_recompress, task_args), len(task_args), "recompress"):
      pass


def transpose_zarr(input_path: str, output_path: str, axes: tuple[int, ...] = (2, 1, 0), *, workers: int | None = None) -> None:
  """Swap axes of a zarr volume (default: reverse all axes)."""
  src = _open_src(input_path)
  if sorted(axes) != list(range(src.ndim)):
    raise ValueError(f"axes must be a permutation of 0..{src.ndim - 1}, got {axes}")
  out_shape = tuple(src.shape[a] for a in axes)
  out_chunks = tuple(src.chunks[a] for a in axes)
  dst = _create_dst(output_path, src, shape=out_shape, chunks=out_chunks)
  coords = _chunk_coords(out_shape, dst.chunks)
  task_args = [(input_path, output_path, c, list(axes)) for c in coords]
  with multiprocessing.Pool(workers) as pool:
    for _ in _progress(pool.imap_unordered(_worker_transpose, task_args), len(task_args), "transpose"):
      pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_scale(args):
  scale_zarr(args.input, args.output, factor=args.factor, workers=args.workers)

def _cli_threshold(args):
  threshold_zarr(args.input, args.output, low=args.low, high=args.high, workers=args.workers)

def _cli_remap(args):
  import json
  lut = json.loads(args.lut)
  remap_zarr(args.input, args.output, lut, workers=args.workers)

def _cli_merge(args):
  merge_zarr(args.input, args.input2, args.output, op=args.op, workers=args.workers)

def _cli_recompress(args):
  recompress_zarr(args.input, args.output, codec=args.codec, level=args.level, workers=args.workers)

def _cli_transpose(args):
  axes = tuple(int(a) for a in args.axes.split(","))
  transpose_zarr(args.input, args.output, axes=axes, workers=args.workers)


def main(argv=None):
  p = argparse.ArgumentParser(prog="python -m volatile.zarr_tasks", description="Chunk-level zarr volume operations")
  p.add_argument("--workers", type=int, default=None, help="Number of worker processes (default: CPU count)")
  sub = p.add_subparsers(dest="command", required=True)

  s = sub.add_parser("scale", help="Downsample by integer factor")
  s.add_argument("input"); s.add_argument("output"); s.add_argument("--factor", type=int, default=2)
  s.set_defaults(func=_cli_scale)

  s = sub.add_parser("threshold", help="Binarise: [low,high] -> 255, else 0")
  s.add_argument("input"); s.add_argument("output")
  s.add_argument("--low",  type=float, default=0);   s.add_argument("--high", type=float, default=255)
  s.set_defaults(func=_cli_threshold)

  s = sub.add_parser("remap", help="Apply 256-entry LUT (JSON array)")
  s.add_argument("input"); s.add_argument("output"); s.add_argument("--lut", required=True)
  s.set_defaults(func=_cli_remap)

  s = sub.add_parser("merge", help="Combine two volumes element-wise")
  s.add_argument("input"); s.add_argument("input2"); s.add_argument("output")
  s.add_argument("--op", default="max", choices=["max", "min", "add", "mean"])
  s.set_defaults(func=_cli_merge)

  s = sub.add_parser("recompress", help="Re-encode with a different codec")
  s.add_argument("input"); s.add_argument("output")
  s.add_argument("--codec", default="blosc"); s.add_argument("--level", type=int, default=5)
  s.set_defaults(func=_cli_recompress)

  s = sub.add_parser("transpose", help="Swap axes (default: reverse all)")
  s.add_argument("input"); s.add_argument("output"); s.add_argument("--axes", default="2,1,0")
  s.set_defaults(func=_cli_transpose)

  args = p.parse_args(argv)
  args.func(args)


if __name__ == "__main__":
  main()
