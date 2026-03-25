"""tifxyz.py — TIF/XYZ format utilities.

Reads and writes 3-channel TIFF files where channels encode (x, y, z) world
coordinates per pixel — the format used by villa's vc_render and related tools
for storing surface parameterizations.

Functions:
  read_tifxyz(path)        -> np.ndarray (rows, cols, 3) float32
  write_tifxyz(path, xyz)  -> None
  tifxyz_to_mesh(xyz)      -> QuadSurface
  mesh_to_tifxyz(surface)  -> np.ndarray (rows, cols, 3) float32
"""
from __future__ import annotations

import struct
from typing import Union

import numpy as np

from .seg import QuadSurface

# ---------------------------------------------------------------------------
# TIFF helpers — minimal read/write for 3-channel float32 TIFFs.
# Uses tifffile when available; falls back to a bare-bones implementation.
# ---------------------------------------------------------------------------

try:
  import tifffile as _tifffile
  _HAS_TIFFFILE = True
except ImportError:
  _HAS_TIFFFILE = False


def read_tifxyz(path: str) -> np.ndarray:
  """Read a 3-channel TIFF as (rows, cols, 3) float32 xyz array.

  Args:
    path: path to a TIFF file with 3 float32 channels.
  Returns:
    Array of shape (rows, cols, 3), dtype float32.
  Raises:
    ImportError: if tifffile is not available and the file isn't a simple flat TIFF.
    ValueError: if the loaded image doesn't have exactly 3 channels.
  """
  if _HAS_TIFFFILE:
    img = _tifffile.imread(path)
  else:
    img = _read_tiff_fallback(path)

  img = np.asarray(img, dtype=np.float32)

  # Support (3, H, W) → (H, W, 3) channel-first layout
  if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
    img = np.moveaxis(img, 0, -1)

  if img.ndim != 3 or img.shape[2] != 3:
    raise ValueError(
      f"Expected a 3-channel image (H, W, 3) or (3, H, W); got shape {img.shape}"
    )
  return img


def write_tifxyz(path: str, xyz: np.ndarray) -> None:
  """Write (rows, cols, 3) float32 xyz array as a 3-channel TIFF.

  Args:
    path: output file path.
    xyz: array of shape (rows, cols, 3), dtype convertible to float32.
  Raises:
    ValueError: if xyz doesn't have shape (H, W, 3).
  """
  xyz = np.asarray(xyz, dtype=np.float32)
  if xyz.ndim != 3 or xyz.shape[2] != 3:
    raise ValueError(f"xyz must have shape (H, W, 3); got {xyz.shape}")

  if _HAS_TIFFFILE:
    # Write channel-first (3, H, W) for compatibility with vc_render readers
    _tifffile.imwrite(path, np.moveaxis(xyz, -1, 0), photometric="minisblack")
  else:
    _write_tiff_fallback(path, xyz)


# ---------------------------------------------------------------------------
# Surface ↔ array conversion
# ---------------------------------------------------------------------------

def tifxyz_to_mesh(xyz: np.ndarray) -> QuadSurface:
  """Convert a (rows, cols, 3) xyz array to a QuadSurface.

  Each pixel becomes a vertex; the grid topology follows the image layout.

  Args:
    xyz: array of shape (rows, cols, 3).
  Returns:
    QuadSurface with rows × cols vertices.
  """
  xyz = np.asarray(xyz, dtype=np.float32)
  if xyz.ndim != 3 or xyz.shape[2] != 3:
    raise ValueError(f"xyz must have shape (H, W, 3); got {xyz.shape}")

  rows, cols = xyz.shape[:2]
  surf = QuadSurface(rows, cols)
  for r in range(rows):
    for c in range(cols):
      surf.set(r, c, (float(xyz[r, c, 0]), float(xyz[r, c, 1]), float(xyz[r, c, 2])))
  return surf


def tifxyz_to_mesh_fast(xyz: np.ndarray) -> QuadSurface:
  """NumPy-accelerated version of tifxyz_to_mesh (avoids Python loop)."""
  xyz = np.asarray(xyz, dtype=np.float32)
  if xyz.ndim != 3 or xyz.shape[2] != 3:
    raise ValueError(f"xyz must have shape (H, W, 3); got {xyz.shape}")
  rows, cols = xyz.shape[:2]
  surf = QuadSurface(rows, cols)
  # QuadSurface._pts is a flat list of [x,y,z]; reshape and assign directly.
  flat = xyz.reshape(-1, 3).tolist()
  surf._pts = flat
  return surf


def mesh_to_tifxyz(surface: QuadSurface) -> np.ndarray:
  """Convert a QuadSurface to a (rows, cols, 3) float32 xyz array.

  Args:
    surface: QuadSurface with rows × cols vertices.
  Returns:
    Array of shape (rows, cols, 3), dtype float32.
  """
  rows, cols = surface.rows, surface.cols
  xyz = np.empty((rows, cols, 3), dtype=np.float32)
  for r in range(rows):
    for c in range(cols):
      xyz[r, c] = surface.get(r, c)
  return xyz


def mesh_to_tifxyz_fast(surface: QuadSurface) -> np.ndarray:
  """NumPy-accelerated version of mesh_to_tifxyz."""
  flat = np.array(surface._pts, dtype=np.float32)
  return flat.reshape(surface.rows, surface.cols, 3)


# ---------------------------------------------------------------------------
# Minimal TIFF fallback (uncompressed float32, single IFD)
# Used only when tifffile is not installed.
# ---------------------------------------------------------------------------

_TIFF_LITTLE_ENDIAN = b"II"
_TIFF_BIG_ENDIAN    = b"MM"
_TIFF_MAGIC         = 42
_FLOAT32_SAMPLE_FMT = 3  # IEEE floating point
_BITS_PER_SAMPLE    = 32


def _write_tiff_fallback(path: str, xyz: np.ndarray) -> None:
  """Write a minimal uncompressed TIFF (little-endian, float32, planar RGB)."""
  rows, cols, _ = xyz.shape
  # Write channel-first planes
  data = np.moveaxis(xyz, -1, 0)  # (3, H, W)

  # IFD entries we need (tag, type, count, value/offset)
  # types: SHORT=3, LONG=4, RATIONAL=5
  ifd_entries: list[tuple[int, int, int, int]] = []

  image_width  = cols
  image_length = rows
  samples_per_pixel = 3
  bits_per_sample   = [32, 32, 32]
  compression       = 1    # no compression
  photometric       = 2    # RGB
  planar_config     = 2    # planar (separate channels)
  sample_format     = [3, 3, 3]  # IEEE float
  rows_per_strip    = rows
  strip_byte_counts = [data[c].nbytes for c in range(3)]

  # Build tag list (must be sorted by tag number)
  # We'll fix up offsets after computing sizes
  # Header: 8 bytes; IFD: 2 + 12*n + 4 bytes; then extra data, then pixel data
  n_tags = 11
  ifd_offset = 8
  ifd_size   = 2 + 12 * n_tags + 4

  # Extra data area: BitsPerSample (3 SHORTs = 6 bytes), SampleFormat (3 SHORTs = 6 bytes),
  # StripOffsets (3 LONGs = 12 bytes), StripByteCounts (3 LONGs = 12 bytes)
  extra_offset = ifd_offset + ifd_size
  bps_offset   = extra_offset
  sfmt_offset  = bps_offset + 6
  soff_offset  = sfmt_offset + 6
  sbc_offset   = soff_offset + 12
  pixel_base   = sbc_offset + 12
  strip_offsets = [pixel_base + sum(strip_byte_counts[:c]) for c in range(3)]

  def pack_short(v: int) -> bytes:
    return struct.pack("<H", v)

  def pack_long(v: int) -> bytes:
    return struct.pack("<L", v)

  def tag_entry(tag: int, typ: int, count: int, value_or_offset: int) -> bytes:
    return struct.pack("<HHLL", tag, typ, count, value_or_offset)

  header = _TIFF_LITTLE_ENDIAN + struct.pack("<HL", _TIFF_MAGIC, ifd_offset)

  ifd  = struct.pack("<H", n_tags)
  ifd += tag_entry(256, 4, 1, image_width)          # ImageWidth
  ifd += tag_entry(257, 4, 1, image_length)         # ImageLength
  ifd += tag_entry(258, 3, 3, bps_offset)           # BitsPerSample
  ifd += tag_entry(259, 3, 1, compression)          # Compression
  ifd += tag_entry(262, 3, 1, photometric)          # PhotometricInterp
  ifd += tag_entry(278, 4, 1, rows_per_strip)       # RowsPerStrip
  ifd += tag_entry(277, 3, 1, samples_per_pixel)    # SamplesPerPixel
  ifd += tag_entry(273, 4, 3, soff_offset)          # StripOffsets
  ifd += tag_entry(279, 4, 3, sbc_offset)           # StripByteCounts
  ifd += tag_entry(284, 3, 1, planar_config)        # PlanarConfig
  ifd += tag_entry(339, 3, 3, sfmt_offset)          # SampleFormat
  ifd += struct.pack("<L", 0)                       # next IFD offset (none)

  extra  = b"".join(pack_short(32) for _ in range(3))        # BitsPerSample
  extra += b"".join(pack_short(3)  for _ in range(3))        # SampleFormat
  extra += b"".join(pack_long(o)   for o in strip_offsets)   # StripOffsets
  extra += b"".join(pack_long(n)   for n in strip_byte_counts)  # StripByteCounts

  with open(path, "wb") as f:
    f.write(header)
    f.write(ifd)
    f.write(extra)
    for c in range(3):
      f.write(data[c].tobytes())


def _read_tiff_fallback(path: str) -> np.ndarray:
  """Read a TIFF written by _write_tiff_fallback; returns (3, H, W) float32."""
  with open(path, "rb") as f:
    raw = f.read()

  endian = raw[:2]
  le = (endian == b"II")
  prefix = "<" if le else ">"

  magic, ifd_offset = struct.unpack_from(f"{prefix}HL", raw, 2)
  if magic != _TIFF_MAGIC:
    raise ValueError(f"Not a valid TIFF (magic={magic})")

  n_entries = struct.unpack_from(f"{prefix}H", raw, ifd_offset)[0]
  tags: dict[int, tuple] = {}
  for i in range(n_entries):
    off = ifd_offset + 2 + i * 12
    tag, typ, count, val = struct.unpack_from(f"{prefix}HHLL", raw, off)
    tags[tag] = (typ, count, val)

  def read_tag_shorts(tag: int) -> list[int]:
    typ, count, val = tags[tag]
    if count == 1:
      return [val & 0xFFFF]
    offs = val
    return [struct.unpack_from(f"{prefix}H", raw, offs + i * 2)[0] for i in range(count)]

  def read_tag_longs(tag: int) -> list[int]:
    typ, count, val = tags[tag]
    if count == 1:
      return [val]
    offs = val
    return [struct.unpack_from(f"{prefix}L", raw, offs + i * 4)[0] for i in range(count)]

  width  = tags[256][2]
  height = tags[257][2]
  strip_offsets    = read_tag_longs(273)
  strip_bytecounts = read_tag_longs(279)

  channels = []
  for offset, nbytes in zip(strip_offsets, strip_bytecounts):
    plane_bytes = raw[offset: offset + nbytes]
    channels.append(np.frombuffer(plane_bytes, dtype=np.float32).reshape(height, width))

  return np.stack(channels, axis=0)  # (3, H, W)
