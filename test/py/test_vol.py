from __future__ import annotations
import os
import json
import tempfile
import pytest


def make_zarr(base: str, levels: list[dict]) -> None:
  """Create a minimal synthetic .zarr directory under base."""
  os.makedirs(base, exist_ok=True)
  for lvl_idx, lvl in enumerate(levels):
    lvl_dir = os.path.join(base, str(lvl_idx))
    os.makedirs(lvl_dir, exist_ok=True)
    zarray = {
      "zarr_format": 2,
      "chunks": lvl["chunks"],
      "shape": lvl["shape"],
      "dtype": "|u1",
      "order": "C",
      "compressor": None,
    }
    with open(os.path.join(lvl_dir, ".zarray"), "w") as f:
      json.dump(zarray, f)
    # write one raw chunk (0.0.0) filled with a known pattern
    chunk_elems = 1
    for s in lvl["chunks"]:
      chunk_elems *= s
    chunk_data = bytes(i % 256 for i in range(chunk_elems))
    with open(os.path.join(lvl_dir, "0.0.0"), "wb") as f:
      f.write(chunk_data)


@pytest.fixture(scope="module")
def zarr_path(tmp_path_factory):
  base = str(tmp_path_factory.mktemp("vol") / "test.zarr")
  make_zarr(base, [
    {"shape": [8, 8, 8],  "chunks": [4, 4, 4]},
    {"shape": [4, 4, 4],  "chunks": [4, 4, 4]},
  ])
  return base


def test_vol_open(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  assert vol is not None
  volatile.vol_free(vol)


def test_vol_num_levels(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  assert volatile.vol_num_levels(vol) == 2
  volatile.vol_free(vol)


def test_vol_shape_level0(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  shape = volatile.vol_shape(vol, 0)
  assert shape == (8, 8, 8)
  volatile.vol_free(vol)


def test_vol_shape_level1(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  shape = volatile.vol_shape(vol, 1)
  assert shape == (4, 4, 4)
  volatile.vol_free(vol)


def test_vol_shape_default_level(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  # default level arg is 0
  shape = volatile.vol_shape(vol)
  assert len(shape) == 3
  volatile.vol_free(vol)


def test_vol_shape_out_of_range(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  with pytest.raises(IndexError):
    volatile.vol_shape(vol, 99)
  volatile.vol_free(vol)


def test_vol_open_nonexistent():
  import volatile
  with pytest.raises(OSError):
    volatile.vol_open("/tmp/no_such_zarr_ever_xyz")


def test_vol_sample_returns_float(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  val = volatile.vol_sample(vol, 0, 2.0, 2.0, 2.0)
  assert isinstance(val, float)
  volatile.vol_free(vol)


def test_vol_capsule_not_reused_after_free(zarr_path):
  import volatile
  vol = volatile.vol_open(zarr_path)
  volatile.vol_free(vol)
  # After explicit free the capsule pointer is NULL; accessing it should raise
  with pytest.raises(TypeError):
    volatile.vol_num_levels(vol)
