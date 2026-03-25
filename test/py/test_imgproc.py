from __future__ import annotations
import struct
import math
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pack_floats(values):
  return struct.pack(f"{len(values)}f", *values)

def _unpack_floats(data):
  n = len(data) // 4
  return struct.unpack(f"{n}f", data)

# ---------------------------------------------------------------------------
# gaussian_blur_2d
# ---------------------------------------------------------------------------

def test_gaussian_blur_2d_output_shape():
  from volatile.imgproc import gaussian_blur_2d
  h, w = 8, 10
  data = _pack_floats([1.0] * (h * w))
  result = gaussian_blur_2d(data, sigma=1.0, height=h, width=w)
  # result is bytes of float32 values, same count as input
  assert len(result) == h * w * 4

def test_gaussian_blur_2d_flat_image_unchanged():
  """Blurring a uniform image should leave values unchanged."""
  from volatile.imgproc import gaussian_blur_2d
  h, w = 6, 6
  data = _pack_floats([3.14] * (h * w))
  result = gaussian_blur_2d(data, sigma=1.0, height=h, width=w)
  values = _unpack_floats(result)
  for v in values:
    assert abs(v - 3.14) < 1e-4, f"expected ~3.14 but got {v}"

def test_gaussian_blur_2d_impulse_spreads():
  """A single bright pixel should spread to neighbours after blur."""
  from volatile.imgproc import gaussian_blur_2d
  h, w = 9, 9
  pixels = [0.0] * (h * w)
  pixels[4 * w + 4] = 1.0  # center pixel
  data = _pack_floats(pixels)
  result = gaussian_blur_2d(data, sigma=1.5, height=h, width=w)
  values = _unpack_floats(result)
  # Center should be less than 1 (energy spread)
  assert values[4 * w + 4] < 1.0
  # Neighbor should be positive
  assert values[4 * w + 5] > 0.0

# ---------------------------------------------------------------------------
# histogram
# ---------------------------------------------------------------------------

def test_histogram_known_data():
  from volatile.imgproc import histogram
  # 100 values uniformly spaced 0..1
  n = 100
  values = [i / (n - 1) for i in range(n)]
  data = _pack_floats(values)
  h = histogram(data, num_bins=10, num_elements=n)

  assert "bins" in h and "min" in h and "max" in h and "mean" in h
  assert len(h["bins"]) == 10
  assert abs(h["min"] - 0.0) < 1e-5
  assert abs(h["max"] - 1.0) < 1e-5
  assert abs(h["mean"] - 0.5) < 0.01
  # Total counts should equal n
  assert sum(h["bins"]) == n

def test_histogram_single_value():
  from volatile.imgproc import histogram
  n = 50
  data = _pack_floats([2.0] * n)
  h = histogram(data, num_bins=5, num_elements=n)
  assert h["min"] == pytest.approx(2.0, abs=1e-5)
  assert h["max"] == pytest.approx(2.0, abs=1e-5)
  assert h["mean"] == pytest.approx(2.0, abs=1e-5)

# ---------------------------------------------------------------------------
# window_level
# ---------------------------------------------------------------------------

def test_window_level_output_length():
  from volatile.imgproc import window_level
  n = 20
  data = _pack_floats([float(i) for i in range(n)])
  result = window_level(data, window=19.0, level=9.5, num_elements=n)
  assert len(result) == n  # uint8 bytes, one per element

def test_window_level_clipping():
  """Values below window-min map to 0; above window-max map to 255."""
  from volatile.imgproc import window_level
  values = [-100.0, 0.0, 50.0, 100.0, 200.0]
  data = _pack_floats(values)
  result = window_level(data, window=100.0, level=50.0, num_elements=len(values))
  out = list(result)  # bytes is iterable as ints
  assert out[0] == 0    # below range -> 0
  assert out[4] == 255  # above range -> 255
  assert 0 < out[2] < 255  # midpoint -> somewhere in between
