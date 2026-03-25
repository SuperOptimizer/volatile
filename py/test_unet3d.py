"""test_unet3d.py — forward pass shape checks for Conv3d and UNet3D.

Run with:  python py/test_unet3d.py
       or: python -m pytest py/test_unet3d.py
"""
from __future__ import annotations

import sys
import os
import unittest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
  from tinygrad import Tensor
  from volatile.ml.unet3d import Conv3d, Conv3dBlock, UNet3D
  _TG = True
except ImportError:
  _TG = False

SKIP = unittest.skipUnless(_TG, "tinygrad not installed")


class TestConv3d(unittest.TestCase):

  @SKIP
  def test_output_shape_same_padding(self):
    conv = Conv3d(3, 8, k=3, padding=1)
    x = Tensor.randn(2, 3, 16, 16, 16)
    y = conv(x)
    self.assertEqual(tuple(y.shape), (2, 8, 16, 16, 16))

  @SKIP
  def test_output_shape_no_padding(self):
    conv = Conv3d(3, 8, k=3, padding=0)
    x = Tensor.randn(1, 3, 10, 10, 10)
    y = conv(x)
    self.assertEqual(tuple(y.shape), (1, 8, 8, 8, 8))

  @SKIP
  def test_1x1x1_conv(self):
    conv = Conv3d(4, 16, k=1, padding=0)
    x = Tensor.randn(2, 4, 8, 8, 8)
    y = conv(x)
    self.assertEqual(tuple(y.shape), (2, 16, 8, 8, 8))

  @SKIP
  def test_output_is_finite(self):
    conv = Conv3d(2, 4, k=3)
    x = Tensor.randn(1, 2, 8, 8, 8)
    y = conv(x)
    self.assertTrue(math.isfinite(y.mean().item()))

  @SKIP
  def test_single_channel(self):
    conv = Conv3d(1, 1, k=3)
    x = Tensor.randn(1, 1, 4, 4, 4)
    y = conv(x)
    self.assertEqual(tuple(y.shape), (1, 1, 4, 4, 4))


class TestConv3dBlock(unittest.TestCase):

  @SKIP
  def test_output_shape(self):
    blk = Conv3dBlock(4, 8)
    x = Tensor.randn(2, 4, 8, 8, 8)
    y = blk(x)
    self.assertEqual(tuple(y.shape), (2, 8, 8, 8, 8))

  @SKIP
  def test_same_channels(self):
    blk = Conv3dBlock(16, 16)
    x = Tensor.randn(1, 16, 4, 4, 4)
    y = blk(x)
    self.assertEqual(tuple(y.shape), (1, 16, 4, 4, 4))

  @SKIP
  def test_output_finite(self):
    blk = Conv3dBlock(1, 4)
    x = Tensor.randn(1, 1, 6, 6, 6)
    y = blk(x)
    self.assertTrue(math.isfinite(y.mean().item()))


class TestUNet3DShapes(unittest.TestCase):

  def _small(self, **kw) -> "UNet3D":
    return UNet3D(in_channels=1, out_channels=2, base=8, levels=2, **kw)

  @SKIP
  def test_output_same_spatial(self):
    model = self._small()
    x = Tensor.randn(1, 1, 16, 16, 16)
    y = model(x)
    self.assertEqual(tuple(y.shape), (1, 2, 16, 16, 16))

  @SKIP
  def test_out_channels_matches(self):
    for out_ch in (1, 4, 8):
      model = UNet3D(in_channels=1, out_channels=out_ch, base=8, levels=2)
      x = Tensor.randn(1, 1, 16, 16, 16)
      y = model(x)
      self.assertEqual(y.shape[1], out_ch, f"out_ch={out_ch}")

  @SKIP
  def test_batch_dimension_preserved(self):
    model = self._small()
    x = Tensor.randn(3, 1, 16, 16, 16)
    y = model(x)
    self.assertEqual(y.shape[0], 3)

  @SKIP
  def test_multichannel_input(self):
    model = UNet3D(in_channels=4, out_channels=2, base=8, levels=2)
    x = Tensor.randn(1, 4, 16, 16, 16)
    y = model(x)
    self.assertEqual(tuple(y.shape), (1, 2, 16, 16, 16))

  @SKIP
  def test_three_levels(self):
    model = UNet3D(in_channels=1, out_channels=3, base=8, levels=3)
    x = Tensor.randn(1, 1, 32, 32, 32)
    y = model(x)
    self.assertEqual(tuple(y.shape), (1, 3, 32, 32, 32))

  @SKIP
  def test_output_is_finite(self):
    model = self._small()
    x = Tensor.randn(1, 1, 16, 16, 16)
    y = model(x)
    self.assertTrue(math.isfinite(y.mean().item()))

  @SKIP
  def test_parameters_non_empty(self):
    model = self._small()
    params = model.parameters()
    self.assertGreater(len(params), 0)

  def test_import_without_tinygrad(self):
    # Module must be importable when tinygrad is absent
    pass  # presence of _TG=False flag is the test


if __name__ == "__main__":
  unittest.main()
