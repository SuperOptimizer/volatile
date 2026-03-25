"""test_nnunet.py — forward pass shape checks for NNUNet and MultiTaskUNet.

Run with:  python py/test_nnunet.py
       or: python -m pytest py/test_nnunet.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

import unittest

def _randn(*shape):
  return Tensor.randn(*shape)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shape(t):
  """Return shape tuple from a Tensor or list of Tensors."""
  if isinstance(t, list):
    return [tuple(x.shape) for x in t]
  return tuple(t.shape)

# ---------------------------------------------------------------------------
# Block-level tests (2D)
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TINYGRAD, "tinygrad not installed")
class TestBlocks(unittest.TestCase):

  def test_plain_conv_block_2d(self):
    from volatile.ml.nnunet import PlainConvBlock
    blk = PlainConvBlock('2d', 4, 8, k=3)
    x = _randn(2, 4, 16, 16)
    y = blk(x)
    self.assertEqual(_shape(y), (2, 8, 16, 16))

  def test_residual_block_same_channels(self):
    from volatile.ml.nnunet import ResidualBlock
    blk = ResidualBlock('2d', 8, 8)
    x = _randn(2, 8, 16, 16)
    y = blk(x)
    self.assertEqual(_shape(y), (2, 8, 16, 16))

  def test_residual_block_channel_change(self):
    from volatile.ml.nnunet import ResidualBlock
    blk = ResidualBlock('2d', 4, 16)
    x = _randn(2, 4, 16, 16)
    y = blk(x)
    self.assertEqual(_shape(y), (2, 16, 16, 16))

  def test_residual_block_stride2(self):
    from volatile.ml.nnunet import ResidualBlock
    blk = ResidualBlock('2d', 8, 16, stride=2)
    x = _randn(2, 8, 16, 16)
    y = blk(x)
    self.assertEqual(_shape(y), (2, 16, 8, 8))

  def test_bottleneck_block_2d(self):
    from volatile.ml.nnunet import BottleneckBlock
    blk = BottleneckBlock('2d', 8, 16, stride=2)
    x = _randn(2, 8, 16, 16)
    y = blk(x)
    self.assertEqual(_shape(y), (2, 16, 8, 8))

  def test_squeeze_excitation_2d(self):
    from volatile.ml.nnunet import SqueezeExcitation
    se = SqueezeExcitation('2d', 16)
    x = _randn(2, 16, 8, 8)
    y = se(x)
    self.assertEqual(_shape(y), (2, 16, 8, 8))

# ---------------------------------------------------------------------------
# Encoder tests
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TINYGRAD, "tinygrad not installed")
class TestEncoder(unittest.TestCase):

  def test_encoder_2d_skip_shapes(self):
    from volatile.ml.nnunet import PlainEncoder
    enc = PlainEncoder('2d', in_channels=1, base_channels=8, num_stages=4, block_type='residual')
    x = _randn(1, 1, 64, 64)
    skips = enc(x)
    self.assertEqual(len(skips), 4)
    # stage 0: stride 1 → (1,8,64,64)
    self.assertEqual(_shape(skips[0]), (1, 8,  64, 64))
    # stage 1: stride 2 → (1,16,32,32)
    self.assertEqual(_shape(skips[1]), (1, 16, 32, 32))
    # stage 2: stride 2 → (1,32,16,16)
    self.assertEqual(_shape(skips[2]), (1, 32, 16, 16))
    # stage 3: stride 2 → (1,64,8,8)
    self.assertEqual(_shape(skips[3]), (1, 64,  8,  8))

  def test_encoder_2d_plain_block(self):
    from volatile.ml.nnunet import PlainEncoder
    enc = PlainEncoder('2d', in_channels=1, base_channels=8, num_stages=3, block_type='plain')
    skips = enc(_randn(1, 1, 32, 32))
    self.assertEqual(len(skips), 3)

  def test_encoder_2d_se(self):
    from volatile.ml.nnunet import PlainEncoder
    enc = PlainEncoder('2d', in_channels=2, base_channels=8, num_stages=3,
                       block_type='residual', use_se=True)
    skips = enc(_randn(1, 2, 32, 32))
    self.assertEqual(len(skips), 3)

# ---------------------------------------------------------------------------
# NNUNet 2D
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TINYGRAD, "tinygrad not installed")
class TestNNUNet2D(unittest.TestCase):

  def test_basic_forward(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3,
                 block_type='residual', conv_op='2d')
    x = _randn(1, 1, 32, 32)
    y = net(x)
    self.assertEqual(_shape(y), (1, 2, 32, 32))

  def test_deep_supervision(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=4,
                 deep_supervision=True, conv_op='2d')
    x = _randn(1, 1, 64, 64)
    y = net(x)
    self.assertIsInstance(y, list)
    self.assertGreater(len(y), 1)
    # largest resolution first
    self.assertEqual(_shape(y[0]), (1, 2, 64, 64))

  def test_plain_block_type(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=1, num_classes=3, base_channels=8, num_stages=3,
                 block_type='plain', conv_op='2d')
    y = net(_randn(1, 1, 32, 32))
    self.assertEqual(_shape(y), (1, 3, 32, 32))

  def test_bottleneck_block_type(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3,
                 block_type='bottleneck', conv_op='2d')
    y = net(_randn(1, 1, 32, 32))
    self.assertEqual(_shape(y), (1, 2, 32, 32))

  def test_multi_channel_input(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=3, num_classes=2, base_channels=8, num_stages=3, conv_op='2d')
    y = net(_randn(2, 3, 32, 32))
    self.assertEqual(_shape(y), (2, 2, 32, 32))

  def test_from_patch_size_2d(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet.from_patch_size((128, 128), in_channels=1, num_classes=2,
                                  base_channels=8, num_stages=3)
    self.assertEqual(net.conv_op, '2d')
    y = net(_randn(1, 1, 32, 32))
    self.assertEqual(_shape(y), (1, 2, 32, 32))

# ---------------------------------------------------------------------------
# NNUNet 3D
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TINYGRAD, "tinygrad not installed")
class TestNNUNet3D(unittest.TestCase):

  def test_basic_forward_3d(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3, conv_op='3d')
    x = _randn(1, 1, 8, 16, 16)
    y = net(x)
    # should return (B, C, D, H, W) with spatial dims at full resolution
    self.assertEqual(_shape(y)[:2], (1, 2))

  def test_from_patch_size_3d(self):
    from volatile.ml.nnunet import NNUNet
    net = NNUNet.from_patch_size((16, 64, 64), in_channels=1, num_classes=2,
                                  base_channels=8, num_stages=3)
    self.assertEqual(net.conv_op, '3d')

# ---------------------------------------------------------------------------
# MultiTaskUNet
# ---------------------------------------------------------------------------

@unittest.skipUnless(_TINYGRAD, "tinygrad not installed")
class TestMultiTaskUNet(unittest.TestCase):

  def _make_config(self, conv_op='2d'):
    return {
      "in_channels": 1, "base_channels": 8, "num_stages": 3,
      "block_type": "residual", "conv_op": conv_op,
      "tasks": {
        "ink":    {"num_classes": 1, "activation": "sigmoid"},
        "label":  {"num_classes": 3, "activation": "softmax"},
        "recon":  {"num_classes": 1, "activation": "none"},
      }
    }

  def test_output_keys(self):
    from volatile.ml.nnunet import MultiTaskUNet
    net = MultiTaskUNet(self._make_config())
    out = net(_randn(1, 1, 32, 32))
    self.assertSetEqual(set(out.keys()), {"ink", "label", "recon"})

  def test_output_shapes_2d(self):
    from volatile.ml.nnunet import MultiTaskUNet
    net = MultiTaskUNet(self._make_config('2d'))
    out = net(_randn(1, 1, 32, 32))
    self.assertEqual(_shape(out["ink"]),   (1, 1, 32, 32))
    self.assertEqual(_shape(out["label"]), (1, 3, 32, 32))
    self.assertEqual(_shape(out["recon"]), (1, 1, 32, 32))

  def test_output_shapes_3d(self):
    from volatile.ml.nnunet import MultiTaskUNet
    cfg = self._make_config('3d')
    net = MultiTaskUNet(cfg)
    out = net(_randn(1, 1, 8, 16, 16))
    self.assertEqual(out["ink"].shape[1], 1)
    self.assertEqual(out["label"].shape[1], 3)

  def test_from_patch_size_auto_3d(self):
    from volatile.ml.nnunet import MultiTaskUNet
    cfg = {
      "in_channels": 1, "base_channels": 8, "num_stages": 3,
      "patch_size": (8, 32, 32),
      "tasks": {"seg": {"num_classes": 2, "activation": "softmax"}},
    }
    net = MultiTaskUNet(cfg)
    self.assertEqual(net.encoder.conv_op, '3d')

  def test_deep_supervision_task(self):
    from volatile.ml.nnunet import MultiTaskUNet
    cfg = {
      "in_channels": 1, "base_channels": 8, "num_stages": 4, "conv_op": "2d",
      "tasks": {
        "seg": {"num_classes": 2, "activation": "softmax", "deep_supervision": True},
        "cls": {"num_classes": 1, "activation": "sigmoid"},
      }
    }
    net = MultiTaskUNet(cfg)
    out = net(_randn(1, 1, 64, 64))
    self.assertIsInstance(out["seg"], list)
    self.assertGreater(len(out["seg"]), 1)
    self.assertNotIsInstance(out["cls"], list)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  if not _TINYGRAD:
    print("SKIP: tinygrad not installed — all tests skipped")
    sys.exit(0)
  unittest.main(verbosity=2)
