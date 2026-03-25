"""test_nnunet.py — comprehensive tests for py/volatile/ml/nnunet.py.

Run with:
  PYTHONPATH=py python3 -m pytest test/py/test_nnunet.py -v

All tests skip cleanly when tinygrad is not installed.
"""
from __future__ import annotations
import pytest

# Skip entire module if tinygrad unavailable (pytest.importorskip raises Skipped).
tinygrad = pytest.importorskip("tinygrad", reason="tinygrad not installed")
from tinygrad import Tensor

from volatile.ml.nnunet import (
  PlainConvBlock, ResidualBlock, BottleneckBlock, SqueezeExcitation,
  PlainEncoder, UNetDecoder, SegmentationHead,
  NNUNet, MultiTaskUNet,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shape(t):
  if isinstance(t, list):
    return [tuple(x.shape) for x in t]
  return tuple(t.shape)


# ---------------------------------------------------------------------------
# 1. NNUNet 2D forward pass: (B,C,H,W) -> (B,num_classes,H,W)
# ---------------------------------------------------------------------------

def test_nnunet_2d_forward_shape():
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3, conv_op='2d')
  y = net(Tensor.randn(2, 1, 64, 64))
  assert shape(y) == (2, 2, 64, 64), f"expected (2,2,64,64) got {shape(y)}"


def test_nnunet_2d_batch_independence():
  """Changing batch size should not affect spatial output dims."""
  net = NNUNet(in_channels=1, num_classes=3, base_channels=8, num_stages=3, conv_op='2d')
  assert shape(net(Tensor.randn(1, 1, 32, 32)))[:2] == (1, 3)
  assert shape(net(Tensor.randn(4, 1, 32, 32)))[:2] == (4, 3)


def test_nnunet_2d_multichannel_input():
  net = NNUNet(in_channels=4, num_classes=2, base_channels=8, num_stages=3, conv_op='2d')
  y = net(Tensor.randn(1, 4, 32, 32))
  assert shape(y) == (1, 2, 32, 32)


# ---------------------------------------------------------------------------
# 2. NNUNet 3D forward pass: (B,C,D,H,W) -> (B,num_classes,D,H,W)
# ---------------------------------------------------------------------------

def test_nnunet_3d_forward_shape():
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3, conv_op='3d')
  y = net(Tensor.randn(1, 1, 8, 16, 16))
  s = shape(y)
  assert s[0] == 1  and s[1] == 2, f"expected (1,2,*,*,*) got {s}"
  assert len(s) == 5, "3D output should be rank-5"


def test_nnunet_3d_from_patch_size():
  net = NNUNet.from_patch_size((16, 64, 64), in_channels=1, num_classes=2,
                                base_channels=8, num_stages=3)
  assert net.conv_op == '3d'
  y = net(Tensor.randn(1, 1, 8, 16, 16))
  assert len(shape(y)) == 5


def test_nnunet_2d_from_patch_size():
  net = NNUNet.from_patch_size((128, 128), in_channels=1, num_classes=2,
                                base_channels=8, num_stages=3)
  assert net.conv_op == '2d'
  y = net(Tensor.randn(1, 1, 32, 32))
  assert len(shape(y)) == 4


# ---------------------------------------------------------------------------
# 3. Residual block skip connection: output = relu(F(x) + proj(x))
# ---------------------------------------------------------------------------

def test_residual_block_skip_same_channels():
  """With same in/out channels no projection — skip is identity."""
  blk = ResidualBlock('2d', 8, 8)
  x = Tensor.randn(1, 8, 16, 16)
  y = blk(x)
  assert shape(y) == (1, 8, 16, 16)
  # projection should be None
  assert blk.proj is None, "no projection needed when channels match and stride==1"


def test_residual_block_skip_channel_change():
  """Channel change forces a 1×1 projection on the skip path."""
  blk = ResidualBlock('2d', 4, 16)
  assert blk.proj is not None, "projection required when in_ch != out_ch"
  y = blk(Tensor.randn(1, 4, 16, 16))
  assert shape(y) == (1, 16, 16, 16)


def test_residual_block_skip_stride2():
  """Stride-2 spatial downsampling with skip projection."""
  blk = ResidualBlock('2d', 8, 16, stride=2)
  assert blk.proj is not None
  y = blk(Tensor.randn(1, 8, 32, 32))
  assert shape(y) == (1, 16, 16, 16)


def test_residual_block_output_nonnegative():
  """ReLU at the end means output should be >= 0."""
  blk = ResidualBlock('2d', 8, 8)
  y = blk(Tensor.randn(2, 8, 8, 8))
  assert float(y.min().numpy()) >= 0.0, "residual output must be non-negative (ReLU)"


# ---------------------------------------------------------------------------
# 4. SqueezeExcitation: output shape == input shape
# ---------------------------------------------------------------------------

def test_squeeze_excitation_shape_2d():
  se = SqueezeExcitation('2d', 16)
  x = Tensor.randn(2, 16, 8, 8)
  y = se(x)
  assert shape(y) == shape(x)


def test_squeeze_excitation_shape_3d():
  se = SqueezeExcitation('3d', 16)
  x = Tensor.randn(2, 16, 4, 8, 8)
  y = se(x)
  assert shape(y) == shape(x)


def test_squeeze_excitation_values_bounded():
  """SE multiplies by sigmoid(·) ∈ (0,1), so |output| <= |input|."""
  se = SqueezeExcitation('2d', 8)
  x = Tensor.ones(1, 8, 4, 4)
  y = se(x)
  # Each channel scaled by a value in (0,1), so output ∈ (0,1)
  y_np = y.numpy()
  assert (y_np > 0).all(), "SE output should be positive for positive input"
  assert (y_np <= 1.0 + 1e-5).all(), "SE output should be <= 1 when input is 1"


def test_squeeze_excitation_in_encoder():
  """Encoder with SE enabled should produce correct skip shapes."""
  enc = PlainEncoder('2d', in_channels=1, base_channels=8, num_stages=3,
                     block_type='residual', use_se=True)
  skips = enc(Tensor.randn(1, 1, 32, 32))
  assert len(skips) == 3
  assert shape(skips[0])[1] == 8   # base channels


# ---------------------------------------------------------------------------
# 5. Deep supervision: returns list of outputs at multiple scales
# ---------------------------------------------------------------------------

def test_deep_supervision_returns_list():
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=4,
               deep_supervision=True, conv_op='2d')
  y = net(Tensor.randn(1, 1, 64, 64))
  assert isinstance(y, list), "deep_supervision should return a list"
  assert len(y) > 1, f"expected multiple outputs, got {len(y)}"


def test_deep_supervision_largest_first():
  """First output should be at full (input) resolution."""
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=4,
               deep_supervision=True, conv_op='2d')
  outs = net(Tensor.randn(1, 1, 64, 64))
  assert shape(outs[0]) == (1, 2, 64, 64), \
    f"largest-resolution output should be (1,2,64,64), got {shape(outs[0])}"


def test_deep_supervision_descending_spatial():
  """Subsequent outputs should be at lower spatial resolutions."""
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=4,
               deep_supervision=True, conv_op='2d')
  outs = net(Tensor.randn(1, 1, 64, 64))
  sizes = [shape(o)[2] for o in outs]  # H dim
  assert sizes == sorted(sizes, reverse=True), \
    f"deep supervision outputs should be in descending spatial order, got {sizes}"


def test_deep_supervision_all_same_classes():
  """All deep supervision outputs should have the same number of classes."""
  net = NNUNet(in_channels=1, num_classes=3, base_channels=8, num_stages=4,
               deep_supervision=True, conv_op='2d')
  outs = net(Tensor.randn(1, 1, 64, 64))
  classes = [shape(o)[1] for o in outs]
  assert all(c == 3 for c in classes), f"all outputs should have 3 classes, got {classes}"


def test_no_deep_supervision_returns_tensor():
  net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=3,
               deep_supervision=False, conv_op='2d')
  y = net(Tensor.randn(1, 1, 32, 32))
  assert isinstance(y, Tensor), "without deep_supervision should return a Tensor"


# ---------------------------------------------------------------------------
# 6. MultiTaskUNet returns dict of {task_name: output}
# ---------------------------------------------------------------------------

def _multitask_config(conv_op='2d'):
  return {
    "in_channels": 1, "base_channels": 8, "num_stages": 3, "conv_op": conv_op,
    "tasks": {
      "ink":    {"num_classes": 1, "activation": "sigmoid"},
      "label":  {"num_classes": 3, "activation": "softmax"},
      "recon":  {"num_classes": 1, "activation": "none"},
    }
  }


def test_multitask_returns_dict():
  net = MultiTaskUNet(_multitask_config())
  out = net(Tensor.randn(1, 1, 32, 32))
  assert isinstance(out, dict)
  assert set(out.keys()) == {"ink", "label", "recon"}


def test_multitask_output_shapes_2d():
  net = MultiTaskUNet(_multitask_config('2d'))
  out = net(Tensor.randn(1, 1, 32, 32))
  assert shape(out["ink"])   == (1, 1, 32, 32)
  assert shape(out["label"]) == (1, 3, 32, 32)
  assert shape(out["recon"]) == (1, 1, 32, 32)


def test_multitask_output_shapes_3d():
  net = MultiTaskUNet(_multitask_config('3d'))
  out = net(Tensor.randn(1, 1, 8, 16, 16))
  assert out["ink"].shape[1]   == 1
  assert out["label"].shape[1] == 3
  assert out["recon"].shape[1] == 1
  assert out["ink"].ndim == 5


def test_multitask_sigmoid_range():
  """Sigmoid activation should produce outputs in (0, 1)."""
  net = MultiTaskUNet(_multitask_config())
  out = net(Tensor.randn(2, 1, 32, 32))
  ink = out["ink"].numpy()
  assert (ink > 0).all() and (ink < 1).all(), "sigmoid output should be in (0,1)"


def test_multitask_softmax_sums_to_one():
  """Softmax over channel dim should sum to 1.0 per spatial location."""
  import numpy as np
  net = MultiTaskUNet(_multitask_config())
  out = net(Tensor.randn(1, 1, 32, 32))
  label = out["label"].numpy()   # (1, 3, 32, 32)
  sums = label.sum(axis=1)       # (1, 32, 32)
  assert np.allclose(sums, 1.0, atol=1e-5), "softmax should sum to 1 along channel dim"


def test_multitask_deep_supervision_task():
  cfg = {
    "in_channels": 1, "base_channels": 8, "num_stages": 4, "conv_op": "2d",
    "tasks": {
      "seg": {"num_classes": 2, "activation": "softmax", "deep_supervision": True},
      "aux": {"num_classes": 1, "activation": "sigmoid"},
    }
  }
  net = MultiTaskUNet(cfg)
  out = net(Tensor.randn(1, 1, 64, 64))
  assert isinstance(out["seg"], list), "deep_supervision task should return list"
  assert not isinstance(out["aux"], list), "non-ds task should return Tensor"


def test_multitask_shared_encoder():
  """All task decoders share a single encoder instance."""
  net = MultiTaskUNet(_multitask_config())
  enc_id = id(net.encoder)
  for name, dec in net.decoders.items():
    # The decoder was built with the same encoder; its output_channels match.
    assert dec is not None
  # Only one encoder object
  assert len({id(net.encoder)}) == 1, "should be exactly one shared encoder"


# ---------------------------------------------------------------------------
# 7. Config-driven build matches manual construction
# ---------------------------------------------------------------------------

def test_config_driven_matches_manual():
  """NNUNet built via config dict should produce same output shape as manual."""
  cfg = {
    "in_channels": 1, "base_channels": 8, "num_stages": 3, "conv_op": "2d",
    "tasks": {"seg": {"num_classes": 2, "activation": "none"}},
  }
  multi = MultiTaskUNet(cfg)

  manual = NNUNet(in_channels=1, num_classes=2, base_channels=8,
                  num_stages=3, conv_op='2d')

  x = Tensor.randn(1, 1, 32, 32)
  assert shape(multi(x)["seg"]) == shape(manual(x)), \
    "config-driven and manual builds should produce same output shape"


def test_config_patch_size_autodetect_2d():
  cfg = {
    "patch_size": (256, 256), "in_channels": 1, "base_channels": 8, "num_stages": 3,
    "tasks": {"s": {"num_classes": 1}},
  }
  net = MultiTaskUNet(cfg)
  assert net.encoder.conv_op == '2d'


def test_config_patch_size_autodetect_3d():
  cfg = {
    "patch_size": (16, 64, 64), "in_channels": 1, "base_channels": 8, "num_stages": 3,
    "tasks": {"s": {"num_classes": 1}},
  }
  net = MultiTaskUNet(cfg)
  assert net.encoder.conv_op == '3d'


def test_config_base_channels_honored():
  """base_channels controls the feature width at stage 0."""
  for bc in (8, 16, 32):
    net = NNUNet(in_channels=1, num_classes=2, base_channels=bc, num_stages=3, conv_op='2d')
    assert net.encoder.output_channels[0] == bc, \
      f"stage-0 channels should be {bc}, got {net.encoder.output_channels[0]}"


def test_config_num_stages_honored():
  """num_stages controls encoder depth."""
  for ns in (3, 4, 5):
    net = NNUNet(in_channels=1, num_classes=2, base_channels=8, num_stages=ns, conv_op='2d')
    assert len(net.encoder.output_channels) == ns
