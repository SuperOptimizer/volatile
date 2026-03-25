from __future__ import annotations

import numpy as np
import pytest

tinygrad = pytest.importorskip("tinygrad", reason="tinygrad not installed")
from tinygrad import Tensor


# ---------------------------------------------------------------------------
# model tests
# ---------------------------------------------------------------------------

from volatile.ml.model import ConvBlock, UNet


def test_conv_block_output_shape():
  block = ConvBlock(1, 32)
  x = Tensor.randn(1, 1, 64, 64)
  y = block(x)
  assert y.shape == (1, 32, 64, 64)


def test_conv_block_preserves_spatial():
  block = ConvBlock(8, 16, kernel_size=3)
  x = Tensor.randn(2, 8, 32, 32)
  y = block(x)
  assert y.shape == (2, 16, 32, 32), f"expected (2,16,32,32), got {y.shape}"


def test_unet_output_shape_default():
  model = UNet(in_channels=1, out_channels=4, base_channels=8, num_levels=3)
  x = Tensor.randn(1, 1, 64, 64)
  y = model(x)
  assert y.shape == (1, 4, 64, 64), f"expected (1,4,64,64) got {y.shape}"


def test_unet_output_shape_multi_channel_input():
  model = UNet(in_channels=3, out_channels=2, base_channels=8, num_levels=2)
  x = Tensor.randn(1, 3, 32, 32)
  y = model(x)
  assert y.shape == (1, 2, 32, 32), f"expected (1,2,32,32) got {y.shape}"


def test_unet_batch_size_two():
  model = UNet(in_channels=1, out_channels=4, base_channels=8, num_levels=2)
  x = Tensor.randn(2, 1, 32, 32)
  y = model(x)
  assert y.shape == (2, 4, 32, 32)


def test_unet_single_level():
  model = UNet(in_channels=1, out_channels=1, base_channels=4, num_levels=1)
  x = Tensor.randn(1, 1, 16, 16)
  y = model(x)
  assert y.shape == (1, 1, 16, 16)


def test_unet_output_is_finite():
  model = UNet(in_channels=1, out_channels=4, base_channels=8, num_levels=2)
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x).numpy()
  assert np.all(np.isfinite(y)), "UNet output contains NaN or Inf"


# ---------------------------------------------------------------------------
# tiled_infer tests
# ---------------------------------------------------------------------------

from volatile.ml.infer import tiled_infer


class _IdentityModel:
  """Trivial model: returns input unchanged (out_ch == in_ch)."""
  def __call__(self, x: Tensor) -> Tensor:
    return x


class _ConstantModel:
  """Returns a fixed constant output with a specified number of channels."""
  def __init__(self, out_ch: int, value: float = 1.0):
    self.out_ch = out_ch
    self.value = value

  def __call__(self, x: Tensor) -> Tensor:
    b, _, h, w = x.shape
    return Tensor.ones(b, self.out_ch, h, w) * self.value


def test_tiled_infer_2d_shape():
  vol = np.random.rand(128, 128).astype(np.float32)
  out = tiled_infer(_IdentityModel(), vol, tile_h=64, tile_w=64, overlap=16)
  # 2-D input → (out_ch, H, W) with out_ch=in_ch=1
  assert out.shape == (1, 128, 128), f"expected (1,128,128) got {out.shape}"


def test_tiled_infer_3d_zchw_shape():
  # ndim=3 is interpreted as (Z, H, W) with C=1; output is (Z, out_ch, H, W)
  vol = np.random.rand(3, 64, 64).astype(np.float32)  # (Z, H, W)
  out = tiled_infer(_IdentityModel(), vol, tile_h=32, tile_w=32, overlap=8)
  assert out.shape == (3, 1, 64, 64), f"expected (3,1,64,64) got {out.shape}"


def test_tiled_infer_3d_shape():
  vol = np.random.rand(4, 64, 64).astype(np.float32)  # (Z, H, W)
  out = tiled_infer(_IdentityModel(), vol, tile_h=32, tile_w=32, overlap=8)
  assert out.shape == (4, 1, 64, 64), f"expected (4,1,64,64) got {out.shape}"


def test_tiled_infer_output_channels():
  vol = np.random.rand(64, 64).astype(np.float32)
  model = _ConstantModel(out_ch=4, value=2.0)
  out = tiled_infer(model, vol, tile_h=32, tile_w=32, overlap=8)
  assert out.shape[0] == 4, f"expected 4 output channels, got {out.shape[0]}"


def test_tiled_infer_constant_model_value():
  # constant model returning 1.0 — blended output should also be ~1.0
  vol = np.zeros((64, 64), dtype=np.float32)
  model = _ConstantModel(out_ch=1, value=1.0)
  out = tiled_infer(model, vol, tile_h=32, tile_w=32, overlap=8)
  assert np.allclose(out, 1.0, atol=1e-4), f"expected uniform 1.0, max diff={np.max(np.abs(out-1.0))}"


def test_tiled_infer_no_tile_boundary_discontinuity():
  # Blending should ensure smooth output for a spatially uniform constant model
  vol = np.zeros((64, 64), dtype=np.float32)
  model = _ConstantModel(out_ch=1, value=5.0)
  out = tiled_infer(model, vol, tile_h=32, tile_w=32, overlap=16)
  assert np.allclose(out, 5.0, atol=0.01), "tile-boundary discontinuity detected"


def test_tiled_infer_small_volume():
  # volume smaller than tile — should still work (clamped to vol size)
  vol = np.random.rand(16, 16).astype(np.float32)
  model = _ConstantModel(out_ch=2, value=1.0)
  out = tiled_infer(model, vol, tile_h=32, tile_w=32, overlap=8)
  assert out.shape == (2, 16, 16)


def test_tiled_infer_unet():
  model = UNet(in_channels=1, out_channels=4, base_channels=8, num_levels=2)
  vol = np.random.rand(64, 64).astype(np.float32)
  out = tiled_infer(model, vol, tile_h=32, tile_w=32, overlap=8)
  assert out.shape == (4, 64, 64)
  assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# model.py — ResBlock / SEBlock / ResUNet
# ---------------------------------------------------------------------------

from volatile.ml.model import ResBlock, SEBlock, ResUNet


def test_se_block_output_shape():
  se = SEBlock(channels=16, reduction_ratio=4)
  x = Tensor.randn(2, 16, 8, 8)
  y = se(x)
  assert y.shape == (2, 16, 8, 8), f"SEBlock shape mismatch: {y.shape}"


def test_se_block_same_channels_as_input():
  se = SEBlock(channels=8, reduction_ratio=2)
  x = Tensor.ones(1, 8, 4, 4)
  y = se(x)
  assert y.shape == x.shape


def test_res_block_same_channels():
  rb = ResBlock(in_ch=16, out_ch=16)
  x = Tensor.randn(1, 16, 8, 8)
  y = rb(x)
  assert y.shape == (1, 16, 8, 8)


def test_res_block_channel_projection():
  rb = ResBlock(in_ch=8, out_ch=16)
  x = Tensor.randn(1, 8, 8, 8)
  y = rb(x)
  assert y.shape == (1, 16, 8, 8)


def test_res_block_with_se():
  rb = ResBlock(in_ch=8, out_ch=8, use_se=True)
  x = Tensor.randn(1, 8, 8, 8)
  y = rb(x)
  assert y.shape == (1, 8, 8, 8)


def test_res_block_output_finite():
  rb = ResBlock(in_ch=4, out_ch=4)
  x = Tensor.randn(2, 4, 16, 16)
  y = rb(x).numpy()
  assert np.all(np.isfinite(y))


def test_resunet_output_shape_default():
  model = ResUNet(in_channels=1, out_channels=4, base_channels=8, num_levels=2)
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x)
  assert y.shape == (1, 4, 32, 32), f"expected (1,4,32,32) got {y.shape}"


def test_resunet_output_shape_with_se():
  model = ResUNet(in_channels=1, out_channels=2, base_channels=8, num_levels=2, use_se=True)
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x)
  assert y.shape == (1, 2, 32, 32)


def test_resunet_output_is_finite():
  model = ResUNet(in_channels=1, out_channels=4, base_channels=8, num_levels=2)
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x).numpy()
  assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# config.py — build_model
# ---------------------------------------------------------------------------

from volatile.ml.config import build_model


def test_build_model_conv():
  model = build_model({"in_channels": 1, "out_channels": 2, "base_channels": 8,
                       "num_levels": 2, "block_type": "conv", "use_se": False})
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x)
  assert y.shape == (1, 2, 32, 32)


def test_build_model_residual():
  model = build_model({"in_channels": 1, "out_channels": 4, "base_channels": 8,
                       "num_levels": 2, "block_type": "residual", "use_se": False})
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x)
  assert y.shape == (1, 4, 32, 32)


def test_build_model_residual_se():
  model = build_model({"in_channels": 1, "out_channels": 2, "base_channels": 8,
                       "num_levels": 2, "block_type": "residual", "use_se": True})
  x = Tensor.randn(1, 1, 32, 32)
  y = model(x)
  assert y.shape == (1, 2, 32, 32)


def test_build_model_unknown_block_type():
  import pytest
  with pytest.raises(ValueError):
    build_model({"block_type": "unknown"})


# ---------------------------------------------------------------------------
# loss.py — DiceLoss / CrossEntropyLoss / DiceCELoss
# ---------------------------------------------------------------------------

from volatile.ml.loss import DiceLoss, CrossEntropyLoss, DiceCELoss


def test_dice_loss_perfect_prediction():
  """Dice loss should be ~0 when pred perfectly matches target."""
  pred = Tensor(np.array([[[[10.0, 10.0], [-10.0, -10.0]]]], dtype=np.float32))  # (1,1,2,2)
  target = Tensor(np.array([[[[1.0, 1.0], [0.0, 0.0]]]], dtype=np.float32))      # (1,1,2,2) one-hot
  loss = DiceLoss()(pred, target)
  assert float(loss.numpy()) < 0.1, f"Expected near-zero Dice loss, got {loss.numpy()}"


def test_dice_loss_shape_scalar():
  pred = Tensor.randn(2, 4, 16, 16)
  target = Tensor(np.random.randint(0, 4, (2, 16, 16)).astype(np.int32))
  loss = DiceLoss()(pred, target)
  assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"


def test_cross_entropy_loss_shape_scalar():
  pred = Tensor.randn(2, 4, 16, 16)
  target = Tensor(np.random.randint(0, 4, (2, 16, 16)).astype(np.int32))
  loss = CrossEntropyLoss()(pred, target)
  assert loss.shape == ()


def test_dice_ce_loss_is_finite():
  pred = Tensor.randn(2, 4, 16, 16)
  target = Tensor(np.random.randint(0, 4, (2, 16, 16)).astype(np.int32))
  loss = DiceCELoss()(pred, target)
  assert np.isfinite(float(loss.numpy()))


# ---------------------------------------------------------------------------
# augment.py — numpy augmentation pipeline
# ---------------------------------------------------------------------------

from volatile.ml.augment import (
  Compose, RandomFlip, RandomRotate90, RandomRotate, RandomScale,
  ElasticDeformation, GaussianNoise, BrightnessJitter, ContrastJitter,
  GammaCorrection, default_train_augmentation,
)


def _make_image(c=1, h=32, w=32):
  return np.random.rand(c, h, w).astype(np.float32)


def test_random_flip_output_shape():
  img = _make_image()
  mask = np.random.randint(0, 2, (32, 32)).astype(np.int32)
  aug = RandomFlip()
  out_img, out_mask = aug(img, mask)
  assert out_img.shape == img.shape
  assert out_mask.shape == mask.shape


def test_random_rotate90_output_shape():
  img = _make_image(h=32, w=32)
  aug = RandomRotate90(p=1.0)
  out_img, _ = aug(img, None)
  assert out_img.shape == img.shape


def test_random_rotate_output_shape():
  img = _make_image()
  aug = RandomRotate(p=1.0)
  out_img, _ = aug(img, None)
  assert out_img.shape == img.shape


def test_random_scale_output_shape():
  img = _make_image(h=32, w=32)
  aug = RandomScale(p=1.0)
  out_img, _ = aug(img, None)
  assert out_img.shape == img.shape


def test_elastic_deformation_output_shape():
  img = _make_image()
  mask = np.random.randint(0, 2, (32, 32)).astype(np.int32)
  aug = ElasticDeformation(p=1.0)
  out_img, out_mask = aug(img, mask)
  assert out_img.shape == img.shape
  assert out_mask.shape == mask.shape


def test_gaussian_noise_output_shape():
  img = _make_image()
  aug = GaussianNoise(std=0.01, p=1.0)
  out_img, _ = aug(img, None)
  assert out_img.shape == img.shape


def test_compose_pipeline():
  img = _make_image()
  mask = np.zeros((32, 32), dtype=np.int32)
  pipeline = default_train_augmentation()
  out_img, out_mask = pipeline(img, mask)
  assert out_img.shape == img.shape


def test_compose_preserves_channel_dim():
  img = _make_image(c=3)
  aug = Compose([RandomFlip(), GaussianNoise(p=1.0)])
  out_img, _ = aug(img, None)
  assert out_img.shape == (3, 32, 32)


def test_augment_none_mask_passthrough():
  img = _make_image()
  aug = Compose([RandomFlip(), RandomRotate90(), GaussianNoise(p=1.0)])
  out_img, out_mask = aug(img, None)
  assert out_img.shape == img.shape
  assert out_mask is None


# ---------------------------------------------------------------------------
# train.py — Trainer smoke test (no real data)
# ---------------------------------------------------------------------------

from volatile.ml.train import Trainer, cosine_annealing_lr


def test_cosine_annealing_lr():
  lr0 = 1e-3
  assert abs(cosine_annealing_lr(lr0, 0, 100) - lr0) < 1e-7
  assert cosine_annealing_lr(lr0, 100, 100) < 1e-5
  for e in range(1, 100):
    assert cosine_annealing_lr(lr0, e, 100) > 0


def test_trainer_one_epoch():
  """Trainer should complete a single epoch without errors."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)

  def _fake_loader():
    for _ in range(2):
      imgs = np.random.rand(1, 1, 16, 16).astype(np.float32)
      masks = np.random.randint(0, 2, (1, 16, 16)).astype(np.int32)
      yield imgs, masks

  trainer = Trainer(
    model=model,
    train_loader=_fake_loader(),
    num_epochs=1,
    learning_rate=1e-3,
    verbose=False,
  )
  history = trainer.train()
  assert "train_loss" in history
  assert len(history["train_loss"]) == 1
  assert np.isfinite(history["train_loss"][0])
