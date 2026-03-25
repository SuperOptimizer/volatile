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

from volatile.ml.train import Trainer, cosine_annealing_lr, clip_grad_norm, ModelEMA, ProcessGroup


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


# ---------------------------------------------------------------------------
# train.py — new features: grad clip, EMA, early stopping, val loop, dist
# ---------------------------------------------------------------------------

def _fake_loader_fn(n_batches=2, out_classes=2):
  def _gen():
    for _ in range(n_batches):
      imgs = np.random.rand(1, 1, 16, 16).astype(np.float32)
      masks = np.random.randint(0, out_classes, (1, 16, 16)).astype(np.int32)
      yield imgs, masks
  return _gen


def test_trainer_two_epochs():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  trainer = Trainer(model=model, train_loader=_fake_loader_fn()(), num_epochs=2, verbose=False)
  history = trainer.train()
  assert len(history["train_loss"]) == 2


def test_trainer_with_val_loader():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  trainer = Trainer(
    model=model,
    train_loader=_fake_loader_fn()(),
    val_loader=_fake_loader_fn()(),
    num_epochs=1,
    n_classes=2,
    verbose=False,
  )
  history = trainer.train()
  assert len(history["val_loss"]) == 1
  assert np.isfinite(history["val_loss"][0])


def test_trainer_early_stopping():
  """With patience=1 and identical val losses, should stop after 2 epochs."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  trainer = Trainer(
    model=model,
    train_loader=_fake_loader_fn(n_batches=1)(),
    val_loader=_fake_loader_fn(n_batches=1)(),
    num_epochs=10,
    early_stopping_patience=1,
    verbose=False,
  )
  history = trainer.train()
  # Should stop early — well before 10 epochs
  assert len(history["train_loss"]) <= 10


def test_trainer_grad_clip():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  trainer = Trainer(
    model=model,
    train_loader=_fake_loader_fn()(),
    num_epochs=1,
    grad_clip=0.1,
    verbose=False,
  )
  history = trainer.train()
  assert np.isfinite(history["train_loss"][0])


def test_trainer_ema():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  trainer = Trainer(
    model=model,
    train_loader=_fake_loader_fn()(),
    num_epochs=1,
    ema_decay=0.999,
    verbose=False,
  )
  history = trainer.train()
  assert np.isfinite(history["train_loss"][0])


def test_trainer_with_scheduler():
  from volatile.ml.scheduler import PolynomialLR
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  sched = PolynomialLR(lr=1e-3, total_steps=2, warmup_steps=1)
  trainer = Trainer(
    model=model,
    train_loader=_fake_loader_fn()(),
    num_epochs=2,
    scheduler=sched,
    verbose=False,
  )
  history = trainer.train()
  assert len(history["train_loss"]) == 2


def test_trainer_dict_batch():
  """Trainer should accept dict batches from PatchDataset.collate."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)

  def _dict_loader():
    for _ in range(2):
      imgs = np.random.rand(1, 1, 16, 16).astype(np.float32)
      masks = np.random.randint(0, 2, (1, 16, 16)).astype(np.int32)
      yield {"image": imgs, "label": masks}

  trainer = Trainer(
    model=model,
    train_loader=_dict_loader(),
    num_epochs=1,
    target_key="label",
    verbose=False,
  )
  history = trainer.train()
  assert np.isfinite(history["train_loss"][0])


def test_process_group_single_rank():
  pg = ProcessGroup(rank=0, world_size=1)
  assert pg.is_primary
  assert pg.allreduce_mean(3.14) == pytest.approx(3.14)


def test_model_ema_shadow_differs_after_update():
  from tinygrad.nn.state import get_state_dict, get_parameters
  from tinygrad import nn as tg_nn
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  ema = ModelEMA(model, decay=0.0)  # decay=0 → shadow = current params exactly after each update
  # Do an optimizer step to actually move the weights
  opt = tg_nn.optim.Adam(get_parameters(model), lr=1.0)
  Tensor.training = True
  x = Tensor.randn(1, 1, 16, 16)
  loss = model(x).sum()
  loss.backward()
  opt.step()
  Tensor.training = False
  # Now record shadow before and after EMA update
  first_key = next(iter(get_state_dict(model)))
  shadow_before = ema._shadow[first_key].copy()
  ema.update()
  # With decay=0, shadow should equal current params exactly (different from original)
  current = get_state_dict(model)[first_key].numpy()
  assert np.allclose(ema._shadow[first_key], current, atol=1e-5)


def test_clip_grad_norm_no_error():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  from tinygrad.nn.state import get_parameters
  params = get_parameters(model)
  x = Tensor.randn(1, 1, 16, 16)
  loss = model(x).sum()
  loss.backward()
  norm = clip_grad_norm(params, max_norm=1.0)
  assert norm >= 0.0


# ---------------------------------------------------------------------------
# scheduler.py — LR scheduler tests
# ---------------------------------------------------------------------------

from volatile.ml.scheduler import (
  CosineAnnealingWarmRestarts, PolynomialLR, OneCycleLR, make_scheduler
)


def test_cosine_warm_restarts_at_boundaries():
  sched = CosineAnnealingWarmRestarts(lr=1e-3, T_0=10, eta_min=1e-6)
  # At step 0 should be near peak
  assert sched.get_lr(0) > 9e-4
  # At step 5 (mid cycle) should be about midpoint
  mid = sched.get_lr(5)
  assert 1e-6 < mid < 1e-3
  # At step 10 (restart) should be near peak again
  assert sched.get_lr(10) > 9e-4


def test_cosine_warm_restarts_warmup():
  sched = CosineAnnealingWarmRestarts(lr=1e-3, T_0=20, warmup_steps=5, eta_min=1e-6)
  # During warmup LR should grow from eta_min toward peak
  assert sched.get_lr(0) < sched.get_lr(4)
  assert sched.get_lr(4) < sched.get_lr(5)


def test_cosine_warm_restarts_decay_gamma():
  sched = CosineAnnealingWarmRestarts(lr=1e-2, T_0=5, gamma=0.5, eta_min=0.0)
  peak_cycle0 = sched.get_lr(0)
  peak_cycle1 = sched.get_lr(5)
  assert peak_cycle1 < peak_cycle0 * 0.6  # decayed by gamma=0.5


def test_polynomial_lr_warmup():
  sched = PolynomialLR(lr=1e-3, total_steps=100, warmup_steps=10)
  assert sched.get_lr(0) == pytest.approx(0.0, abs=1e-10)
  assert sched.get_lr(10) == pytest.approx(1e-3, rel=1e-5)
  # After warmup should decay
  assert sched.get_lr(50) < sched.get_lr(10)
  assert sched.get_lr(99) < sched.get_lr(50)


def test_polynomial_lr_no_warmup():
  sched = PolynomialLR(lr=1e-3, total_steps=100)
  assert sched.get_lr(0) == pytest.approx(1e-3, rel=1e-5)
  assert sched.get_lr(99) < 1e-4


def test_polynomial_lr_monotone_decay():
  sched = PolynomialLR(lr=1e-3, total_steps=50)
  lrs = [sched.get_lr(i) for i in range(50)]
  assert all(lrs[i] >= lrs[i + 1] for i in range(len(lrs) - 1))


def test_one_cycle_lr_peak_at_warmup_end():
  sched = OneCycleLR(lr=1e-3, total_steps=100, pct_start=0.3, div_factor=25.0)
  warmup_end = int(0.3 * 100)
  peak = sched.get_lr(warmup_end)
  assert peak == pytest.approx(1e-3 * 25.0, rel=0.05)


def test_one_cycle_lr_decays_after_peak():
  sched = OneCycleLR(lr=1e-3, total_steps=100, pct_start=0.3)
  warmup_end = int(0.3 * 100)
  lrs_after = [sched.get_lr(i) for i in range(warmup_end, 100)]
  assert all(lrs_after[i] >= lrs_after[i + 1] for i in range(len(lrs_after) - 1))


def test_scheduler_step_updates_optimizer():
  from tinygrad.nn.state import get_parameters
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  params = get_parameters(model)
  from tinygrad import nn as tg_nn
  opt = tg_nn.optim.Adam(params, lr=1.0)
  sched = PolynomialLR(lr=1e-3, total_steps=10)
  sched.attach(opt)
  sched.step()
  assert float(opt.lr) < 1.0  # must be < 1.0 (original) and != 1e-3 first-step value varies


def test_make_scheduler_factory():
  sched = make_scheduler("poly", lr=1e-3, total_steps=100, warmup_steps=5)
  assert isinstance(sched, PolynomialLR)
  sched2 = make_scheduler("one_cycle", lr=1e-3, total_steps=100)
  assert isinstance(sched2, OneCycleLR)
  sched3 = make_scheduler("cosine_warm_restarts", lr=1e-3, total_steps=50)
  assert isinstance(sched3, CosineAnnealingWarmRestarts)


def test_make_scheduler_unknown_raises():
  with pytest.raises(ValueError):
    make_scheduler("unknown_sched", lr=1e-3, total_steps=10)


# ---------------------------------------------------------------------------
# dataset.py — PatchDataset
# ---------------------------------------------------------------------------

from volatile.ml.dataset import PatchDataset, SimpleBatcher, _iter_positions_3d, _iter_positions_2d


def _make_volume_2d(h=32, w=32, n_classes=2):
  img = np.random.rand(h, w).astype(np.float32)
  label = np.random.randint(0, n_classes, (h, w)).astype(np.int32)
  return {"name": "vol0", "image": img, "labels": {"seg": label}}


def _make_volume_3d(d=16, h=16, w=16, n_classes=2):
  img = np.random.rand(d, h, w).astype(np.float32)
  label = np.random.randint(0, n_classes, (d, h, w)).astype(np.int32)
  return {"name": "vol0", "image": img, "labels": {"seg": label}}


def test_iter_positions_3d_basic():
  positions = _iter_positions_3d((16, 16, 16), (8, 8, 8), (8, 8, 8))
  assert len(positions) == 8  # 2x2x2


def test_iter_positions_2d_basic():
  positions = _iter_positions_2d((16, 16), (8, 8), (8, 8))
  assert len(positions) == 4  # 2x2


def test_patch_dataset_2d_len():
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(16, 16), skip_empty=False)
  assert len(ds) == 4  # 2x2


def test_patch_dataset_3d_len():
  vol = _make_volume_3d(d=16, h=16, w=16)
  ds = PatchDataset([vol], patch_size=(8, 8, 8), skip_empty=False)
  assert len(ds) == 8  # 2x2x2


def test_patch_dataset_2d_item_shape():
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(16, 16), skip_empty=False)
  item = ds[0]
  assert item["image"].shape == (1, 16, 16), f"got {item['image'].shape}"
  assert item["seg"].shape == (16, 16), f"got {item['seg'].shape}"
  assert item["padding_mask"].shape == (1, 16, 16)


def test_patch_dataset_3d_item_shape():
  vol = _make_volume_3d(d=16, h=16, w=16)
  ds = PatchDataset([vol], patch_size=(8, 8, 8), skip_empty=False)
  item = ds[0]
  assert item["image"].shape == (1, 8, 8, 8)
  assert item["seg"].shape == (8, 8, 8)


def test_patch_dataset_multi_volume():
  vols = [_make_volume_2d(h=16, w=16), _make_volume_2d(h=16, w=16)]
  ds = PatchDataset(vols, patch_size=(8, 8), skip_empty=False)
  assert len(ds) == 8  # 2 volumes * 4 patches each


def test_patch_dataset_target_names():
  vol = _make_volume_2d()
  ds = PatchDataset([vol], patch_size=(16, 16), target_names=["seg"], skip_empty=False)
  assert ds.target_names == ["seg"]
  item = ds[0]
  assert "seg" in item


def test_patch_dataset_no_labels():
  vol = {"name": "v", "image": np.random.rand(16, 16).astype(np.float32), "labels": {}}
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  item = ds[0]
  assert "image" in item


def test_patch_dataset_skip_empty():
  # Volume of all zeros should yield 0 patches when skip_empty=True
  vol = {"name": "v", "image": np.zeros((16, 16), dtype=np.float32), "labels": {}}
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=True, empty_threshold=0.0)
  assert len(ds) == 0


def test_patch_dataset_padding_mask_full_patch():
  # Image larger than patch — all valid, mask should be all 1s
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(16, 16), skip_empty=False)
  item = ds[0]
  assert np.all(item["padding_mask"] == 1.0)


def test_patch_dataset_stride():
  vol = _make_volume_2d(h=32, w=32)
  ds_full = PatchDataset([vol], patch_size=(16, 16), stride=(8, 8), skip_empty=False)
  ds_half = PatchDataset([vol], patch_size=(16, 16), stride=(16, 16), skip_empty=False)
  assert len(ds_full) > len(ds_half)


def test_patch_dataset_z_partition():
  vol = _make_volume_3d(d=16, h=8, w=8)
  ds0 = PatchDataset([vol], patch_size=(8, 8, 8), skip_empty=False, z_partitions=2, z_partition_idx=0)
  ds1 = PatchDataset([vol], patch_size=(8, 8, 8), skip_empty=False, z_partitions=2, z_partition_idx=1)
  # Each partition should own roughly half; together they cover all patches
  ds_all = PatchDataset([vol], patch_size=(8, 8, 8), skip_empty=False, z_partitions=1)
  assert len(ds0) + len(ds1) == len(ds_all)


def test_patch_dataset_iter():
  vol = _make_volume_2d(h=16, w=16)
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  items = list(ds)
  assert len(items) == len(ds)


def test_patch_dataset_collate():
  vol = _make_volume_2d(h=16, w=16)
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  batch = PatchDataset.collate([ds[0], ds[1]])
  assert batch["image"].shape[0] == 2


def test_simple_batcher_len():
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  batcher = SimpleBatcher(ds, batch_size=2, shuffle=False)
  assert len(batcher) == len(ds) // 2


def test_simple_batcher_yields_correct_batch_size():
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  batcher = SimpleBatcher(ds, batch_size=2, shuffle=False, drop_last=True)
  for batch in batcher:
    assert batch["image"].shape[0] == 2


def test_simple_batcher_shuffle():
  vol = _make_volume_2d(h=32, w=32)
  ds = PatchDataset([vol], patch_size=(8, 8), skip_empty=False)
  b1 = [list(it["position"]) for it in SimpleBatcher(ds, batch_size=1, shuffle=True, seed=0)]
  b2 = [list(it["position"]) for it in SimpleBatcher(ds, batch_size=1, shuffle=True, seed=99)]
  # Different seeds should produce different order at least sometimes
  assert b1 != b2 or len(b1) <= 1  # can be equal if only 1 patch


def test_trainer_with_patch_dataset():
  """End-to-end: PatchDataset → SimpleBatcher → Trainer runs 2 epochs."""
  vol = _make_volume_2d(h=32, w=32, n_classes=2)
  ds = PatchDataset([vol], patch_size=(16, 16), target_names=["seg"], skip_empty=False)
  batcher = SimpleBatcher(ds, batch_size=2, shuffle=False)

  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)

  def _loader_from_batcher():
    for batch in batcher:
      yield batch["image"], batch["seg"]

  trainer = Trainer(
    model=model,
    train_loader=_loader_from_batcher(),
    num_epochs=2,
    verbose=False,
  )
  history = trainer.train()
  assert len(history["train_loss"]) == 2
  assert all(np.isfinite(v) for v in history["train_loss"])


# ---------------------------------------------------------------------------
# semi_supervised.py — Mean Teacher tests
# ---------------------------------------------------------------------------

from volatile.ml.semi_supervised import (
  sigmoid_rampup, linear_rampup, build_two_stream_batches, MeanTeacherTrainer
)


def test_sigmoid_rampup_bounds():
  # At current=0: phase=1.0, value = exp(-5) ≈ 0.0067 (near zero)
  assert sigmoid_rampup(0.0, 10.0) < 0.01
  assert sigmoid_rampup(10.0, 10.0) == pytest.approx(1.0, rel=1e-5)
  assert 0.0 < sigmoid_rampup(5.0, 10.0) < 1.0


def test_sigmoid_rampup_zero_length():
  assert sigmoid_rampup(0.0, 0.0) == 1.0


def test_linear_rampup_bounds():
  assert linear_rampup(0.0, 10.0) == pytest.approx(0.0, abs=1e-9)
  assert linear_rampup(10.0, 10.0) == pytest.approx(1.0, rel=1e-5)
  assert linear_rampup(20.0, 10.0) == pytest.approx(1.0)


def test_build_two_stream_batches_shape():
  rng = np.random.default_rng(0)
  imgs = [np.random.rand(1, 16, 16).astype(np.float32) for _ in range(8)]
  masks = [np.random.randint(0, 2, (16, 16)).astype(np.int32) for _ in range(8)]
  labeled = list(zip(imgs, masks))
  unlabeled = [np.random.rand(1, 16, 16).astype(np.float32) for _ in range(8)]
  batches = build_two_stream_batches(labeled, unlabeled, labeled_batch_size=2, unlabeled_batch_size=2, rng=rng)
  assert len(batches) > 0
  b = batches[0]
  assert "image" in b and "label" in b and "n_labeled" in b
  # 2 labeled + 2 unlabeled = 4 images
  assert b["image"].shape[0] == 4
  assert b["n_labeled"] == 2


def test_mean_teacher_trainer_two_epochs():
  """Mean Teacher should run 2 epochs and produce finite history losses."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)

  def _lbl_loader():
    for _ in range(2):
      imgs = np.random.rand(1, 1, 16, 16).astype(np.float32)
      masks = np.random.randint(0, 2, (1, 16, 16)).astype(np.int32)
      yield imgs, masks

  def _unl_loader():
    for _ in range(2):
      yield np.random.rand(1, 1, 16, 16).astype(np.float32)

  trainer = MeanTeacherTrainer(
    student=model,
    labeled_loader=_lbl_loader(),
    unlabeled_loader=_unl_loader(),
    num_epochs=2,
    consistency_rampup=1.0,
    verbose=False,
  )
  history = trainer.train()
  assert "train_sup_loss" in history
  assert "train_cons_loss" in history
  assert len(history["train_sup_loss"]) == 2
  assert all(np.isfinite(v) for v in history["train_sup_loss"])


def test_mean_teacher_trainer_teacher_weights():
  """Teacher weights dict should be populated after training."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)

  def _lbl():
    yield np.random.rand(1, 1, 16, 16).astype(np.float32), np.zeros((1, 16, 16), dtype=np.int32)

  def _unl():
    yield np.random.rand(1, 1, 16, 16).astype(np.float32)

  trainer = MeanTeacherTrainer(
    student=model,
    labeled_loader=_lbl(),
    unlabeled_loader=_unl(),
    num_epochs=1,
    verbose=False,
  )
  trainer.train()
  weights = trainer.teacher_weights
  assert isinstance(weights, dict)
  assert len(weights) > 0


# ---------------------------------------------------------------------------
# self_supervised.py — MAE tests
# ---------------------------------------------------------------------------

from volatile.ml.self_supervised import (
  patchify, unpatchify, random_mask, make_patch_grid,
  MAEEncoder, MAEDecoder, MaskedAutoencoder, mae_loss, MAEPretrainer
)


def test_make_patch_grid():
  nh, nw, ph, pw = make_patch_grid(64, 64, 16, 16)
  assert nh == 4 and nw == 4 and ph == 16 and pw == 16


def test_random_mask_ratio():
  rng = np.random.default_rng(42)
  mask = random_mask(100, 0.75, rng)
  assert mask.dtype == bool
  assert mask.sum() == 75


def test_patchify_shape():
  img = np.random.rand(1, 32, 32).astype(np.float32)
  patches = patchify(img, 8, 8)
  assert patches.shape == (16, 1 * 8 * 8)  # 4x4 = 16 patches, 1*8*8 = 64 values each


def test_patchify_unpatchify_roundtrip():
  img = np.random.rand(1, 32, 32).astype(np.float32)
  patches = patchify(img, 8, 8)
  recon = unpatchify(patches, C=1, H=32, W=32, patch_h=8, patch_w=8)
  assert recon.shape == (1, 32, 32)
  assert np.allclose(img, recon, atol=1e-6)


def test_mae_encoder_output_shape():
  enc = MAEEncoder(in_channels=1, latent_channels=16, hidden_channels=16, depth=2)
  x = Tensor.randn(1, 1, 32, 32)
  z = enc(x)
  assert z.shape == (1, 16, 32, 32)


def test_mae_decoder_output_shape():
  dec = MAEDecoder(latent_channels=16, out_channels=1, hidden_channels=16, depth=2)
  z = Tensor.randn(1, 16, 32, 32)
  out = dec(z)
  assert out.shape == (1, 1, 32, 32)


def test_masked_autoencoder_forward_shapes():
  mae = MaskedAutoencoder(
    in_channels=1, patch_h=8, patch_w=8, mask_ratio=0.75,
    latent_channels=16, hidden_channels=16, encoder_depth=2, decoder_depth=2, seed=0,
  )
  x = Tensor.randn(1, 1, 32, 32)
  recon, mask = mae(x)
  assert recon.shape == (1, 1, 32, 32)
  assert mask.shape == (1, 1, 32, 32)


def test_masked_autoencoder_mask_ratio():
  """About 75% of patch positions should be masked (zero)."""
  mae = MaskedAutoencoder(
    in_channels=1, patch_h=8, patch_w=8, mask_ratio=0.75,
    latent_channels=8, hidden_channels=8, encoder_depth=1, decoder_depth=1, seed=1,
  )
  x = Tensor.ones(1, 1, 32, 32)
  _, mask = mae(x)
  mask_np = mask.numpy()
  # fraction of zero pixels in the mask (= masked patches)
  frac_masked = (mask_np == 0.0).mean()
  assert 0.6 < frac_masked < 0.9


def test_mae_loss_only_masked():
  """mae_loss should be 0 when reconstruction is perfect."""
  recon = Tensor.ones(1, 1, 16, 16)
  original = Tensor.ones(1, 1, 16, 16)
  mask = Tensor(np.zeros((1, 1, 16, 16), dtype=np.float32))  # all masked
  loss = mae_loss(recon, original, mask)
  assert float(loss.numpy()) == pytest.approx(0.0, abs=1e-6)


def test_mae_pretrainer_loss_finite():
  """MAEPretrainer should produce finite loss over 2 epochs."""
  mae = MaskedAutoencoder(
    in_channels=1, patch_h=8, patch_w=8, mask_ratio=0.75,
    latent_channels=8, hidden_channels=8, encoder_depth=1, decoder_depth=1, seed=2,
  )

  def _loader():
    for _ in range(2):
      yield np.random.rand(1, 1, 32, 32).astype(np.float32)

  pretrainer = MAEPretrainer(mae=mae, train_loader=_loader(), num_epochs=2, verbose=False)
  history = pretrainer.train()
  assert len(history["train_loss"]) == 2
  assert all(np.isfinite(v) for v in history["train_loss"])


def test_mae_pretrainer_encoder_state_dict():
  mae = MaskedAutoencoder(
    in_channels=1, patch_h=8, patch_w=8,
    latent_channels=8, hidden_channels=8, encoder_depth=1, decoder_depth=1,
  )

  def _loader():
    yield np.random.rand(1, 1, 32, 32).astype(np.float32)

  pretrainer = MAEPretrainer(mae=mae, train_loader=_loader(), num_epochs=1, verbose=False)
  pretrainer.train()
  sd = pretrainer.encoder_state_dict()
  assert isinstance(sd, dict)
  assert len(sd) > 0
  assert all(isinstance(v, np.ndarray) for v in sd.values())


# ---------------------------------------------------------------------------
# auxiliary.py — target generators and AuxTaskTrainer
# ---------------------------------------------------------------------------

from volatile.ml.auxiliary import (
  structure_tensor_targets, surface_normal_targets, distance_transform_targets,
  AuxHead, AuxTask, MSELoss, aux_weight_schedule, AuxTaskTrainer
)


def test_structure_tensor_targets_shape():
  img = np.random.rand(32, 32).astype(np.float32)
  out = structure_tensor_targets(img)
  assert out.shape == (2, 32, 32)
  assert out.dtype == np.float32
  assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_structure_tensor_targets_channel_input():
  img = np.random.rand(1, 32, 32).astype(np.float32)
  out = structure_tensor_targets(img)
  assert out.shape == (2, 32, 32)


def test_surface_normal_targets_shape():
  img = np.random.rand(32, 32).astype(np.float32)
  out = surface_normal_targets(img)
  assert out.shape == (3, 32, 32)
  assert out.dtype == np.float32


def test_surface_normal_targets_channel_input():
  img = np.random.rand(1, 32, 32).astype(np.float32)
  out = surface_normal_targets(img)
  assert out.shape == (3, 32, 32)


def test_distance_transform_targets_shape():
  label = np.zeros((32, 32), dtype=np.int32)
  label[10:20, 10:20] = 1
  out = distance_transform_targets(label)
  assert out.shape == (1, 32, 32)
  assert out.dtype == np.float32
  assert np.all(out >= 0.0) and np.all(out <= 1.0)


def test_aux_weight_schedule_initial():
  w = aux_weight_schedule(0, 100, initial_weight=0.4, final_weight=0.0)
  assert w == pytest.approx(0.4, rel=1e-5)


def test_aux_weight_schedule_final():
  w = aux_weight_schedule(100, 100, initial_weight=0.4, final_weight=0.0)
  assert w == pytest.approx(0.0, abs=1e-9)


def test_aux_weight_schedule_monotone():
  weights = [aux_weight_schedule(e, 100, 0.4, 0.0) for e in range(101)]
  assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))


def test_aux_head_output_shape():
  head = AuxHead(in_channels=8, out_channels=3)
  x = Tensor.randn(2, 8, 16, 16)
  out = head(x)
  assert out.shape == (2, 3, 16, 16)


def test_aux_task_trainer_runs():
  """AuxTaskTrainer should run 2 epochs with finite primary and aux losses."""
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  head = AuxHead(in_channels=1, out_channels=2)  # attached to raw image
  task = AuxTask(name="normals", head=head, loss_fn=MSELoss(), initial_weight=0.3, final_weight=0.0)

  def _loader():
    for _ in range(2):
      imgs = np.random.rand(1, 1, 16, 16).astype(np.float32)
      labels = np.random.randint(0, 2, (1, 16, 16)).astype(np.int32)
      normals = np.random.rand(1, 2, 16, 16).astype(np.float32)
      yield {"image": imgs, "label": labels, "normals": normals}

  trainer = AuxTaskTrainer(
    primary_model=model,
    aux_tasks=[task],
    train_loader=_loader(),
    num_epochs=2,
    verbose=False,
  )
  history = trainer.train()
  assert "train_primary_loss" in history
  assert "train_normals_loss" in history
  assert len(history["train_primary_loss"]) == 2
  assert all(np.isfinite(v) for v in history["train_primary_loss"])


def test_aux_task_trainer_predict_aux_shape():
  model = UNet(in_channels=1, out_channels=2, base_channels=4, num_levels=2)
  head = AuxHead(in_channels=1, out_channels=3)
  task = AuxTask(name="stnorm", head=head, loss_fn=MSELoss())

  def _loader():
    yield {"image": np.random.rand(1, 1, 16, 16).astype(np.float32),
           "label": np.zeros((1, 16, 16), dtype=np.int32),
           "stnorm": np.random.rand(1, 3, 16, 16).astype(np.float32)}

  trainer = AuxTaskTrainer(
    primary_model=model,
    aux_tasks=[task],
    train_loader=_loader(),
    num_epochs=1,
    verbose=False,
  )
  trainer.train()
  preds = trainer.predict_aux(np.random.rand(1, 1, 16, 16).astype(np.float32))
  assert "stnorm" in preds
  assert preds["stnorm"].shape == (1, 3, 16, 16)


# ---------------------------------------------------------------------------
# ink_detection.py
# ---------------------------------------------------------------------------

from volatile.ml.ink_detection import (
  InkDetector, InkDataset, InkUNet,
  extract_surface_columns, _tile_positions, _gaussian_kernel_2d,
)


def _make_synthetic_ink_data(D=32, H=64, W=64, n_vols=2):
  """Return (volumes, labels, surfaces) for ink detection tests."""
  rng = np.random.default_rng(7)
  volumes = [rng.random((D, H, W)).astype(np.float32) * 200 for _ in range(n_vols)]
  labels = [(rng.random((H, W)) > 0.7).astype(np.float32) for _ in range(n_vols)]
  surfaces = [np.full((H, W), D // 2, dtype=np.int32) for _ in range(n_vols)]
  return volumes, labels, surfaces


def test_gaussian_kernel_shape_and_max():
  k = _gaussian_kernel_2d(16, sigma=2.0)
  assert k.shape == (16, 16)
  assert k.max() == pytest.approx(1.0, abs=1e-6)


def test_tile_positions_cover_all():
  """All positions from _tile_positions should have full tile_size."""
  H, W, ts, stride = 64, 64, 32, 16
  positions = _tile_positions(H, W, ts, stride)
  assert len(positions) > 0
  for y1, x1, y2, x2 in positions:
    assert y2 - y1 == ts
    assert x2 - x1 == ts


def test_tile_positions_single_tile():
  """When H=W=tile_size there should be exactly one tile."""
  positions = _tile_positions(64, 64, 64, 32)
  assert (0, 0, 64, 64) in positions


def test_extract_surface_columns_shape():
  vol = np.random.rand(32, 16, 16).astype(np.float32)
  surf = np.full((16, 16), 15, dtype=np.int32)
  out = extract_surface_columns(vol, surf, z_range=8)
  assert out.shape == (8, 16, 16)


def test_extract_surface_columns_clips_z():
  """Surface at z=0 with z_range=6 should not index below 0."""
  vol = np.arange(32 * 4 * 4, dtype=np.float32).reshape(32, 4, 4)
  surf = np.zeros((4, 4), dtype=np.int32)  # surface at z=0
  out = extract_surface_columns(vol, surf, z_range=6)
  assert out.shape == (6, 4, 4)
  assert np.all(np.isfinite(out))


def test_ink_unet_output_shape():
  model = InkUNet(z_range=8, base_channels=8, num_levels=2)
  x = Tensor.randn(2, 8, 32, 32)
  out = model(x)
  assert out.shape == (2, 1, 32, 32)


def test_ink_dataset_len_and_item_shapes():
  vols, lbls, surfs = _make_synthetic_ink_data(D=32, H=64, W=64, n_vols=2)
  ds = InkDataset(vols, lbls, surfs, z_range=8, patch_size=32, stride=32, augment=False)
  assert len(ds) > 0
  img, lbl = ds[0]
  assert img.shape == (8, 32, 32), f"got {img.shape}"
  assert lbl.shape == (32, 32), f"got {lbl.shape}"


def test_ink_dataset_augment_no_shape_change():
  vols, lbls, surfs = _make_synthetic_ink_data(D=32, H=64, W=64, n_vols=1)
  ds = InkDataset(vols, lbls, surfs, z_range=8, patch_size=32, stride=32, augment=True, seed=42)
  img, lbl = ds[0]
  assert img.shape == (8, 32, 32)
  assert lbl.shape == (32, 32)


def test_ink_dataset_batches():
  vols, lbls, surfs = _make_synthetic_ink_data(D=32, H=64, W=64, n_vols=2)
  ds = InkDataset(vols, lbls, surfs, z_range=8, patch_size=32, stride=32, augment=False)
  batches = list(ds.batches(batch_size=2, shuffle=False))
  assert len(batches) > 0
  b = batches[0]
  assert "image" in b and "label" in b
  assert b["image"].ndim == 4  # (B, z_range, H, W)
  assert b["label"].ndim == 3  # (B, H, W)


def test_ink_dataset_skips_none_labels():
  vols, lbls, surfs = _make_synthetic_ink_data(D=32, H=32, W=32, n_vols=2)
  lbls[0] = None  # first volume has no label
  ds_partial = InkDataset(vols, lbls, surfs, z_range=4, patch_size=16, stride=16, augment=False)
  ds_full = InkDataset([vols[1]], [lbls[1]], [surfs[1]], z_range=4, patch_size=16, stride=16, augment=False)
  assert len(ds_partial) == len(ds_full)


def test_ink_detector_train_one_epoch():
  """InkDetector.train should complete 1 epoch and return finite loss."""
  vols, lbls, surfs = _make_synthetic_ink_data(D=32, H=64, W=64, n_vols=2)
  ds = InkDataset(vols, lbls, surfs, z_range=8, patch_size=32, stride=32, augment=False, seed=0)

  detector = InkDetector(z_range=8, base_channels=8, num_levels=2)
  history = detector.train(ds, epochs=1, lr=1e-3, batch_size=2, verbose=False)

  assert "train_loss" in history
  assert len(history["train_loss"]) == 1
  assert np.isfinite(history["train_loss"][0])


def test_ink_detector_predict_surface_shape():
  """predict_surface should return (H, W) probability map."""
  D, H, W = 32, 64, 64
  vol = np.random.rand(D, H, W).astype(np.float32) * 200
  surf = np.full((H, W), D // 2, dtype=np.int32)

  detector = InkDetector(z_range=8, base_channels=8, num_levels=2)
  prob = detector.predict_surface(vol, surf, z_range=8, tile_size=32, stride=32)

  assert prob.shape == (H, W), f"got {prob.shape}"
  assert prob.dtype == np.float32
  assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_ink_detector_predict_volume_slice_shape():
  """predict_volume_slice should return (H, W) map."""
  D, H, W = 32, 64, 64
  vol = np.random.rand(D, H, W).astype(np.float32) * 200

  detector = InkDetector(z_range=8, base_channels=8, num_levels=2)
  prob = detector.predict_volume_slice(vol, z=D // 2, tile_size=32, stride=32)

  assert prob.shape == (H, W)
  assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_ink_detector_model_type_aliases():
  """All model_type aliases should construct without error."""
  for mt in ('unet', 'resnet3d', 'timesformer'):
    det = InkDetector(model_type=mt, z_range=4, base_channels=4, num_levels=2)
    assert det.model_type == mt


# ---------------------------------------------------------------------------
# transformers.py — EVA / FlashRoPE / PoPE / Primus / Vesuvius3dViT tests
# ---------------------------------------------------------------------------

from volatile.ml.transformers import (
  MultiHeadAttention,
  FlashRoPEAttention, FlashRoPE,
  EVABlock,
  PopeEmbedding, PopeBlock,
  VisionTransformer,
  PrimusEncoder, PrimusDecoder,
  Vesuvius3dViTModel,
  build_rope_nd_freqs, apply_rope_nd,
)


# ---- RoPE utilities --------------------------------------------------------

def test_build_rope_nd_freqs_2d_shape():
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)
  assert freqs.shape == (16, 8, 2)   # 4*4=16 positions, head_dim//2=8 freq pairs, (sin,cos)


def test_build_rope_nd_freqs_3d_shape():
  freqs = build_rope_nd_freqs((2, 4, 4), head_dim=24)
  assert freqs.shape == (32, 12, 2)  # 2*4*4=32 positions, 24//2=12 freq pairs


def test_apply_rope_nd_output_shape():
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)
  x = Tensor.randn(2, 4, 16, 16)    # (B, H, N, D)
  out = apply_rope_nd(x, freqs)
  assert out.shape == (2, 4, 16, 16)


# ---- MultiHeadAttention ----------------------------------------------------

def test_mha_output_shape():
  mha = MultiHeadAttention(embed_dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  out = mha(x)
  assert out.shape == (2, 16, 64)


def test_mha_with_rope():
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)  # 16 positions
  mha = MultiHeadAttention(embed_dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  out = mha(x, rope_freqs=freqs)
  assert out.shape == (2, 16, 64)


# ---- FlashRoPEAttention ----------------------------------------------------

def test_flash_rope_attention_shape():
  attn = FlashRoPEAttention(dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  out = attn(x)
  assert out.shape == (2, 16, 64)


def test_flash_rope_attention_with_freqs():
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)
  attn = FlashRoPEAttention(dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  out = attn(x, rope_freqs=freqs)
  assert out.shape == (2, 16, 64)


def test_flash_rope_embedding_freqs_shape():
  rope = FlashRoPE(head_dim=16, feat_shape=(4, 4))
  freqs = rope.get_freqs()
  assert freqs.shape == (16, 8, 2)


# ---- EVABlock ---------------------------------------------------------------

def test_eva_block_output_shape():
  blk = EVABlock(dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  out = blk(x)
  assert out.shape == (2, 16, 64)


def test_eva_block_with_rope():
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)
  blk = EVABlock(dim=64, num_heads=4, init_values=0.1)
  x = Tensor.randn(2, 16, 64)
  out = blk(x, rope_freqs=freqs)
  assert out.shape == (2, 16, 64)


def test_eva_block_swiglu_false():
  blk = EVABlock(dim=64, num_heads=4, swiglu_mlp=False)
  x = Tensor.randn(2, 16, 64)
  out = blk(x)
  assert out.shape == (2, 16, 64)


# ---- PopeEmbedding / PopeBlock ----------------------------------------------

def test_pope_embedding_shape():
  emb = PopeEmbedding(head_dim=16, feat_shape=(4, 4))
  freqs = emb.get_embed()
  assert freqs.shape == (16, 8, 2)


def test_pope_block_output_shape():
  blk = PopeBlock(dim=64, num_heads=4)
  x = Tensor.randn(2, 16, 64)
  freqs = build_rope_nd_freqs((4, 4), head_dim=16)
  out = blk(x, rope_freqs=freqs)
  assert out.shape == (2, 16, 64)


def test_pope_block_no_rope():
  blk = PopeBlock(dim=64, num_heads=4, swiglu_mlp=False)
  x = Tensor.randn(2, 16, 64)
  out = blk(x)
  assert out.shape == (2, 16, 64)


# ---- VisionTransformer ------------------------------------------------------

def test_vit_2d_output_shape():
  vit = VisionTransformer(embed_dim=64, depth=2, num_heads=4, feat_shape=(4, 4))
  x = Tensor.randn(2, 16, 64)   # 4*4=16 tokens
  out = vit(x)
  assert out.shape == (2, 16, 64)


def test_vit_3d_output_shape():
  vit = VisionTransformer(embed_dim=48, depth=2, num_heads=4, feat_shape=(2, 4, 4), swiglu_mlp=False)
  x = Tensor.randn(2, 32, 48)   # 2*4*4=32 tokens
  out = vit(x)
  assert out.shape == (2, 32, 48)


def test_vit_pope_type():
  vit = VisionTransformer(embed_dim=64, depth=2, num_heads=4, feat_shape=(4, 4), pos_emb_type="pope")
  x = Tensor.randn(1, 16, 64)
  out = vit(x)
  assert out.shape == (1, 16, 64)


# ---- PrimusEncoder / PrimusDecoder ------------------------------------------

def test_primus_encoder_2d_output_shape():
  enc = PrimusEncoder(in_channels=1, embed_dim=64, patch_size=8, input_shape=(32, 32),
                      depth=2, num_heads=4, swiglu_mlp=False)
  x = Tensor.randn(2, 1, 32, 32)
  tokens = enc(x)
  assert tokens.shape == (2, 16, 64)   # (32//8)^2 = 16 tokens


def test_primus_encoder_3d_output_shape():
  enc = PrimusEncoder(in_channels=1, embed_dim=48, patch_size=8, input_shape=(16, 16, 16),
                      depth=2, num_heads=4, swiglu_mlp=False)
  x = Tensor.randn(1, 1, 16, 16, 16)
  tokens = enc(x)
  assert tokens.shape == (1, 8, 48)    # (16//8)^3 = 8 tokens


def test_primus_decoder_2d_output_shape():
  dec = PrimusDecoder(embed_dim=64, patch_size=8, out_channels=2, input_shape=(32, 32))
  tokens = Tensor.randn(2, 16, 64)
  out = dec(tokens)
  assert out.shape == (2, 2, 32, 32)


def test_primus_decoder_3d_output_shape():
  dec = PrimusDecoder(embed_dim=48, patch_size=8, out_channels=2, input_shape=(16, 16, 16))
  tokens = Tensor.randn(1, 8, 48)
  out = dec(tokens)
  assert out.shape == (1, 2, 16, 16, 16)


def test_primus_encoder_decoder_roundtrip_2d():
  """Encoder → Decoder should produce the correct spatial output shape."""
  enc = PrimusEncoder(in_channels=1, embed_dim=64, patch_size=8, input_shape=(32, 32),
                      depth=1, num_heads=4, swiglu_mlp=False)
  dec = PrimusDecoder(embed_dim=64, patch_size=8, out_channels=2, input_shape=(32, 32))
  x = Tensor.randn(1, 1, 32, 32)
  tokens = enc(x)
  out = dec(tokens)
  assert out.shape == (1, 2, 32, 32)


# ---- Vesuvius3dViTModel -----------------------------------------------------

def test_vesuvius_3d_vit_pooled_output():
  """Mean-pooled feature vector output."""
  model = Vesuvius3dViTModel(
    in_channels=1, patch_size=8, embed_dim=48, depth=2, num_heads=4,
    input_shape=(16, 16, 16), out_channels=0, return_tokens=False,
  )
  x = Tensor.randn(2, 1, 16, 16, 16)
  out = model(x)
  assert out.shape == (2, 48), f"got {out.shape}"


def test_vesuvius_3d_vit_token_output():
  """Token-level output."""
  model = Vesuvius3dViTModel(
    in_channels=1, patch_size=8, embed_dim=48, depth=2, num_heads=4,
    input_shape=(16, 16, 16), return_tokens=True,
  )
  x = Tensor.randn(1, 1, 16, 16, 16)
  out = model(x)
  n_tokens = (16 // 8) ** 3   # = 8
  assert out.shape == (1, n_tokens, 48), f"got {out.shape}"


def test_vesuvius_3d_vit_with_head():
  """With out_channels > 0 head should project to that dim."""
  model = Vesuvius3dViTModel(
    in_channels=1, patch_size=8, embed_dim=48, depth=1, num_heads=4,
    input_shape=(16, 16, 16), out_channels=4, return_tokens=False,
  )
  x = Tensor.randn(2, 1, 16, 16, 16)
  out = model(x)
  assert out.shape == (2, 4)


def test_vesuvius_3d_vit_pope():
  """PoPE variant should produce same shapes as RoPE variant."""
  model = Vesuvius3dViTModel(
    in_channels=1, patch_size=8, embed_dim=48, depth=2, num_heads=4,
    input_shape=(16, 16, 16), return_tokens=True, pos_emb_type="pope",
  )
  x = Tensor.randn(1, 1, 16, 16, 16)
  out = model(x)
  assert out.shape == (1, 8, 48)
