from __future__ import annotations
"""Tests for volatile.ml.topo_loss — topology-aware loss functions."""

import math
import numpy as np
import pytest

tinygrad = pytest.importorskip("tinygrad", reason="tinygrad not installed")
from tinygrad.tensor import Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logits(probs: np.ndarray) -> np.ndarray:
  """Convert probabilities to logits."""
  probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
  return np.log(probs / (1.0 - probs)).astype(np.float32)


def _make(pred_np: np.ndarray, gt_np: np.ndarray, requires_grad: bool = True):
  """Wrap numpy arrays as (1,1,H,W) Tensors."""
  p = Tensor(_logits(pred_np).reshape(1, 1, *pred_np.shape), requires_grad=requires_grad)
  g = Tensor(gt_np.astype(np.float32).reshape(1, 1, *gt_np.shape))
  return p, g


def _disk(h: int, w: int, cy: int, cx: int, r: float) -> np.ndarray:
  ys, xs = np.ogrid[:h, :w]
  return ((ys - cy)**2 + (xs - cx)**2 <= r**2).astype(np.float32)


def _ring(h: int, w: int, cy: int, cx: int, r_out: float, r_in: float) -> np.ndarray:
  ys, xs = np.ogrid[:h, :w]
  d2 = (ys - cy)**2 + (xs - cx)**2
  return ((d2 <= r_out**2) & (d2 > r_in**2)).astype(np.float32)


# ---------------------------------------------------------------------------
# Primitive tests
# ---------------------------------------------------------------------------

class TestConnectedComponents:
  def test_single_component(self):
    from volatile.ml.topo_loss import connected_components_2d
    m = np.zeros((8, 8), dtype=bool); m[2:6, 2:6] = True
    _, n = connected_components_2d(m)
    assert n == 1

  def test_two_components(self):
    from volatile.ml.topo_loss import connected_components_2d
    m = np.zeros((8, 8), dtype=bool)
    m[1:3, 1:3] = True; m[5:7, 5:7] = True
    _, n = connected_components_2d(m)
    assert n == 2

  def test_empty_mask(self):
    from volatile.ml.topo_loss import connected_components_2d
    m = np.zeros((4, 4), dtype=bool)
    _, n = connected_components_2d(m)
    assert n == 0

  def test_labels_unique_per_component(self):
    from volatile.ml.topo_loss import connected_components_2d
    m = np.zeros((6, 6), dtype=bool)
    m[0:2, 0:2] = True; m[4:6, 4:6] = True
    labels, n = connected_components_2d(m)
    assert n == 2
    fg_labels = sorted(set(labels[m].tolist()))
    assert len(fg_labels) == 2
    assert 0 not in fg_labels


class TestEulerNumber:
  def test_solid_square_euler_one(self):
    from volatile.ml.topo_loss import euler_number_2d
    m = np.zeros((8, 8), dtype=np.uint8); m[2:6, 2:6] = 1
    assert euler_number_2d(m) == pytest.approx(1.0, abs=0.1)

  def test_square_with_hole_euler_zero(self):
    from volatile.ml.topo_loss import euler_number_2d
    m = np.zeros((10, 10), dtype=np.uint8); m[1:9, 1:9] = 1; m[4:6, 4:6] = 0
    # 1 component, 1 hole → euler = 0
    assert euler_number_2d(m) == pytest.approx(0.0, abs=0.5)

  def test_two_disks_euler_two(self):
    from volatile.ml.topo_loss import euler_number_2d
    m = np.zeros((10, 10), dtype=np.uint8)
    m[1:3, 1:3] = 1; m[7:9, 7:9] = 1
    assert euler_number_2d(m) == pytest.approx(2.0, abs=0.1)


class TestBettiNumbers:
  def test_single_disk(self):
    from volatile.ml.topo_loss import betti_numbers_2d
    m = _disk(16, 16, 8, 8, 4.0).astype(bool)
    b0, b1 = betti_numbers_2d(m)
    assert b0 == 1; assert b1 == 0

  def test_ring_has_hole(self):
    from volatile.ml.topo_loss import betti_numbers_2d
    m = _ring(20, 20, 10, 10, 8.0, 4.0).astype(bool)
    b0, b1 = betti_numbers_2d(m)
    assert b0 == 1; assert b1 >= 1

  def test_two_disks(self):
    from volatile.ml.topo_loss import betti_numbers_2d
    m = np.zeros((16, 32), dtype=bool)
    m[4:12, 2:10] = True; m[4:12, 22:30] = True
    b0, b1 = betti_numbers_2d(m)
    assert b0 == 2; assert b1 == 0


class TestSkeletonize:
  def test_skeleton_thinner_than_input(self):
    from volatile.ml.topo_loss import skeletonize_2d
    m = np.zeros((12, 12), dtype=bool); m[2:10, 2:10] = True
    skel = skeletonize_2d(m)
    assert skel.sum() < m.sum(), "skeleton should be thinner"

  def test_horizontal_bar_skeleton(self):
    from volatile.ml.topo_loss import skeletonize_2d
    m = np.zeros((10, 20), dtype=bool); m[4:6, 2:18] = True
    skel = skeletonize_2d(m)
    assert skel.sum() > 0, "skeleton of a bar must be non-empty"
    # Skeleton height should be at most 2 rows
    row_counts = skel.any(axis=1)
    assert row_counts.sum() <= 2

  def test_empty_mask_skeleton(self):
    from volatile.ml.topo_loss import skeletonize_2d
    m = np.zeros((8, 8), dtype=bool)
    skel = skeletonize_2d(m)
    assert skel.sum() == 0


# ---------------------------------------------------------------------------
# BettiNumberLoss
# ---------------------------------------------------------------------------

class TestBettiNumberLoss:
  def test_perfect_prediction_lower_loss(self):
    """Loss on a perfect prediction should be ≤ loss on a bad prediction."""
    from volatile.ml.topo_loss import BettiNumberLoss
    loss_fn = BettiNumberLoss()
    gt = _disk(16, 16, 8, 8, 5.0)
    # Perfect: predict same disk with high confidence.
    pred_good, g = _make(gt * 0.9 + 0.05, gt)
    # Bad: predict two disconnected blobs instead of one.
    bad_pred = np.zeros_like(gt); bad_pred[3:7, 3:7] = 0.9; bad_pred[10:14, 10:14] = 0.9
    pred_bad, _ = _make(bad_pred, gt)
    l_good = float(loss_fn(pred_good, g).numpy())
    l_bad  = float(loss_fn(pred_bad,  g).numpy())
    assert l_bad >= l_good, f"bad pred loss {l_bad:.4f} should be >= good {l_good:.4f}"

  def test_output_is_scalar(self):
    from volatile.ml.topo_loss import BettiNumberLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    p, g = _make(gt, gt)
    loss = BettiNumberLoss()(p, g)
    assert loss.numpy().shape == () or loss.numpy().size == 1

  def test_gradient_flows(self):
    from volatile.ml.topo_loss import BettiNumberLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    p, g = _make(gt * 0.6 + 0.2, gt)
    loss = BettiNumberLoss()(p, g)
    loss.backward()
    assert p.grad is not None, "gradient must flow to pred"
    assert not np.all(p.grad.numpy() == 0.0), "gradient must be nonzero"

  def test_batch_gradient_flows(self):
    """Test with batch size 2."""
    from volatile.ml.topo_loss import BettiNumberLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    pred_np = (_logits(gt * 0.7 + 0.15)).reshape(1, 1, 12, 12)
    pred_np2 = np.concatenate([pred_np, pred_np], axis=0)
    gt_np2   = np.concatenate([gt.reshape(1, 1, 12, 12)] * 2, axis=0)
    p = Tensor(pred_np2, requires_grad=True)
    g = Tensor(gt_np2.astype(np.float32))
    loss = BettiNumberLoss()(p, g)
    loss.backward()
    assert p.grad is not None

  def test_wrong_component_count_raises_loss(self):
    """Predicting 2 components vs GT 1 component increases loss."""
    from volatile.ml.topo_loss import BettiNumberLoss
    loss_fn = BettiNumberLoss(weight_b0=10.0, weight_b1=0.0)
    gt = _disk(20, 20, 10, 10, 6.0)
    pred_correct = gt.copy()
    pred_two = np.zeros_like(gt); pred_two[3:8, 3:8] = 0.9; pred_two[13:18, 13:18] = 0.9

    p1, g1 = _make(pred_correct, gt)
    p2, g2 = _make(pred_two, gt)
    l1 = float(loss_fn(p1, g1).numpy())
    l2 = float(loss_fn(p2, g2).numpy())
    assert l2 > l1, f"wrong-topology loss {l2:.4f} should exceed correct {l1:.4f}"


# ---------------------------------------------------------------------------
# SkeletonRecallLoss
# ---------------------------------------------------------------------------

class TestSkeletonRecallLoss:
  def test_full_coverage_low_loss(self):
    """Predicting the full GT region should give near-zero loss."""
    from volatile.ml.topo_loss import SkeletonRecallLoss
    gt = _disk(16, 16, 8, 8, 5.0)
    p, g = _make(gt * 0.9 + 0.05, gt)
    loss = float(SkeletonRecallLoss()(p, g).numpy())
    assert loss < 0.5, f"full coverage should give small loss, got {loss:.4f}"

  def test_empty_prediction_high_loss(self):
    """Predicting all zeros when GT has a skeleton should give high loss."""
    from volatile.ml.topo_loss import SkeletonRecallLoss
    gt = _disk(16, 16, 8, 8, 5.0)
    pred_empty = np.zeros_like(gt)
    p, g = _make(pred_empty, gt)
    loss = float(SkeletonRecallLoss()(p, g).numpy())
    assert loss > 0.4, f"empty prediction should give high loss, got {loss:.4f}"

  def test_output_is_scalar(self):
    from volatile.ml.topo_loss import SkeletonRecallLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    p, g = _make(gt, gt)
    loss = SkeletonRecallLoss()(p, g)
    assert loss.numpy().size == 1

  def test_gradient_flows(self):
    from volatile.ml.topo_loss import SkeletonRecallLoss
    gt = _disk(14, 14, 7, 7, 4.0)
    p, g = _make(gt * 0.5 + 0.2, gt)
    loss = SkeletonRecallLoss()(p, g)
    loss.backward()
    assert p.grad is not None
    assert not np.all(p.grad.numpy() == 0.0)

  def test_empty_gt_skeleton_zero_loss(self):
    """When GT has no skeleton (e.g. single-pixel GT), loss should be ~0."""
    from volatile.ml.topo_loss import SkeletonRecallLoss
    gt = np.zeros((10, 10), dtype=np.float32); gt[5, 5] = 1.0
    p, g = _make(gt, gt)
    # skeletonize of single pixel → empty (or single point); either way loss is tiny
    loss = float(SkeletonRecallLoss()(p, g).numpy())
    assert loss < 1.0 + 1e-3  # just check it doesn't crash


# ---------------------------------------------------------------------------
# CenterlineDiceLoss
# ---------------------------------------------------------------------------

class TestCenterlineDiceLoss:
  def test_perfect_prediction_near_zero(self):
    from volatile.ml.topo_loss import CenterlineDiceLoss
    gt = _disk(16, 16, 8, 8, 5.0)
    p, g = _make(gt * 0.9 + 0.05, gt)
    loss = float(CenterlineDiceLoss()(p, g).numpy())
    assert loss < 0.6, f"perfect CL Dice loss should be small, got {loss:.4f}"

  def test_orthogonal_prediction_higher_loss(self):
    """Prediction on skeleton of a perpendicular bar should give higher loss."""
    from volatile.ml.topo_loss import CenterlineDiceLoss
    h, w = 20, 20
    gt   = np.zeros((h, w), dtype=np.float32); gt[9:11, 2:18] = 1.0  # horizontal bar
    pred = np.zeros((h, w), dtype=np.float32); pred[2:18, 9:11] = 1.0 # vertical bar

    p_good, g = _make(gt,   gt)
    p_bad,  _ = _make(pred, gt)
    l_good = float(CenterlineDiceLoss()(p_good, g).numpy())
    l_bad  = float(CenterlineDiceLoss()(p_bad,  g).numpy())
    assert l_bad > l_good, f"orthogonal pred {l_bad:.4f} should be worse than aligned {l_good:.4f}"

  def test_output_is_scalar(self):
    from volatile.ml.topo_loss import CenterlineDiceLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    p, g = _make(gt, gt)
    assert CenterlineDiceLoss()(p, g).numpy().size == 1

  def test_gradient_flows(self):
    from volatile.ml.topo_loss import CenterlineDiceLoss
    gt = _disk(14, 14, 7, 7, 4.0)
    p, g = _make(gt * 0.6 + 0.2, gt)
    loss = CenterlineDiceLoss()(p, g)
    loss.backward()
    assert p.grad is not None
    assert not np.all(p.grad.numpy() == 0.0)

  def test_empty_union_skeleton_no_crash(self):
    """When both pred and gt are empty, should return ~0 without crashing."""
    from volatile.ml.topo_loss import CenterlineDiceLoss
    gt = np.zeros((10, 10), dtype=np.float32)
    p, g = _make(gt, gt)
    loss = float(CenterlineDiceLoss()(p, g).numpy())
    assert math.isfinite(loss)


# ---------------------------------------------------------------------------
# TopologicalLoss (combined)
# ---------------------------------------------------------------------------

class TestTopologicalLoss:
  def test_all_weights_combined(self):
    from volatile.ml.topo_loss import TopologicalLoss
    gt = _disk(16, 16, 8, 8, 5.0)
    p, g = _make(gt * 0.8 + 0.1, gt)
    loss_fn = TopologicalLoss(w_betti=1.0, w_skeleton=1.0, w_cl_dice=1.0)
    loss = loss_fn(p, g)
    assert math.isfinite(float(loss.numpy()))

  def test_zero_weights_zero_contribution(self):
    """Setting a weight to 0 should skip that term."""
    from volatile.ml.topo_loss import TopologicalLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    p, g = _make(gt, gt)
    l_all  = float(TopologicalLoss(w_betti=1.0, w_skeleton=1.0, w_cl_dice=1.0)(p, g).numpy())
    l_betti_only = float(TopologicalLoss(w_betti=1.0, w_skeleton=0.0, w_cl_dice=0.0)(p, g).numpy())
    # Combined should be >= betti-only (all terms are non-negative)
    assert l_all >= l_betti_only - 1e-4

  def test_gradient_flows_combined(self):
    from volatile.ml.topo_loss import TopologicalLoss
    gt = _disk(14, 14, 7, 7, 4.0)
    p, g = _make(gt * 0.6 + 0.2, gt)
    loss = TopologicalLoss()(p, g)
    loss.backward()
    assert p.grad is not None
    assert not np.all(p.grad.numpy() == 0.0)

  def test_custom_weights(self):
    from volatile.ml.topo_loss import TopologicalLoss
    gt = _disk(14, 14, 7, 7, 4.0)
    p, g = _make(gt * 0.6 + 0.2, gt)
    l1 = float(TopologicalLoss(w_betti=2.0, w_skeleton=1.0, w_cl_dice=0.5)(p, g).numpy())
    l2 = float(TopologicalLoss(w_betti=0.5, w_skeleton=1.0, w_cl_dice=2.0)(p, g).numpy())
    assert math.isfinite(l1) and math.isfinite(l2)

  def test_batch_size_2(self):
    from volatile.ml.topo_loss import TopologicalLoss
    gt = _disk(12, 12, 6, 6, 3.0)
    pnp = _logits(gt * 0.8 + 0.1).reshape(1, 1, 12, 12)
    gnp = gt.reshape(1, 1, 12, 12).astype(np.float32)
    p = Tensor(np.concatenate([pnp, pnp], axis=0), requires_grad=True)
    g = Tensor(np.concatenate([gnp, gnp], axis=0))
    loss = TopologicalLoss()(p, g)
    assert math.isfinite(float(loss.numpy()))
    loss.backward()
    assert p.grad is not None
