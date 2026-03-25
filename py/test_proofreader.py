"""test_proofreader.py — unit tests for Proofreader (headless, no napari required).

Run with:  python -m pytest py/test_proofreader.py
       or: python py/test_proofreader.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from volatile.proofreader import Proofreader, _extract_patch, _patch_count

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pred(shape=(128, 128, 128)) -> str:
  """Write a random float32 prediction .npy; return path."""
  tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
  np.save(tmp.name, (np.random.rand(*shape) > 0.5).astype(np.float32))
  return tmp.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPatchHelpers(unittest.TestCase):

  def _vol(self, d=128, h=128, w=128):
    return np.zeros((d, h, w), dtype=np.float32)

  def test_patch_count(self):
    vol = self._vol()
    self.assertEqual(_patch_count(vol, 64), 8)  # 2×2×2

  def test_extract_patch_shape(self):
    vol = np.arange(128 * 128 * 128, dtype=np.float32).reshape(128, 128, 128)
    patch = _extract_patch(vol, 0, 64)
    self.assertEqual(patch.shape, (64, 64, 64))

  def test_extract_patch_last(self):
    vol = self._vol()
    patch = _extract_patch(vol, 7, 64)  # last of 8 patches
    self.assertEqual(patch.shape, (64, 64, 64))

  def test_extract_patch_out_of_range(self):
    vol = self._vol()
    with self.assertRaises(IndexError):
      _extract_patch(vol, 8, 64)

  def test_extract_patch_negative(self):
    vol = self._vol()
    with self.assertRaises(IndexError):
      _extract_patch(vol, -1, 64)


class TestProofreaderLifecycle(unittest.TestCase):

  def setUp(self):
    self.pred_path = _make_pred()
    self.pr = Proofreader(self.pred_path, patch_size=64)

  def tearDown(self):
    os.unlink(self.pred_path)

  def test_initial_stats_all_pending(self):
    s = self.pr.stats()
    self.assertEqual(s["total"], 8)
    self.assertEqual(s["pending"], 8)
    self.assertEqual(s["approved"], 0)
    self.assertEqual(s["rejected"], 0)

  def test_set_decision_approve(self):
    self.pr.set_decision(0, "approve")
    s = self.pr.stats()
    self.assertEqual(s["approved"], 1)
    self.assertEqual(s["pending"], 7)

  def test_set_decision_reject(self):
    self.pr.set_decision(1, "reject")
    self.assertEqual(self.pr._decisions[1], "reject")

  def test_set_decision_invalid(self):
    with self.assertRaises(ValueError):
      self.pr.set_decision(0, "maybe")

  def test_set_decision_out_of_range(self):
    with self.assertRaises(IndexError):
      self.pr.set_decision(99, "approve")


class TestProofreaderExport(unittest.TestCase):

  def setUp(self):
    self.pred_path = _make_pred()
    self.pr = Proofreader(self.pred_path, patch_size=64)

  def tearDown(self):
    os.unlink(self.pred_path)

  def test_export_approved_shape(self):
    self.pr.set_decision(0, "approve")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
      out_path = f.name
    try:
      out = self.pr.export_approved(out_path)
      self.assertEqual(out.shape, self.pr._predictions.shape)
    finally:
      os.unlink(out_path)

  def test_export_approved_zeros_rejected(self):
    # Approve patch 0, reject all others → only patch 0 region non-zero allowed
    for i in range(8):
      self.pr.set_decision(i, "reject")
    self.pr.set_decision(0, "approve")
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
      out_path = f.name
    try:
      out = self.pr.export_approved(out_path)
      # Patch 1 region (zs=0, ze=64, ys=0, ye=64, xs=64, xe=128) should be zero
      self.assertEqual(float(out[0:64, 0:64, 64:128].sum()), 0.0)
    finally:
      os.unlink(out_path)

  def test_export_all_pending_is_all_zeros(self):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
      out_path = f.name
    try:
      out = self.pr.export_approved(out_path)
      self.assertEqual(float(out.sum()), 0.0)
    finally:
      os.unlink(out_path)


class TestProofreaderPersistence(unittest.TestCase):

  def setUp(self):
    self.pred_path = _make_pred()

  def tearDown(self):
    os.unlink(self.pred_path)

  def test_save_load_state(self):
    pr = Proofreader(self.pred_path, patch_size=64)
    pr.set_decision(0, "approve")
    pr.set_decision(1, "reject")

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
      state_path = f.name

    try:
      pr.save_state(state_path)
      pr2 = Proofreader(self.pred_path, patch_size=64, state_path=state_path)
      self.assertEqual(pr2._decisions[0], "approve")
      self.assertEqual(pr2._decisions[1], "reject")
      self.assertEqual(pr2._decisions[2], "pending")
    finally:
      os.unlink(state_path)

  def test_load_state_partial(self):
    pr = Proofreader(self.pred_path, patch_size=64)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
      json.dump({"0": "approve"}, f)
      state_path = f.name
    try:
      pr.load_state(state_path)
      self.assertEqual(pr._decisions[0], "approve")
      self.assertEqual(pr._decisions[1], "pending")  # unchanged
    finally:
      os.unlink(state_path)


class TestProofreaderWithGroundTruth(unittest.TestCase):

  def setUp(self):
    self.pred_path = _make_pred()
    self.gt_path   = _make_pred()

  def tearDown(self):
    os.unlink(self.pred_path)
    os.unlink(self.gt_path)

  def test_construct_with_gt(self):
    pr = Proofreader(self.pred_path, ground_truth_path=self.gt_path, patch_size=64)
    self.assertIsNotNone(pr._ground_truth)

  def test_stats_total(self):
    pr = Proofreader(self.pred_path, ground_truth_path=self.gt_path, patch_size=64)
    self.assertEqual(pr.stats()["total"], 8)


if __name__ == "__main__":
  unittest.main()
