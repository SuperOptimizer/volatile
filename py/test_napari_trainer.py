"""test_napari_trainer.py — unit tests for NapariTrainer (headless, no napari/tinygrad required).

Run with:  python -m pytest py/test_napari_trainer.py
       or: python py/test_napari_trainer.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from volatile.napari_trainer import NapariTrainer, _load_volume

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npy(shape=(4, 4, 4)) -> str:
  """Write a random float32 .npy file; return path."""
  tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
  np.save(tmp.name, np.random.rand(*shape).astype(np.float32))
  return tmp.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLoadVolume(unittest.TestCase):

  def test_load_npy(self):
    path = _make_npy()
    try:
      vol = _load_volume(path)
      self.assertEqual(vol.dtype, np.float32)
      self.assertEqual(vol.shape, (4, 4, 4))
    finally:
      os.unlink(path)

  def test_load_npy_converts_dtype(self):
    tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
    np.save(tmp.name, np.ones((2, 2, 2), dtype=np.uint16))
    try:
      vol = _load_volume(tmp.name)
      self.assertEqual(vol.dtype, np.float32)
    finally:
      os.unlink(tmp.name)


class TestNapariTrainerLifecycle(unittest.TestCase):

  def setUp(self):
    self.vol_path = _make_npy()

  def tearDown(self):
    os.unlink(self.vol_path)

  def test_construct_no_model(self):
    trainer = NapariTrainer(self.vol_path)
    self.assertIsNone(trainer._model)
    self.assertIsNone(trainer._prediction)

  def test_construct_nonexistent_model_path(self):
    # Should not raise; model_path is loaded only if the file exists
    trainer = NapariTrainer(self.vol_path, model_path="/nonexistent/model.safetensors")
    self.assertIsNone(trainer._model)

  def test_launch_without_napari_raises(self):
    import volatile.napari_trainer as nt_mod
    orig = nt_mod._HAS_NAPARI
    nt_mod._HAS_NAPARI = False
    try:
      trainer = NapariTrainer(self.vol_path)
      with self.assertRaises(RuntimeError):
        trainer.launch()
    finally:
      nt_mod._HAS_NAPARI = orig

  def test_train_without_tinygrad_raises(self):
    import volatile.napari_trainer as nt_mod
    orig = nt_mod._HAS_TINYGRAD
    nt_mod._HAS_TINYGRAD = False
    try:
      trainer = NapariTrainer(self.vol_path)
      with self.assertRaises(ImportError):
        trainer.train_on_labels()
    finally:
      nt_mod._HAS_TINYGRAD = orig

  def test_predict_without_tinygrad_raises(self):
    import volatile.napari_trainer as nt_mod
    orig = nt_mod._HAS_TINYGRAD
    nt_mod._HAS_TINYGRAD = False
    try:
      trainer = NapariTrainer(self.vol_path)
      with self.assertRaises(ImportError):
        trainer.predict()
    finally:
      nt_mod._HAS_TINYGRAD = orig

  def test_train_no_labels_raises(self):
    import volatile.napari_trainer as nt_mod
    orig = nt_mod._HAS_TINYGRAD
    nt_mod._HAS_TINYGRAD = True
    try:
      trainer = NapariTrainer(self.vol_path)
      trainer._viewer = None
      trainer._labels = None
      with self.assertRaises(ValueError):
        trainer.train_on_labels()
    finally:
      nt_mod._HAS_TINYGRAD = orig

  def test_set_labels(self):
    trainer = NapariTrainer(self.vol_path)
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    trainer.set_labels(labels)
    self.assertIsNotNone(trainer._labels)
    self.assertEqual(trainer._labels.dtype, np.int64)

  def test_ensure_volume_loads_once(self):
    trainer = NapariTrainer(self.vol_path)
    v1 = trainer._ensure_volume()
    v2 = trainer._ensure_volume()
    self.assertIs(v1, v2)   # same object — not reloaded


if __name__ == "__main__":
  unittest.main()
