"""test_smoke.py — verify every volatile Python module imports without error."""
from __future__ import annotations


def test_import_volatile():
  import volatile
  assert hasattr(volatile, "__version__")


def test_import_imgproc():
  import volatile.imgproc


def test_import_seg():
  import volatile.seg


def test_import_ink():
  import volatile.ink


def test_import_zarr_tasks():
  import volatile.zarr_tasks


def test_import_structure_tensor():
  import volatile.structure_tensor


def test_import_eval():
  import volatile.eval


def test_import_metrics():
  import volatile.metrics


def test_import_fit():
  import volatile.fit


def test_import_ml():
  import volatile.ml


def test_import_ml_model():
  import volatile.ml.model


def test_import_ml_infer():
  import volatile.ml.infer


def test_import_ml_augment():
  import volatile.ml.augment


def test_import_ml_loss():
  import volatile.ml.loss


def test_import_ml_config():
  import volatile.ml.config


def test_import_ml_train():
  import volatile.ml.train


def test_import_ml_data():
  import volatile.ml.data
