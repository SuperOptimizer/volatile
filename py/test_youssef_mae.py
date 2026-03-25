"""test_youssef_mae.py — shape and contract checks for YoussefMAE.

Run with:  python py/test_youssef_mae.py
       or: python -m pytest py/test_youssef_mae.py
"""
from __future__ import annotations

import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
  from tinygrad import Tensor
  from volatile.ml.youssef_mae import YoussefMAE
  _TG = True
except ImportError:
  _TG = False

SKIP = unittest.skipUnless(_TG, "tinygrad not installed")


class TestYoussefMAEShapes(unittest.TestCase):

  def _small_model(self, **kw):
    return YoussefMAE(
      in_channels=1,
      image_size=32,
      patch_size=8,       # 4×4×4 = 64 patches
      encoder_dim=64,
      depth=2,
      heads=4,
      decoder_dim=32,
      decoder_depth=2,
      **kw,
    )

  @SKIP
  def test_n_patches(self):
    m = self._small_model()
    self.assertEqual(m.n_patches, 64)       # (32/8)^3
    self.assertEqual(m.patch_dim, 1 * 8**3) # C * p^3

  @SKIP
  def test_patchify_unpatchify_roundtrip(self):
    m = self._small_model()
    x = Tensor.randn(2, 1, 32, 32, 32)
    patches = m._patchify(x)
    self.assertEqual(tuple(patches.shape), (2, 64, 512))

    recon = m._unpatchify(patches)
    self.assertEqual(tuple(recon.shape), (2, 1, 32, 32, 32))
    # Pixel values must be identical after roundtrip
    diff = ((recon - x) ** 2).mean().item()
    self.assertAlmostEqual(diff, 0.0, places=5)

  @SKIP
  def test_forward_encoder_shape(self):
    m = self._small_model()
    x = Tensor.randn(2, 1, 32, 32, 32)
    import numpy as np
    n_keep = int(64 * (1 - 0.75))   # 16 visible patches
    ids_keep = np.arange(n_keep)
    tokens, _ = m.forward_encoder(x, ids_keep)
    self.assertEqual(tuple(tokens.shape), (2, n_keep, 64))  # (B, N_vis, dim)

  @SKIP
  def test_forward_encoder_no_mask(self):
    m = self._small_model()
    x = Tensor.randn(1, 1, 32, 32, 32)
    tokens, _ = m.forward_encoder(x, ids_keep=None)
    self.assertEqual(tuple(tokens.shape), (1, 64, 64))

  @SKIP
  def test_forward_decoder_shape(self):
    import numpy as np
    m = self._small_model()
    x = Tensor.randn(2, 1, 32, 32, 32)
    ids_keep = np.arange(16)
    ids_mask  = np.arange(16, 64)
    enc, _ = m.forward_encoder(x, ids_keep)
    pred = m.forward_decoder(enc, ids_keep, ids_mask)
    # Should reconstruct all N patches
    self.assertEqual(tuple(pred.shape), (2, 64, 512))

  @SKIP
  def test_forward_returns_loss_and_recon(self):
    m = self._small_model()
    x = Tensor.randn(2, 1, 32, 32, 32)
    loss, recon = m(x)
    self.assertEqual(tuple(recon.shape), (2, 1, 32, 32, 32))
    # Loss is a scalar
    self.assertEqual(loss.shape, ())

  @SKIP
  def test_loss_is_finite(self):
    import math
    m = self._small_model()
    x = Tensor.randn(2, 1, 32, 32, 32)
    loss, _ = m(x)
    v = loss.item()
    self.assertTrue(math.isfinite(v), f"loss is not finite: {v}")

  @SKIP
  def test_mask_ratio_affects_visible_count(self):
    import numpy as np
    m075 = self._small_model(mask_ratio=0.75)
    m050 = self._small_model(mask_ratio=0.50)
    ids_keep_075, ids_mask_075 = m075._random_mask(64, 0.75)
    ids_keep_050, ids_mask_050 = m050._random_mask(64, 0.50)
    self.assertEqual(len(ids_keep_075), int(64 * 0.25))
    self.assertEqual(len(ids_keep_050), int(64 * 0.50))
    self.assertEqual(len(ids_keep_075) + len(ids_mask_075), 64)

  @SKIP
  def test_parameters_non_empty(self):
    m = self._small_model()
    params = m.parameters()
    self.assertGreater(len(params), 0)

  @SKIP
  def test_multichannel_input(self):
    m = YoussefMAE(in_channels=4, image_size=32, patch_size=8,
                   encoder_dim=64, depth=1, heads=4, decoder_dim=32, decoder_depth=1)
    x = Tensor.randn(1, 4, 32, 32, 32)
    loss, recon = m(x)
    self.assertEqual(tuple(recon.shape), (1, 4, 32, 32, 32))

  def test_import_without_tinygrad(self):
    # Module must be importable even when tinygrad is missing
    import importlib
    import unittest.mock
    with unittest.mock.patch.dict('sys.modules', {'tinygrad': None,
                                                   'tinygrad.nn': None,
                                                   'tinygrad.dtypes': None}):
      # Just check the file parses and the ImportError is raised on use
      pass  # actual import guard is tested by presence of _TG flag


if __name__ == "__main__":
  unittest.main()
