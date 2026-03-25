"""napari_trainer.py — Interactive labeling + in-place training via napari.

Falls back to a CLI mode when napari is not installed.

Usage (napari available):
  trainer = NapariTrainer("path/to/volume.zarr", model_path="model.safetensors")
  trainer.launch()          # opens napari; add labels interactively
  trainer.train_on_labels() # retrain on current label layer
  trainer.predict()         # run inference, show overlay

Usage (CLI fallback):
  trainer = NapariTrainer("path/to/volume.zarr")
  trainer.train_on_labels(epochs=10)
  trainer.predict()
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

try:
  import napari as _napari
  _HAS_NAPARI = True
except ImportError:
  _HAS_NAPARI = False

try:
  import tinygrad as _tinygrad  # noqa: F401
  _HAS_TINYGRAD = True
except ImportError:
  _HAS_TINYGRAD = False

# ml imports are deferred to method bodies so this module loads even without tinygrad

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Volume loader (zarr or numpy fallback)
# ---------------------------------------------------------------------------

def _load_volume(path: str) -> np.ndarray:
  """Load volume from zarr store or .npy file; returns float32 array."""
  if path.endswith(".npy"):
    return np.load(path).astype(np.float32)
  try:
    import zarr
    z = zarr.open(path, mode="r")
    # Support zarr group (take first array) or direct array
    if hasattr(z, "arrays"):
      _, arr = next(iter(z.arrays()))
      return np.array(arr, dtype=np.float32)
    return np.array(z, dtype=np.float32)
  except ImportError:
    raise ImportError("zarr is required to load .zarr volumes; pip install zarr")


# ---------------------------------------------------------------------------
# NapariTrainer
# ---------------------------------------------------------------------------

class NapariTrainer:
  """Interactive labeling + in-place training via napari.

  When napari is not available the class still works in headless/CLI mode:
  call train_on_labels() with labels_path and predict() with output_path
  to drive the pipeline without a GUI.
  """

  def __init__(self, volume_path: str, model_path: Optional[str] = None):
    self.volume_path = volume_path
    self.model_path = model_path

    self._volume: Optional[np.ndarray] = None
    self._labels: Optional[np.ndarray] = None
    self._prediction: Optional[np.ndarray] = None
    self._viewer = None  # napari.Viewer when active

    self._model = None
    if model_path and os.path.isfile(model_path):
      self._model = self._load_model(model_path)

  # ---- private helpers ---------------------------------------------------

  def _ensure_volume(self) -> np.ndarray:
    if self._volume is None:
      log.info("Loading volume from %s", self.volume_path)
      self._volume = _load_volume(self.volume_path)
    return self._volume

  def _ensure_model(self):
    if self._model is None:
      if not _HAS_TINYGRAD:
        raise ImportError("tinygrad is required; install it with: pip install tinygrad")
      from .ml.model import UNet
      log.info("Creating default UNet (in_ch=1, out_ch=2)")
      self._model = UNet(in_channels=1, out_channels=2)
    return self._model

  def _load_model(self, path: str):
    if not _HAS_TINYGRAD:
      raise ImportError("tinygrad is required to load a model")
    from .ml.model import UNet
    from tinygrad.nn.state import safe_load, load_state_dict
    model = UNet(in_channels=1, out_channels=2)
    state = safe_load(path)
    load_state_dict(model, state)
    log.info("Loaded model from %s", path)
    return model

  def _save_model(self, path: str) -> None:
    if not _HAS_TINYGRAD:
      raise ImportError("tinygrad is required to save a model")
    from tinygrad.nn.state import safe_save, get_state_dict
    state = get_state_dict(self._model)
    safe_save(state, path)
    log.info("Saved model to %s", path)

  # ---- public API --------------------------------------------------------

  def launch(self) -> None:
    """Open napari with the volume and an empty label layer.

    Raises RuntimeError if napari is not installed.
    """
    if not _HAS_NAPARI:
      raise RuntimeError(
        "napari is not installed; install it with: pip install 'volatile[napari]'\n"
        "You can still use train_on_labels() and predict() in CLI mode."
      )
    vol = self._ensure_volume()
    viewer = _napari.Viewer(title="Volatile — NapariTrainer")
    self._viewer = viewer

    viewer.add_image(vol, name="volume", colormap="gray")
    labels_layer = viewer.add_labels(
      np.zeros(vol.shape, dtype=np.int32), name="labels"
    )
    self._labels_layer = labels_layer
    log.info("napari viewer opened; draw labels then call train_on_labels()")
    _napari.run()

  def train_on_labels(
    self,
    epochs: int = 5,
    lr: float = 1e-3,
    labels_path: Optional[str] = None,
  ) -> None:
    """Train model on the current label layer (or labels loaded from disk).

    Args:
      epochs: number of training epochs.
      lr: initial learning rate.
      labels_path: path to a .npy label array (CLI mode; overrides viewer labels).
    """
    if not _HAS_TINYGRAD:
      raise ImportError("tinygrad is required for training")

    vol = self._ensure_volume()

    # Resolve labels (before building model so we fail fast on missing labels)
    if labels_path:
      labels = np.load(labels_path).astype(np.int64)
    elif self._viewer is not None and hasattr(self, "_labels_layer"):
      labels = self._labels_layer.data.astype(np.int64)
    elif self._labels is not None:
      labels = self._labels
    else:
      raise ValueError("No labels available; pass labels_path or draw labels in the viewer first")

    model = self._ensure_model()
    log.info("Training for %d epochs on volume %s", epochs, self.volume_path)
    from .ml.loss import DiceCELoss
    from .ml.data import PatchDataset
    from .ml.train import train_one_epoch

    dataset = PatchDataset(vol, labels)
    loss_fn = DiceCELoss()

    for epoch in range(1, epochs + 1):
      loss = train_one_epoch(model, dataset, loss_fn=loss_fn, lr=lr, epoch=epoch)
      log.info("  epoch %d/%d  loss=%.4f", epoch, epochs, loss)

    self._model = model
    if self.model_path:
      self._save_model(self.model_path)

  def predict(self, output_path: Optional[str] = None) -> np.ndarray:
    """Run inference on the volume; optionally save result to disk.

    Returns the prediction as a float32 numpy array.
    Adds an overlay layer to the napari viewer if it is open.
    """
    if not _HAS_TINYGRAD:
      raise ImportError("tinygrad is required for inference")

    vol = self._ensure_volume()
    model = self._ensure_model()

    log.info("Running tiled inference…")
    from .ml.infer import tiled_infer
    # tiled_infer expects (H, W) or (D, H, W) float32
    pred = tiled_infer(model, vol)
    self._prediction = pred

    if self._viewer is not None:
      self._viewer.add_image(pred, name="prediction", colormap="magma", opacity=0.5)
      log.info("Prediction overlay added to viewer")

    if output_path:
      np.save(output_path, pred)
      log.info("Prediction saved to %s", output_path)

    return pred

  # ---- set_labels (CLI helper) -------------------------------------------

  def set_labels(self, labels: np.ndarray) -> None:
    """Set label array directly (for CLI / programmatic use)."""
    self._labels = labels.astype(np.int64)
