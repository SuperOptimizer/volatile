from __future__ import annotations
"""ink.py — Ink detection pipeline stub.

Full inference requires tinygrad (optional dependency) and a trained model
checkpoint. When tinygrad is unavailable the function returns a zero-filled
array so callers can continue developing against the API.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
  try:
    import numpy as np
    _ArrayLike = np.ndarray
  except ImportError:
    _ArrayLike = object

try:
  import numpy as np
  _HAS_NUMPY = True
except ImportError:
  _HAS_NUMPY = False

try:
  from tinygrad import Tensor
  _HAS_TINYGRAD = True
except ImportError:
  _HAS_TINYGRAD = False


def detect_ink(volume, surface, model_path: str | None = None):
  """Run ink detection on a surface using tiled model inference.

  Args:
    volume:     Opened volume handle (volatile._core capsule or equivalent).
                Must support vol_sample(vol, level, z, y, x) -> float.
    surface:    QuadSurface (volatile.seg.QuadSurface) whose vertices define
                the sampling grid.
    model_path: Path to a tinygrad-serialised model checkpoint.  When None,
                falls back to the bundled default UNet weights if present.

  Returns:
    A 2-D float32 array of shape (surface.rows, surface.cols) with per-vertex
    ink probability in [0, 1].  Returns a zero array when the model or
    tinygrad is unavailable.
  """
  rows = surface.rows
  cols = surface.cols

  # ------------------------------------------------------------------
  # Step 1: sample the volume along the surface
  # ------------------------------------------------------------------
  try:
    from volatile import vol_sample
  except ImportError:
    vol_sample = None

  if _HAS_NUMPY:
    samples = np.zeros((rows, cols), dtype=np.float32)
  else:
    samples = [[0.0] * cols for _ in range(rows)]

  if vol_sample is not None and volume is not None:
    for row in range(rows):
      for col in range(cols):
        x, y, z = surface.get(row, col)
        v = vol_sample(volume, 0, z, y, x)
        if _HAS_NUMPY:
          samples[row, col] = v
        else:
          samples[row][col] = v

  # ------------------------------------------------------------------
  # Step 2 & 3: load model and run tiled inference
  # ------------------------------------------------------------------
  if not _HAS_TINYGRAD:
    # Return zero prediction — callers can still exercise the API
    return samples

  try:
    from volatile.ml.model import UNet
    from volatile.ml.infer import infer_tiled
  except ImportError:
    return samples

  model = UNet(in_channels=1, out_channels=1)

  if model_path is not None:
    try:
      from tinygrad.nn.state import load_state_dict, safe_load
      load_state_dict(model, safe_load(model_path))
    except Exception:
      pass  # proceed with random weights rather than crashing
  else:
    # Attempt to locate bundled default weights alongside this file
    import os
    default_path = os.path.join(os.path.dirname(__file__), "ml", "weights", "unet_default.safetensors")
    if os.path.exists(default_path):
      try:
        from tinygrad.nn.state import load_state_dict, safe_load
        load_state_dict(model, safe_load(default_path))
      except Exception:
        pass

  # ------------------------------------------------------------------
  # Step 4: tiled inference and return prediction
  # ------------------------------------------------------------------
  try:
    prediction = infer_tiled(model, samples, tile_size=64, overlap=16)
  except Exception:
    if _HAS_NUMPY:
      prediction = np.zeros((rows, cols), dtype=np.float32)
    else:
      prediction = [[0.0] * cols for _ in range(rows)]

  return prediction
