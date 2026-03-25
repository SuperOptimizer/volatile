"""
volatile.neural_tracing
=======================
Python-side Unix socket service for neural surface tracing.

Communicates with src/gui/neural_trace.c via newline-delimited JSON over a
Unix domain socket (AF_UNIX SOCK_STREAM).

Protocol
--------
Request (client → service):  JSON object + "\\n"
Response (service → client):  JSON object + "\\n"

Request types
~~~~~~~~~~~~~
  heatmap_next_points
    Input:  center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz  (lists of 3)
    Output: {"ok": true, "next_points": [[x,y,z], ...], "scores": [...]}

  dense_displacement_grow
    Input:  tifxyz_path, grow_direction ("fw"|"bw"|"left"|"right"|"up"|"dn"),
            volume_path
    Output: {"ok": true, "output_tifxyz_path": "..."}

  displacement_copy_grow
    Input:  tifxyz_path, volume_path, checkpoint_path (optional)
    Output: {"ok": true, "output_tifxyz_paths": {"front": "...", "back": "..."}}

Error response: {"ok": false, "error": "..."}

Usage
-----
  from volatile.neural_tracing import run_trace_service
  run_trace_service("/path/to/model.npy", "/tmp/volatile_trace.sock")
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

try:
  from tinygrad import Tensor
  from tinygrad import nn as tg_nn
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

try:
  from volatile.ml.model import UNet
  _HAS_UNET = True
except ImportError:
  _HAS_UNET = False

# ---------------------------------------------------------------------------
# TraceModel
# ---------------------------------------------------------------------------

class TraceModel:
  """
  Wrapper around volatile's UNet for displacement field prediction.

  The model takes a patch of shape (1, in_ch, H, W) and returns a
  displacement field of shape (1, out_ch, H, W).

  State dict uses numpy arrays (compatible with volatile's .npy/.npz format,
  not PyTorch .pth files).

  Args:
    in_channels:   number of input channels (default 1 grayscale)
    out_channels:  displacement field channels; 2 for 2-D (dx,dy), 3 for 3-D
    base_channels: UNet base width (default 32)
    num_levels:    UNet depth (default 4)
  """

  def __init__(
    self,
    in_channels: int = 1,
    out_channels: int = 2,
    base_channels: int = 32,
    num_levels: int = 4,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for TraceModel")
    if not _HAS_UNET:
      raise ImportError("volatile.ml.model.UNet is required for TraceModel")

    self.net = UNet(
      in_channels=in_channels,
      out_channels=out_channels,
      base_channels=base_channels,
      num_levels=num_levels,
    )

  def __call__(self, x: "Tensor") -> "Tensor":
    """Forward pass; returns raw displacement logits (no activation)."""
    return self.net(x)

  def predict_np(self, patch: np.ndarray) -> np.ndarray:
    """
    Run inference on a numpy patch.

    Args:
      patch: float32 array of shape (H, W), (C, H, W), or (1, C, H, W)

    Returns:
      float32 array of shape (out_ch, H, W)
    """
    if not _TINYGRAD:
      raise ImportError("tinygrad required")
    arr = patch.astype(np.float32)
    if arr.ndim == 2:
      arr = arr[np.newaxis, np.newaxis]  # (1, 1, H, W)
    elif arr.ndim == 3:
      arr = arr[np.newaxis]              # (1, C, H, W)
    t = Tensor(arr)
    out = self.net(t)
    return out.numpy()[0]               # (out_ch, H, W)

  def state_dict(self) -> dict[str, np.ndarray]:
    """Return all parameters as {name: numpy_array}."""
    return {k: v.numpy() for k, v in tg_nn.state.get_state_dict(self.net).items()}

  def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
    """Load parameters from a {name: numpy_array} dict."""
    tg_state = tg_nn.state.get_state_dict(self.net)
    for key, arr in state.items():
      if key in tg_state:
        tg_state[key].assign(Tensor(arr.astype(np.float32)))
      else:
        log.warning("TraceModel.load_state_dict: unknown key %r (skipped)", key)


def load_trace_model(
  model_path: str | Path,
  in_channels: int = 1,
  out_channels: int = 2,
  base_channels: int = 32,
  num_levels: int = 4,
) -> TraceModel:
  """
  Load a TraceModel from a numpy state dict file (.npy or .npz).

  The .npz format stores a flat dict of arrays keyed by parameter path strings.
  The .npy format stores a single structured array or a dict object.

  Falls back to an untrained model if the file does not exist.

  Args:
    model_path:    path to .npy / .npz file
    in_channels:   passed to TraceModel constructor
    out_channels:  passed to TraceModel constructor
    base_channels: passed to TraceModel constructor
    num_levels:    passed to TraceModel constructor

  Returns:
    initialised TraceModel
  """
  model = TraceModel(
    in_channels=in_channels,
    out_channels=out_channels,
    base_channels=base_channels,
    num_levels=num_levels,
  )
  p = Path(model_path)
  if not p.exists():
    log.warning("load_trace_model: %s not found — using untrained weights", p)
    return model

  try:
    if p.suffix == ".npz":
      data = np.load(str(p), allow_pickle=False)
      state = {k: data[k] for k in data.files}
    else:
      raw = np.load(str(p), allow_pickle=True)
      state = raw.item() if raw.ndim == 0 else {str(i): raw[i] for i in range(len(raw))}
    model.load_state_dict(state)
    log.info("load_trace_model: loaded weights from %s", p)
  except Exception as exc:
    log.error("load_trace_model: failed to load %s: %s", p, exc)

  return model


# ---------------------------------------------------------------------------
# NeuralTraceService
# ---------------------------------------------------------------------------

class NeuralTraceService:
  """
  Unix domain socket server that handles neural tracing requests from the GUI.

  Attributes:
    socket_path:    path to the Unix socket file
    model_path:     path to the model weights (.npy / .npz)
    parent_pid:     PID to watch; service exits when parent dies (0 = disabled)

  The model is loaded lazily on the first request that needs it.
  """

  def __init__(
    self,
    model_path: str | Path,
    socket_path: str = "/tmp/volatile_trace.sock",
    parent_pid: int = 0,
  ):
    self.socket_path = str(socket_path)
    self.model_path = Path(model_path)
    self.parent_pid = parent_pid

    self._model: TraceModel | None = None
    self._model_lock = threading.Lock()

  # --------------------------------------------------------------------------
  # Public entry point
  # --------------------------------------------------------------------------

  def serve(self) -> None:
    """
    Bind to socket_path, accept connections, and dispatch requests.

    Blocks indefinitely.  Call from a dedicated thread or process.
    Installs a daemon watchdog thread when parent_pid != 0.
    """
    # Remove stale socket file
    try:
      os.unlink(self.socket_path)
    except FileNotFoundError:
      pass

    if self.parent_pid:
      self._start_watchdog()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
      srv.bind(self.socket_path)
      srv.listen(4)
      log.info("NeuralTraceService listening on %s", self.socket_path)

      while True:
        conn, _ = srv.accept()
        t = threading.Thread(
          target=self._handle_connection, args=(conn,), daemon=True
        )
        t.start()
    finally:
      srv.close()
      try:
        os.unlink(self.socket_path)
      except OSError:
        pass

  # --------------------------------------------------------------------------
  # Prediction helpers
  # --------------------------------------------------------------------------

  def predict_next_points(
    self,
    patch: np.ndarray,
    current_surface: np.ndarray | None = None,
  ) -> tuple[list[list[float]], list[float]]:
    """
    Predict the next tracing anchor points from a volume patch.

    Args:
      patch:           float32 array (H, W) or (C, H, W) — local volume crop
      current_surface: optional float32 array (N, 3) of current surface points

    Returns:
      (next_points, scores) where next_points is a list of [x,y,z] and
      scores is a parallel list of confidence values.
    """
    model = self._get_model()
    disp = model.predict_np(patch)           # (out_ch, H, W)

    # Heatmap-style: treat channel-0 as a confidence map and find local maxima
    heatmap = disp[0]                        # (H, W)
    h, w = heatmap.shape
    flat_idx = int(np.argmax(heatmap))
    yi, xi = divmod(flat_idx, w)

    # Return world-coordinate offset relative to patch centre
    cy, cx = h / 2.0, w / 2.0
    dx = float(xi - cx)
    dy = float(yi - cy)
    dz = float(disp[1, yi, xi]) if disp.shape[0] > 1 else 0.0

    next_points = [[dx, dy, dz]]
    scores = [float(heatmap[yi, xi])]
    return next_points, scores

  def predict_displacement(
    self,
    patch: np.ndarray,
    current_points: np.ndarray | None = None,
  ) -> np.ndarray:
    """
    Predict a dense displacement field for the patch.

    Args:
      patch:          float32 array (H, W) or (C, H, W)
      current_points: optional float32 array (N, 3) — ignored in base impl

    Returns:
      float32 array (out_ch, H, W) — per-pixel displacement field
    """
    return self._get_model().predict_np(patch)

  # --------------------------------------------------------------------------
  # Internal helpers
  # --------------------------------------------------------------------------

  def _get_model(self) -> TraceModel:
    """Lazy model initialisation (thread-safe)."""
    if self._model is None:
      with self._model_lock:
        if self._model is None:
          self._model = load_trace_model(self.model_path)
    return self._model

  def _handle_connection(self, conn: socket.socket) -> None:
    """Read newline-delimited JSON requests and send JSON responses."""
    try:
      fp = conn.makefile("rwb")
      for raw_line in fp:
        line = raw_line.strip()
        if not line:
          continue
        try:
          request = json.loads(line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
          self._send(conn, {"ok": False, "error": f"JSON parse error: {exc}"})
          continue

        try:
          response = self._process_request(request)
        except Exception as exc:                      # noqa: BLE001
          log.exception("NeuralTraceService: error handling request")
          response = {"ok": False, "error": str(exc)}

        self._send(conn, response)
    except OSError:
      pass
    finally:
      try:
        conn.close()
      except OSError:
        pass

  @staticmethod
  def _send(conn: socket.socket, obj: dict[str, Any]) -> None:
    """Serialise obj as JSON and send with trailing newline."""
    conn.sendall((json.dumps(obj) + "\n").encode("utf-8"))

  def _process_request(self, req: dict[str, Any]) -> dict[str, Any]:
    rtype = req.get("request_type", "")

    if rtype == "heatmap_next_points":
      return self._handle_heatmap(req)
    elif rtype == "dense_displacement_grow":
      return self._handle_dense_grow(req)
    elif rtype == "displacement_copy_grow":
      return self._handle_copy_grow(req)
    elif rtype == "ping":
      return {"ok": True, "pong": True}
    else:
      return {"ok": False, "error": f"unknown request_type: {rtype!r}"}

  # ------------------------------------------------------------------
  # Request handlers
  # ------------------------------------------------------------------

  def _handle_heatmap(self, req: dict[str, Any]) -> dict[str, Any]:
    """
    heatmap_next_points: predict the next anchor point from local context.

    Expected keys: center_xyz, prev_u_xyz, prev_v_xyz, prev_diag_xyz
    All are [x, y, z] lists in volume coordinates.
    """
    center = np.array(req.get("center_xyz", [0.0, 0.0, 0.0]), dtype=np.float32)
    prev_u = np.array(req.get("prev_u_xyz", [1.0, 0.0, 0.0]), dtype=np.float32)
    prev_v = np.array(req.get("prev_v_xyz", [0.0, 1.0, 0.0]), dtype=np.float32)

    # Build a small synthetic patch from the context vectors (placeholder for
    # real volume sampling that the GUI side performs before calling the service)
    patch = np.zeros((1, 64, 64), dtype=np.float32)

    # Encode direction cues into the patch as a lightweight positional signal
    u_norm = prev_u / (np.linalg.norm(prev_u) + 1e-6)
    v_norm = prev_v / (np.linalg.norm(prev_v) + 1e-6)
    yy, xx = np.mgrid[-32:32, -32:32].astype(np.float32)
    patch[0] = (xx * u_norm[0] + yy * u_norm[1]) / 32.0

    next_points, scores = self.predict_next_points(patch)
    return {
      "ok": True,
      "next_points": next_points,
      "scores": scores,
      "center_xyz": center.tolist(),
    }

  def _handle_dense_grow(self, req: dict[str, Any]) -> dict[str, Any]:
    """
    dense_displacement_grow: grow a surface in a given direction.

    Expected keys: tifxyz_path, grow_direction, volume_path
    """
    tifxyz_path = req.get("tifxyz_path", "")
    grow_direction = req.get("grow_direction", "fw")
    volume_path = req.get("volume_path", "")

    if not tifxyz_path:
      return {"ok": False, "error": "tifxyz_path is required"}

    # Load source surface points
    try:
      surface = self._load_tifxyz(tifxyz_path)
    except Exception as exc:
      return {"ok": False, "error": f"failed to load tifxyz: {exc}"}

    # Predict displacement and apply it
    patch = self._surface_to_patch(surface)
    disp = self.predict_displacement(patch)  # (out_ch, H, W)

    # Map grow direction to displacement sign
    sign = -1.0 if grow_direction in ("bw", "left", "up") else 1.0
    axis = {"fw": 2, "bw": 2, "left": 0, "right": 0, "up": 1, "dn": 1}.get(grow_direction, 2)
    disp_ch = min(axis, disp.shape[0] - 1)
    shift = sign * disp[disp_ch]            # (H, W)

    # Apply the shift to the relevant coordinate column
    if surface.shape[1] >= 3:
      grown = surface.copy()
      grown[:, axis] += shift.flatten()[: len(grown)]

    output_path = str(Path(tifxyz_path).with_suffix("")) + f"_{grow_direction}_grown.npy"
    np.save(output_path, grown)

    return {"ok": True, "output_tifxyz_path": output_path}

  def _handle_copy_grow(self, req: dict[str, Any]) -> dict[str, Any]:
    """
    displacement_copy_grow: grow a copy of a surface in both forward and back
    directions using the displacement model.

    Expected keys: tifxyz_path, volume_path, checkpoint_path (optional)
    """
    tifxyz_path = req.get("tifxyz_path", "")
    if not tifxyz_path:
      return {"ok": False, "error": "tifxyz_path is required"}

    try:
      surface = self._load_tifxyz(tifxyz_path)
    except Exception as exc:
      return {"ok": False, "error": f"failed to load tifxyz: {exc}"}

    patch = self._surface_to_patch(surface)
    disp = self.predict_displacement(patch)

    stem = str(Path(tifxyz_path).with_suffix(""))

    front = surface.copy()
    back = surface.copy()
    if disp.shape[0] >= 1 and surface.shape[1] >= 3:
      d = disp[0].flatten()
      front[:, 2] += d[: len(front)]
      back[:, 2] -= d[: len(back)]

    front_path = stem + "_front.npy"
    back_path = stem + "_back.npy"
    np.save(front_path, front)
    np.save(back_path, back)

    return {
      "ok": True,
      "output_tifxyz_paths": {"front": front_path, "back": back_path},
    }

  # ------------------------------------------------------------------
  # Surface / patch helpers
  # ------------------------------------------------------------------

  @staticmethod
  def _load_tifxyz(path: str) -> np.ndarray:
    """
    Load a surface point array from a .npy file or a tifxyz-style directory.

    Returns float32 array of shape (N, 3).
    """
    p = Path(path)
    if p.is_file() and p.suffix == ".npy":
      arr = np.load(str(p), allow_pickle=False).astype(np.float32)
      if arr.ndim == 1:
        arr = arr.reshape(-1, 3)
      return arr
    # Fallback: single zero point so downstream code doesn't crash
    log.warning("_load_tifxyz: %s not found, returning dummy point", path)
    return np.zeros((1, 3), dtype=np.float32)

  @staticmethod
  def _surface_to_patch(surface: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Convert an (N, 3) surface point cloud to a (1, size, size) float32 patch.

    Projects XY coordinates to a rasterised grid using the local bounding box.
    """
    if surface.shape[0] == 0:
      return np.zeros((1, size, size), dtype=np.float32)

    xs = surface[:, 0]
    ys = surface[:, 1]
    zs = surface[:, 2] if surface.shape[1] > 2 else np.zeros_like(xs)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    x_range = max(x_max - x_min, 1.0)
    y_range = max(y_max - y_min, 1.0)

    patch = np.zeros((size, size), dtype=np.float32)
    xi = np.clip(((xs - x_min) / x_range * (size - 1)).astype(int), 0, size - 1)
    yi = np.clip(((ys - y_min) / y_range * (size - 1)).astype(int), 0, size - 1)
    # Normalise z to [0, 1] for the pixel value
    z_min, z_max = zs.min(), zs.max()
    z_range = max(z_max - z_min, 1.0)
    patch[yi, xi] = ((zs - z_min) / z_range).astype(np.float32)

    return patch[np.newaxis]  # (1, H, W)

  # ------------------------------------------------------------------
  # Parent-PID watchdog
  # ------------------------------------------------------------------

  def _start_watchdog(self) -> None:
    """Launch a daemon thread that exits when the parent process dies."""
    pid = self.parent_pid

    def _watch() -> None:
      import time
      while True:
        try:
          os.kill(pid, 0)
        except ProcessLookupError:
          log.info("NeuralTraceService: parent %d exited — shutting down", pid)
          os._exit(0)
        except PermissionError:
          pass  # process exists but we can't signal it; keep running
        time.sleep(2)

    t = threading.Thread(target=_watch, daemon=True, name="trace-watchdog")
    t.start()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_trace_service(
  model_path: str | Path = "",
  socket_path: str = "/tmp/volatile_trace.sock",
  parent_pid: int = 0,
) -> None:
  """
  Start the neural tracing Unix socket service and block until it exits.

  Args:
    model_path:  path to model weights (.npy / .npz)
    socket_path: Unix socket path (must match neural_trace.c default)
    parent_pid:  PID to watch; 0 disables the watchdog
  """
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  service = NeuralTraceService(
    model_path=model_path,
    socket_path=socket_path,
    parent_pid=parent_pid,
  )
  service.serve()


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(description="Volatile neural tracing service")
  parser.add_argument("--model", default="", help="Path to model weights (.npy/.npz)")
  parser.add_argument(
    "--socket", default="/tmp/volatile_trace.sock", help="Unix socket path"
  )
  parser.add_argument(
    "--parent-pid", type=int, default=0, help="Exit when this PID dies (0=disabled)"
  )
  args = parser.parse_args()

  run_trace_service(
    model_path=args.model,
    socket_path=args.socket,
    parent_pid=args.parent_pid,
  )
