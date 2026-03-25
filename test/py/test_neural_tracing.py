"""
Tests for volatile.neural_tracing
"""

from __future__ import annotations

import json
import os
import socket
import tempfile
import threading
import time

import numpy as np
import pytest

# Skip entire module if tinygrad is unavailable
tinygrad = pytest.importorskip("tinygrad", reason="tinygrad not installed")

from volatile.neural_tracing import (
  NeuralTraceService,
  TraceModel,
  load_trace_model,
  run_trace_service,
)


# ---------------------------------------------------------------------------
# TraceModel
# ---------------------------------------------------------------------------

class TestTraceModel:
  def test_forward_shape(self):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    patch = np.zeros((1, 1, 32, 32), dtype=np.float32)
    from tinygrad import Tensor
    out = m(Tensor(patch))
    assert out.shape == (1, 2, 32, 32), f"unexpected shape {out.shape}"

  def test_predict_np_2d_input(self):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    patch = np.zeros((32, 32), dtype=np.float32)
    out = m.predict_np(patch)
    assert out.shape == (2, 32, 32)
    assert out.dtype == np.float32

  def test_predict_np_3d_input(self):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    patch = np.zeros((1, 32, 32), dtype=np.float32)
    out = m.predict_np(patch)
    assert out.shape == (2, 32, 32)

  def test_state_dict_roundtrip(self):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    sd = m.state_dict()
    assert isinstance(sd, dict)
    assert len(sd) > 0
    # All values should be numpy arrays
    for v in sd.values():
      assert isinstance(v, np.ndarray)

    # Load back and verify shapes unchanged
    m2 = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    m2.load_state_dict(sd)
    sd2 = m2.state_dict()
    for k in sd:
      assert sd[k].shape == sd2[k].shape

  def test_load_state_dict_ignores_unknown_keys(self):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    sd = m.state_dict()
    sd["nonexistent_key"] = np.zeros(4, dtype=np.float32)
    m.load_state_dict(sd)  # should not raise

  def test_different_out_channels(self):
    m = TraceModel(in_channels=1, out_channels=3, base_channels=8, num_levels=2)
    out = m.predict_np(np.zeros((16, 16), dtype=np.float32))
    assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# load_trace_model
# ---------------------------------------------------------------------------

class TestLoadTraceModel:
  def test_missing_file_returns_model(self):
    m = load_trace_model("/nonexistent/path.npy", base_channels=8, num_levels=2)
    assert isinstance(m, TraceModel)

  def test_load_npz(self, tmp_path):
    # Save a model's state dict, then reload it
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    sd = m.state_dict()
    p = tmp_path / "weights.npz"
    np.savez(str(p), **sd)

    m2 = load_trace_model(p, base_channels=8, num_levels=2)
    sd2 = m2.state_dict()
    for k in sd:
      np.testing.assert_allclose(sd[k], sd2[k], rtol=1e-5)

  def test_load_npy_dict(self, tmp_path):
    m = TraceModel(in_channels=1, out_channels=2, base_channels=8, num_levels=2)
    sd = m.state_dict()
    p = tmp_path / "weights.npy"
    np.save(str(p), sd)

    m2 = load_trace_model(p, base_channels=8, num_levels=2)
    assert isinstance(m2, TraceModel)


# ---------------------------------------------------------------------------
# NeuralTraceService — helpers
# ---------------------------------------------------------------------------

class TestServiceHelpers:
  def test_surface_to_patch_shape(self):
    surface = np.random.rand(50, 3).astype(np.float32)
    patch = NeuralTraceService._surface_to_patch(surface, size=32)
    assert patch.shape == (1, 32, 32)
    assert patch.dtype == np.float32

  def test_surface_to_patch_empty(self):
    surface = np.zeros((0, 3), dtype=np.float32)
    patch = NeuralTraceService._surface_to_patch(surface, size=16)
    assert patch.shape == (1, 16, 16)
    np.testing.assert_array_equal(patch, 0)

  def test_load_tifxyz_missing_returns_dummy(self):
    arr = NeuralTraceService._load_tifxyz("/nonexistent/surface.npy")
    assert arr.shape == (1, 3)
    np.testing.assert_array_equal(arr, 0)

  def test_load_tifxyz_npy(self, tmp_path):
    pts = np.random.rand(20, 3).astype(np.float32)
    p = tmp_path / "surf.npy"
    np.save(str(p), pts)
    loaded = NeuralTraceService._load_tifxyz(str(p))
    assert loaded.shape == (20, 3)
    np.testing.assert_allclose(loaded, pts)

  def test_surface_to_patch_values_in_range(self):
    surface = np.array([[0, 0, 0], [10, 10, 1]], dtype=np.float32)
    patch = NeuralTraceService._surface_to_patch(surface, size=16)
    assert patch.min() >= 0.0
    assert patch.max() <= 1.0


# ---------------------------------------------------------------------------
# NeuralTraceService — request dispatch (no socket, call _process_request)
# ---------------------------------------------------------------------------

class TestProcessRequest:
  def _svc(self):
    return NeuralTraceService(
      model_path="/nonexistent/model.npy",
      socket_path="/tmp/test_volatile_trace_proc.sock",
    )

  def test_ping(self):
    svc = self._svc()
    r = svc._process_request({"request_type": "ping"})
    assert r["ok"] is True
    assert r.get("pong") is True

  def test_unknown_request_type(self):
    svc = self._svc()
    r = svc._process_request({"request_type": "bogus"})
    assert r["ok"] is False
    assert "unknown" in r["error"]

  def test_heatmap_next_points_returns_ok(self):
    svc = self._svc()
    r = svc._process_request({
      "request_type": "heatmap_next_points",
      "center_xyz": [100.0, 200.0, 50.0],
      "prev_u_xyz": [1.0, 0.0, 0.0],
      "prev_v_xyz": [0.0, 1.0, 0.0],
      "prev_diag_xyz": [1.0, 1.0, 0.0],
    })
    assert r["ok"] is True
    assert "next_points" in r
    assert "scores" in r
    assert len(r["next_points"]) == len(r["scores"])
    assert len(r["next_points"][0]) == 3

  def test_dense_grow_missing_path(self):
    svc = self._svc()
    r = svc._process_request({
      "request_type": "dense_displacement_grow",
      "tifxyz_path": "",
      "grow_direction": "fw",
      "volume_path": "",
    })
    assert r["ok"] is False

  def test_dense_grow_with_surface(self, tmp_path):
    svc = self._svc()
    pts = np.random.rand(30, 3).astype(np.float32)
    p = tmp_path / "surf.npy"
    np.save(str(p), pts)
    r = svc._process_request({
      "request_type": "dense_displacement_grow",
      "tifxyz_path": str(p),
      "grow_direction": "fw",
      "volume_path": "",
    })
    assert r["ok"] is True
    assert os.path.exists(r["output_tifxyz_path"])

  def test_copy_grow_missing_path(self):
    svc = self._svc()
    r = svc._process_request({
      "request_type": "displacement_copy_grow",
      "tifxyz_path": "",
      "volume_path": "",
    })
    assert r["ok"] is False

  def test_copy_grow_with_surface(self, tmp_path):
    svc = self._svc()
    pts = np.random.rand(30, 3).astype(np.float32)
    p = tmp_path / "surf.npy"
    np.save(str(p), pts)
    r = svc._process_request({
      "request_type": "displacement_copy_grow",
      "tifxyz_path": str(p),
      "volume_path": "",
    })
    assert r["ok"] is True
    assert "output_tifxyz_paths" in r
    assert os.path.exists(r["output_tifxyz_paths"]["front"])
    assert os.path.exists(r["output_tifxyz_paths"]["back"])


# ---------------------------------------------------------------------------
# NeuralTraceService — full socket roundtrip
# ---------------------------------------------------------------------------

def _pick_socket_path(tmp_path) -> str:
  return str(tmp_path / "test_trace.sock")


def _start_service_thread(socket_path: str) -> NeuralTraceService:
  svc = NeuralTraceService(
    model_path="/nonexistent/model.npy",
    socket_path=socket_path,
  )
  t = threading.Thread(target=svc.serve, daemon=True)
  t.start()
  # Wait for the socket to appear
  for _ in range(50):
    if os.path.exists(socket_path):
      break
    time.sleep(0.05)
  return svc


def _send_request(socket_path: str, req: dict) -> dict:
  c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
  c.connect(socket_path)
  c.sendall((json.dumps(req) + "\n").encode("utf-8"))
  c.shutdown(socket.SHUT_WR)
  buf = b""
  while True:
    chunk = c.recv(4096)
    if not chunk:
      break
    buf += chunk
  c.close()
  return json.loads(buf.strip().decode("utf-8"))


class TestSocketRoundtrip:
  def test_ping(self, tmp_path):
    sock = _pick_socket_path(tmp_path)
    _start_service_thread(sock)
    r = _send_request(sock, {"request_type": "ping"})
    assert r["ok"] is True

  def test_heatmap_request(self, tmp_path):
    sock = _pick_socket_path(tmp_path)
    _start_service_thread(sock)
    r = _send_request(sock, {
      "request_type": "heatmap_next_points",
      "center_xyz": [50.0, 50.0, 25.0],
      "prev_u_xyz": [1.0, 0.0, 0.0],
      "prev_v_xyz": [0.0, 1.0, 0.0],
      "prev_diag_xyz": [1.0, 1.0, 0.0],
    })
    assert r["ok"] is True
    assert isinstance(r["next_points"], list)

  def test_bad_json(self, tmp_path):
    sock = _pick_socket_path(tmp_path)
    _start_service_thread(sock)
    c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    c.connect(sock)
    c.sendall(b"not valid json\n")
    c.shutdown(socket.SHUT_WR)
    buf = b""
    while True:
      chunk = c.recv(4096)
      if not chunk:
        break
      buf += chunk
    c.close()
    r = json.loads(buf.strip())
    assert r["ok"] is False

  def test_unknown_type_over_socket(self, tmp_path):
    sock = _pick_socket_path(tmp_path)
    _start_service_thread(sock)
    r = _send_request(sock, {"request_type": "whatever"})
    assert r["ok"] is False


# ---------------------------------------------------------------------------
# predict_next_points / predict_displacement
# ---------------------------------------------------------------------------

class TestPredictionHelpers:
  def _svc(self):
    return NeuralTraceService(model_path="/nonexistent/model.npy",
                              socket_path="/tmp/unused.sock")

  def test_predict_next_points_shape(self):
    svc = self._svc()
    patch = np.random.rand(64, 64).astype(np.float32)
    pts, scores = svc.predict_next_points(patch)
    assert len(pts) > 0
    assert len(pts) == len(scores)
    assert len(pts[0]) == 3

  def test_predict_displacement_shape(self):
    svc = self._svc()
    patch = np.random.rand(32, 32).astype(np.float32)
    disp = svc.predict_displacement(patch)
    assert disp.ndim == 3         # (out_ch, H, W)
    assert disp.shape[1] == 32
    assert disp.shape[2] == 32

  def test_predict_next_points_finite(self):
    svc = self._svc()
    patch = np.random.rand(64, 64).astype(np.float32)
    pts, scores = svc.predict_next_points(patch)
    for pt in pts:
      assert all(np.isfinite(v) for v in pt)
    assert all(np.isfinite(s) for s in scores)
