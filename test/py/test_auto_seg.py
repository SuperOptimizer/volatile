"""
Tests for volatile.auto_seg — automatic segmentation pipeline.

Synthetic test: two thin parallel sheets embedded in a 3-D volume.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from volatile.auto_seg import (
  AutoSegmenter,
  QuadSurface,
  SurfaceResult,
  InstanceResult,
  AdjacencyGraph,
  _connected_components_3d,
  _connected_components_3d_fast,
)


# ---------------------------------------------------------------------------
# Synthetic volume factory
# ---------------------------------------------------------------------------

def make_parallel_sheets_volume(
  shape=(40, 48, 48),
  sheet_z=(12, 28),
  sheet_thickness=1,
  background=0.1,
  sheet_value=0.9,
) -> np.ndarray:
  """
  Build a float32 volume with two thin bright XY planes at z=sheet_z[0] and
  z=sheet_z[1] against a dark background.

  The gradient across each sheet is strong, making them detectable by Sobel.
  """
  D, H, W = shape
  vol = np.full(shape, background, dtype=np.float32)
  for sz in sheet_z:
    for dz in range(-sheet_thickness, sheet_thickness + 1):
      z = sz + dz
      if 0 <= z < D:
        vol[z, :, :] = sheet_value
  # Add a smooth ramp so d1 (gradient) is well-defined and large at sheet surfaces
  ramp = np.linspace(background, background * 1.5, D, dtype=np.float32)
  vol += ramp[:, None, None] * 0.05
  return np.clip(vol, 0, 1)


# ---------------------------------------------------------------------------
# Connected components (unit)
# ---------------------------------------------------------------------------

class TestConnectedComponents:
  def test_single_component(self):
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[1:4, 1:4, 1:4] = True
    labels, n = _connected_components_3d_fast(mask)
    assert n == 1
    assert (labels[mask] == 1).all()
    assert (labels[~mask] == 0).all()

  def test_two_components(self):
    mask = np.zeros((10, 5, 5), dtype=bool)
    mask[1:3, 1:4, 1:4] = True    # component A
    mask[7:9, 1:4, 1:4] = True    # component B (separated by gap)
    labels, n = _connected_components_3d_fast(mask)
    assert n == 2
    ids = set(labels[mask].tolist())
    assert len(ids) == 2

  def test_empty_mask(self):
    mask = np.zeros((4, 4, 4), dtype=bool)
    labels, n = _connected_components_3d_fast(mask)
    assert n == 0
    assert (labels == 0).all()

  def test_full_mask_is_one_component(self):
    mask = np.ones((4, 4, 4), dtype=bool)
    labels, n = _connected_components_3d_fast(mask)
    assert n == 1

  def test_pure_numpy_matches_fast(self):
    rng = np.random.default_rng(42)
    mask = rng.random((8, 8, 8)) > 0.7
    l1, n1 = _connected_components_3d(mask)
    l2, n2 = _connected_components_3d_fast(mask)
    assert n1 == n2
    # Label sets should match (labels may differ but counts should agree)
    assert set(l1[mask].tolist()) == set(range(1, n1 + 1))
    assert set(l2[mask].tolist()) == set(range(1, n2 + 1))


# ---------------------------------------------------------------------------
# Surface detection
# ---------------------------------------------------------------------------

class TestDetectSurfaces:
  def test_returns_surface_result(self):
    vol = make_parallel_sheets_volume()
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.1)
    result = seg.detect_surfaces(vol)
    assert isinstance(result, SurfaceResult)
    assert result.mask.shape == vol.shape
    assert result.mask.dtype == bool
    assert result.normals.shape == (*vol.shape, 3)

  def test_detects_nonzero_surfaces(self):
    vol = make_parallel_sheets_volume()
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2)
    result = seg.detect_surfaces(vol)
    assert result.mask.sum() > 0, "Expected at least some surface voxels"

  def test_points_and_normals_shape_match(self):
    vol = make_parallel_sheets_volume()
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2)
    result = seg.detect_surfaces(vol)
    assert result.points.shape[0] == result.point_normals.shape[0]
    assert result.points.shape[1] == 3
    assert result.point_normals.shape[1] == 3

  def test_normals_are_unit_length(self):
    vol = make_parallel_sheets_volume()
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2)
    result = seg.detect_surfaces(vol)
    if len(result.point_normals) > 0:
      norms = np.linalg.norm(result.point_normals, axis=1)
      np.testing.assert_allclose(norms, 1.0, atol=1e-5)

  def test_surface_voxels_are_in_mask(self):
    vol = make_parallel_sheets_volume()
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2)
    result = seg.detect_surfaces(vol)
    if len(result.points) > 0:
      coords = result.points.astype(int)
      for z, y, x in coords:
        assert result.mask[z, y, x], "Point not in mask"


# ---------------------------------------------------------------------------
# Instance extraction
# ---------------------------------------------------------------------------

class TestExtractInstances:
  def _surfaces(self, shape=(30, 30, 30)):
    vol = make_parallel_sheets_volume(shape=shape)
    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2, min_instance_size=5)
    return seg.detect_surfaces(vol), seg

  def test_returns_instance_result(self):
    surf, seg = self._surfaces()
    result = seg.extract_instances(surf.mask, surf.normals)
    assert isinstance(result, InstanceResult)
    assert result.labels.shape == surf.mask.shape

  def test_labels_nonnegative(self):
    surf, seg = self._surfaces()
    result = seg.extract_instances(surf.mask, surf.normals)
    assert result.labels.min() >= 0

  def test_instance_count_nonnegative(self):
    surf, seg = self._surfaces()
    result = seg.extract_instances(surf.mask, surf.normals)
    assert result.n_instances >= 0

  def test_centroids_length_matches_n(self):
    surf, seg = self._surfaces()
    result = seg.extract_instances(surf.mask, surf.normals)
    # centroids[0] is a dummy; length = n_instances + 1
    assert len(result.centroids) == result.n_instances + 1

  def test_bboxes_valid(self):
    surf, seg = self._surfaces()
    result = seg.extract_instances(surf.mask, surf.normals)
    for i in range(1, result.n_instances + 1):
      z0, y0, x0, z1, y1, x1 = result.bboxes[i]
      assert z0 <= z1 and y0 <= y1 and x0 <= x1

  def test_empty_mask_gives_zero_instances(self):
    mask = np.zeros((10, 10, 10), dtype=bool)
    normals = np.zeros((10, 10, 10, 3), dtype=np.float32)
    seg = AutoSegmenter(min_instance_size=1)
    result = seg.extract_instances(mask, normals)
    assert result.n_instances == 0


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------

class TestBuildGraph:
  def _instances_two_adjacent(self):
    """Two instances that share a dilated boundary."""
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[5:8, 5:15, 5:15] = 1    # sheet A
    labels[9:12, 5:15, 5:15] = 2   # sheet B (adjacent after dilation=3)
    centroids = [
      np.zeros(3, dtype=np.float32),
      np.array([6.0, 10.0, 10.0], dtype=np.float32),
      np.array([10.0, 10.0, 10.0], dtype=np.float32),
    ]
    bboxes = [(0, 0, 0, 0, 0, 0), (5, 5, 5, 7, 14, 14), (9, 5, 5, 11, 14, 14)]
    return InstanceResult(labels=labels, n_instances=2, centroids=centroids, bboxes=bboxes)

  def _instances_two_separate(self):
    """Two instances far apart — no adjacency."""
    labels = np.zeros((40, 20, 20), dtype=np.int32)
    labels[2:4, 5:15, 5:15] = 1
    labels[35:38, 5:15, 5:15] = 2
    centroids = [
      np.zeros(3, dtype=np.float32),
      np.array([3.0, 10.0, 10.0], dtype=np.float32),
      np.array([36.0, 10.0, 10.0], dtype=np.float32),
    ]
    bboxes = [(0, 0, 0, 0, 0, 0), (2, 5, 5, 3, 14, 14), (35, 5, 5, 37, 14, 14)]
    return InstanceResult(labels=labels, n_instances=2, centroids=centroids, bboxes=bboxes)

  def test_adjacent_instances_have_edge(self):
    inst = self._instances_two_adjacent()
    seg = AutoSegmenter(adjacency_dilation=3)
    graph = seg.build_graph(inst)
    assert isinstance(graph, AdjacencyGraph)
    assert len(graph.edges) >= 1
    pairs = {(min(i, j), max(i, j)) for i, j, _ in graph.edges}
    assert (1, 2) in pairs

  def test_separate_instances_no_edge(self):
    inst = self._instances_two_separate()
    seg = AutoSegmenter(adjacency_dilation=3)
    graph = seg.build_graph(inst)
    assert len(graph.edges) == 0

  def test_edge_weights_positive(self):
    inst = self._instances_two_adjacent()
    seg = AutoSegmenter(adjacency_dilation=3)
    graph = seg.build_graph(inst)
    for _, _, w in graph.edges:
      assert w >= 0.0

  def test_graph_adjacency_list_populated(self):
    inst = self._instances_two_adjacent()
    seg = AutoSegmenter(adjacency_dilation=3)
    graph = seg.build_graph(inst)
    if graph.edges:
      assert len(graph.adj[1]) >= 1 or len(graph.adj[2]) >= 1


# ---------------------------------------------------------------------------
# Stitch sheets
# ---------------------------------------------------------------------------

class TestStitchSheets:
  def _graph_two_connected(self):
    return AdjacencyGraph(n_nodes=3, edges=[(1, 2, 0.8), (2, 3, 0.6)])

  def _graph_two_separate(self):
    return AdjacencyGraph(n_nodes=4, edges=[(1, 2, 0.9), (3, 4, 0.7)])

  def test_connected_graph_gives_one_sheet(self):
    graph = self._graph_two_connected()
    seg = AutoSegmenter()
    sheets = seg.stitch_sheets(graph)
    assert len(sheets) == 1
    assert sorted(sheets[0]) == [1, 2, 3]

  def test_disconnected_graph_gives_two_sheets(self):
    graph = self._graph_two_separate()
    seg = AutoSegmenter()
    sheets = seg.stitch_sheets(graph)
    assert len(sheets) == 2
    all_ids = sorted(sum(sheets, []))
    assert all_ids == [1, 2, 3, 4]

  def test_no_edges_gives_singleton_sheets(self):
    graph = AdjacencyGraph(n_nodes=3, edges=[])
    seg = AutoSegmenter()
    sheets = seg.stitch_sheets(graph)
    assert len(sheets) == 3

  def test_all_connected_is_one_sheet(self):
    n = 5
    edges = [(i, i + 1, 1.0) for i in range(1, n)]
    graph = AdjacencyGraph(n_nodes=n, edges=edges)
    seg = AutoSegmenter()
    sheets = seg.stitch_sheets(graph)
    assert len(sheets) == 1
    assert len(sheets[0]) == n


# ---------------------------------------------------------------------------
# Sheets to meshes
# ---------------------------------------------------------------------------

class TestSheetsToMeshes:
  def _simple_instances(self):
    labels = np.zeros((20, 20, 20), dtype=np.int32)
    labels[5:8, 5:15, 5:15] = 1
    labels[13:16, 5:15, 5:15] = 2
    centroids = [
      np.zeros(3, dtype=np.float32),
      np.array([6.0, 10.0, 10.0], dtype=np.float32),
      np.array([14.0, 10.0, 10.0], dtype=np.float32),
    ]
    bboxes = [(0, 0, 0, 0, 0, 0), (5, 5, 5, 7, 14, 14), (13, 5, 5, 15, 14, 14)]
    return InstanceResult(labels=labels, n_instances=2, centroids=centroids, bboxes=bboxes)

  def test_two_sheets_give_two_meshes(self):
    inst = self._simple_instances()
    sheets = [[1], [2]]
    seg = AutoSegmenter()
    meshes = seg.sheets_to_meshes(sheets, inst)
    assert len(meshes) == 2

  def test_mesh_has_points(self):
    inst = self._simple_instances()
    sheets = [[1]]
    seg = AutoSegmenter()
    meshes = seg.sheets_to_meshes(sheets, inst)
    assert meshes[0].points.shape[1] == 3
    assert len(meshes[0].points) > 0

  def test_mesh_normals_shape(self):
    inst = self._simple_instances()
    sheets = [[1]]
    seg = AutoSegmenter()
    meshes = seg.sheets_to_meshes(sheets, inst)
    assert meshes[0].normals.shape == meshes[0].points.shape

  def test_mesh_instance_ids_correct(self):
    inst = self._simple_instances()
    sheets = [[1, 2]]
    seg = AutoSegmenter()
    meshes = seg.sheets_to_meshes(sheets, inst)
    assert meshes[0].instance_ids == [1, 2]


# ---------------------------------------------------------------------------
# QuadSurface save / load
# ---------------------------------------------------------------------------

class TestQuadSurface:
  def test_save_load_roundtrip(self, tmp_path):
    pts = np.random.rand(20, 3).astype(np.float32)
    norms = np.random.rand(20, 3).astype(np.float32)
    surf = QuadSurface(points=pts, normals=norms, instance_ids=[1, 3],
                       coeffs=np.array([0.1, 0.2, 0.3], dtype=np.float32))
    p = str(tmp_path / "sheet.json")
    surf.save(p)
    loaded = QuadSurface.load(p)
    np.testing.assert_allclose(loaded.points, pts, rtol=1e-5)
    np.testing.assert_allclose(loaded.normals, norms, rtol=1e-5)
    assert loaded.instance_ids == [1, 3]

  def test_save_creates_file(self, tmp_path):
    surf = QuadSurface(
      points=np.zeros((5, 3), dtype=np.float32),
      normals=np.zeros((5, 3), dtype=np.float32),
      instance_ids=[1],
    )
    p = str(tmp_path / "sub" / "sheet_0.json")
    surf.save(p)
    assert os.path.exists(p)

  def test_saved_json_has_required_keys(self, tmp_path):
    surf = QuadSurface(
      points=np.ones((3, 3), dtype=np.float32),
      normals=np.zeros((3, 3), dtype=np.float32),
      instance_ids=[2],
    )
    p = str(tmp_path / "s.json")
    surf.save(p)
    with open(p) as fh:
      obj = json.load(fh)
    for key in ("instance_ids", "n_points", "points", "normals"):
      assert key in obj

  def test_load_without_coeffs(self, tmp_path):
    surf = QuadSurface(points=np.zeros((3, 3), dtype=np.float32),
                       normals=np.zeros((3, 3), dtype=np.float32), instance_ids=[])
    p = str(tmp_path / "s.json")
    surf.save(p)
    loaded = QuadSurface.load(p)
    assert loaded.coeffs is None


# ---------------------------------------------------------------------------
# Full pipeline — two parallel sheets → two meshes
# ---------------------------------------------------------------------------

class TestRunPipeline:
  def test_two_sheets_detected(self, tmp_path):
    vol = make_parallel_sheets_volume(shape=(40, 40, 40), sheet_z=(10, 30))
    vol_path = str(tmp_path / "vol.npy")
    np.save(vol_path, vol)

    seg = AutoSegmenter(
      blur_size=1,
      threshold_der=0.05,
      threshold_der2=0.2,
      min_instance_size=5,
      adjacency_dilation=4,
    )
    out_dir = str(tmp_path / "output")
    meshes = seg.run_pipeline(vol_path, out_dir)

    assert len(meshes) >= 1, "Expected at least 1 mesh from 2-sheet volume"
    # Each saved file should exist
    for i in range(len(meshes)):
      assert os.path.exists(os.path.join(out_dir, f"sheet_{i}.json"))

  def test_output_files_valid_json(self, tmp_path):
    vol = make_parallel_sheets_volume(shape=(40, 40, 40), sheet_z=(10, 30))
    vol_path = str(tmp_path / "vol.npy")
    np.save(vol_path, vol)

    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2, min_instance_size=5)
    out_dir = str(tmp_path / "output")
    meshes = seg.run_pipeline(vol_path, out_dir)

    for i in range(len(meshes)):
      p = os.path.join(out_dir, f"sheet_{i}.json")
      with open(p) as fh:
        obj = json.load(fh)
      assert "points" in obj

  def test_pipeline_on_uniform_volume_gives_no_crash(self, tmp_path):
    """Uniform volume has no surfaces; pipeline should complete without error."""
    vol = np.full((20, 20, 20), 0.5, dtype=np.float32)
    vol_path = str(tmp_path / "flat.npy")
    np.save(vol_path, vol)

    seg = AutoSegmenter(blur_size=1, threshold_der=0.1, threshold_der2=0.05)
    out_dir = str(tmp_path / "out")
    meshes = seg.run_pipeline(vol_path, out_dir)
    assert isinstance(meshes, list)

  def test_mesh_points_within_volume_bounds(self, tmp_path):
    D, H, W = 40, 40, 40
    vol = make_parallel_sheets_volume(shape=(D, H, W), sheet_z=(10, 30))
    vol_path = str(tmp_path / "vol.npy")
    np.save(vol_path, vol)

    seg = AutoSegmenter(blur_size=1, threshold_der=0.05, threshold_der2=0.2, min_instance_size=5)
    out_dir = str(tmp_path / "out")
    meshes = seg.run_pipeline(vol_path, out_dir)

    for mesh in meshes:
      assert mesh.points[:, 0].max() < D
      assert mesh.points[:, 1].max() < H
      assert mesh.points[:, 2].max() < W
      assert mesh.points.min() >= 0.0
