"""
volatile.auto_seg
=================
Automatic scroll segmentation pipeline — a pure-numpy port of the core
ThaumatoAnakalyptor algorithms (Julian Schilliger, Vesuvius Challenge 2023/24).

Pipeline
--------
  1. detect_surfaces  — 3-D Sobel + bell-curve thresholding on 1st/2nd derivatives
  2. extract_instances — 3-D connected components on the surface mask
  3. build_graph       — spatial adjacency graph between instance bounding boxes
  4. stitch_sheets     — belief-propagation / greedy spanning-tree stitching
  5. sheets_to_meshes  — polynomial surface fit per sheet → QuadSurface

No torch / open3d / numba required; all operations use numpy (+ optional scipy
for the polynomial least-squares fit).

Style: 2-space indent, 150-char lines.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers — 3-D Sobel (pure numpy)
# ---------------------------------------------------------------------------

# 3×3×3 Sobel kernels identical to ThaumatoAnakalyptor (lukeboi formulation)
_SOBEL_X = np.array([
  [[[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
   [[ 2,  0, -2], [ 4,  0, -4], [ 2,  0, -2]],
   [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]]],
], dtype=np.float32).squeeze(0)  # (3,3,3)

_SOBEL_Y = _SOBEL_X.transpose(2, 1, 0)  # transpose axes 0,2
_SOBEL_Z = _SOBEL_X.transpose(0, 2, 1)  # transpose axes 1,2


def _conv3d_full(vol: np.ndarray, kernel: np.ndarray) -> np.ndarray:
  """3-D separable-ish convolution via sliding window (same padding)."""
  from scipy.ndimage import convolve
  return convolve(vol.astype(np.float32), kernel, mode="reflect")


def _sobel_gradients(vol: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Return (Gx, Gy, Gz) gradient volumes via 3-D Sobel."""
  try:
    Gx = _conv3d_full(vol, _SOBEL_X)
    Gy = _conv3d_full(vol, _SOBEL_Y)
    Gz = _conv3d_full(vol, _SOBEL_Z)
  except ImportError:
    # fallback: numpy gradient (less accurate at boundaries)
    Gz, Gy, Gx = np.gradient(vol.astype(np.float32))
  return Gx, Gy, Gz


def _scale_to_m1_1(arr: np.ndarray) -> np.ndarray:
  """Clip to [-1000,1000] then normalise to [-1, 1]."""
  clipped = np.clip(arr, -1000.0, 1000.0)
  scale = max(float(np.abs(clipped).max()), 1e-8)
  return clipped / scale


def _uniform_blur3d(vol: np.ndarray, size: int = 3) -> np.ndarray:
  """Box (uniform) blur in 3-D."""
  try:
    from scipy.ndimage import uniform_filter
    return uniform_filter(vol.astype(np.float32), size=size, mode="reflect")
  except ImportError:
    return vol.astype(np.float32)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SurfaceResult:
  """Output of detect_surfaces."""
  mask: np.ndarray         # bool (D,H,W) — surface voxels
  normals: np.ndarray      # float32 (D,H,W,3) — unit normal per voxel (0 for non-surface)
  points: np.ndarray       # float32 (N,3) — (z,y,x) coords of surface voxels
  point_normals: np.ndarray  # float32 (N,3) — normals at those points


@dataclass
class InstanceResult:
  """Output of extract_instances."""
  labels: np.ndarray       # int32 (D,H,W); 0 = background, 1..K = instances
  n_instances: int
  # Per-instance metadata (length K+1; index 0 unused)
  centroids: list[np.ndarray]     # (3,) float32 zyx centroid
  bboxes: list[tuple]            # (z0,y0,x0,z1,y1,x1) int


@dataclass
class AdjacencyGraph:
  """Output of build_graph.  Nodes are instance IDs 1..K."""
  n_nodes: int
  edges: list[tuple[int, int, float]]  # (i, j, overlap_score)
  # adjacency list for fast lookup
  adj: dict[int, list[tuple[int, float]]] = field(default_factory=dict)

  def __post_init__(self):
    self.adj = {i: [] for i in range(1, self.n_nodes + 1)}
    for i, j, w in self.edges:
      self.adj[i].append((j, w))
      self.adj[j].append((i, w))


@dataclass
class QuadSurface:
  """A surface described by a polynomial fit over a regular grid."""
  points: np.ndarray     # float32 (N,3) — (z,y,x) surface sample points
  normals: np.ndarray    # float32 (N,3) — normals at sample points
  instance_ids: list[int]
  coeffs: np.ndarray | None = None  # polynomial coefficients (if fitted)
  metadata: dict[str, Any] = field(default_factory=dict)

  def save(self, path: str) -> None:
    """Serialise to JSON (points + normals as nested lists)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    obj = {
      "instance_ids": self.instance_ids,
      "n_points": len(self.points),
      "points": self.points.tolist(),
      "normals": self.normals.tolist(),
      "coeffs": self.coeffs.tolist() if self.coeffs is not None else None,
      "metadata": self.metadata,
    }
    with open(path, "w") as fh:
      json.dump(obj, fh)
    log.info("Saved surface with %d points → %s", len(self.points), path)

  @classmethod
  def load(cls, path: str) -> "QuadSurface":
    with open(path) as fh:
      obj = json.load(fh)
    return cls(
      points=np.array(obj["points"], dtype=np.float32),
      normals=np.array(obj["normals"], dtype=np.float32),
      instance_ids=obj["instance_ids"],
      coeffs=np.array(obj["coeffs"], dtype=np.float32) if obj.get("coeffs") else None,
      metadata=obj.get("metadata", {}),
    )


# ---------------------------------------------------------------------------
# Connected components (3-D, 6-connectivity) without scipy
# ---------------------------------------------------------------------------

def _connected_components_3d(mask: np.ndarray) -> tuple[np.ndarray, int]:
  """
  Label connected components in a boolean 3-D mask (6-connectivity).

  Returns (labels int32 array same shape as mask, n_components).
  Uses union-find with path compression for efficiency.
  """
  D, H, W = mask.shape
  labels = np.zeros((D, H, W), dtype=np.int32)
  parent = [0]          # index 0 is unused sentinel; component IDs start at 1
  next_id = [1]

  def find(x: int) -> int:
    while parent[x] != x:
      parent[x] = parent[parent[x]]
      x = parent[x]
    return x

  def union(a: int, b: int) -> None:
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[rb] = ra

  # First pass: assign provisional labels
  for z in range(D):
    for y in range(H):
      for x in range(W):
        if not mask[z, y, x]:
          continue
        neighbours = []
        if z > 0 and labels[z - 1, y, x]:
          neighbours.append(labels[z - 1, y, x])
        if y > 0 and labels[z, y - 1, x]:
          neighbours.append(labels[z, y - 1, x])
        if x > 0 and labels[z, y, x - 1]:
          neighbours.append(labels[z, y, x - 1])
        if not neighbours:
          lid = next_id[0]
          next_id[0] += 1
          parent.append(lid)   # parent[lid] = lid
          labels[z, y, x] = lid
        else:
          root = find(neighbours[0])
          for n in neighbours[1:]:
            union(root, find(n))
          labels[z, y, x] = root

  # Second pass: flatten labels to contiguous IDs
  remap: dict[int, int] = {}
  new_id = [1]
  for z in range(D):
    for y in range(H):
      for x in range(W):
        if labels[z, y, x] == 0:
          continue
        root = find(labels[z, y, x])
        if root not in remap:
          remap[root] = new_id[0]
          new_id[0] += 1
        labels[z, y, x] = remap[root]

  return labels, new_id[0] - 1


# Vectorised connected components via scipy when available (much faster)
def _connected_components_3d_fast(mask: np.ndarray) -> tuple[np.ndarray, int]:
  try:
    from scipy.ndimage import label
    struct = np.ones((3, 3, 3), dtype=int)  # 26-connectivity (superset of 6)
    labels, n = label(mask, structure=struct)
    return labels.astype(np.int32), int(n)
  except ImportError:
    return _connected_components_3d(mask)


# ---------------------------------------------------------------------------
# AutoSegmenter
# ---------------------------------------------------------------------------

class AutoSegmenter:
  """
  Automatic scroll segmentation pipeline (ThaumatoAnakalyptor port).

  All steps operate on numpy arrays only; no GPU required.

  Args:
    blur_size:          box-blur radius for pre-smoothing (voxels)
    threshold_der:      minimum first-derivative magnitude to flag a surface voxel
    threshold_der2:     maximum second-derivative magnitude (near-zero = inflection)
    min_instance_size:  discard connected components smaller than this (voxels)
    adjacency_dilation: voxel dilation radius for bbox overlap detection
    poly_degree:        polynomial degree for surface fitting (1=plane, 2=quadratic)
  """

  def __init__(
    self,
    blur_size: int = 3,
    threshold_der: float = 0.1,
    threshold_der2: float = 0.05,
    min_instance_size: int = 10,
    adjacency_dilation: int = 3,
    poly_degree: int = 2,
  ):
    self.blur_size = blur_size
    self.threshold_der = threshold_der
    self.threshold_der2 = threshold_der2
    self.min_instance_size = min_instance_size
    self.adjacency_dilation = adjacency_dilation
    self.poly_degree = poly_degree

  # --------------------------------------------------------------------------
  # Step 1: Surface detection
  # --------------------------------------------------------------------------

  def detect_surfaces(self, volume: np.ndarray, threshold: float = 0.5) -> SurfaceResult:
    """
    Detect surface voxels using 3-D Sobel gradient analysis.

    Mirrors ThaumatoAnakalyptor surface_detection():
      1. Box-blur the volume
      2. Compute 3-D Sobel gradient → gradient vectors at each voxel
      3. Compute per-voxel mean local normal direction via a coarse window
      4. Project gradient onto normals → signed first derivative along normal
      5. Compute second derivative in same direction
      6. Flag voxels where |d2| < threshold_der2 and |d1| > threshold_der
         (inflection point along normal = surface)

    Args:
      volume:    float32 (D,H,W) — normalised to [0,1]
      threshold: unused in the primary path; kept for API compatibility

    Returns:
      SurfaceResult with mask, normals, points, point_normals
    """
    vol = volume.astype(np.float32)

    # 1. Blur
    blurred = _uniform_blur3d(vol, self.blur_size)

    # 2. Sobel gradients
    Gx, Gy, Gz = _sobel_gradients(blurred)
    grad_vec = np.stack([Gz, Gy, Gx], axis=-1)          # (D,H,W,3)  → z,y,x order

    # 3. Compute mean normal direction: subsample and interpolate back
    normal_field = self._compute_normal_field(grad_vec)  # (D,H,W,3)

    # 4. First derivative (projection of gradient onto normals)
    d1 = self._project_onto_normals(grad_vec, normal_field)  # (D,H,W)
    d1 = _scale_to_m1_1(d1)

    # 5. Second derivative: re-apply Sobel to d1, project onto same normals
    d1_Gx, d1_Gy, d1_Gz = _sobel_gradients(d1)
    d1_grad = np.stack([d1_Gz, d1_Gy, d1_Gx], axis=-1)
    d2 = self._project_onto_normals(d1_grad, normal_field)   # (D,H,W)
    d2 = _scale_to_m1_1(d2)

    # 6. Bell-curve threshold: inflection (|d2| small) with strong gradient (|d1| large)
    mask_recto = (np.abs(d2) < self.threshold_der2) & (d1 > self.threshold_der)
    mask_verso = (np.abs(d2) < self.threshold_der2) & (d1 < -self.threshold_der)
    mask = mask_recto | mask_verso

    # Normals: use the precomputed normal field where surface; flip verso side
    surface_normals = normal_field.copy()
    surface_normals[mask_verso] *= -1.0
    # Zero out non-surface normals (keep memory, avoids a separate sparse store)
    surface_normals[~mask] = 0.0

    # Collect point coords
    coords = np.argwhere(mask).astype(np.float32)     # (N,3) zyx
    if len(coords) > 0:
      pn = surface_normals[mask].astype(np.float32)   # (N,3)
      norms = np.linalg.norm(pn, axis=1, keepdims=True)
      pn /= np.where(norms > 1e-8, norms, 1.0)
    else:
      pn = np.zeros((0, 3), dtype=np.float32)

    log.debug("detect_surfaces: %d surface voxels (recto=%d, verso=%d)",
              int(mask.sum()), int(mask_recto.sum()), int(mask_verso.sum()))

    return SurfaceResult(mask=mask, normals=surface_normals.astype(np.float32),
                         points=coords, point_normals=pn)

  def _compute_normal_field(self, grad_vec: np.ndarray, window: int = 16, stride: int = 16) -> np.ndarray:
    """
    Estimate mean normal direction at each voxel by averaging gradient vectors
    in non-overlapping windows, then interpolating back to full resolution.

    This is a simplified version of ThaumatoAnakalyptor's vector_convolution.
    """
    D, H, W, _ = grad_vec.shape
    wz = max(1, window)
    wy = max(1, window)
    wx = max(1, window)
    sz = max(1, stride)
    sy = max(1, stride)
    sx = max(1, stride)

    out_z = max(1, (D - wz) // sz + 1)
    out_y = max(1, (H - wy) // sy + 1)
    out_x = max(1, (W - wx) // sx + 1)
    coarse = np.zeros((out_z, out_y, out_x, 3), dtype=np.float32)

    for iz in range(out_z):
      z0, z1 = iz * sz, min(D, iz * sz + wz)
      for iy in range(out_y):
        y0, y1 = iy * sy, min(H, iy * sy + wy)
        for ix in range(out_x):
          x0, x1 = ix * sx, min(W, ix * sx + wx)
          patch = grad_vec[z0:z1, y0:y1, x0:x1].reshape(-1, 3)
          nz = np.any(patch != 0, axis=1)
          if nz.any():
            v = patch[nz].mean(axis=0)
          else:
            v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
          norm = float(np.linalg.norm(v))
          coarse[iz, iy, ix] = v / (norm if norm > 1e-8 else 1.0)

    # Trilinear upsample coarse → full resolution
    return self._trilinear_upsample(coarse, (D, H, W))

  @staticmethod
  def _trilinear_upsample(coarse: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Trilinear interpolation from (Dc,Hc,Wc,3) to (D,H,W,3)."""
    Dc, Hc, Wc, _ = coarse.shape
    D, H, W = target_shape
    zi = np.linspace(0, Dc - 1, D)
    yi = np.linspace(0, Hc - 1, H)
    xi = np.linspace(0, Wc - 1, W)

    try:
      from scipy.ndimage import map_coordinates
      out = np.zeros((D, H, W, 3), dtype=np.float32)
      zg, yg, xg = np.meshgrid(zi, yi, xi, indexing="ij")
      for c in range(3):
        out[..., c] = map_coordinates(coarse[..., c], [zg, yg, xg], order=1, mode="nearest")
      # Renormalise
      n = np.linalg.norm(out, axis=-1, keepdims=True)
      out /= np.where(n > 1e-8, n, 1.0)
      return out
    except ImportError:
      # Nearest-neighbour fallback
      zi_idx = np.round(zi).astype(int).clip(0, Dc - 1)
      yi_idx = np.round(yi).astype(int).clip(0, Hc - 1)
      xi_idx = np.round(xi).astype(int).clip(0, Wc - 1)
      out = coarse[np.ix_(zi_idx, yi_idx, xi_idx)]
      n = np.linalg.norm(out, axis=-1, keepdims=True)
      out /= np.where(n > 1e-8, n, 1.0)
      return out

  @staticmethod
  def _project_onto_normals(grad_vec: np.ndarray, normal_field: np.ndarray) -> np.ndarray:
    """Signed projection of gradient vectors onto normal field; returns (D,H,W) scalar."""
    # dot product along last axis
    return (grad_vec * normal_field).sum(axis=-1)

  # --------------------------------------------------------------------------
  # Step 2: Instance extraction
  # --------------------------------------------------------------------------

  def extract_instances(self, surface_mask: np.ndarray, normals: np.ndarray) -> InstanceResult:
    """
    Cluster surface voxels into sheet instances via 3-D connected components.

    Small components (< min_instance_size) are discarded.

    Args:
      surface_mask: bool (D,H,W)
      normals:      float32 (D,H,W,3)

    Returns:
      InstanceResult with labels, n_instances, centroids, bboxes
    """
    raw_labels, n_raw = _connected_components_3d_fast(surface_mask)
    log.debug("extract_instances: %d raw components", n_raw)

    # Filter small components
    labels = np.zeros_like(raw_labels)
    new_id = 1
    centroids: list[np.ndarray] = [np.zeros(3, dtype=np.float32)]   # index 0 unused
    bboxes: list[tuple] = [(0, 0, 0, 0, 0, 0)]

    for old_id in range(1, n_raw + 1):
      vox = np.argwhere(raw_labels == old_id)
      if len(vox) < self.min_instance_size:
        continue
      labels[raw_labels == old_id] = new_id
      cen = vox.mean(axis=0).astype(np.float32)
      z0, y0, x0 = vox.min(axis=0)
      z1, y1, x1 = vox.max(axis=0)
      centroids.append(cen)
      bboxes.append((int(z0), int(y0), int(x0), int(z1), int(y1), int(x1)))
      new_id += 1

    n_inst = new_id - 1
    log.debug("extract_instances: %d instances after size filter (min=%d)",
              n_inst, self.min_instance_size)
    return InstanceResult(labels=labels, n_instances=n_inst, centroids=centroids, bboxes=bboxes)

  # --------------------------------------------------------------------------
  # Step 3: Build adjacency graph
  # --------------------------------------------------------------------------

  def build_graph(self, instances: InstanceResult) -> AdjacencyGraph:
    """
    Build a stitching graph where nodes = sheet instances, edges = spatial
    adjacency (bounding boxes overlap after dilation).

    Edge weight = fraction of bounding box volume that overlaps.

    Args:
      instances: InstanceResult from extract_instances

    Returns:
      AdjacencyGraph
    """
    K = instances.n_instances
    d = self.adjacency_dilation
    edges: list[tuple[int, int, float]] = []

    for i in range(1, K + 1):
      z0i, y0i, x0i, z1i, y1i, x1i = instances.bboxes[i]
      # Dilate bbox
      az0, ay0, ax0 = z0i - d, y0i - d, x0i - d
      az1, ay1, ax1 = z1i + d, y1i + d, x1i + d

      for j in range(i + 1, K + 1):
        z0j, y0j, x0j, z1j, y1j, x1j = instances.bboxes[j]

        # Check overlap of dilated bbox i with bbox j
        if (az0 <= z1j and az1 >= z0j and
            ay0 <= y1j and ay1 >= y0j and
            ax0 <= x1j and ax1 >= x0j):
          # Overlap volume as fraction of smaller instance box
          oz = max(0, min(az1, z1j) - max(az0, z0j))
          oy = max(0, min(ay1, y1j) - max(ay0, y0j))
          ox = max(0, min(ax1, x1j) - max(ax0, x0j))
          overlap = float(oz * oy * ox)
          vi = (z1i - z0i + 1) * (y1i - y0i + 1) * (x1i - x0i + 1)
          vj = (z1j - z0j + 1) * (y1j - y0j + 1) * (x1j - x0j + 1)
          score = overlap / max(min(vi, vj), 1)
          edges.append((i, j, score))

    log.debug("build_graph: %d nodes, %d edges", K, len(edges))
    return AdjacencyGraph(n_nodes=K, edges=edges)

  # --------------------------------------------------------------------------
  # Step 4: Stitch sheets
  # --------------------------------------------------------------------------

  def stitch_sheets(self, graph: AdjacencyGraph) -> list[list[int]]:
    """
    Group instances into coherent sheets by finding connected components in
    the adjacency graph (simplified belief propagation via greedy union-find).

    Each connected component in the graph corresponds to one sheet.

    Args:
      graph: AdjacencyGraph from build_graph

    Returns:
      list of sheets; each sheet is a sorted list of instance IDs
    """
    K = graph.n_nodes
    parent = list(range(K + 1))

    def find(x: int) -> int:
      while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
      return x

    def union(a: int, b: int) -> None:
      ra, rb = find(a), find(b)
      if ra != rb:
        parent[rb] = ra

    # Merge by adjacency (edges already represent spatial connectivity)
    for i, j, _ in graph.edges:
      union(i, j)

    # Group by root
    sheet_map: dict[int, list[int]] = {}
    for i in range(1, K + 1):
      root = find(i)
      sheet_map.setdefault(root, []).append(i)

    sheets = sorted(sheet_map.values(), key=len, reverse=True)
    log.debug("stitch_sheets: %d sheets from %d instances", len(sheets), K)
    return sheets

  # --------------------------------------------------------------------------
  # Step 5: Sheets to meshes
  # --------------------------------------------------------------------------

  def sheets_to_meshes(
    self,
    sheets: list[list[int]],
    instances: InstanceResult,
    surface: SurfaceResult | None = None,
  ) -> list[QuadSurface]:
    """
    Convert each sheet (list of instance IDs) to a QuadSurface by fitting a
    polynomial surface to the instance voxel centroids.

    Args:
      sheets:    list of sheets from stitch_sheets
      instances: InstanceResult (for labels and centroids)
      surface:   optional SurfaceResult to get per-voxel normals

    Returns:
      list of QuadSurface, one per sheet
    """
    meshes: list[QuadSurface] = []
    for sheet_ids in sheets:
      pts, norms = self._collect_sheet_points(sheet_ids, instances, surface)
      if len(pts) < 4:
        log.debug("sheets_to_meshes: sheet %s has only %d points — skipping", sheet_ids, len(pts))
        continue
      coeffs = self._fit_polynomial_surface(pts)
      surf = QuadSurface(points=pts, normals=norms, instance_ids=sheet_ids,
                         coeffs=coeffs, metadata={"n_instances": len(sheet_ids)})
      meshes.append(surf)

    log.info("sheets_to_meshes: %d meshes from %d sheets", len(meshes), len(sheets))
    return meshes

  def _collect_sheet_points(
    self,
    sheet_ids: list[int],
    instances: InstanceResult,
    surface: SurfaceResult | None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Gather (N,3) point cloud and (N,3) normals for all instances in a sheet."""
    pts_list: list[np.ndarray] = []
    norms_list: list[np.ndarray] = []

    for iid in sheet_ids:
      vox = np.argwhere(instances.labels == iid).astype(np.float32)  # (M,3) zyx
      if len(vox) == 0:
        continue
      pts_list.append(vox)
      if surface is not None and surface.normals is not None:
        # Collect normals at those voxels
        idx = vox.astype(int)
        n = surface.normals[idx[:, 0], idx[:, 1], idx[:, 2]]
        norms_list.append(n)
      else:
        norms_list.append(np.zeros((len(vox), 3), dtype=np.float32))

    if not pts_list:
      return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    pts = np.concatenate(pts_list, axis=0)
    norms = np.concatenate(norms_list, axis=0)
    return pts.astype(np.float32), norms.astype(np.float32)

  def _fit_polynomial_surface(self, points: np.ndarray) -> np.ndarray | None:
    """
    Fit a degree-n polynomial z = f(y, x) to the given (N,3) points in zyx order.

    Returns coefficient vector or None if the fit fails.
    """
    if len(points) < 3:
      return None
    z, y, x = points[:, 0], points[:, 1], points[:, 2]
    n = self.poly_degree
    # Build Vandermonde matrix with terms y^i * x^j, i+j <= n
    cols = []
    for i in range(n + 1):
      for j in range(n + 1 - i):
        cols.append((y ** i) * (x ** j))
    A = np.column_stack(cols)
    try:
      from scipy.linalg import lstsq as sp_lstsq
      coeffs, _, _, _ = sp_lstsq(A, z)
    except ImportError:
      coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs.astype(np.float32)

  # --------------------------------------------------------------------------
  # Full pipeline
  # --------------------------------------------------------------------------

  def run_pipeline(self, volume_path: str, output_dir: str) -> list[QuadSurface]:
    """
    Run the full segmentation pipeline end-to-end.

    Args:
      volume_path: path to a .npy (D,H,W) float32 volume or zarr directory
      output_dir:  directory to write sheet_N.json files

    Returns:
      list of QuadSurface objects
    """
    vol = self._load_volume(volume_path)
    log.info("run_pipeline: volume shape %s", vol.shape)

    surfaces = self.detect_surfaces(vol)
    log.info("run_pipeline: %d surface voxels", int(surfaces.mask.sum()))

    inst = self.extract_instances(surfaces.mask, surfaces.normals)
    log.info("run_pipeline: %d instances", inst.n_instances)

    graph = self.build_graph(inst)
    log.info("run_pipeline: graph has %d edges", len(graph.edges))

    sheets = self.stitch_sheets(graph)
    log.info("run_pipeline: %d sheets", len(sheets))

    meshes = self.sheets_to_meshes(sheets, inst, surfaces)
    log.info("run_pipeline: %d meshes", len(meshes))

    os.makedirs(output_dir, exist_ok=True)
    for i, mesh in enumerate(meshes):
      mesh.save(f"{output_dir}/sheet_{i}.json")

    return meshes

  # --------------------------------------------------------------------------
  # I/O helpers
  # --------------------------------------------------------------------------

  @staticmethod
  def _load_volume(path: str) -> np.ndarray:
    """Load a volume from .npy, .npz, or zarr store."""
    p = Path(path)
    if p.suffix == ".npy":
      return np.load(str(p)).astype(np.float32)
    if p.suffix == ".npz":
      data = np.load(str(p))
      key = list(data.files)[0]
      return data[key].astype(np.float32)
    # Try zarr
    try:
      import zarr
      store = zarr.open(str(p), mode="r")
      arr = store[:]
      return np.asarray(arr, dtype=np.float32)
    except Exception as exc:
      raise ValueError(f"Cannot load volume from {path}: {exc}") from exc
