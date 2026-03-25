from __future__ import annotations
import json
import math
import struct
from typing import Sequence

try:
  import numpy as np
  _HAS_NUMPY = True
except ImportError:
  _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gauss(d2: float, sigma: float) -> float:
  """Unnormalized Gaussian weight for squared distance d2."""
  return math.exp(-0.5 * d2 / (sigma * sigma)) if sigma > 0 else (1.0 if d2 == 0 else 0.0)


def _clamp(v: float, lo: float, hi: float) -> float:
  return lo if v < lo else (hi if v > hi else v)


# ---------------------------------------------------------------------------
# SurfaceEdit — undo record
# ---------------------------------------------------------------------------

class SurfaceEdit:
  """Records the vertices changed by one tool application; supports undo."""

  def __init__(self, saved: dict[tuple[int, int], tuple[float, float, float]]):
    # saved: {(row, col): original_xyz}
    self._saved = saved

  def undo(self, surf: QuadSurface) -> None:
    """Restore the surface to its state before this edit."""
    for (row, col), xyz in self._saved.items():
      surf.set(row, col, xyz)


# ---------------------------------------------------------------------------
# QuadSurface
# ---------------------------------------------------------------------------

class QuadSurface:
  """Regular rows×cols grid of 3-D points (u,v parameterisation).

  Vertices are stored row-major as flat lists of (x, y, z) tuples for
  zero-dependency operation. When numpy is available, array views are used
  for batch operations.
  """

  def __init__(self, rows: int, cols: int):
    if rows < 1 or cols < 1:
      raise ValueError(f"rows and cols must be >= 1, got {rows}×{cols}")
    self.rows = rows
    self.cols = cols
    # Flat list of (x, y, z); default all zeros
    self._pts: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(rows * cols)]

  # ---- index helpers -------------------------------------------------------

  def _idx(self, row: int, col: int) -> int:
    if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
      raise IndexError(f"({row},{col}) out of bounds for {self.rows}×{self.cols} surface")
    return row * self.cols + col

  # ---- point access --------------------------------------------------------

  def get(self, row: int, col: int) -> tuple[float, float, float]:
    p = self._pts[self._idx(row, col)]
    return (p[0], p[1], p[2])

  def set(self, row: int, col: int, xyz: Sequence[float]) -> None:
    idx = self._idx(row, col)
    self._pts[idx] = [float(xyz[0]), float(xyz[1]), float(xyz[2])]

  # ---- normal at a grid point (central differences) -----------------------

  def normal(self, row: int, col: int) -> tuple[float, float, float]:
    """Unit normal at (row, col) via central-difference cross-product."""
    r0, c0 = max(0, row - 1), max(0, col - 1)
    r1, c1 = min(self.rows - 1, row + 1), min(self.cols - 1, col + 1)
    du = [self._pts[row * self.cols + c1][i] - self._pts[row * self.cols + c0][i] for i in range(3)]
    dv = [self._pts[r1 * self.cols + col][i] - self._pts[r0 * self.cols + col][i] for i in range(3)]
    # cross product du × dv
    nx = du[1] * dv[2] - du[2] * dv[1]
    ny = du[2] * dv[0] - du[0] * dv[2]
    nz = du[0] * dv[1] - du[1] * dv[0]
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-9:
      return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)

  # ---- serialisation -------------------------------------------------------

  def save(self, path: str) -> None:
    """Save to a JSON file."""
    data = {
      "rows": self.rows,
      "cols": self.cols,
      "points": self._pts,
    }
    with open(path, "w", encoding="utf-8") as f:
      json.dump(data, f)

  @classmethod
  def load(cls, path: str) -> QuadSurface:
    """Load from a JSON file previously written by save()."""
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    surf = cls(data["rows"], data["cols"])
    pts = data["points"]
    if len(pts) != surf.rows * surf.cols:
      raise ValueError(f"expected {surf.rows * surf.cols} points, got {len(pts)}")
    surf._pts = [[float(p[0]), float(p[1]), float(p[2])] for p in pts]
    return surf

  # ---- misc ----------------------------------------------------------------

  def __repr__(self) -> str:
    return f"QuadSurface(rows={self.rows}, cols={self.cols})"


# ---------------------------------------------------------------------------
# Tool: brush_apply
# ---------------------------------------------------------------------------

def brush_apply(surf: QuadSurface, u: float, v: float, delta: float,
                radius: float = 5.0, sigma: float = 2.0) -> SurfaceEdit:
  """Displace vertices near grid position (u, v) by delta along their normal.

  Args:
    surf:   The surface to modify in-place.
    u, v:   Centre of the brush in grid coordinates (float row/col).
    delta:  Displacement amount along the vertex normal.
    radius: Influence radius in grid units.
    sigma:  Gaussian falloff standard deviation.

  Returns:
    A SurfaceEdit that can be passed to edit.undo(surf).
  """
  r_int = math.ceil(radius)
  row_c = int(round(u))
  col_c = int(round(v))

  saved: dict[tuple[int, int], tuple[float, float, float]] = {}

  for row in range(max(0, row_c - r_int), min(surf.rows, row_c + r_int + 1)):
    for col in range(max(0, col_c - r_int), min(surf.cols, col_c + r_int + 1)):
      dr = row - u
      dc = col - v
      d2 = dr * dr + dc * dc
      if d2 > radius * radius:
        continue
      w = _gauss(d2, sigma)
      if w < 1e-6:
        continue
      key = (row, col)
      saved[key] = surf.get(row, col)
      nx, ny, nz = surf.normal(row, col)
      p = surf._pts[surf._idx(row, col)]
      p[0] += w * delta * nx
      p[1] += w * delta * ny
      p[2] += w * delta * nz

  return SurfaceEdit(saved)


# ---------------------------------------------------------------------------
# Tool: line_apply
# ---------------------------------------------------------------------------

def line_apply(surf: QuadSurface, u0: float, v0: float, u1: float, v1: float,
               delta: float, radius: float = 5.0, sigma: float = 2.0) -> SurfaceEdit:
  """Displace vertices near the line segment (u0,v0)→(u1,v1) by delta along normals.

  Samples the line at sub-grid intervals and accumulates Gaussian weights so
  that each vertex is displaced at most once (by its peak weight).
  """
  length = math.sqrt((u1 - u0) ** 2 + (v1 - v0) ** 2)
  steps = max(1, int(math.ceil(length)))
  peak_weight: dict[tuple[int, int], float] = {}

  for s in range(steps + 1):
    t = s / steps
    pu = u0 + t * (u1 - u0)
    pv = v0 + t * (v1 - v0)
    r_int = math.ceil(radius)
    row_c = int(round(pu))
    col_c = int(round(pv))
    for row in range(max(0, row_c - r_int), min(surf.rows, row_c + r_int + 1)):
      for col in range(max(0, col_c - r_int), min(surf.cols, col_c + r_int + 1)):
        dr = row - pu
        dc = col - pv
        d2 = dr * dr + dc * dc
        if d2 > radius * radius:
          continue
        w = _gauss(d2, sigma)
        key = (row, col)
        if w > peak_weight.get(key, 0.0):
          peak_weight[key] = w

  saved: dict[tuple[int, int], tuple[float, float, float]] = {}
  for (row, col), w in peak_weight.items():
    if w < 1e-6:
      continue
    key = (row, col)
    saved[key] = surf.get(row, col)
    nx, ny, nz = surf.normal(row, col)
    p = surf._pts[surf._idx(row, col)]
    p[0] += w * delta * nx
    p[1] += w * delta * ny
    p[2] += w * delta * nz

  return SurfaceEdit(saved)


# ---------------------------------------------------------------------------
# Tool: pushpull_apply
# ---------------------------------------------------------------------------

def pushpull_apply(surf: QuadSurface, u: float, v: float,
                   push_amount: float, radius: float = 5.0, sigma: float = 2.0) -> SurfaceEdit:
  """Uniformly displace a region along each vertex's normal (push/pull).

  Identical to brush_apply but the intent is a uniform region displacement
  rather than a sculpting stroke.
  """
  return brush_apply(surf, u, v, push_amount, radius=radius, sigma=sigma)
