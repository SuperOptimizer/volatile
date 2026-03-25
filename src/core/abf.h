#pragma once
#include "core/geom.h"
#include <stddef.h>

// ---------------------------------------------------------------------------
// uv_coords — flat array of (u, v) pairs, one per grid vertex.
// Indexed as uv[row * cols + col].
// ---------------------------------------------------------------------------

typedef struct {
  float *u;
  float *v;
  int    count;   // rows * cols
  int    rows;
  int    cols;
} uv_coords;

void uv_coords_free(uv_coords *uv);

// ---------------------------------------------------------------------------
// abf_flatten — Angle-Based Flattening (ABF++) on a quad_surface.
//
// Triangulates the quad grid (each cell → 2 triangles), optimizes interior
// angles with ABF++ Newton steps, then recovers UV positions via LSCM least-
// squares. Uses the built-in sparse CG solver for the symmetric LSCM step;
// the ABF angle-update system is solved analytically per-triangle/vertex.
//
// Returns NULL on failure (degenerate mesh, alloc failure, no valid cells).
// Caller owns the returned uv_coords.
// ---------------------------------------------------------------------------

uv_coords *abf_flatten(const quad_surface *surf);
