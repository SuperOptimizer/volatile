#pragma once
#include "core/geom.h"
#include "core/vol.h"

// ---------------------------------------------------------------------------
// slicing — sample a volume along an arbitrary plane or quad surface.
// ---------------------------------------------------------------------------

// Sample the volume along an infinite plane at (width x height) pixels.
// Each pixel maps to a world point:
//   world = plane->origin + (col - width/2)*scale*u_axis
//                         + (row - height/2)*scale*v_axis
// out must be caller-allocated float[rows * cols].
void slice_volume_plane(const volume *vol, const plane_surface *plane,
                        float *out, int width, int height, float scale);

// Sample the volume at every (row,col) point of a quad_surface.
// out must be caller-allocated float[surf->rows * surf->cols].
void slice_volume_quad(const volume *vol, const quad_surface *surf,
                       float *out, int rows, int cols);
