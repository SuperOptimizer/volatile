#pragma once
#include "core/geom.h"
#include "core/math.h"

// ---------------------------------------------------------------------------
// surface_index — grid-based spatial hash for fast surface queries.
//
// Divides the bounding box into cells. Each cell stores a list of vertex
// indices from the quad_surface. The grid cell size is chosen automatically
// (~2x median edge length) to keep expected bucket occupancy near 1–4.
// ---------------------------------------------------------------------------

typedef struct surface_index surface_index;

// Build the index. Returns NULL on alloc failure or empty surface.
surface_index *surface_index_build(const quad_surface *surf);
void           surface_index_free(surface_index *idx);

// Find the single nearest surface vertex to `query`.
// Returns the flat vertex index (row*cols + col), or -1 on error.
// Sets *out_dist to the Euclidean distance if out_dist != NULL.
int surface_index_nearest(const surface_index *idx, vec3f query, float *out_dist);

// Find all surface vertices within `radius` of `query`.
// Writes up to `max` vertex indices into out_indices.
// Returns number of results written.
int surface_index_radius(const surface_index *idx, vec3f query, float radius,
                          int *out_indices, int max);
