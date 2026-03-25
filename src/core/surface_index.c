#include "core/surface_index.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Internal grid structure
// ---------------------------------------------------------------------------

struct surface_index {
  // Bounding box
  vec3f     min_pt, max_pt;
  // Grid dimensions
  int       gx, gy, gz;   // cells per axis
  float     cell_size;
  // Cell storage: sorted list of vertex indices per cell
  int      *cell_start;   // cell_start[flat_cell]     = offset into data[]
  int      *cell_count;   // cell_count[flat_cell]     = number of entries
  int      *data;          // packed vertex indices
  int       total_cells;
  // Reference to vertices (not owned)
  const vec3f *pts;
  int          npts;
};

static int cell_flat(const surface_index *idx, int cx, int cy, int cz) {
  return cz * idx->gy * idx->gx + cy * idx->gx + cx;
}

static void world_to_cell(const surface_index *idx, vec3f p,
                           int *cx, int *cy, int *cz) {
  float cs = idx->cell_size;
  *cx = (int)((p.x - idx->min_pt.x) / cs);
  *cy = (int)((p.y - idx->min_pt.y) / cs);
  *cz = (int)((p.z - idx->min_pt.z) / cs);
  if (*cx < 0) *cx = 0; else if (*cx >= idx->gx) *cx = idx->gx-1;
  if (*cy < 0) *cy = 0; else if (*cy >= idx->gy) *cy = idx->gy-1;
  if (*cz < 0) *cz = 0; else if (*cz >= idx->gz) *cz = idx->gz-1;
}

// ---------------------------------------------------------------------------
// Build
// ---------------------------------------------------------------------------

surface_index *surface_index_build(const quad_surface *surf) {
  if (!surf || surf->rows < 1 || surf->cols < 1) return NULL;
  int N = surf->rows * surf->cols;

  // Compute bounding box
  vec3f mn = surf->points[0], mx = surf->points[0];
  for (int i = 1; i < N; i++) {
    vec3f p = surf->points[i];
    if (p.x < mn.x) mn.x = p.x;  if (p.x > mx.x) mx.x = p.x;
    if (p.y < mn.y) mn.y = p.y;  if (p.y > mx.y) mx.y = p.y;
    if (p.z < mn.z) mn.z = p.z;  if (p.z > mx.z) mx.z = p.z;
  }
  // Pad slightly to avoid edge cases
  mn.x -= 0.5f; mn.y -= 0.5f; mn.z -= 0.5f;
  mx.x += 0.5f; mx.y += 0.5f; mx.z += 0.5f;

  // Estimate cell size from median edge length (just use bbox / sqrt(N))
  float dx = mx.x - mn.x, dy = mx.y - mn.y, dz = mx.z - mn.z;
  float cell_size = fmaxf(fmaxf(dx, dy), dz) / (float)sqrtf((float)N) * 2.0f;
  if (cell_size < 1e-6f) cell_size = 1.0f;

  int gx = (int)(dx / cell_size) + 1;
  int gy = (int)(dy / cell_size) + 1;
  int gz = (int)(dz / cell_size) + 1;
  // Cap grid to avoid huge allocations on degenerate surfaces
  if (gx > 512) gx = 512;
  if (gy > 512) gy = 512;
  if (gz > 512) gz = 512;
  int ncells = gx * gy * gz;

  surface_index *idx = calloc(1, sizeof(*idx));
  int *cell_count = calloc((size_t)ncells, sizeof(int));
  int *cell_start = malloc((size_t)ncells * sizeof(int));
  if (!idx || !cell_count || !cell_start) {
    free(idx); free(cell_count); free(cell_start); return NULL;
  }
  idx->min_pt = mn; idx->max_pt = mx;
  idx->gx = gx; idx->gy = gy; idx->gz = gz;
  idx->cell_size = cell_size;
  idx->cell_count = cell_count;
  idx->cell_start = cell_start;
  idx->pts = surf->points;
  idx->npts = N;
  idx->total_cells = ncells;

  // Count points per cell
  for (int i = 0; i < N; i++) {
    int cx, cy, cz;
    world_to_cell(idx, surf->points[i], &cx, &cy, &cz);
    cell_count[cell_flat(idx, cx, cy, cz)]++;
  }
  // Prefix sum → cell_start
  int total = 0;
  for (int c = 0; c < ncells; c++) {
    cell_start[c] = total;
    total += cell_count[c];
  }
  // Fill data array
  int *data = malloc((size_t)N * sizeof(int));
  if (!data) { surface_index_free(idx); return NULL; }
  idx->data = data;
  // Reuse cell_count as insertion cursor
  memset(cell_count, 0, (size_t)ncells * sizeof(int));
  for (int i = 0; i < N; i++) {
    int cx, cy, cz;
    world_to_cell(idx, surf->points[i], &cx, &cy, &cz);
    int fc = cell_flat(idx, cx, cy, cz);
    data[cell_start[fc] + cell_count[fc]++] = i;
  }
  return idx;
}

void surface_index_free(surface_index *idx) {
  if (!idx) return;
  free(idx->cell_start);
  free(idx->cell_count);
  free(idx->data);
  free(idx);
}

// ---------------------------------------------------------------------------
// Nearest-neighbor query
// ---------------------------------------------------------------------------

int surface_index_nearest(const surface_index *idx, vec3f query, float *out_dist) {
  if (!idx) return -1;
  int cx, cy, cz;
  world_to_cell(idx, query, &cx, &cy, &cz);

  int best_vi = -1;
  float best_d2 = FLT_MAX;

  // Expand search radius in cell-space until we find at least one candidate
  for (int radius_cells = 0; radius_cells <= idx->gx + idx->gy + idx->gz; radius_cells++) {
    int x0 = cx-radius_cells, x1 = cx+radius_cells;
    int y0 = cy-radius_cells, y1 = cy+radius_cells;
    int z0 = cz-radius_cells, z1 = cz+radius_cells;
    if (x0 < 0) x0=0;  if (x1 >= idx->gx) x1=idx->gx-1;
    if (y0 < 0) y0=0;  if (y1 >= idx->gy) y1=idx->gy-1;
    if (z0 < 0) z0=0;  if (z1 >= idx->gz) z1=idx->gz-1;

    for (int iz = z0; iz <= z1; iz++)
      for (int iy = y0; iy <= y1; iy++)
        for (int ix = x0; ix <= x1; ix++) {
          // Only visit shell cells on the first pass (shell = at least one coord at boundary)
          if (radius_cells > 0 &&
              iz != z0 && iz != z1 &&
              iy != y0 && iy != y1 &&
              ix != x0 && ix != x1) continue;
          int fc = cell_flat(idx, ix, iy, iz);
          int base = idx->cell_start[fc], cnt = idx->cell_count[fc];
          for (int k = 0; k < cnt; k++) {
            int vi = idx->data[base + k];
            vec3f d = vec3f_sub(idx->pts[vi], query);
            float d2 = vec3f_dot(d, d);
            if (d2 < best_d2) { best_d2 = d2; best_vi = vi; }
          }
        }

    if (best_vi >= 0) {
      // Check that no closer point can exist in outer shells
      float shell_dist = (float)(radius_cells) * idx->cell_size;
      if (best_d2 <= shell_dist * shell_dist) break;
    }
  }
  if (out_dist && best_vi >= 0) *out_dist = sqrtf(best_d2);
  return best_vi;
}

// ---------------------------------------------------------------------------
// Radius query
// ---------------------------------------------------------------------------

int surface_index_radius(const surface_index *idx, vec3f query, float radius,
                          int *out_indices, int max) {
  if (!idx || !out_indices || max <= 0 || radius <= 0.0f) return 0;
  int cx, cy, cz;
  world_to_cell(idx, query, &cx, &cy, &cz);
  int cells = (int)(radius / idx->cell_size) + 1;

  int x0=cx-cells, x1=cx+cells, y0=cy-cells, y1=cy+cells, z0=cz-cells, z1=cz+cells;
  if (x0<0) x0=0;  if (x1>=idx->gx) x1=idx->gx-1;
  if (y0<0) y0=0;  if (y1>=idx->gy) y1=idx->gy-1;
  if (z0<0) z0=0;  if (z1>=idx->gz) z1=idx->gz-1;

  float r2 = radius * radius;
  int count = 0;
  for (int iz=z0; iz<=z1 && count<max; iz++)
    for (int iy=y0; iy<=y1 && count<max; iy++)
      for (int ix=x0; ix<=x1 && count<max; ix++) {
        int fc = cell_flat(idx, ix, iy, iz);
        int base = idx->cell_start[fc], cnt = idx->cell_count[fc];
        for (int k = 0; k < cnt && count < max; k++) {
          int vi = idx->data[base + k];
          vec3f d = vec3f_sub(idx->pts[vi], query);
          if (vec3f_dot(d, d) <= r2) out_indices[count++] = vi;
        }
      }
  return count;
}
