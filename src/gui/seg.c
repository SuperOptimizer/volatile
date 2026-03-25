#include "seg.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Edit record: stores indices + old positions for undo
// ---------------------------------------------------------------------------

typedef struct {
  int   index;   // flat index into quad_surface->points
  vec3f old_pt;
} edit_entry;

struct seg_edit {
  edit_entry *entries;
  int         count;
  int         cap;
};

static seg_edit *edit_new(void) {
  seg_edit *e = calloc(1, sizeof(*e));
  if (!e) return NULL;
  e->cap = 32;
  e->entries = malloc((size_t)e->cap * sizeof(edit_entry));
  if (!e->entries) { free(e); return NULL; }
  return e;
}

void seg_edit_free(seg_edit *e) {
  if (!e) return;
  free(e->entries);
  free(e);
}

// Record the old value of point at flat index, if not already recorded.
static void edit_record(seg_edit *e, const quad_surface *s, int idx) {
  // linear scan is fine: edits are typically O(radius^2) entries
  for (int i = 0; i < e->count; i++) {
    if (e->entries[i].index == idx) return;
  }
  if (e->count == e->cap) {
    int new_cap = e->cap * 2;
    edit_entry *buf = realloc(e->entries, (size_t)new_cap * sizeof(edit_entry));
    if (!buf) return;
    e->entries = buf;
    e->cap = new_cap;
  }
  e->entries[e->count++] = (edit_entry){ .index = idx, .old_pt = s->points[idx] };
}

void seg_edit_undo(quad_surface *s, const seg_edit *e) {
  assert(s && e);
  for (int i = 0; i < e->count; i++) {
    s->points[e->entries[i].index] = e->entries[i].old_pt;
  }
  // invalidate cached normals
  if (s->normals) { free(s->normals); s->normals = NULL; }
}

// ---------------------------------------------------------------------------
// Gaussian weight: exp(-r^2 / (2*sigma^2)), clamped to [0,1]
// ---------------------------------------------------------------------------

static inline float gauss_weight(float dist, float sigma) {
  if (sigma <= 0.0f) return dist <= 0.0f ? 1.0f : 0.0f;
  return expf(-(dist * dist) / (2.0f * sigma * sigma));
}

// ---------------------------------------------------------------------------
// Convert (u,v) in [0,1] to nearest grid (row, col)
// ---------------------------------------------------------------------------

static inline void uv_to_grid(const quad_surface *s, float u, float v, float *out_col, float *out_row) {
  *out_col = u * (float)(s->cols - 1);
  *out_row = v * (float)(s->rows - 1);
}

// ---------------------------------------------------------------------------
// Brush apply: gaussian-weighted displacement along normal
// ---------------------------------------------------------------------------

seg_edit *seg_brush_apply(quad_surface *s, float u, float v, float delta,
                          const seg_tool_params *params) {
  assert(s && params);

  // Ensure normals are available.
  if (!s->normals) quad_surface_compute_normals(s);

  seg_edit *e = edit_new();
  if (!e) return NULL;

  float gc, gr;
  uv_to_grid(s, u, v, &gc, &gr);

  float radius = params->radius > 0.0f ? params->radius : 1.0f;
  float sigma  = params->sigma  > 0.0f ? params->sigma  : radius * 0.5f;
  int ri = (int)ceilf(radius);

  int r0 = (int)(gr - ri); if (r0 < 0) r0 = 0;
  int r1 = (int)(gr + ri); if (r1 >= s->rows) r1 = s->rows - 1;
  int c0 = (int)(gc - ri); if (c0 < 0) c0 = 0;
  int c1 = (int)(gc + ri); if (c1 >= s->cols) c1 = s->cols - 1;

  for (int r = r0; r <= r1; r++) {
    for (int c = c0; c <= c1; c++) {
      float dr = (float)r - gr;
      float dc = (float)c - gc;
      float dist = sqrtf(dr * dr + dc * dc);
      if (dist > radius) continue;

      int idx = r * s->cols + c;
      edit_record(e, s, idx);

      float w = gauss_weight(dist, sigma);
      vec3f n = s->normals[idx];
      s->points[idx] = vec3f_add(s->points[idx], vec3f_scale(n, delta * w));
    }
  }

  // normals are now stale
  if (s->normals) { free(s->normals); s->normals = NULL; }
  return e;
}

// ---------------------------------------------------------------------------
// Line apply: sample points along (u0,v0)->(u1,v1), apply brush at each
// ---------------------------------------------------------------------------

seg_edit *seg_line_apply(quad_surface *s, float u0, float v0, float u1, float v1,
                         float delta, const seg_tool_params *params) {
  assert(s && params);

  if (!s->normals) quad_surface_compute_normals(s);

  // Number of sample steps: at least 1, scaled to arc length in grid units.
  float gc0, gr0, gc1, gr1;
  uv_to_grid(s, u0, v0, &gc0, &gr0);
  uv_to_grid(s, u1, v1, &gc1, &gr1);
  float dx = gc1 - gc0, dy = gr1 - gr0;
  float len = sqrtf(dx * dx + dy * dy);
  int steps = (int)(len * 2.0f) + 1;  // 2 samples per grid unit

  seg_edit *e = edit_new();
  if (!e) return NULL;

  float radius = params->radius > 0.0f ? params->radius : 1.0f;
  float sigma  = params->sigma  > 0.0f ? params->sigma  : radius * 0.5f;
  int ri = (int)ceilf(radius);

  for (int step = 0; step <= steps; step++) {
    float t  = steps > 0 ? (float)step / (float)steps : 0.0f;
    float gc = gc0 + t * (gc1 - gc0);
    float gr = gr0 + t * (gr1 - gr0);

    int r0 = (int)(gr - ri); if (r0 < 0) r0 = 0;
    int r1 = (int)(gr + ri); if (r1 >= s->rows) r1 = s->rows - 1;
    int c0 = (int)(gc - ri); if (c0 < 0) c0 = 0;
    int c1 = (int)(gc + ri); if (c1 >= s->cols) c1 = s->cols - 1;

    for (int r = r0; r <= r1; r++) {
      for (int c = c0; c <= c1; c++) {
        float dr = (float)r - gr;
        float dc = (float)c - gc;
        float dist = sqrtf(dr * dr + dc * dc);
        if (dist > radius) continue;

        int idx = r * s->cols + c;
        edit_record(e, s, idx);  // no-op if already recorded

        float w = gauss_weight(dist, sigma);
        vec3f n = s->normals[idx];
        s->points[idx] = vec3f_add(s->points[idx], vec3f_scale(n, delta * w));
      }
    }
  }

  if (s->normals) { free(s->normals); s->normals = NULL; }
  return e;
}

// ---------------------------------------------------------------------------
// Push-pull: uniform displacement within radius (no gaussian falloff)
// ---------------------------------------------------------------------------

seg_edit *seg_pushpull_apply(quad_surface *s, float u, float v,
                             const seg_tool_params *params) {
  assert(s && params);

  if (!s->normals) quad_surface_compute_normals(s);

  seg_edit *e = edit_new();
  if (!e) return NULL;

  float gc, gr;
  uv_to_grid(s, u, v, &gc, &gr);

  float radius = params->radius > 0.0f ? params->radius : 1.0f;
  float amount = params->push_amount;
  int ri = (int)ceilf(radius);

  int r0 = (int)(gr - ri); if (r0 < 0) r0 = 0;
  int r1 = (int)(gr + ri); if (r1 >= s->rows) r1 = s->rows - 1;
  int c0 = (int)(gc - ri); if (c0 < 0) c0 = 0;
  int c1 = (int)(gc + ri); if (c1 >= s->cols) c1 = s->cols - 1;

  for (int r = r0; r <= r1; r++) {
    for (int c = c0; c <= c1; c++) {
      float dr = (float)r - gr;
      float dc = (float)c - gc;
      if (sqrtf(dr * dr + dc * dc) > radius) continue;

      int idx = r * s->cols + c;
      edit_record(e, s, idx);

      vec3f n = s->normals[idx];
      s->points[idx] = vec3f_add(s->points[idx], vec3f_scale(n, amount));
    }
  }

  if (s->normals) { free(s->normals); s->normals = NULL; }
  return e;
}
