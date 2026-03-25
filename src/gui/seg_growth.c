#include "seg_growth.h"
#include "core/thread.h"
#include "core/math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

#define MAX_CORRECTIONS 256
#define TRACER_SEARCH_RADIUS 8   // voxels to search along normal
#define TRACER_SEARCH_STEPS  32  // samples along normal

typedef struct {
  float u, v;
  vec3f target;
} correction_point;

struct seg_grower {
  volume        *vol;
  quad_surface  *surface;   // current surface (may be replaced after step)
  atomic_int     busy;      // 1 while background thread is running

  correction_point corrections[MAX_CORRECTIONS];
  int              num_corrections;

  threadpool    *pool;
  future        *pending;
};

// ---------------------------------------------------------------------------
// Thread task arg
// ---------------------------------------------------------------------------

typedef struct {
  seg_grower  *grower;
  growth_params params;
} step_arg;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vec3f surface_normal_at(const quad_surface *s, int row, int col) {
  int r0 = row > 0 ? row - 1 : row;
  int r1 = row < s->rows - 1 ? row + 1 : row;
  int c0 = col > 0 ? col - 1 : col;
  int c1 = col < s->cols - 1 ? col + 1 : col;

  vec3f pr = quad_surface_get(s, r1, col);
  vec3f pl = quad_surface_get(s, r0, col);
  vec3f pu = quad_surface_get(s, row, c1);
  vec3f pd = quad_surface_get(s, row, c0);

  vec3f drow = vec3f_sub(pr, pl);
  vec3f dcol = vec3f_sub(pu, pd);
  return vec3f_normalize(vec3f_cross(drow, dcol));
}

// Determine whether a grid position is on the boundary of the surface.
// For direction-constrained growth, only process the relevant boundary row/col.
static bool is_boundary(const quad_surface *s, int row, int col,
                         growth_direction dir) {
  switch (dir) {
    case GROWTH_DIR_UP:    return row == 0;
    case GROWTH_DIR_DOWN:  return row == s->rows - 1;
    case GROWTH_DIR_LEFT:  return col == 0;
    case GROWTH_DIR_RIGHT: return col == s->cols - 1;
    case GROWTH_DIR_ALL:
    default:
      return row == 0 || row == s->rows - 1 ||
             col == 0 || col == s->cols - 1;
  }
}

// ---------------------------------------------------------------------------
// TRACER growth: walk along surface normal, find intensity peak.
// ---------------------------------------------------------------------------

static vec3f tracer_new_pos(volume *vol, vec3f pos, vec3f normal,
                             float step_size) {
  float best_val = -1.0f;
  vec3f best_pos = pos;

  for (int i = 1; i <= TRACER_SEARCH_STEPS; i++) {
    float t = (float)i * step_size / (float)TRACER_SEARCH_STEPS *
              (float)TRACER_SEARCH_RADIUS;
    vec3f candidate = vec3f_add(pos, vec3f_scale(normal, t));
    float val = vol_sample(vol, 0, candidate.z, candidate.y, candidate.x);
    if (val > best_val) {
      best_val = val;
      best_pos = candidate;
    }
  }
  return best_pos;
}

static void grow_tracer(seg_grower *g, const growth_params *p) {
  quad_surface *s = g->surface;
  quad_surface *next = quad_surface_clone(s);
  if (!next) return;

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      if (!is_boundary(s, r, c, p->direction)) continue;

      vec3f pos    = quad_surface_get(s, r, c);
      vec3f normal = surface_normal_at(s, r, c);
      vec3f newpos = tracer_new_pos(g->vol, pos, normal, p->step_size);
      quad_surface_set(next, r, c, newpos);
    }
  }

  quad_surface_free(g->surface);
  g->surface = next;
}

// ---------------------------------------------------------------------------
// EXTRAPOLATION growth: linear extrapolation of boundary vertices.
// Mirrors VC3D ExtrapolationGrowth: estimate velocity from 2 interior rows/cols
// and step the boundary forward by step_size.
// ---------------------------------------------------------------------------

static vec3f extrapolate_velocity(const quad_surface *s, int row, int col,
                                   growth_direction dir) {
  vec3f p0, p1;
  switch (dir) {
    case GROWTH_DIR_UP:
      p0 = quad_surface_get(s, 1, col);
      p1 = quad_surface_get(s, 0, col);
      break;
    case GROWTH_DIR_DOWN:
      p0 = quad_surface_get(s, s->rows - 2, col);
      p1 = quad_surface_get(s, s->rows - 1, col);
      break;
    case GROWTH_DIR_LEFT:
      p0 = quad_surface_get(s, row, 1);
      p1 = quad_surface_get(s, row, 0);
      break;
    case GROWTH_DIR_RIGHT:
      p0 = quad_surface_get(s, row, s->cols - 2);
      p1 = quad_surface_get(s, row, s->cols - 1);
      break;
    case GROWTH_DIR_ALL:
    default:
      // Use the outward normal direction
      p0 = quad_surface_get(s, row, col);
      p1 = p0;  // no velocity; handled below
      break;
  }
  return vec3f_sub(p1, p0);
}

static void grow_extrapolation(seg_grower *g, const growth_params *p) {
  quad_surface *s = g->surface;
  quad_surface *next = quad_surface_clone(s);
  if (!next) return;

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      if (!is_boundary(s, r, c, p->direction)) continue;

      vec3f pos = quad_surface_get(s, r, c);
      vec3f vel = extrapolate_velocity(s, r, c, p->direction);
      float vlen = vec3f_len(vel);

      vec3f delta;
      if (vlen > 1e-6f) {
        delta = vec3f_scale(vec3f_normalize(vel), p->step_size);
      } else {
        // Fall back to surface normal
        vec3f n = surface_normal_at(s, r, c);
        delta = vec3f_scale(n, p->step_size);
      }

      // Straightness: blend toward the original velocity direction
      if (p->straightness_weight > 0.0f && vlen > 1e-6f) {
        vec3f straight = vec3f_scale(vel, p->step_size / vlen);
        float w = fminf(p->straightness_weight, 1.0f);
        delta = vec3f_lerp(delta, straight, w);
      }

      quad_surface_set(next, r, c, vec3f_add(pos, delta));
    }
  }

  quad_surface_free(g->surface);
  g->surface = next;
}

// ---------------------------------------------------------------------------
// CORRECTIONS growth: move boundary toward nearest correction anchor point,
// weighted by distance_weight; otherwise tracer step.
// ---------------------------------------------------------------------------

static void grow_corrections(seg_grower *g, const growth_params *p) {
  if (g->num_corrections == 0) {
    // No corrections: fall back to tracer
    grow_tracer(g, p);
    return;
  }

  quad_surface *s = g->surface;
  quad_surface *next = quad_surface_clone(s);
  if (!next) return;

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      if (!is_boundary(s, r, c, p->direction)) continue;

      vec3f pos    = quad_surface_get(s, r, c);
      vec3f normal = surface_normal_at(s, r, c);

      // Find closest correction point
      float best_dist = 1e18f;
      vec3f best_target = pos;
      for (int i = 0; i < g->num_corrections; i++) {
        vec3f t = g->corrections[i].target;
        float d = vec3f_len(vec3f_sub(t, pos));
        if (d < best_dist) {
          best_dist = d;
          best_target = t;
        }
      }

      // Blend tracer step with correction anchor
      vec3f tracer_pos = tracer_new_pos(g->vol, pos, normal, p->step_size);
      vec3f to_target  = vec3f_sub(best_target, pos);
      float tlen = vec3f_len(to_target);

      vec3f correction_pos = pos;
      if (tlen > 1e-6f) {
        float move = fminf(p->step_size, tlen);
        correction_pos = vec3f_add(pos,
          vec3f_scale(vec3f_normalize(to_target), move));
      }

      float w = fminf(p->distance_weight, 1.0f);
      vec3f newpos = vec3f_lerp(tracer_pos, correction_pos, w);
      quad_surface_set(next, r, c, newpos);
    }
  }

  quad_surface_free(g->surface);
  g->surface = next;
}

// ---------------------------------------------------------------------------
// Background thread entry point
// ---------------------------------------------------------------------------

static void *growth_thread(void *arg) {
  step_arg *a = arg;
  seg_grower *g = a->grower;
  const growth_params *p = &a->params;

  int gens = p->generations < 1 ? 1 : p->generations;
  for (int i = 0; i < gens; i++) {
    switch (p->method) {
      case GROWTH_TRACER:
        grow_tracer(g, p);
        break;
      case GROWTH_EXTRAPOLATION:
        grow_extrapolation(g, p);
        break;
      case GROWTH_CORRECTIONS:
        grow_corrections(g, p);
        break;
    }
  }

  atomic_store(&g->busy, 0);
  free(a);
  return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

seg_grower *seg_grower_new(volume *vol, quad_surface *seed) {
  assert(vol && seed);
  seg_grower *g = calloc(1, sizeof(*g));
  if (!g) return NULL;

  g->vol     = vol;
  g->surface = quad_surface_clone(seed);
  if (!g->surface) { free(g); return NULL; }

  atomic_init(&g->busy, 0);
  g->pool = threadpool_new(1);
  if (!g->pool) { quad_surface_free(g->surface); free(g); return NULL; }

  return g;
}

void seg_grower_free(seg_grower *g) {
  if (!g) return;
  // Wait for any pending work before freeing
  if (g->pending) {
    future_get(g->pending, 30000);
    future_free(g->pending);
  }
  threadpool_free(g->pool);
  quad_surface_free(g->surface);
  free(g);
}

bool seg_grower_step(seg_grower *g, const growth_params *params) {
  assert(g && params);
  if (atomic_load(&g->busy)) return false;

  step_arg *a = malloc(sizeof(*a));
  if (!a) return false;
  a->grower = g;
  a->params = *params;

  atomic_store(&g->busy, 1);

  if (g->pending) {
    future_free(g->pending);
    g->pending = NULL;
  }
  g->pending = threadpool_submit(g->pool, growth_thread, a);
  if (!g->pending) {
    atomic_store(&g->busy, 0);
    free(a);
    return false;
  }
  return true;
}

quad_surface *seg_grower_surface(seg_grower *g) {
  assert(g);
  return g->surface;
}

bool seg_grower_busy(const seg_grower *g) {
  assert(g);
  return atomic_load(&g->busy) != 0;
}

void seg_grower_add_correction(seg_grower *g, float u, float v, vec3f target) {
  assert(g);
  if (g->num_corrections >= MAX_CORRECTIONS) return;
  g->corrections[g->num_corrections++] = (correction_point){ u, v, target };
}
