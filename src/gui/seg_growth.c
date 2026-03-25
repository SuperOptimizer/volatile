#define _POSIX_C_SOURCE 200809L
#include "seg_growth.h"
#include "core/thread.h"
#include "core/math.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdatomic.h>
#include <assert.h>
#include <stdint.h>
#include <time.h>

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

#define MAX_EXCLUSION_SURFACES 16

struct seg_grower {
  volume        *vol;
  quad_surface  *surface;   // current surface (may be replaced after step)
  atomic_int     busy;      // 1 while background thread is running

  correction_point corrections[MAX_CORRECTIONS];
  int              num_corrections;

  quad_surface  *exclusion[MAX_EXCLUSION_SURFACES];
  int            nexclusion;

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

// ---------------------------------------------------------------------------
// SplitMix64: deterministic PRNG matching villa's implementation
// ---------------------------------------------------------------------------

uint64_t splitmix64(uint64_t *state) {
  uint64_t z = (*state += UINT64_C(0x9E3779B97F4A7C15));
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}

// Jitter a position by jitter_amount using deterministic hash of grid coords.
static vec3f jitter_pos(vec3f p, float amount, uint64_t seed) {
  if (amount <= 0.0f) return p;
  uint64_t s = seed;
  float jx = (float)(splitmix64(&s) & 0xFFFF) / 32767.5f - 1.0f;
  float jy = (float)(splitmix64(&s) & 0xFFFF) / 32767.5f - 1.0f;
  float jz = (float)(splitmix64(&s) & 0xFFFF) / 32767.5f - 1.0f;
  return (vec3f){ p.x + jx * amount, p.y + jy * amount, p.z + jz * amount };
}

// ---------------------------------------------------------------------------
// Per-vertex cost function (lower = better candidate position)
// ---------------------------------------------------------------------------

// straightness_2d: penalise deviation of the XY projection from a straight
// line by comparing (candidate - prev) direction to (prev - prev_prev).
// straightness_3d: same but in full 3D.
// distance_weight: penalise candidate if its distance from the seed origin
// changes significantly.
// z_location_weight: penalise vertical drift from the seed Z plane.
static float vertex_cost(vec3f candidate, vec3f origin, vec3f prev,
                         vec3f prev_dir_3d, vec3f prev_dir_2d,
                         const advanced_growth_params *p) {
  float cost = 0.0f;

  // Intensity reward: we want bright voxels — handled externally via sampling.
  // The cost here is purely geometric.

  if (p->straightness_3d > 0.0f) {
    vec3f dir3d = vec3f_sub(candidate, prev);
    float len = vec3f_len(dir3d);
    if (len > 1e-6f) {
      dir3d = vec3f_scale(dir3d, 1.0f / len);
      float dot = vec3f_dot(dir3d, prev_dir_3d);
      cost += p->straightness_3d * (1.0f - dot);
    }
  }

  if (p->straightness_2d > 0.0f) {
    vec3f dir2d = { candidate.x - prev.x, candidate.y - prev.y, 0.0f };
    float len2d = vec3f_len(dir2d);
    if (len2d > 1e-6f) {
      dir2d = vec3f_scale(dir2d, 1.0f / len2d);
      float dot2d = dir2d.x * prev_dir_2d.x + dir2d.y * prev_dir_2d.y;
      cost += p->straightness_2d * (1.0f - dot2d);
    }
  }

  if (p->distance_weight > 0.0f) {
    float d = vec3f_len(vec3f_sub(candidate, origin));
    float d0 = vec3f_len(vec3f_sub(prev, origin));
    float dd = fabsf(d - d0);
    cost += p->distance_weight * dd;
  }

  if (p->z_location_weight > 0.0f) {
    cost += p->z_location_weight * fabsf(candidate.z - origin.z);
  }

  return cost;
}

// ---------------------------------------------------------------------------
// Exclusion surface check: returns true if pos is within 1 voxel of any
// point on any exclusion surface (approximate: scan all vertices).
// ---------------------------------------------------------------------------

static bool is_excluded(const seg_grower *g, vec3f pos) {
  const float EXCL_DIST = 1.0f;
  for (int k = 0; k < g->nexclusion; k++) {
    const quad_surface *ex = g->exclusion[k];
    int n = ex->rows * ex->cols;
    for (int i = 0; i < n; i++) {
      if (vec3f_len(vec3f_sub(pos, ex->points[i])) < EXCL_DIST)
        return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// One generation of advanced growth
// ---------------------------------------------------------------------------

static void grow_advanced_one(seg_grower *g, const advanced_growth_params *p,
                              int gen) {
  quad_surface *s = g->surface;
  if (!s->normals) quad_surface_compute_normals(s);
  quad_surface *next = quad_surface_clone(s);
  if (!next) return;

  int steps  = p->search_steps  > 0 ? p->search_steps  : 16;
  float radius = p->search_radius > 0.0f ? p->search_radius : 8.0f;

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      vec3f pos    = quad_surface_get(s, r, c);
      vec3f normal = (s->normals) ? s->normals[r * s->cols + c]
                                  : surface_normal_at(s, r, c);

      // Deterministic jitter seed from grid coordinates + generation
      uint64_t jseed = (uint64_t)((gen + 1) * 73856093)
                     ^ (uint64_t)(r * 19349663)
                     ^ (uint64_t)(c * 83492791);
      vec3f search_pos = jitter_pos(pos, p->jitter_amount, jseed);

      // Sample along normal; pick best intensity position.
      float best_val  = -1.0f;
      float best_cost = 1e18f;
      vec3f best_pos  = pos;

      // Direction vectors for straightness cost: estimate from neighbors.
      vec3f prev3d = { 0.0f, 0.0f, 1.0f };
      vec3f prev2d = { 1.0f, 0.0f, 0.0f };
      if (r > 0) {
        vec3f nb = quad_surface_get(s, r - 1, c);
        vec3f d3d = vec3f_sub(pos, nb);
        float l3d = vec3f_len(d3d);
        if (l3d > 1e-6f) prev3d = vec3f_scale(d3d, 1.0f / l3d);
        prev2d = (vec3f){ d3d.x / (l3d + 1e-9f), d3d.y / (l3d + 1e-9f), 0.0f };
      }

      for (int i = 0; i <= steps; i++) {
        float t = (i == 0) ? 0.0f
                           : ((float)i / (float)steps) * radius;
        vec3f cand = vec3f_add(search_pos, vec3f_scale(normal, t));

        if (is_excluded(g, cand)) continue;

        float val  = vol_sample(g->vol, 0, cand.z, cand.y, cand.x);
        float cost = vertex_cost(cand, pos, pos, prev3d, prev2d, p);

        // Score = intensity reward minus geometric cost (normalised to [0,1]).
        // Prefer bright + low-cost.
        float score = val - 50.0f * cost;
        if (score > best_val) {
          best_val = score;
          best_cost = cost;
          best_pos  = cand;
        }
      }
      (void)best_cost;

      // Honor correction anchors if requested.
      if (p->use_corrections && g->num_corrections > 0) {
        float nearest = 1e18f;
        vec3f anchor  = best_pos;
        for (int k = 0; k < g->num_corrections; k++) {
          float d = vec3f_len(vec3f_sub(g->corrections[k].target, pos));
          if (d < nearest) { nearest = d; anchor = g->corrections[k].target; }
        }
        if (nearest < radius) {
          float w = fminf(p->distance_weight, 1.0f);
          best_pos = vec3f_lerp(best_pos, anchor, w);
        }
      }

      quad_surface_set(next, r, c, best_pos);
    }
  }

  quad_surface_free(g->surface);
  g->surface = next;
  quad_surface_compute_normals(g->surface);
}

// ---------------------------------------------------------------------------
// Public: seg_grower_grow_advanced (synchronous — runs on caller thread)
// ---------------------------------------------------------------------------

bool seg_grower_grow_advanced(seg_grower *g, const advanced_growth_params *params) {
  assert(g && params);
  // Block if a background step is still running.
  while (atomic_load(&g->busy)) {
    struct timespec ts = { 0, 500000 };
    nanosleep(&ts, NULL);
  }

  int gens = (params->max_generations > 0) ? params->max_generations : 1;
  for (int gen = 0; gen < gens; gen++)
    grow_advanced_one(g, params, gen);

  return true;
}

// ---------------------------------------------------------------------------
// Public: exclusion surfaces
// ---------------------------------------------------------------------------

void seg_grower_set_exclusion_surfaces(seg_grower *g,
                                       quad_surface **others, int count) {
  assert(g);
  g->nexclusion = 0;
  for (int i = 0; i < count && i < MAX_EXCLUSION_SURFACES; i++) {
    g->exclusion[g->nexclusion++] = others[i];
  }
}
