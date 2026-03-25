#include "core/tracer.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// SplitMix64 deterministic jitter (from villa's GrowSurface.cpp)
// ---------------------------------------------------------------------------

static uint64_t mix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

// Returns value in [-1, 1).
static float det_jitter(int row, int col, int salt) {
  uint64_t h = mix64(((uint64_t)(uint32_t)row << 32) ^ (uint64_t)(uint32_t)col ^ (uint64_t)(uint32_t)salt);
  return (float)((double)(h >> 11) / (double)(1ULL << 52)) * 2.0f - 1.0f;
}

// ---------------------------------------------------------------------------
// Exclusion surface list
// ---------------------------------------------------------------------------

#define MAX_EXCLUSIONS 64

struct tracer {
  volume       *vol;
  volume       *dir_vol;    // direction field (optional)
  volume       *edt_vol;    // EDT distance volume (optional)

  const quad_surface *exclusions[MAX_EXCLUSIONS];
  int           n_exclusions;
};

tracer *tracer_new(volume *vol) {
  assert(vol);
  tracer *t = calloc(1, sizeof(*t));
  REQUIRE(t, "tracer_new: calloc failed");
  t->vol = vol;
  return t;
}

void tracer_free(tracer *t) {
  if (!t) return;
  free(t);
}

void tracer_set_direction_field(tracer *t, volume *dir_vol) {
  REQUIRE(t, "tracer_set_direction_field: null tracer");
  t->dir_vol = dir_vol;
}

void tracer_set_edt(tracer *t, volume *edt_vol) {
  REQUIRE(t, "tracer_set_edt: null tracer");
  t->edt_vol = edt_vol;
}

void tracer_add_exclusion(tracer *t, const quad_surface *other) {
  REQUIRE(t && other, "tracer_add_exclusion: null arg");
  if (t->n_exclusions < MAX_EXCLUSIONS)
    t->exclusions[t->n_exclusions++] = other;
}

tracer_params tracer_params_default(void) {
  return (tracer_params){
    .straightness_2d    = 0.7f,
    .straightness_3d    = 4.0f,
    .distance_weight    = 1.0f,
    .z_location_weight  = 0.1f,
    .search_radius      = 5.0f,
    .search_steps       = 16,
    .jitter             = 0.05f,
    .use_direction_field = false,
    .use_edt            = false,
    .use_neural         = false,
    .falloff_sigma      = 2.0f,
  };
}

// ---------------------------------------------------------------------------
// Surface normal at grid point (finite differences, clamped)
// ---------------------------------------------------------------------------

static vec3f surface_normal_at(const quad_surface *s, int row, int col) {
  int rows = s->rows, cols = s->cols;
  int r0 = row > 0        ? row - 1 : row;
  int r1 = row < rows - 1 ? row + 1 : row;
  int c0 = col > 0        ? col - 1 : col;
  int c1 = col < cols - 1 ? col + 1 : col;
  vec3f du = vec3f_sub(quad_surface_get(s, row, c1), quad_surface_get(s, row, c0));
  vec3f dv = vec3f_sub(quad_surface_get(s, r1, col), quad_surface_get(s, r0, col));
  vec3f n  = vec3f_normalize(vec3f_cross(du, dv));
  return n;
}

// ---------------------------------------------------------------------------
// tracer_check_overlap
// ---------------------------------------------------------------------------

bool tracer_check_overlap(const tracer *t, vec3f pos, float threshold) {
  assert(t);
  float thr2 = threshold * threshold;
  for (int e = 0; e < t->n_exclusions; e++) {
    const quad_surface *s = t->exclusions[e];
    for (int r = 0; r < s->rows; r++) {
      for (int c = 0; c < s->cols; c++) {
        vec3f v = quad_surface_get(s, r, c);
        vec3f d = vec3f_sub(pos, v);
        if (vec3f_dot(d, d) < thr2) return true;
      }
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// tracer_cost
// ---------------------------------------------------------------------------
// NOTE: Lower cost = better candidate.
// Components mirror villa's LOSS_STRAIGHT (2D), LOSS_3DNORMALLINE, LOSS_DIST, z_loc.

float tracer_cost(const tracer *t, const quad_surface *surf,
                  int row, int col, vec3f cand,
                  const tracer_params *p) {
  assert(t && surf && p);
  float cost = 0.0f;
  int rows = surf->rows, cols = surf->cols;

  // --- 2D straightness: candidate should be close to where a straight
  //     extrapolation of the grid would place it. ---
  if (p->straightness_2d > 0.0f) {
    // Average position of valid grid-neighbors
    vec3f avg = {0, 0, 0};
    int n = 0;
    int nr[4] = {row-1, row+1, row,   row  };
    int nc[4] = {col,   col,   col-1, col+1};
    for (int i = 0; i < 4; i++) {
      if (nr[i] >= 0 && nr[i] < rows && nc[i] >= 0 && nc[i] < cols) {
        avg = vec3f_add(avg, quad_surface_get(surf, nr[i], nc[i]));
        n++;
      }
    }
    if (n > 0) {
      avg = vec3f_scale(avg, 1.0f / (float)n);
      vec3f d = vec3f_sub(cand, avg);
      cost += p->straightness_2d * vec3f_len(d);
    }
  }

  // --- 3D straightness: candidate normal should align with neighbors' normals ---
  if (p->straightness_3d > 0.0f && rows > 1 && cols > 1) {
    vec3f n_here = surface_normal_at(surf, row, col);
    vec3f n_cand = vec3f_normalize(vec3f_sub(cand, quad_surface_get(surf, row, col)));
    float dot = vec3f_dot(n_here, n_cand);
    // dot near -1 means candidate is on the wrong side
    cost += p->straightness_3d * fmaxf(0.0f, -dot);
  }

  // --- Distance regularization: expected step = search_radius / search_steps ---
  if (p->distance_weight > 0.0f) {
    float expected = p->search_radius / (float)(p->search_steps > 0 ? p->search_steps : 1);
    vec3f base = quad_surface_get(surf, row, col);
    float actual = vec3f_len(vec3f_sub(cand, base));
    float delta = actual - expected;
    cost += p->distance_weight * delta * delta;
  }

  // --- Z-location smoothness: prefer z close to neighbor average ---
  if (p->z_location_weight > 0.0f) {
    float z_avg = 0.0f;
    int n = 0;
    int nr[4] = {row-1, row+1, row,   row  };
    int nc[4] = {col,   col,   col-1, col+1};
    for (int i = 0; i < 4; i++) {
      if (nr[i] >= 0 && nr[i] < rows && nc[i] >= 0 && nc[i] < cols) {
        z_avg += quad_surface_get(surf, nr[i], nc[i]).z;
        n++;
      }
    }
    if (n > 0) {
      z_avg /= (float)n;
      float dz = cand.z - z_avg;
      cost += p->z_location_weight * dz * dz;
    }
  }

  // --- EDT cost: prefer positions with low EDT value (near bright voxels) ---
  if (p->use_edt && t->edt_vol) {
    float edt = vol_sample(t->edt_vol, 0, cand.z, cand.y, cand.x);
    cost += edt;
  }

  // --- Direction field: penalize deviation from field direction ---
  if (p->use_direction_field && t->dir_vol) {
    float dx = vol_sample(t->dir_vol, 0, cand.z, cand.y, cand.x);
    // NOTE: scalar direction field — use as intensity guidance (lower cost where field is high)
    cost -= dx * 0.5f;
  }

  return cost;
}

// ---------------------------------------------------------------------------
// tracer_grow_patch — BFS fringe expansion
// ---------------------------------------------------------------------------

// State flags per grid cell
#define ST_EMPTY    0
#define ST_VALID    1
#define ST_FRINGE   2

quad_surface *tracer_grow_patch(tracer *t, const quad_surface *seed,
                                const tracer_params *p,
                                int generations, growth_direction dir) {
  assert(t && seed && p && generations > 0);

  int seed_rows = seed->rows, seed_cols = seed->cols;

  // Allocate output surface: pad by `generations` in the active growth directions.
  // GROWTH_ROW  → expand row count (pad top/bottom), cols fixed.
  // GROWTH_COL  → expand col count (pad left/right), rows fixed.
  // GROWTH_ALL  → pad both.
  int grow_r = (dir == GROWTH_COL) ? 0 : generations;
  int grow_c = (dir == GROWTH_ROW) ? 0 : generations;
  int out_rows = seed_rows + 2 * grow_r;
  int out_cols = seed_cols + 2 * grow_c;

  quad_surface *out = quad_surface_new(out_rows, out_cols);
  if (!out) return NULL;

  uint8_t *state = calloc((size_t)(out_rows * out_cols), 1);
  if (!state) { quad_surface_free(out); return NULL; }

  // Seed offset in the output grid
  int r_off = grow_r;
  int c_off = grow_c;

  // Copy seed into the output surface and mark valid
  for (int r = 0; r < seed_rows; r++) {
    for (int c = 0; c < seed_cols; c++) {
      int or_ = r + r_off, oc = c + c_off;
      quad_surface_set(out, or_, oc, quad_surface_get(seed, r, c));
      state[or_ * out_cols + oc] = ST_VALID;
    }
  }

  // Build initial fringe: boundary cells of the seed that are adjacent to empty
  int *fringe     = malloc((size_t)(out_rows * out_cols) * sizeof(int));  // flat indices
  int *next_fringe = malloc((size_t)(out_rows * out_cols) * sizeof(int));
  if (!fringe || !next_fringe) {
    free(fringe); free(next_fringe); free(state); quad_surface_free(out);
    return NULL;
  }

  int drow[4] = {-1, 1, 0, 0};
  int dcol[4] = {0, 0, -1, 1};

  int fringe_n = 0;
  for (int r = r_off; r < r_off + seed_rows; r++) {
    for (int c = c_off; c < c_off + seed_cols; c++) {
      bool is_boundary = false;
      for (int d = 0; d < 4; d++) {
        int nr = r + drow[d], nc = c + dcol[d];
        if (nr < 0 || nr >= out_rows || nc < 0 || nc >= out_cols) continue;
        if (state[nr * out_cols + nc] == ST_EMPTY) { is_boundary = true; break; }
      }
      if (is_boundary) fringe[fringe_n++] = r * out_cols + c;
    }
  }

  int salt = 0x1337;

  for (int gen = 0; gen < generations && fringe_n > 0; gen++) {
    int next_n = 0;

    for (int fi = 0; fi < fringe_n; fi++) {
      int fr = fringe[fi] / out_cols;
      int fc = fringe[fi] % out_cols;

      for (int d = 0; d < 4; d++) {
        // Respect growth direction
        if (dir == GROWTH_ROW && d < 2) continue;   // skip row-direction steps
        if (dir == GROWTH_COL && d >= 2) continue;  // skip col-direction steps

        int nr = fr + drow[d], nc = fc + dcol[d];
        if (nr < 0 || nr >= out_rows || nc < 0 || nc >= out_cols) continue;
        if (state[nr * out_cols + nc] != ST_EMPTY) continue;

        // Mark processing
        state[nr * out_cols + nc] = ST_FRINGE;

        // Gather neighbor positions to estimate starting point + normal
        vec3f avg = {0, 0, 0};
        int n_neighbors = 0;
        for (int d2 = 0; d2 < 4; d2++) {
          int ar = nr + drow[d2], ac = nc + dcol[d2];
          if (ar < 0 || ar >= out_rows || ac < 0 || ac >= out_cols) continue;
          if (state[ar * out_cols + ac] != ST_VALID) continue;
          avg = vec3f_add(avg, quad_surface_get(out, ar, ac));
          n_neighbors++;
        }
        if (n_neighbors == 0) continue;
        avg = vec3f_scale(avg, 1.0f / (float)n_neighbors);

        // Surface normal from the closest valid neighbor
        vec3f src_pos = quad_surface_get(out, fr, fc);
        vec3f normal  = surface_normal_at(out, fr, fc);
        // Normalize; if degenerate, use simple z-step
        float nlen = vec3f_len(normal);
        if (nlen < 1e-6f) normal = (vec3f){0, 0, 1};

        // Sample N candidates along the normal direction
        float best_cost = FLT_MAX;
        vec3f best_cand = avg;  // fallback

        float step = p->search_radius / (float)(p->search_steps > 0 ? p->search_steps : 1);
        for (int si = 1; si <= p->search_steps; si++) {
          float t_val = (float)si * step;

          // SplitMix64 jitter
          float jx = det_jitter(nr, nc + si * 17, salt)     * p->jitter;
          float jy = det_jitter(nr + si * 13, nc, salt + 1) * p->jitter;
          float jz = det_jitter(nr * 7, nc + si, salt + 2)  * p->jitter;

          vec3f cand = {
            src_pos.x + normal.x * t_val + jx,
            src_pos.y + normal.y * t_val + jy,
            src_pos.z + normal.z * t_val + jz,
          };

          // Exclusion check
          if (tracer_check_overlap(t, cand, 1.5f)) continue;

          // Volume bounds check
          int64_t shape[3] = {0};
          vol_shape(t->vol, 0, shape);
          if (cand.z < 0 || cand.z >= (float)shape[0] ||
              cand.y < 0 || cand.y >= (float)shape[1] ||
              cand.x < 0 || cand.x >= (float)shape[2]) continue;

          // Volume intensity — prefer bright voxels (papyrus ink signal)
          float intensity = vol_sample(t->vol, 0, cand.z, cand.y, cand.x);

          // Cost: lower is better; subtract intensity so bright = cheaper
          float c = tracer_cost(t, out, nr, nc, cand, p) - intensity * 0.01f;

          if (c < best_cost) {
            best_cost = c;
            best_cand = cand;
          }
        }

        // Also try the simple extrapolation direction (step straight along normal)
        {
          vec3f cand = vec3f_add(src_pos, vec3f_scale(normal, step));
          if (!tracer_check_overlap(t, cand, 1.5f)) {
            float c = tracer_cost(t, out, nr, nc, cand, p);
            if (c < best_cost) { best_cost = c; best_cand = cand; }
          }
        }

        quad_surface_set(out, nr, nc, best_cand);
        state[nr * out_cols + nc] = ST_VALID;
        next_fringe[next_n++]     = nr * out_cols + nc;
      }
    }

    // Swap fringe
    memcpy(fringe, next_fringe, (size_t)next_n * sizeof(int));
    fringe_n = next_n;
    salt++;
  }

  free(fringe);
  free(next_fringe);
  free(state);
  return out;
}
