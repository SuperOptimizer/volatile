#include "greatest.h"
#include "core/tracer.h"
#include "core/vol.h"
#include "core/geom.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers: build a synthetic volume with a bright plane at z=16
// ---------------------------------------------------------------------------

static char g_vol_path[256];

// Create a 32x32x32 uint8 zarr with high values on the plane z==16.
static volume *make_bright_plane_volume(void) {
  snprintf(g_vol_path, sizeof(g_vol_path),
           "/tmp/test_tracer_vol_%d.zarr", (int)getpid());

  vol_create_params cp = {
    .zarr_version = 2,
    .ndim         = 3,
    .shape        = {32, 32, 32},
    .chunk_shape  = {32, 32, 32},
    .dtype        = DTYPE_U8,
    .compressor   = NULL,
    .clevel       = 0,
    .sharded      = false,
  };
  volume *v = vol_create(g_vol_path, cp);
  if (!v) return NULL;

  uint8_t chunk[32 * 32 * 32];
  memset(chunk, 20, sizeof(chunk));          // background intensity
  for (int y = 0; y < 32; y++)
    for (int x = 0; x < 32; x++)
      chunk[16 * 32 * 32 + y * 32 + x] = 200;  // bright plane at z=16

  int64_t coords[3] = {0, 0, 0};
  vol_write_chunk(v, 0, coords, chunk, sizeof(chunk));
  return v;
}

// Build a flat 5×5 seed surface at z=16, y=[8..12], x=[8..12]
static quad_surface *make_flat_seed(void) {
  quad_surface *s = quad_surface_new(5, 5);
  if (!s) return NULL;
  for (int r = 0; r < 5; r++) {
    for (int c = 0; c < 5; c++) {
      vec3f p = { (float)(8 + c), (float)(8 + r), 16.0f };
      quad_surface_set(s, r, c, p);
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_tracer_new_free(void) {
  volume *v = make_bright_plane_volume();
  ASSERT(v != NULL);
  tracer *t = tracer_new(v);
  ASSERT(t != NULL);
  tracer_free(t);
  vol_free(v);
  PASS();
}

TEST test_tracer_params_default(void) {
  tracer_params p = tracer_params_default();
  ASSERT(p.search_steps > 0);
  ASSERT(p.search_radius > 0.0f);
  ASSERT(p.straightness_2d >= 0.0f);
  PASS();
}

// grow_patch should return a larger surface than the seed
TEST test_grow_patch_expands(void) {
  volume *v = make_bright_plane_volume();
  ASSERT(v != NULL);

  tracer *t = tracer_new(v);
  ASSERT(t != NULL);

  quad_surface *seed = make_flat_seed();
  ASSERT(seed != NULL);

  tracer_params p = tracer_params_default();
  p.search_radius = 3.0f;
  p.search_steps  = 8;

  quad_surface *out = tracer_grow_patch(t, seed, &p, 3, GROWTH_ALL);
  ASSERT(out != NULL);

  // Output must be larger than seed
  ASSERT(out->rows > seed->rows);
  ASSERT(out->cols > seed->cols);

  quad_surface_free(seed);
  quad_surface_free(out);
  tracer_free(t);
  vol_free(v);
  PASS();
}

// Grown surface should follow the bright plane: z values stay near 16
TEST test_grow_patch_follows_plane(void) {
  volume *v = make_bright_plane_volume();
  ASSERT(v != NULL);

  tracer *t = tracer_new(v);
  quad_surface *seed = make_flat_seed();
  ASSERT(seed != NULL);

  tracer_params p = tracer_params_default();
  p.search_radius    = 4.0f;
  p.search_steps     = 8;
  p.straightness_2d  = 0.7f;
  p.z_location_weight = 0.1f;

  quad_surface *out = tracer_grow_patch(t, seed, &p, 2, GROWTH_ALL);
  ASSERT(out != NULL);

  // All vertices should be within ±4 voxels of z=16
  int rows = out->rows, cols = out->cols;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      vec3f pt = quad_surface_get(out, r, c);
      if (pt.x == 0.0f && pt.y == 0.0f && pt.z == 0.0f) continue; // unfilled
      ASSERT(fabsf(pt.z - 16.0f) < 6.0f);
    }
  }

  quad_surface_free(seed);
  quad_surface_free(out);
  tracer_free(t);
  vol_free(v);
  PASS();
}

// GROWTH_ROW only expands along columns, not rows
TEST test_grow_patch_direction_row(void) {
  volume *v = make_bright_plane_volume();
  tracer *t = tracer_new(v);
  quad_surface *seed = make_flat_seed();
  tracer_params p = tracer_params_default();

  quad_surface *out = tracer_grow_patch(t, seed, &p, 2, GROWTH_ROW);
  ASSERT(out != NULL);
  // Row count unchanged, cols expanded
  ASSERT_EQ(seed->rows, out->rows);
  ASSERT(out->cols > seed->cols);

  quad_surface_free(seed);
  quad_surface_free(out);
  tracer_free(t);
  vol_free(v);
  PASS();
}

TEST test_tracer_cost_nonnegative_baseline(void) {
  volume *v = make_bright_plane_volume();
  tracer *t = tracer_new(v);
  quad_surface *s = make_flat_seed();
  tracer_params p = tracer_params_default();

  // Cost at the center of the seed on a neighbor position — should be finite
  vec3f cand = { 10.0f, 10.0f, 16.0f };
  float c = tracer_cost(t, s, 2, 2, cand, &p);
  ASSERT(isfinite(c));

  quad_surface_free(s);
  tracer_free(t);
  vol_free(v);
  PASS();
}

TEST test_check_overlap(void) {
  volume *v = make_bright_plane_volume();
  tracer *t = tracer_new(v);
  quad_surface *excl = make_flat_seed();  // exclusion at y=[8..12], x=[8..12], z=16

  tracer_add_exclusion(t, excl);

  // Point right on the exclusion surface
  vec3f inside = { 10.0f, 10.0f, 16.0f };
  ASSERT(tracer_check_overlap(t, inside, 2.0f));

  // Point far away
  vec3f outside = { 0.0f, 0.0f, 0.0f };
  ASSERT(!tracer_check_overlap(t, outside, 2.0f));

  quad_surface_free(excl);
  tracer_free(t);
  vol_free(v);
  PASS();
}

TEST test_tracer_add_exclusion_limit(void) {
  volume *v = make_bright_plane_volume();
  tracer *t = tracer_new(v);
  quad_surface *s = make_flat_seed();

  // Add more than MAX_EXCLUSIONS — should not crash
  for (int i = 0; i < 100; i++)
    tracer_add_exclusion(t, s);

  quad_surface_free(s);
  tracer_free(t);
  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(tracer_suite) {
  RUN_TEST(test_tracer_new_free);
  RUN_TEST(test_tracer_params_default);
  RUN_TEST(test_grow_patch_expands);
  RUN_TEST(test_grow_patch_follows_plane);
  RUN_TEST(test_grow_patch_direction_row);
  RUN_TEST(test_tracer_cost_nonnegative_baseline);
  RUN_TEST(test_check_overlap);
  RUN_TEST(test_tracer_add_exclusion_limit);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(tracer_suite);
  GREATEST_MAIN_END();
}
