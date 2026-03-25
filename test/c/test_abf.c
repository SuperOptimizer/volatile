#include "greatest.h"
#include "core/abf.h"
#include "core/geom.h"
#include "core/math.h"

#include <math.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a flat R×C grid in the XY plane with unit spacing.
static quad_surface *make_flat_grid(int R, int C) {
  quad_surface *s = quad_surface_new(R, C);
  if (!s) return NULL;
  for (int r = 0; r < R; r++)
    for (int c = 0; c < C; c++)
      quad_surface_set(s, r, c, (vec3f){(float)c, (float)r, 0.0f});
  return s;
}

// Build a cylindrical surface (bend around Z axis) to test non-flat input.
static quad_surface *make_cylinder(int R, int C) {
  quad_surface *s = quad_surface_new(R, C);
  if (!s) return NULL;
  float radius = 10.0f;
  for (int r = 0; r < R; r++)
    for (int c = 0; c < C; c++) {
      float theta = (float)c / (float)(C-1) * (float)M_PI;
      quad_surface_set(s, r, c, (vec3f){
        radius * cosf(theta), radius * sinf(theta), (float)r
      });
    }
  return s;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_uv_coords_free_null(void) {
  uv_coords_free(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Degenerate inputs
// ---------------------------------------------------------------------------

TEST test_null_surface(void) {
  ASSERT(abf_flatten(NULL) == NULL);
  PASS();
}

TEST test_too_small(void) {
  quad_surface *s = quad_surface_new(1, 1);
  ASSERT(abf_flatten(s) == NULL);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Flat grid — UVs should roughly span expected range
// ---------------------------------------------------------------------------

TEST test_flat_grid_returns_uv(void) {
  quad_surface *s = make_flat_grid(4, 5);
  ASSERT(s != NULL);
  uv_coords *uv = abf_flatten(s);
  ASSERT(uv != NULL);
  ASSERT_EQ(uv->count, 4 * 5);
  ASSERT_EQ(uv->rows,  4);
  ASSERT_EQ(uv->cols,  5);
  uv_coords_free(uv);
  quad_surface_free(s);
  PASS();
}

TEST test_flat_grid_pin0_at_origin(void) {
  quad_surface *s = make_flat_grid(3, 4);
  ASSERT(s != NULL);
  uv_coords *uv = abf_flatten(s);
  ASSERT(uv != NULL);
  // pin0 = vertex 0 should be at (0,0)
  ASSERT_IN_RANGE(-1e-4f, uv->u[0], 1e-4f);
  ASSERT_IN_RANGE(-1e-4f, uv->v[0], 1e-4f);
  uv_coords_free(uv);
  quad_surface_free(s);
  PASS();
}

TEST test_flat_grid_pin1_positive_u(void) {
  quad_surface *s = make_flat_grid(3, 4);
  ASSERT(s != NULL);
  uv_coords *uv = abf_flatten(s);
  ASSERT(uv != NULL);
  // pin1 = vertex (0, cols-1): should have u > 0
  ASSERT(uv->u[3] > 0.5f);
  ASSERT_IN_RANGE(-1e-3f, uv->v[3], 1e-3f);
  uv_coords_free(uv);
  quad_surface_free(s);
  PASS();
}

TEST test_flat_grid_all_finite(void) {
  quad_surface *s = make_flat_grid(5, 6);
  ASSERT(s != NULL);
  uv_coords *uv = abf_flatten(s);
  ASSERT(uv != NULL);
  for (int i = 0; i < uv->count; i++) {
    ASSERT(isfinite(uv->u[i]));
    ASSERT(isfinite(uv->v[i]));
  }
  uv_coords_free(uv);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Cylinder — non-planar surface, should still produce finite UVs
// ---------------------------------------------------------------------------

TEST test_cylinder_finite_uv(void) {
  quad_surface *s = make_cylinder(4, 6);
  ASSERT(s != NULL);
  uv_coords *uv = abf_flatten(s);
  ASSERT(uv != NULL);
  for (int i = 0; i < uv->count; i++) {
    ASSERT(isfinite(uv->u[i]));
    ASSERT(isfinite(uv->v[i]));
  }
  uv_coords_free(uv);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(abf_suite) {
  RUN_TEST(test_uv_coords_free_null);
  RUN_TEST(test_null_surface);
  RUN_TEST(test_too_small);
  RUN_TEST(test_flat_grid_returns_uv);
  RUN_TEST(test_flat_grid_pin0_at_origin);
  RUN_TEST(test_flat_grid_pin1_positive_u);
  RUN_TEST(test_flat_grid_all_finite);
  RUN_TEST(test_cylinder_finite_uv);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(abf_suite);
  GREATEST_MAIN_END();
}
