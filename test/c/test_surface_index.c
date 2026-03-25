#include "greatest.h"
#include "core/surface_index.h"
#include "core/geom.h"
#include "core/math.h"

#include <math.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helper: flat R×C grid
// ---------------------------------------------------------------------------

static quad_surface *make_grid(int R, int C) {
  quad_surface *s = quad_surface_new(R, C);
  if (!s) return NULL;
  for (int r = 0; r < R; r++)
    for (int c = 0; c < C; c++)
      quad_surface_set(s, r, c, (vec3f){(float)c, (float)r, 0.0f});
  return s;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_build_free(void) {
  quad_surface *s = make_grid(4, 5);
  ASSERT(s != NULL);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);
  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_free_null(void) {
  surface_index_free(NULL);  // must not crash
  PASS();
}

TEST test_build_null(void) {
  ASSERT(surface_index_build(NULL) == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Nearest-neighbor queries
// ---------------------------------------------------------------------------

TEST test_nearest_exact_vertex(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Query exactly at vertex (1,2) → flat index 1*3+2 = 5
  float dist;
  int vi = surface_index_nearest(idx, (vec3f){2.0f, 1.0f, 0.0f}, &dist);
  ASSERT_EQ(vi, 5);
  ASSERT_IN_RANGE(0.0f, dist, 1e-4f);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_nearest_midpoint(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Query between (0,0) and (0,1) — should find one of them
  int vi = surface_index_nearest(idx, (vec3f){0.5f, 0.0f, 0.0f}, NULL);
  ASSERT(vi == 0 || vi == 1);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_nearest_far_outside(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Query far outside bounding box — should still return valid vertex
  float dist;
  int vi = surface_index_nearest(idx, (vec3f){1000.0f, 1000.0f, 1000.0f}, &dist);
  ASSERT(vi >= 0);
  ASSERT(dist > 100.0f);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_nearest_returns_corner(void) {
  quad_surface *s = make_grid(4, 4);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Query at exactly top-right corner (row=3, col=3) → index 15
  float dist;
  int vi = surface_index_nearest(idx, (vec3f){3.0f, 3.0f, 0.0f}, &dist);
  ASSERT_EQ(vi, 15);
  ASSERT_IN_RANGE(0.0f, dist, 1e-4f);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_nearest_no_dist_out(void) {
  quad_surface *s = make_grid(2, 2);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);
  // Should not crash with NULL out_dist
  int vi = surface_index_nearest(idx, (vec3f){0.0f, 0.0f, 0.0f}, NULL);
  ASSERT(vi >= 0);
  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Radius queries
// ---------------------------------------------------------------------------

TEST test_radius_single_cell(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Query at (0,0) with small radius — only vertex 0 at distance 0
  int out[16];
  int n = surface_index_radius(idx, (vec3f){0.0f,0.0f,0.0f}, 0.1f, out, 16);
  ASSERT_EQ(n, 1);
  ASSERT_EQ(out[0], 0);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_radius_neighbourhood(void) {
  quad_surface *s = make_grid(5, 5);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Center of 5×5 grid is at (2,2), unit spacing.
  // Radius 1.5 should include immediate 4-neighbors + center = 5 pts at most
  // But corners at (1,1),(1,3),(3,1),(3,3) are sqrt(2) ≈ 1.41 away, also inside
  int out[32];
  int n = surface_index_radius(idx, (vec3f){2.0f,2.0f,0.0f}, 1.5f, out, 32);
  ASSERT(n >= 5);   // at least center + 4 direct neighbors
  ASSERT(n <= 9);   // at most 3×3 block

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_radius_max_cap(void) {
  quad_surface *s = make_grid(4, 4);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Large radius catches all 16 vertices, but max=3 caps output
  int out[3];
  int n = surface_index_radius(idx, (vec3f){1.5f,1.5f,0.0f}, 100.0f, out, 3);
  ASSERT_EQ(n, 3);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_radius_zero_radius(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);
  int out[4];
  int n = surface_index_radius(idx, (vec3f){0.0f,0.0f,0.0f}, 0.0f, out, 4);
  ASSERT_EQ(n, 0);
  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

TEST test_radius_outside_bbox(void) {
  quad_surface *s = make_grid(3, 3);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);
  int out[4];
  // Query far away with small radius → no results
  int n = surface_index_radius(idx, (vec3f){100.0f,100.0f,0.0f}, 0.5f, out, 4);
  ASSERT_EQ(n, 0);
  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Larger grid stress test
// ---------------------------------------------------------------------------

TEST test_large_grid_nearest(void) {
  quad_surface *s = make_grid(20, 20);
  ASSERT(s != NULL);
  surface_index *idx = surface_index_build(s);
  ASSERT(idx != NULL);

  // Nearest to (10.0, 10.0, 0) should be vertex row=10,col=10 → 10*20+10=210
  float dist;
  int vi = surface_index_nearest(idx, (vec3f){10.0f, 10.0f, 0.0f}, &dist);
  ASSERT_EQ(vi, 10*20+10);
  ASSERT_IN_RANGE(0.0f, dist, 1e-4f);

  surface_index_free(idx);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(surface_index_suite) {
  RUN_TEST(test_build_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_build_null);
  RUN_TEST(test_nearest_exact_vertex);
  RUN_TEST(test_nearest_midpoint);
  RUN_TEST(test_nearest_far_outside);
  RUN_TEST(test_nearest_returns_corner);
  RUN_TEST(test_nearest_no_dist_out);
  RUN_TEST(test_radius_single_cell);
  RUN_TEST(test_radius_neighbourhood);
  RUN_TEST(test_radius_max_cap);
  RUN_TEST(test_radius_zero_radius);
  RUN_TEST(test_radius_outside_bbox);
  RUN_TEST(test_large_grid_nearest);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(surface_index_suite);
  GREATEST_MAIN_END();
}
