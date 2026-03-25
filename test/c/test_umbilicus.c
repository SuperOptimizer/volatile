#include "greatest.h"
#include "core/umbilicus.h"

#include <math.h>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  vec3f ctrl[] = { {100, 200, 0}, {110, 210, 50}, {120, 220, 99} };
  umbilicus *u = umbilicus_from_points(ctrl, 3, 100);
  ASSERT(u != NULL);
  ASSERT_EQ(100, u->count);
  umbilicus_free(u);
  PASS();
}

TEST test_null_inputs(void) {
  ASSERT(umbilicus_from_points(NULL, 3, 100) == NULL);
  ASSERT(umbilicus_from_points((vec3f[]){ {0,0,0} }, 0, 100) == NULL);
  ASSERT(umbilicus_from_points((vec3f[]){ {0,0,0} }, 1,   0) == NULL);
  umbilicus_free(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Single control point — center is constant in XY across all slices
// ---------------------------------------------------------------------------

TEST test_single_ctrl_constant_xy(void) {
  vec3f ctrl[] = { {50.0f, 60.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 1, 200);
  ASSERT(u != NULL);
  ASSERT_IN_RANGE(49.9f, u->points[0].x,   50.1f);
  ASSERT_IN_RANGE(49.9f, u->points[100].x, 50.1f);
  ASSERT_IN_RANGE(49.9f, u->points[199].x, 50.1f);
  umbilicus_free(u);
  PASS();
}

// ---------------------------------------------------------------------------
// Two control points — center at midpoint z should interpolate correctly
// ---------------------------------------------------------------------------

TEST test_two_ctrl_interpolation(void) {
  vec3f ctrl[] = { {0.0f, 0.0f, 0.0f}, {100.0f, 100.0f, 100.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 2, 101);
  ASSERT(u != NULL);
  // At z=50 center should be ~(50, 50)
  ASSERT_IN_RANGE(48.0f, u->points[50].x, 52.0f);
  ASSERT_IN_RANGE(48.0f, u->points[50].y, 52.0f);
  umbilicus_free(u);
  PASS();
}

// ---------------------------------------------------------------------------
// Clamping: z below/above control range uses boundary center
// ---------------------------------------------------------------------------

TEST test_clamp_below(void) {
  vec3f ctrl[] = { {20.0f, 30.0f, 10.0f}, {40.0f, 50.0f, 90.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 2, 100);
  ASSERT(u != NULL);
  // z=0 is below first control z=10: should use first xy
  ASSERT_IN_RANGE(19.0f, u->points[0].x, 21.0f);
  umbilicus_free(u);
  PASS();
}

// ---------------------------------------------------------------------------
// umbilicus_distance: point on center should be 0
// ---------------------------------------------------------------------------

TEST test_distance_at_center_zero(void) {
  vec3f ctrl[] = { {100.0f, 100.0f, 0.0f}, {100.0f, 100.0f, 99.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 2, 100);
  ASSERT(u != NULL);
  vec3f center = u->points[50];  // exact center
  float d = umbilicus_distance(u, center);
  ASSERT_IN_RANGE(-0.01f, d, 0.01f);
  umbilicus_free(u);
  PASS();
}

TEST test_distance_nonzero(void) {
  vec3f ctrl[] = { {0.0f, 0.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 1, 10);
  ASSERT(u != NULL);
  vec3f p = {3.0f, 4.0f, 5.0f};
  float d = umbilicus_distance(u, p);
  // sqrt(3^2 + 4^2) = 5
  ASSERT_IN_RANGE(4.9f, d, 5.1f);
  umbilicus_free(u);
  PASS();
}

// ---------------------------------------------------------------------------
// umbilicus_winding_angle
// ---------------------------------------------------------------------------

TEST test_winding_pos_x(void) {
  vec3f ctrl[] = { {0.0f, 0.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 1, 10);
  ASSERT(u != NULL);
  // +X direction: atan2(0, 3) = 0 degrees
  vec3f p = {3.0f, 0.0f, 0.0f};
  float ang = umbilicus_winding_angle(u, p);
  ASSERT_IN_RANGE(-0.5f, ang, 0.5f);
  umbilicus_free(u);
  PASS();
}

TEST test_winding_pos_y(void) {
  vec3f ctrl[] = { {0.0f, 0.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 1, 10);
  ASSERT(u != NULL);
  // +Y direction: atan2(3, 0) = 90 degrees
  vec3f p = {0.0f, 3.0f, 0.0f};
  float ang = umbilicus_winding_angle(u, p);
  ASSERT_IN_RANGE(89.5f, ang, 90.5f);
  umbilicus_free(u);
  PASS();
}

TEST test_winding_neg_x_wraps(void) {
  vec3f ctrl[] = { {0.0f, 0.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 1, 10);
  ASSERT(u != NULL);
  // -X direction: atan2(0,-1) = -180 -> mapped to 180
  vec3f p = {-3.0f, 0.0f, 0.0f};
  float ang = umbilicus_winding_angle(u, p);
  ASSERT_IN_RANGE(179.0f, ang, 181.0f);
  umbilicus_free(u);
  PASS();
}

TEST test_winding_null_returns_zero(void) {
  ASSERT_IN_RANGE(-0.01f, umbilicus_winding_angle(NULL, (vec3f){1,2,3}), 0.01f);
  PASS();
}

// ---------------------------------------------------------------------------
// Unsorted control points are sorted internally
// ---------------------------------------------------------------------------

TEST test_unsorted_ctrl_sorted_internally(void) {
  // Provide in reversed z order
  vec3f ctrl[] = { {80.0f, 80.0f, 80.0f}, {40.0f, 40.0f, 40.0f}, {0.0f, 0.0f, 0.0f} };
  umbilicus *u = umbilicus_from_points(ctrl, 3, 100);
  ASSERT(u != NULL);
  // z=0 should have center ~(0,0) and z=80 should have center ~(80,80)
  ASSERT_IN_RANGE(-1.0f, u->points[0].x, 1.0f);
  ASSERT_IN_RANGE(79.0f, u->points[80].x, 81.0f);
  umbilicus_free(u);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(umbilicus_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_null_inputs);
  RUN_TEST(test_single_ctrl_constant_xy);
  RUN_TEST(test_two_ctrl_interpolation);
  RUN_TEST(test_clamp_below);
  RUN_TEST(test_distance_at_center_zero);
  RUN_TEST(test_distance_nonzero);
  RUN_TEST(test_winding_pos_x);
  RUN_TEST(test_winding_pos_y);
  RUN_TEST(test_winding_neg_x_wraps);
  RUN_TEST(test_winding_null_returns_zero);
  RUN_TEST(test_unsorted_ctrl_sorted_internally);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(umbilicus_suite);
  GREATEST_MAIN_END();
}
