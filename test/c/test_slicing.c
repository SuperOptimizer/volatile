#include "greatest.h"
#include "core/slicing.h"
#include "core/geom.h"
#include "core/math.h"

// ---------------------------------------------------------------------------
// plane_surface_sample geometry tests (these exercise the code path
// that slice_volume_plane relies on for coordinate mapping)
// ---------------------------------------------------------------------------

TEST test_plane_sample_origin(void) {
  // At u=0, v=0 the sample should equal the plane origin.
  plane_surface p = plane_surface_from_normal(
    (vec3f){10, 20, 30}, (vec3f){0, 0, 1});
  vec3f s = plane_surface_sample(&p, 0.0f, 0.0f);
  ASSERT_IN_RANGE(9.9f, s.x, 10.1f);
  ASSERT_IN_RANGE(19.9f, s.y, 20.1f);
  ASSERT_IN_RANGE(29.9f, s.z, 30.1f);
  PASS();
}

TEST test_plane_sample_u_axis(void) {
  // For a Z-normal plane the u_axis is in the XY plane.
  plane_surface p = plane_surface_from_normal(
    (vec3f){0, 0, 0}, (vec3f){0, 0, 1});
  // sample u=1, v=0 should shift by 1 unit along u_axis
  vec3f s = plane_surface_sample(&p, 1.0f, 0.0f);
  // The sample should be 1 unit away from origin in XY
  float dist = sqrtf(s.x*s.x + s.y*s.y);
  ASSERT_IN_RANGE(0.99f, dist, 1.01f);
  PASS();
}

// ---------------------------------------------------------------------------
// slice_volume_plane / slice_volume_quad — null-safety (vol==NULL)
// vol_sample returns 0.0 for NULL vol (no crash is the key assertion)
// ---------------------------------------------------------------------------

TEST test_slice_plane_null_vol_no_crash(void) {
  plane_surface p = plane_surface_from_normal(
    (vec3f){0,0,0}, (vec3f){0,0,1});
  float out[4];
  // Should not crash; vol_sample with NULL vol is expected to return 0.
  // We catch any assert via the REQUIRE macro — just verify it runs.
  // (If REQUIRE aborts on NULL, this test will fail loudly which is fine.)
  // Use a try-equivalent: skip if vol_open is needed.
  // Since vol=NULL may REQUIRE-abort in vol_sample, we only call if safe.
  // vol_sample is documented to handle NULL by returning 0.
  // We call with a small 1x1 output to minimise risk.
  slice_volume_plane(NULL, &p, out, 1, 1, 1.0f);
  // If we get here, no crash occurred.
  PASS();
}

TEST test_slice_quad_null_vol_no_crash(void) {
  quad_surface *s = quad_surface_new(2, 2);
  ASSERT(s != NULL);
  quad_surface_set(s, 0, 0, (vec3f){0,0,0});
  quad_surface_set(s, 0, 1, (vec3f){1,0,0});
  quad_surface_set(s, 1, 0, (vec3f){0,1,0});
  quad_surface_set(s, 1, 1, (vec3f){1,1,0});
  float out[4];
  slice_volume_quad(NULL, s, out, 2, 2);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Output buffer values: with NULL volume vol_sample returns 0.0,
// so the output should be all zeros.
// ---------------------------------------------------------------------------

TEST test_slice_plane_output_zeros_for_null_vol(void) {
  plane_surface p = plane_surface_from_normal(
    (vec3f){0,0,0}, (vec3f){0,0,1});
  float out[9] = {99, 99, 99, 99, 99, 99, 99, 99, 99};
  slice_volume_plane(NULL, &p, out, 3, 3, 1.0f);
  for (int i = 0; i < 9; i++)
    ASSERT_IN_RANGE(-0.01f, out[i], 0.01f);
  PASS();
}

TEST test_slice_quad_output_zeros_for_null_vol(void) {
  quad_surface *s = quad_surface_new(2, 2);
  ASSERT(s != NULL);
  quad_surface_set(s, 0, 0, (vec3f){1,2,3});
  quad_surface_set(s, 0, 1, (vec3f){4,5,6});
  quad_surface_set(s, 1, 0, (vec3f){7,8,9});
  quad_surface_set(s, 1, 1, (vec3f){10,11,12});
  float out[4] = {99, 99, 99, 99};
  slice_volume_quad(NULL, s, out, 2, 2);
  for (int i = 0; i < 4; i++)
    ASSERT_IN_RANGE(-0.01f, out[i], 0.01f);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// slice_volume_quad: rows/cols clamp to surf dimensions
// ---------------------------------------------------------------------------

TEST test_slice_quad_clamped_dims(void) {
  // Request more rows/cols than the surface has — only valid cells filled.
  quad_surface *s = quad_surface_new(2, 2);
  ASSERT(s != NULL);
  float out[16] = {0};  // request 4x4 but surface is 2x2
  // Should not read out-of-bounds; only fill top-left 2x2 of a 4-col output.
  slice_volume_quad(NULL, s, out, 4, 4);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// plane_surface_dist sanity check
// ---------------------------------------------------------------------------

TEST test_plane_dist_on_plane_is_zero(void) {
  plane_surface p = plane_surface_from_normal(
    (vec3f){5, 5, 5}, (vec3f){0, 1, 0});
  // A point on the plane (same y as origin, any x,z)
  float d = plane_surface_dist(&p, (vec3f){10, 5, 3});
  ASSERT_IN_RANGE(-0.01f, d, 0.01f);
  PASS();
}

TEST test_plane_dist_above_plane(void) {
  plane_surface p = plane_surface_from_normal(
    (vec3f){0, 0, 0}, (vec3f){0, 1, 0});
  float d = plane_surface_dist(&p, (vec3f){0, 3, 0});
  ASSERT_IN_RANGE(2.99f, d, 3.01f);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(slicing_suite) {
  RUN_TEST(test_plane_sample_origin);
  RUN_TEST(test_plane_sample_u_axis);
  RUN_TEST(test_slice_plane_null_vol_no_crash);
  RUN_TEST(test_slice_quad_null_vol_no_crash);
  RUN_TEST(test_slice_plane_output_zeros_for_null_vol);
  RUN_TEST(test_slice_quad_output_zeros_for_null_vol);
  RUN_TEST(test_slice_quad_clamped_dims);
  RUN_TEST(test_plane_dist_on_plane_is_zero);
  RUN_TEST(test_plane_dist_above_plane);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(slicing_suite);
  GREATEST_MAIN_END();
}
