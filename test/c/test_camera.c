#include "greatest.h"
#include "render/camera.h"
#include <math.h>

#define EPS 1e-5f

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static viewer_camera make_cam(void) {
  viewer_camera cam;
  camera_init(&cam);
  return cam;
}

static viewport make_vp(int w, int h) {
  return (viewport){ .screen_w = w, .screen_h = h, .tile_size = 256.0f };
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

TEST test_init_defaults(void) {
  viewer_camera cam = make_cam();
  ASSERT_IN_RANGE(0.5f, cam.center.x, EPS);
  ASSERT_IN_RANGE(0.5f, cam.center.y, EPS);
  ASSERT_IN_RANGE(0.0f, cam.center.z, EPS);
  ASSERT_IN_RANGE(1.0f, cam.scale, EPS);
  ASSERT_IN_RANGE(0.0f, cam.z_offset, EPS);
  ASSERT_EQ(0, cam.pyramid_level);
  ASSERT_EQ(0u, cam.epoch);
  PASS();
}

// ---------------------------------------------------------------------------
// pan
// ---------------------------------------------------------------------------

TEST test_pan_moves_center(void) {
  viewer_camera cam = make_cam();
  // scale=1 so 10px pan = 10 surface units shift
  camera_pan(&cam, 10.0f, 0.0f);
  ASSERT_IN_RANGE(0.5f - 10.0f, cam.center.x, EPS);
  ASSERT_IN_RANGE(0.5f, cam.center.y, EPS);
  PASS();
}

TEST test_pan_bumps_epoch(void) {
  viewer_camera cam = make_cam();
  uint64_t prev = cam.epoch;
  camera_pan(&cam, 1.0f, 1.0f);
  ASSERT(cam.epoch > prev);
  PASS();
}

TEST test_pan_both_axes(void) {
  viewer_camera cam = make_cam();
  camera_pan(&cam, -5.0f, 3.0f);
  ASSERT_IN_RANGE(0.5f + 5.0f, cam.center.x, EPS);
  ASSERT_IN_RANGE(0.5f - 3.0f, cam.center.y, EPS);
  PASS();
}

// ---------------------------------------------------------------------------
// zoom
// ---------------------------------------------------------------------------

TEST test_zoom_in_decreases_scale(void) {
  viewer_camera cam = make_cam();
  viewport      vp  = make_vp(800, 600);
  float old_scale = cam.scale;
  camera_zoom(&cam, &vp, 2.0f, 400.0f, 300.0f);
  ASSERT(cam.scale < old_scale);
  PASS();
}

TEST test_zoom_out_increases_scale(void) {
  viewer_camera cam = make_cam();
  viewport      vp  = make_vp(800, 600);
  float old_scale = cam.scale;
  camera_zoom(&cam, &vp, 0.5f, 400.0f, 300.0f);
  ASSERT(cam.scale > old_scale);
  PASS();
}

TEST test_zoom_clamped_min(void) {
  viewer_camera cam = make_cam();
  viewport      vp  = make_vp(800, 600);
  for (int i = 0; i < 20; i++) camera_zoom(&cam, &vp, 2.0f, 400.0f, 300.0f);
  ASSERT(cam.scale >= 1.0f / 32.0f - EPS);
  PASS();
}

TEST test_zoom_clamped_max(void) {
  viewer_camera cam = make_cam();
  viewport      vp  = make_vp(800, 600);
  for (int i = 0; i < 20; i++) camera_zoom(&cam, &vp, 0.25f, 400.0f, 300.0f);
  ASSERT(cam.scale <= 4.0f + EPS);
  PASS();
}

// Zoom centered on screen-space pivot: the surface coord under (cx,cy) must
// remain unchanged after a zoom.
TEST test_zoom_pivot_invariant(void) {
  viewer_camera cam = make_cam();
  viewport      vp  = make_vp(800, 600);

  float cx = 200.0f, cy = 150.0f;
  // surface coord under pivot before zoom
  float su0, sv0;
  viewport_screen_to_surface(&cam, &vp, cx, cy, &su0, &sv0);

  camera_zoom(&cam, &vp, 2.0f, cx, cy);

  float su1, sv1;
  viewport_screen_to_surface(&cam, &vp, cx, cy, &su1, &sv1);

  ASSERT_IN_RANGE(su0, su1, 1e-4f);
  ASSERT_IN_RANGE(sv0, sv1, 1e-4f);
  PASS();
}

// ---------------------------------------------------------------------------
// z offset
// ---------------------------------------------------------------------------

TEST test_set_z_offset(void) {
  viewer_camera cam = make_cam();
  camera_set_z_offset(&cam, 42.5f);
  ASSERT_IN_RANGE(42.5f, cam.z_offset, EPS);
  ASSERT(cam.epoch > 0u);
  PASS();
}

TEST test_step_z(void) {
  viewer_camera cam = make_cam();
  camera_step_z(&cam, 1.0f);
  ASSERT_IN_RANGE(1.0f, cam.z_offset, EPS);
  camera_step_z(&cam, -0.5f);
  ASSERT_IN_RANGE(0.5f, cam.z_offset, EPS);
  PASS();
}

// ---------------------------------------------------------------------------
// pyramid level
// ---------------------------------------------------------------------------

TEST test_pyramid_level_no_levels(void) {
  viewer_camera cam = make_cam();
  ASSERT_EQ(0, camera_calc_pyramid_level(&cam, 1));
  PASS();
}

TEST test_pyramid_level_scale1(void) {
  viewer_camera cam = make_cam();
  cam.scale = 1.0f;  // 1 texel/pixel => level 0
  ASSERT_EQ(0, camera_calc_pyramid_level(&cam, 5));
  PASS();
}

TEST test_pyramid_level_scale_half(void) {
  viewer_camera cam = make_cam();
  cam.scale = 0.5f;  // 2 texels/pixel => level 1
  int level = camera_calc_pyramid_level(&cam, 5);
  ASSERT_EQ(1, level);
  PASS();
}

TEST test_pyramid_level_scale_quarter(void) {
  viewer_camera cam = make_cam();
  cam.scale = 0.25f;  // 4 texels/pixel => level 2
  int level = camera_calc_pyramid_level(&cam, 5);
  ASSERT_EQ(2, level);
  PASS();
}

TEST test_pyramid_level_clamped(void) {
  viewer_camera cam = make_cam();
  cam.scale = 0.01f;  // very zoomed out => clamp to max level
  int level = camera_calc_pyramid_level(&cam, 3);
  ASSERT_EQ(2, level);
  PASS();
}

// ---------------------------------------------------------------------------
// invalidate
// ---------------------------------------------------------------------------

TEST test_invalidate_bumps_epoch(void) {
  viewer_camera cam = make_cam();
  uint64_t e0 = cam.epoch;
  camera_invalidate(&cam);
  ASSERT(cam.epoch == e0 + 1);
  PASS();
}

// ---------------------------------------------------------------------------
// surface <-> screen roundtrip
// ---------------------------------------------------------------------------

TEST test_roundtrip_center(void) {
  viewer_camera cam = make_cam();
  viewport vp = make_vp(800, 600);

  // screen center should map to cam->center
  float su, sv;
  viewport_screen_to_surface(&cam, &vp, 400.0f, 300.0f, &su, &sv);
  ASSERT_IN_RANGE(cam.center.x, su, EPS);
  ASSERT_IN_RANGE(cam.center.y, sv, EPS);
  PASS();
}

TEST test_roundtrip_surface_to_screen_and_back(void) {
  viewer_camera cam = make_cam();
  cam.scale = 0.5f;
  cam.center.x = 0.3f;
  cam.center.y = 0.7f;
  viewport vp = make_vp(1024, 768);

  float su_in = 0.12f, sv_in = 0.88f;
  float sx, sy;
  viewport_surface_to_screen(&cam, &vp, su_in, sv_in, &sx, &sy);

  float su_out, sv_out;
  viewport_screen_to_surface(&cam, &vp, sx, sy, &su_out, &sv_out);

  ASSERT_IN_RANGE(su_in, su_out, EPS);
  ASSERT_IN_RANGE(sv_in, sv_out, EPS);
  PASS();
}

TEST test_roundtrip_screen_to_surface_and_back(void) {
  viewer_camera cam = make_cam();
  cam.scale = 2.0f;
  viewport vp = make_vp(640, 480);

  float sx_in = 100.0f, sy_in = 200.0f;
  float su, sv;
  viewport_screen_to_surface(&cam, &vp, sx_in, sy_in, &su, &sv);

  float sx_out, sy_out;
  viewport_surface_to_screen(&cam, &vp, su, sv, &sx_out, &sy_out);

  ASSERT_IN_RANGE(sx_in, sx_out, EPS);
  ASSERT_IN_RANGE(sy_in, sy_out, EPS);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(camera_suite) {
  RUN_TEST(test_init_defaults);
  RUN_TEST(test_pan_moves_center);
  RUN_TEST(test_pan_bumps_epoch);
  RUN_TEST(test_pan_both_axes);
  RUN_TEST(test_zoom_in_decreases_scale);
  RUN_TEST(test_zoom_out_increases_scale);
  RUN_TEST(test_zoom_clamped_min);
  RUN_TEST(test_zoom_clamped_max);
  RUN_TEST(test_zoom_pivot_invariant);
  RUN_TEST(test_set_z_offset);
  RUN_TEST(test_step_z);
  RUN_TEST(test_pyramid_level_no_levels);
  RUN_TEST(test_pyramid_level_scale1);
  RUN_TEST(test_pyramid_level_scale_half);
  RUN_TEST(test_pyramid_level_scale_quarter);
  RUN_TEST(test_pyramid_level_clamped);
  RUN_TEST(test_invalidate_bumps_epoch);
  RUN_TEST(test_roundtrip_center);
  RUN_TEST(test_roundtrip_surface_to_screen_and_back);
  RUN_TEST(test_roundtrip_screen_to_surface_and_back);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(camera_suite);
  GREATEST_MAIN_END();
}
