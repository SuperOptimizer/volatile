#include "greatest.h"
#include "gui/viewer3d.h"
#include "core/math.h"

#include <stdint.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static viewer3d_config default_cfg(void) {
  return (viewer3d_config){
    .mode        = RENDER3D_MIP,
    .iso_value   = 0.5f,
    .step_size   = 0.5f,
    .fov_degrees = 45.0f,
    .cmap_id     = 0,
    .window      = 1.0f,
    .level       = 0.5f,
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  ASSERT(v != NULL);
  viewer3d_free(v);
  PASS();
}

TEST test_defaults_filled(void) {
  // Zero step_size and fov should be replaced with defaults
  viewer3d_config cfg = {0};
  viewer3d *v = viewer3d_new(cfg);
  ASSERT(v != NULL);
  viewer3d_free(v);
  PASS();
}

TEST test_set_camera_no_crash(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  viewer3d_set_camera(v,
    (vec3f){0.0f, 0.0f, -3.0f},
    (vec3f){0.5f, 0.5f,  0.5f},
    (vec3f){0.0f, 1.0f,  0.0f});
  viewer3d_free(v);
  PASS();
}

TEST test_orbit_no_crash(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  viewer3d_orbit(v, 0.1f, 0.05f);
  viewer3d_orbit(v, -0.2f, -0.1f);
  viewer3d_free(v);
  PASS();
}

TEST test_dolly_no_crash(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  viewer3d_dolly(v, 0.5f);
  viewer3d_dolly(v, -1.0f);
  viewer3d_free(v);
  PASS();
}

TEST test_screen_ray_direction_normalised(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  viewer3d_set_camera(v,
    (vec3f){0.0f, 0.0f, -3.0f},
    (vec3f){0.5f, 0.5f,  0.5f},
    (vec3f){0.0f, 1.0f,  0.0f});

  vec3f origin, dir;
  viewer3d_screen_ray(v, 32.0f, 32.0f, 64, 64, &origin, &dir);

  float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
  ASSERT_IN_RANGE(len, 0.999f, 1.001f);

  viewer3d_free(v);
  PASS();
}

TEST test_screen_ray_centre_points_at_target(void) {
  // Centre pixel ray should point roughly toward the target
  viewer3d *v = viewer3d_new(default_cfg());
  vec3f eye    = (vec3f){0.0f, 0.0f, -3.0f};
  vec3f target = (vec3f){0.5f, 0.5f,  0.5f};
  viewer3d_set_camera(v, eye, target, (vec3f){0.0f, 1.0f, 0.0f});

  vec3f origin, dir;
  int w = 64, h = 64;
  // Centre pixel = (w/2 - 0.5, h/2 - 0.5) but closest is pixel (31,31)
  viewer3d_screen_ray(v, 31.5f, 31.5f, w, h, &origin, &dir);

  vec3f expected = {target.x - eye.x, target.y - eye.y, target.z - eye.z};
  float len = sqrtf(expected.x*expected.x + expected.y*expected.y + expected.z*expected.z);
  expected.x /= len; expected.y /= len; expected.z /= len;

  // Dot product should be > 0.99 (within ~8 degrees)
  float dot = dir.x*expected.x + dir.y*expected.y + dir.z*expected.z;
  ASSERT(dot > 0.99f);

  viewer3d_free(v);
  PASS();
}

TEST test_render_null_vol_black(void) {
  // Rendering without a volume should produce all-zero (black) output
  viewer3d *v = viewer3d_new(default_cfg());

  int w = 16, h = 16;
  uint8_t pixels[16 * 16 * 4];
  memset(pixels, 0xFF, sizeof(pixels));

  viewer3d_render_cpu(v, pixels, w, h);

  bool all_zero = true;
  for (int i = 0; i < w * h * 4; i++) {
    if (pixels[i] != 0) { all_zero = false; break; }
  }
  ASSERT(all_zero);

  viewer3d_free(v);
  PASS();
}

TEST test_set_volume_null_safe(void) {
  viewer3d *v = viewer3d_new(default_cfg());
  viewer3d_set_volume(v, NULL);  // must not crash
  viewer3d_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(viewer3d_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_defaults_filled);
  RUN_TEST(test_set_camera_no_crash);
  RUN_TEST(test_orbit_no_crash);
  RUN_TEST(test_dolly_no_crash);
  RUN_TEST(test_screen_ray_direction_normalised);
  RUN_TEST(test_screen_ray_centre_points_at_target);
  RUN_TEST(test_render_null_vol_black);
  RUN_TEST(test_set_volume_null_safe);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(viewer3d_suite);
  GREATEST_MAIN_END();
}
