#include "greatest.h"
#include "gui/viewer.h"
#include "render/camera.h"
#include "render/composite.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static viewer_config make_config(int axis) {
  viewer_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.view_axis = axis;
  cfg.cmap_id   = 0;  // CMAP_GRAYSCALE
  cfg.window    = 1.0f;
  cfg.level     = 0.5f;
  camera_init(&cfg.camera);
  composite_params_default(&cfg.composite);
  return cfg;
}

// ---------------------------------------------------------------------------
// Creation / free
// ---------------------------------------------------------------------------

TEST test_viewer_new_free(void) {
  viewer_config cfg = make_config(0);
  slice_viewer *v = viewer_new(cfg, NULL);
  ASSERT(v != NULL);
  viewer_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Render with NULL volume -> fills grey, all alpha=255
// ---------------------------------------------------------------------------

TEST test_render_null_volume(void) {
  viewer_config cfg = make_config(0);
  slice_viewer *v = viewer_new(cfg, NULL);
  ASSERT(v != NULL);

  int W = 64, H = 64;
  uint8_t *pixels = calloc((size_t)W * H * 4, 1);
  ASSERT(pixels != NULL);

  viewer_render(v, pixels, W, H);

  // Output should be non-zero (grey fill) with alpha=255
  int nonzero = 0;
  for (int i = 0; i < W * H * 4; i++) {
    if (pixels[i] != 0) nonzero++;
  }
  ASSERT(nonzero > 0);

  // Alpha channel (every 4th byte starting at index 3) should be 255
  for (int i = 3; i < W * H * 4; i += 4) {
    ASSERT_EQ(255, pixels[i]);
  }

  free(pixels);
  viewer_free(v);
  PASS();
}

TEST test_render_produces_pixels(void) {
  viewer_config cfg = make_config(0);
  slice_viewer *v = viewer_new(cfg, NULL);

  int W = 128, H = 128;
  uint8_t *pixels = malloc((size_t)W * H * 4);
  memset(pixels, 0, (size_t)W * H * 4);

  viewer_render(v, pixels, W, H);

  // At least some pixels should be set
  int set = 0;
  for (int i = 0; i < W * H * 4; i++) {
    if (pixels[i] != 0) set++;
  }
  ASSERT(set > 0);

  free(pixels);
  viewer_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Render different axes (should all produce output)
// ---------------------------------------------------------------------------

TEST test_render_all_axes(void) {
  int W = 32, H = 32;
  uint8_t *pixels = malloc((size_t)W * H * 4);

  for (int axis = 0; axis < 3; axis++) {
    viewer_config cfg = make_config(axis);
    slice_viewer *v = viewer_new(cfg, NULL);
    memset(pixels, 0, (size_t)W * H * 4);
    viewer_render(v, pixels, W, H);

    int nonzero = 0;
    for (int i = 0; i < W * H * 4; i++) if (pixels[i]) nonzero++;
    ASSERT(nonzero > 0);
    viewer_free(v);
  }

  free(pixels);
  PASS();
}

// ---------------------------------------------------------------------------
// Pan / zoom / scroll don't crash
// ---------------------------------------------------------------------------

TEST test_camera_ops(void) {
  viewer_config cfg = make_config(0);
  slice_viewer *v = viewer_new(cfg, NULL);

  viewer_pan(v, 10.0f, -5.0f);
  viewer_zoom(v, 2.0f, 100.0f, 100.0f);
  viewer_scroll_slice(v, 1.0f);
  viewer_scroll_slice(v, -1.0f);

  // Should still render without crash
  int W = 32, H = 32;
  uint8_t *pixels = calloc((size_t)W * H * 4, 1);
  viewer_render(v, pixels, W, H);
  free(pixels);

  viewer_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// viewer_current_slice / viewer_current_level
// ---------------------------------------------------------------------------

TEST test_current_slice(void) {
  viewer_config cfg = make_config(0);
  cfg.camera.z_offset = 42.5f;
  slice_viewer *v = viewer_new(cfg, NULL);

  ASSERT_IN_RANGE(42.5f, viewer_current_slice(v), 1e-5f);
  ASSERT_EQ(0, viewer_current_level(v));  // no volume -> level 0

  viewer_free(v);
  PASS();
}

TEST test_scroll_updates_slice(void) {
  viewer_config cfg = make_config(0);
  cfg.camera.z_offset = 0.0f;
  slice_viewer *v = viewer_new(cfg, NULL);

  viewer_scroll_slice(v, 5.0f);
  float after = viewer_current_slice(v);
  ASSERT(after != 0.0f);  // scroll changed the value

  viewer_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// screen_to_world: axes produce correct world positions
// ---------------------------------------------------------------------------

TEST test_screen_to_world_xy(void) {
  viewer_config cfg = make_config(0);  // XY: z_offset -> Z component
  cfg.camera.z_offset = 7.0f;
  slice_viewer *v = viewer_new(cfg, NULL);

  vec3f w = viewer_screen_to_world(v, 0.0f, 0.0f);
  // z component should reflect z_offset
  ASSERT_IN_RANGE(7.0f, w.z, 1e-4f);

  viewer_free(v);
  PASS();
}

TEST test_screen_to_world_xz(void) {
  viewer_config cfg = make_config(1);  // XZ: z_offset -> Y component
  cfg.camera.z_offset = 3.0f;
  slice_viewer *v = viewer_new(cfg, NULL);

  vec3f w = viewer_screen_to_world(v, 0.0f, 0.0f);
  ASSERT_IN_RANGE(3.0f, w.y, 1e-4f);

  viewer_free(v);
  PASS();
}

TEST test_screen_to_world_yz(void) {
  viewer_config cfg = make_config(2);  // YZ: z_offset -> X component
  cfg.camera.z_offset = 5.5f;
  slice_viewer *v = viewer_new(cfg, NULL);

  vec3f w = viewer_screen_to_world(v, 0.0f, 0.0f);
  ASSERT_IN_RANGE(5.5f, w.x, 1e-4f);

  viewer_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_viewer) {
  RUN_TEST(test_viewer_new_free);
  RUN_TEST(test_render_null_volume);
  RUN_TEST(test_render_produces_pixels);
  RUN_TEST(test_render_all_axes);
  RUN_TEST(test_camera_ops);
  RUN_TEST(test_current_slice);
  RUN_TEST(test_scroll_updates_slice);
  RUN_TEST(test_screen_to_world_xy);
  RUN_TEST(test_screen_to_world_xz);
  RUN_TEST(test_screen_to_world_yz);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_viewer);
  GREATEST_MAIN_END();
}
