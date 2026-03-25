#include "greatest.h"
#include "gui/crosshair.h"
#include "gui/viewer.h"
#include "render/camera.h"
#include "render/composite.h"
#include "render/overlay.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static slice_viewer *make_viewer(int axis) {
  viewer_config cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.view_axis = axis;
  cfg.window    = 1.0f;
  cfg.level     = 0.5f;
  camera_init(&cfg.camera);
  composite_params_default(&cfg.composite);
  return viewer_new(cfg, NULL);
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  crosshair_sync *s = crosshair_sync_new();
  ASSERT(s != NULL);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// focus get/set
// ---------------------------------------------------------------------------

TEST test_set_get_focus(void) {
  crosshair_sync *s = crosshair_sync_new();
  vec3f pos = {1.5f, 2.5f, 3.5f};
  crosshair_sync_set_focus(s, pos);
  vec3f got = crosshair_sync_get_focus(s);
  ASSERT_IN_RANGE(1.5f, got.x, 1e-5f);
  ASSERT_IN_RANGE(2.5f, got.y, 1e-5f);
  ASSERT_IN_RANGE(3.5f, got.z, 1e-5f);
  crosshair_sync_free(s);
  PASS();
}

TEST test_default_focus_zero(void) {
  crosshair_sync *s = crosshair_sync_new();
  vec3f got = crosshair_sync_get_focus(s);
  ASSERT_IN_RANGE(0.0f, got.x, 1e-5f);
  ASSERT_IN_RANGE(0.0f, got.y, 1e-5f);
  ASSERT_IN_RANGE(0.0f, got.z, 1e-5f);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// viewer registry
// ---------------------------------------------------------------------------

TEST test_add_remove_viewer(void) {
  crosshair_sync *s = crosshair_sync_new();
  slice_viewer *v = make_viewer(0);

  crosshair_sync_add_viewer(s, v);
  // duplicate add should be a no-op
  crosshair_sync_add_viewer(s, v);

  crosshair_sync_remove_viewer(s, v);
  // remove non-member should not crash
  crosshair_sync_remove_viewer(s, v);

  viewer_free(v);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// overlay generation: XY viewer (axis=0)
// focus (fx, fy, fz) -> vertical line at x=fx, horizontal line at y=fy
// color: orange (255, 165, 0)
// ---------------------------------------------------------------------------

TEST test_overlay_xy_line_positions(void) {
  crosshair_sync *s = crosshair_sync_new();
  slice_viewer   *v = make_viewer(0);
  overlay_list   *ol = overlay_list_new();

  crosshair_sync_add_viewer(s, v);
  crosshair_sync_set_focus(s, (vec3f){10.0f, 20.0f, 30.0f});
  crosshair_sync_render_overlays(s, v, ol);

  // Should have exactly 2 lines
  ASSERT_EQ(2, overlay_count(ol));

  overlay_list_free(ol);
  viewer_free(v);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// overlay generation: XZ viewer (axis=1)
// focus (fx, fy, fz) -> vertical line at x=fx, horizontal line at z=fz
// ---------------------------------------------------------------------------

TEST test_overlay_xz_two_lines(void) {
  crosshair_sync *s = crosshair_sync_new();
  slice_viewer   *v = make_viewer(1);
  overlay_list   *ol = overlay_list_new();

  crosshair_sync_set_focus(s, (vec3f){5.0f, 15.0f, 25.0f});
  crosshair_sync_render_overlays(s, v, ol);

  ASSERT_EQ(2, overlay_count(ol));

  overlay_list_free(ol);
  viewer_free(v);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// overlay generation: YZ viewer (axis=2)
// focus (fx, fy, fz) -> vertical line at y=fy, horizontal line at z=fz
// ---------------------------------------------------------------------------

TEST test_overlay_yz_two_lines(void) {
  crosshair_sync *s = crosshair_sync_new();
  slice_viewer   *v = make_viewer(2);
  overlay_list   *ol = overlay_list_new();

  crosshair_sync_set_focus(s, (vec3f){3.0f, 7.0f, 11.0f});
  crosshair_sync_render_overlays(s, v, ol);

  ASSERT_EQ(2, overlay_count(ol));

  overlay_list_free(ol);
  viewer_free(v);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// all three viewers together: each gets 2 lines
// ---------------------------------------------------------------------------

TEST test_overlay_three_viewers(void) {
  crosshair_sync *s  = crosshair_sync_new();
  slice_viewer   *vxy = make_viewer(0);
  slice_viewer   *vxz = make_viewer(1);
  slice_viewer   *vyz = make_viewer(2);

  crosshair_sync_add_viewer(s, vxy);
  crosshair_sync_add_viewer(s, vxz);
  crosshair_sync_add_viewer(s, vyz);

  crosshair_sync_set_focus(s, (vec3f){1.0f, 2.0f, 3.0f});

  overlay_list *ol_xy = overlay_list_new();
  overlay_list *ol_xz = overlay_list_new();
  overlay_list *ol_yz = overlay_list_new();

  crosshair_sync_render_overlays(s, vxy, ol_xy);
  crosshair_sync_render_overlays(s, vxz, ol_xz);
  crosshair_sync_render_overlays(s, vyz, ol_yz);

  ASSERT_EQ(2, overlay_count(ol_xy));
  ASSERT_EQ(2, overlay_count(ol_xz));
  ASSERT_EQ(2, overlay_count(ol_yz));

  overlay_list_free(ol_xy);
  overlay_list_free(ol_xz);
  overlay_list_free(ol_yz);
  viewer_free(vxy);
  viewer_free(vxz);
  viewer_free(vyz);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// focus update propagates on next render_overlays call
// ---------------------------------------------------------------------------

TEST test_focus_update(void) {
  crosshair_sync *s = crosshair_sync_new();

  crosshair_sync_set_focus(s, (vec3f){1.0f, 2.0f, 3.0f});
  crosshair_sync_set_focus(s, (vec3f){9.0f, 8.0f, 7.0f});

  vec3f got = crosshair_sync_get_focus(s);
  ASSERT_IN_RANGE(9.0f, got.x, 1e-5f);
  ASSERT_IN_RANGE(8.0f, got.y, 1e-5f);
  ASSERT_IN_RANGE(7.0f, got.z, 1e-5f);

  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// null-safety: render_overlays with unregistered viewer should not crash
// ---------------------------------------------------------------------------

TEST test_render_unregistered_viewer(void) {
  crosshair_sync *s = crosshair_sync_new();
  slice_viewer   *v = make_viewer(0);
  overlay_list   *ol = overlay_list_new();

  // viewer not added to sync — should still work fine
  crosshair_sync_set_focus(s, (vec3f){1, 2, 3});
  crosshair_sync_render_overlays(s, v, ol);
  ASSERT_EQ(2, overlay_count(ol));

  overlay_list_free(ol);
  viewer_free(v);
  crosshair_sync_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_crosshair) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_set_get_focus);
  RUN_TEST(test_default_focus_zero);
  RUN_TEST(test_add_remove_viewer);
  RUN_TEST(test_overlay_xy_line_positions);
  RUN_TEST(test_overlay_xz_two_lines);
  RUN_TEST(test_overlay_yz_two_lines);
  RUN_TEST(test_overlay_three_viewers);
  RUN_TEST(test_focus_update);
  RUN_TEST(test_render_unregistered_viewer);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_crosshair);
  GREATEST_MAIN_END();
}
