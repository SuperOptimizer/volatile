#include "greatest.h"
#include "gui/layout.h"

#include <math.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static int feq(float a, float b) { return fabsf(a - b) < 1e-4f; }

// ---------------------------------------------------------------------------
// Creation / defaults
// ---------------------------------------------------------------------------

TEST test_new_default_not_null(void) {
  app_layout *l = layout_new_default();
  ASSERT(l != NULL);
  layout_free(l);
  PASS();
}

TEST test_default_xy_visible(void) {
  app_layout *l = layout_new_default();
  panel_rect r = layout_get_panel(l, PANEL_VIEWER_XY);
  ASSERT(r.visible);
  ASSERT(r.w > 0.0f && r.h > 0.0f);
  layout_free(l);
  PASS();
}

TEST test_default_console_spans_full_width(void) {
  app_layout *l = layout_new_default();
  panel_rect r = layout_get_panel(l, PANEL_CONSOLE);
  ASSERT(r.visible);
  ASSERT(feq(r.x, 0.0f));
  ASSERT(feq(r.w, 1.0f));
  layout_free(l);
  PASS();
}

TEST test_default_hidden_panels(void) {
  app_layout *l = layout_new_default();
  panel_rect s = layout_get_panel(l, PANEL_SETTINGS);
  panel_rect a = layout_get_panel(l, PANEL_ANNOTATIONS);
  ASSERT(!s.visible);
  ASSERT(!a.visible);
  layout_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Panel set / toggle
// ---------------------------------------------------------------------------

TEST test_set_panel_roundtrip(void) {
  app_layout *l = layout_new_default();
  panel_rect orig = layout_get_panel(l, PANEL_VIEWER_XY);
  panel_rect mod = orig;
  mod.x = 0.1f; mod.y = 0.2f; mod.w = 0.3f; mod.h = 0.4f;

  layout_set_panel(l, PANEL_VIEWER_XY, mod);
  panel_rect got = layout_get_panel(l, PANEL_VIEWER_XY);

  ASSERT(feq(got.x, 0.1f));
  ASSERT(feq(got.y, 0.2f));
  ASSERT(feq(got.w, 0.3f));
  ASSERT(feq(got.h, 0.4f));
  ASSERT_EQ(got.id, PANEL_VIEWER_XY);
  layout_free(l);
  PASS();
}

TEST test_toggle_visibility(void) {
  app_layout *l = layout_new_default();
  panel_rect r = layout_get_panel(l, PANEL_VIEWER_XY);
  ASSERT(r.visible);

  layout_toggle_panel(l, PANEL_VIEWER_XY);
  r = layout_get_panel(l, PANEL_VIEWER_XY);
  ASSERT(!r.visible);

  layout_toggle_panel(l, PANEL_VIEWER_XY);
  r = layout_get_panel(l, PANEL_VIEWER_XY);
  ASSERT(r.visible);

  layout_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

TEST test_hit_test_xy_top_left(void) {
  app_layout *l = layout_new_default();
  // Top-left corner of a 1000x1000 window should be inside XY viewer
  panel_id hit = layout_hit_test(l, 5.0f, 5.0f, 1000, 1000);
  ASSERT_EQ(hit, PANEL_VIEWER_XY);
  layout_free(l);
  PASS();
}

TEST test_hit_test_console_bottom(void) {
  app_layout *l = layout_new_default();
  // Bottom strip of the window should be the console (y > 0.85 * h)
  panel_id hit = layout_hit_test(l, 500.0f, 900.0f, 1000, 1000);
  ASSERT_EQ(hit, PANEL_CONSOLE);
  layout_free(l);
  PASS();
}

TEST test_hit_test_hidden_panel_not_hit(void) {
  app_layout *l = layout_new_default();
  // PANEL_SETTINGS is hidden with zero geometry — hit test should never return it
  for (int i = 0; i < PANEL_COUNT; i++) {
    panel_id hit = layout_hit_test(l, 1.0f, 1.0f, 1000, 1000);
    ASSERT(hit != PANEL_SETTINGS);
  }
  layout_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Preset switching
// ---------------------------------------------------------------------------

TEST test_preset_single_only_one_viewer_visible(void) {
  app_layout *l = layout_new_default();
  layout_preset_single(l);

  panel_rect xy = layout_get_panel(l, PANEL_VIEWER_XY);
  panel_rect xz = layout_get_panel(l, PANEL_VIEWER_XZ);
  panel_rect yz = layout_get_panel(l, PANEL_VIEWER_YZ);
  panel_rect v3 = layout_get_panel(l, PANEL_VIEWER_3D);

  ASSERT(xy.visible);
  ASSERT(!xz.visible);
  ASSERT(!yz.visible);
  ASSERT(!v3.visible);

  layout_free(l);
  PASS();
}

TEST test_preset_quad_no_side_panel(void) {
  app_layout *l = layout_new_default();
  layout_preset_quad(l);

  panel_rect st = layout_get_panel(l, PANEL_SURFACE_TREE);
  panel_rect sg = layout_get_panel(l, PANEL_SEGMENTATION);
  ASSERT(!st.visible);
  ASSERT(!sg.visible);

  // All four viewers visible
  ASSERT(layout_get_panel(l, PANEL_VIEWER_XY).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_XZ).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_YZ).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_3D).visible);

  layout_free(l);
  PASS();
}

TEST test_preset_vc3d_restores_four_viewers(void) {
  app_layout *l = layout_new_default();
  layout_preset_single(l);  // collapse first
  layout_preset_vc3d(l);    // restore

  ASSERT(layout_get_panel(l, PANEL_VIEWER_XY).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_XZ).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_YZ).visible);
  ASSERT(layout_get_panel(l, PANEL_VIEWER_3D).visible);

  layout_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Save / load roundtrip
// ---------------------------------------------------------------------------

TEST test_save_load_roundtrip(void) {
  const char *path = "/tmp/test_layout.json";

  app_layout *orig = layout_new_default();
  ASSERT(orig != NULL);

  // Modify a panel to differentiate from defaults
  panel_rect r = layout_get_panel(orig, PANEL_VIEWER_XY);
  r.x = 0.05f; r.y = 0.05f;
  layout_set_panel(orig, PANEL_VIEWER_XY, r);

  layout_toggle_panel(orig, PANEL_CONSOLE);  // hide console

  ASSERT(layout_save(orig, path));

  app_layout *loaded = layout_load(path);
  ASSERT(loaded != NULL);

  panel_rect xy = layout_get_panel(loaded, PANEL_VIEWER_XY);
  ASSERT(feq(xy.x, 0.05f));
  ASSERT(feq(xy.y, 0.05f));

  panel_rect con = layout_get_panel(loaded, PANEL_CONSOLE);
  ASSERT(!con.visible);

  layout_free(orig);
  layout_free(loaded);
  PASS();
}

TEST test_load_nonexistent_returns_null(void) {
  app_layout *l = layout_load("/tmp/does_not_exist_layout_xyzzy.json");
  ASSERT(l == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_layout) {
  RUN_TEST(test_new_default_not_null);
  RUN_TEST(test_default_xy_visible);
  RUN_TEST(test_default_console_spans_full_width);
  RUN_TEST(test_default_hidden_panels);
  RUN_TEST(test_set_panel_roundtrip);
  RUN_TEST(test_toggle_visibility);
  RUN_TEST(test_hit_test_xy_top_left);
  RUN_TEST(test_hit_test_console_bottom);
  RUN_TEST(test_hit_test_hidden_panel_not_hit);
  RUN_TEST(test_preset_single_only_one_viewer_visible);
  RUN_TEST(test_preset_quad_no_side_panel);
  RUN_TEST(test_preset_vc3d_restores_four_viewers);
  RUN_TEST(test_save_load_roundtrip);
  RUN_TEST(test_load_nonexistent_returns_null);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_layout);
  GREATEST_MAIN_END();
}
