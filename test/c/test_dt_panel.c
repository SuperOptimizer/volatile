#define _POSIX_C_SOURCE 200809L
#include "greatest.h"
#include "gui/dt_panel.h"

#include <stdlib.h>

// ---------------------------------------------------------------------------
// Recompute callback helper
// ---------------------------------------------------------------------------

static bool g_recompute_called = false;
static float g_recompute_thresh = 0.0f;

static void test_recompute_cb(float threshold, cmap_id cmap, void *ctx) {
  (void)cmap; (void)ctx;
  g_recompute_called = true;
  g_recompute_thresh = threshold;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_lifecycle(void) {
  dt_panel *p = dt_panel_new();
  ASSERT_NEQ(NULL, p);

  // Default state
  ASSERT(dt_panel_visible(p));
  ASSERT(dt_panel_auto_compute(p));
  ASSERT(!dt_panel_dirty(p));
  ASSERT(dt_panel_threshold(p) > 0.0f);
  ASSERT_EQ(CMAP_VIRIDIS, dt_panel_cmap(p));

  dt_panel_free(p);
  PASS();
}

TEST test_dirty_on_surface_change(void) {
  dt_panel *p = dt_panel_new();
  ASSERT_NEQ(NULL, p);

  ASSERT(!dt_panel_dirty(p));
  dt_panel_on_surface_changed(p);
  ASSERT(dt_panel_dirty(p));

  dt_panel_clear_dirty(p);
  ASSERT(!dt_panel_dirty(p));

  dt_panel_free(p);
  PASS();
}

TEST test_recompute_callback_fires(void) {
  dt_panel *p = dt_panel_new();
  ASSERT_NEQ(NULL, p);

  g_recompute_called = false;
  dt_panel_set_recompute_cb(p, test_recompute_cb, NULL);
  dt_panel_on_surface_changed(p);

  ASSERT(g_recompute_called);
  ASSERT(g_recompute_thresh > 0.0f);

  dt_panel_free(p);
  PASS();
}

TEST test_null_safe(void) {
  // NULL inputs must not crash
  ASSERT(!dt_panel_visible(NULL));
  ASSERT(!dt_panel_auto_compute(NULL));
  ASSERT(!dt_panel_dirty(NULL));
  ASSERT_EQ(0.0f, dt_panel_threshold(NULL));
  dt_panel_clear_dirty(NULL);
  dt_panel_on_surface_changed(NULL);
  dt_panel_render(NULL, NULL, "x");
  PASS();
}

TEST test_render_noop(void) {
  // render with NULL ctx must not crash (NK_STUB path)
  dt_panel *p = dt_panel_new();
  ASSERT_NEQ(NULL, p);
  dt_panel_render(p, NULL, "Distance Transform");
  dt_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(dt_panel_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_dirty_on_surface_change);
  RUN_TEST(test_recompute_callback_fires);
  RUN_TEST(test_null_safe);
  RUN_TEST(test_render_noop);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(dt_panel_suite);
  GREATEST_MAIN_END();
}
