#include "greatest.h"
#include "gui/viewer_controls.h"
#include "render/composite.h"
#include "render/cmap.h"

#include <string.h>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  viewer_controls *c = viewer_controls_new();
  ASSERT(c != NULL);
  viewer_controls_free(c);
  PASS();
}

TEST test_free_null(void) {
  viewer_controls_free(NULL);   // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Initial state
// ---------------------------------------------------------------------------

TEST test_default_cmap(void) {
  viewer_controls *c = viewer_controls_new();
  ASSERT(c != NULL);
  // Default colormap is index 0 (CMAP_GRAYSCALE)
  ASSERT_EQ(CMAP_GRAYSCALE, viewer_controls_get_cmap(c));
  viewer_controls_free(c);
  PASS();
}

TEST test_default_composite(void) {
  viewer_controls *c = viewer_controls_new();
  ASSERT(c != NULL);

  composite_params cp = viewer_controls_get_composite(c);

  // composite_params_default should set COMPOSITE_MAX
  ASSERT_EQ((int)COMPOSITE_MAX, (int)cp.mode);
  ASSERT(cp.num_layers_front  >= 0);
  ASSERT(cp.num_layers_behind >= 0);

  viewer_controls_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// render with NULL ctx (must not crash — early return)
// ---------------------------------------------------------------------------

TEST test_render_null_ctx(void) {
  viewer_controls *c = viewer_controls_new();
  ASSERT(c != NULL);
  // ctx = NULL and viewer = NULL: should return false without crashing
  bool changed = viewer_controls_render(c, NULL, NULL);
  ASSERT(!changed);
  viewer_controls_free(c);
  PASS();
}

TEST test_render_null_controls(void) {
  // NULL controls: should return false without crashing
  bool changed = viewer_controls_render(NULL, NULL, NULL);
  ASSERT(!changed);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(suite_viewer_controls) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_default_cmap);
  RUN_TEST(test_default_composite);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_null_controls);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_viewer_controls);
  GREATEST_MAIN_END();
}
