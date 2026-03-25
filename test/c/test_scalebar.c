// test_scalebar.c — unit tests for scalebar widget (no-display, NULL-ctx)

#include "greatest.h"
#include "gui/scalebar.h"

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST lifecycle(void) {
  scalebar *s = scalebar_new(1.0f);
  ASSERT(s != NULL);
  scalebar_free(s);
  PASS();
}

TEST free_null(void) {
  scalebar_free(NULL);   // must not crash
  PASS();
}

TEST new_zero_voxel(void) {
  // voxel_size_um <= 0 should clamp to 1.0 internally
  scalebar *s = scalebar_new(0.0f);
  ASSERT(s != NULL);
  scalebar_free(s);
  PASS();
}

TEST new_negative_voxel(void) {
  scalebar *s = scalebar_new(-5.0f);
  ASSERT(s != NULL);
  scalebar_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// set_voxel_size
// ---------------------------------------------------------------------------

TEST set_voxel_size_basic(void) {
  scalebar *s = scalebar_new(1.0f);
  ASSERT(s != NULL);
  scalebar_set_voxel_size(s, 7.91f);
  scalebar_free(s);
  PASS();
}

TEST set_voxel_size_null(void) {
  scalebar_set_voxel_size(NULL, 5.0f);   // must not crash
  PASS();
}

TEST set_voxel_size_zero(void) {
  scalebar *s = scalebar_new(1.0f);
  ASSERT(s != NULL);
  scalebar_set_voxel_size(s, 0.0f);   // ignored — must not crash
  scalebar_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Render with NULL context (early-return guard)
// ---------------------------------------------------------------------------

TEST render_null_ctx(void) {
  scalebar *s = scalebar_new(7.91f);
  ASSERT(s != NULL);
  scalebar_render(s, NULL, 1.0f, 0);
  scalebar_free(s);
  PASS();
}

TEST render_null_scalebar(void) {
  scalebar_render(NULL, NULL, 1.0f, 120);   // must not crash
  PASS();
}

TEST render_zero_zoom(void) {
  scalebar *s = scalebar_new(7.91f);
  ASSERT(s != NULL);
  scalebar_render(s, NULL, 0.0f, 120);   // zoom clamped to 1.0, NULL ctx guard
  scalebar_free(s);
  PASS();
}

TEST render_various_voxel_sizes(void) {
  float sizes[] = {1.0f, 7.91f, 0.5f, 100.0f, 0.01f};
  for (int i = 0; i < 5; i++) {
    scalebar *s = scalebar_new(sizes[i]);
    ASSERT(s != NULL);
    scalebar_render(s, NULL, 2.0f, 120);
    scalebar_free(s);
  }
  PASS();
}

TEST render_default_bar_width(void) {
  scalebar *s = scalebar_new(7.91f);
  ASSERT(s != NULL);
  scalebar_render(s, NULL, 1.5f, 0);   // bar_width_px=0 → default 120
  scalebar_free(s);
  PASS();
}

TEST render_mm_label(void) {
  // voxel 1000 µm at zoom 1 → um_per_px = 1000; nice scale >= 1000 → mm label
  scalebar *s = scalebar_new(1000.0f);
  ASSERT(s != NULL);
  scalebar_render(s, NULL, 1.0f, 200);
  scalebar_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(scalebar_suite) {
  RUN_TEST(lifecycle);
  RUN_TEST(free_null);
  RUN_TEST(new_zero_voxel);
  RUN_TEST(new_negative_voxel);
  RUN_TEST(set_voxel_size_basic);
  RUN_TEST(set_voxel_size_null);
  RUN_TEST(set_voxel_size_zero);
  RUN_TEST(render_null_ctx);
  RUN_TEST(render_null_scalebar);
  RUN_TEST(render_zero_zoom);
  RUN_TEST(render_various_voxel_sizes);
  RUN_TEST(render_default_bar_width);
  RUN_TEST(render_mm_label);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(scalebar_suite);
  GREATEST_MAIN_END();
}
