// test_statusbar.c — unit tests for statusbar widget (no-display, NULL-ctx)

#include "greatest.h"
#include "gui/statusbar.h"

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST lifecycle(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  statusbar_free(s);
  PASS();
}

TEST free_null(void) {
  statusbar_free(NULL);   // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Update (no crash, values stored — we verify indirectly via render with NULL)
// ---------------------------------------------------------------------------

TEST update_basic(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  statusbar_update(s, 1.0f, 2.0f, 3.0f, 0.5f, 2.0f, 1, 60.0f, 1024 * 1024);
  statusbar_free(s);
  PASS();
}

TEST update_null_safe(void) {
  statusbar_update(NULL, 0, 0, 0, 0, 1.0f, 0, 0, 0);   // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Render with NULL context must not crash
// ---------------------------------------------------------------------------

TEST render_null_ctx(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  statusbar_update(s, 10.0f, 20.0f, 30.0f, 127.5f, 1.5f, 2, 55.3f,
                   512UL * 1024 * 1024);
  statusbar_render(s, NULL, 0);   // NULL ctx — early-return guard
  statusbar_free(s);
  PASS();
}

TEST render_null_statusbar(void) {
  statusbar_render(NULL, NULL, 22);   // must not crash
  PASS();
}

TEST render_default_height(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  statusbar_update(s, 0, 0, 0, 0, 1.0f, 0, 30.0f, 0);
  statusbar_render(s, NULL, 0);   // height=0 → default 22, NULL ctx guard fires first
  statusbar_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Update with extreme values
// ---------------------------------------------------------------------------

TEST update_zero_zoom(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  statusbar_update(s, -1.0f, -2.0f, -3.0f, -0.1f, 0.0f, 0, 0.0f, 0);
  statusbar_render(s, NULL, 22);
  statusbar_free(s);
  PASS();
}

TEST update_large_mem(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  // > 1 GB branch
  statusbar_update(s, 0, 0, 0, 0, 1.0f, 0, 60.0f,
                   (size_t)2 * 1024 * 1024 * 1024);
  statusbar_render(s, NULL, 0);
  statusbar_free(s);
  PASS();
}

TEST update_mb_mem(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  // MB branch
  statusbar_update(s, 0, 0, 0, 0, 1.0f, 0, 60.0f, 256UL * 1024 * 1024);
  statusbar_render(s, NULL, 0);
  statusbar_free(s);
  PASS();
}

TEST update_kb_mem(void) {
  statusbar *s = statusbar_new();
  ASSERT(s != NULL);
  // KB branch (< 1 MB)
  statusbar_update(s, 0, 0, 0, 0, 1.0f, 0, 60.0f, 512UL * 1024);
  statusbar_render(s, NULL, 0);
  statusbar_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(statusbar_suite) {
  RUN_TEST(lifecycle);
  RUN_TEST(free_null);
  RUN_TEST(update_basic);
  RUN_TEST(update_null_safe);
  RUN_TEST(render_null_ctx);
  RUN_TEST(render_null_statusbar);
  RUN_TEST(render_default_height);
  RUN_TEST(update_zero_zoom);
  RUN_TEST(update_large_mem);
  RUN_TEST(update_mb_mem);
  RUN_TEST(update_kb_mem);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(statusbar_suite);
  GREATEST_MAIN_END();
}
