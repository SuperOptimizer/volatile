#include "greatest.h"
#include "gui/console.h"
#include "core/log.h"

#include <stdbool.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void add_n(log_console *c, int n) {
  for (int i = 0; i < n; i++) {
    log_console_add(c, 1 /* INFO */, "test.c", i + 1, "message");
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  log_console *c = log_console_new(100);
  ASSERT(c != NULL);
  ASSERT_EQ(0, log_console_count(c));
  log_console_free(c);
  PASS();
}

TEST test_add_count(void) {
  log_console *c = log_console_new(100);
  ASSERT(c != NULL);
  add_n(c, 5);
  ASSERT_EQ(5, log_console_count(c));
  log_console_free(c);
  PASS();
}

TEST test_clear(void) {
  log_console *c = log_console_new(100);
  add_n(c, 10);
  ASSERT_EQ(10, log_console_count(c));
  log_console_clear(c);
  ASSERT_EQ(0, log_console_count(c));
  log_console_free(c);
  PASS();
}

TEST test_ring_buffer_wraps(void) {
  log_console *c = log_console_new(10);
  // Add more entries than capacity
  add_n(c, 25);
  // Count should stay at cap
  ASSERT_EQ(10, log_console_count(c));
  log_console_free(c);
  PASS();
}

TEST test_default_max_lines(void) {
  // max_lines <= 0 should default to 1000
  log_console *c = log_console_new(0);
  ASSERT(c != NULL);
  add_n(c, 50);
  ASSERT_EQ(50, log_console_count(c));
  log_console_free(c);
  PASS();
}

TEST test_filter_by_level(void) {
  log_console *c = log_console_new(100);
  log_console_add(c, 0 /* DEBUG */, "a.c", 1, "debug msg");
  log_console_add(c, 1 /* INFO  */, "b.c", 2, "info msg");
  log_console_add(c, 2 /* WARN  */, "c.c", 3, "warn msg");

  // All three entries are stored regardless of min_level filter
  ASSERT_EQ(3, log_console_count(c));

  // Setting min_level doesn't remove entries; render skips them.
  // We test count is unchanged — the filter only affects rendering.
  log_console_set_min_level(c, 2 /* WARN */);
  ASSERT_EQ(3, log_console_count(c));

  log_console_free(c);
  PASS();
}

TEST test_substring_filter_set(void) {
  log_console *c = log_console_new(100);
  // Just verify set/clear doesn't crash
  log_console_set_filter(c, "error");
  log_console_set_filter(c, NULL);
  log_console_set_filter(c, "");
  log_console_free(c);
  PASS();
}

TEST test_auto_scroll_default(void) {
  log_console *c = log_console_new(100);
  ASSERT(log_console_auto_scroll(c) == true);
  log_console_set_auto_scroll(c, false);
  ASSERT(log_console_auto_scroll(c) == false);
  log_console_set_auto_scroll(c, true);
  ASSERT(log_console_auto_scroll(c) == true);
  log_console_free(c);
  PASS();
}

// Wire up via log_set_callback and verify entries arrive
static log_console *g_cb_console = NULL;
static void test_callback(void *ctx, log_level_t level, const char *file, int line, const char *msg) {
  log_console_add((log_console *)ctx, (int)level, file, line, msg);
}

TEST test_log_callback_integration(void) {
  log_console *c = log_console_new(50);
  g_cb_console = c;

  log_set_callback(test_callback, c);
  log_set_level(LOG_DEBUG);

  LOG_INFO("hello from test");
  LOG_WARN("warning from test");

  log_set_callback(NULL, NULL);
  log_set_level(LOG_INFO);

  ASSERT(log_console_count(c) >= 2);
  log_console_free(c);
  g_cb_console = NULL;
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(console_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_add_count);
  RUN_TEST(test_clear);
  RUN_TEST(test_ring_buffer_wraps);
  RUN_TEST(test_default_max_lines);
  RUN_TEST(test_filter_by_level);
  RUN_TEST(test_substring_filter_set);
  RUN_TEST(test_auto_scroll_default);
  RUN_TEST(test_log_callback_integration);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(console_suite);
  GREATEST_MAIN_END();
}
