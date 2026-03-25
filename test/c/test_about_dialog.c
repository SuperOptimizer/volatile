#include "greatest.h"
#include "gui/about_dialog.h"

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  about_dialog *d = about_dialog_new();
  ASSERT(d != NULL);
  about_dialog_free(d);
  PASS();
}

TEST test_initially_hidden(void) {
  about_dialog *d = about_dialog_new();
  ASSERT(!about_dialog_is_visible(d));
  about_dialog_free(d);
  PASS();
}

TEST test_show_makes_visible(void) {
  about_dialog *d = about_dialog_new();
  about_dialog_show(d);
  ASSERT(about_dialog_is_visible(d));
  about_dialog_free(d);
  PASS();
}

TEST test_show_idempotent(void) {
  about_dialog *d = about_dialog_new();
  about_dialog_show(d);
  about_dialog_show(d);
  ASSERT(about_dialog_is_visible(d));
  about_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// Null-safety
// ---------------------------------------------------------------------------

TEST test_null_context_render_returns_false(void) {
  about_dialog *d = about_dialog_new();
  about_dialog_show(d);
  // ctx is NULL — must not crash, must return false
  bool open = about_dialog_render(d, NULL);
  ASSERT(!open);
  about_dialog_free(d);
  PASS();
}

TEST test_null_dialog_is_visible_returns_false(void) {
  ASSERT(!about_dialog_is_visible(NULL));
  PASS();
}

TEST test_render_hidden_returns_false(void) {
  about_dialog *d = about_dialog_new();
  // Not shown — render should return false without crash
  bool open = about_dialog_render(d, NULL);
  ASSERT(!open);
  about_dialog_free(d);
  PASS();
}

TEST test_free_null_no_crash(void) {
  about_dialog_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(about_dialog_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_initially_hidden);
  RUN_TEST(test_show_makes_visible);
  RUN_TEST(test_show_idempotent);
  RUN_TEST(test_null_context_render_returns_false);
  RUN_TEST(test_null_dialog_is_visible_returns_false);
  RUN_TEST(test_render_hidden_returns_false);
  RUN_TEST(test_free_null_no_crash);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(about_dialog_suite);
  GREATEST_MAIN_END();
}
