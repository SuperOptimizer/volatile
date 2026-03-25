#include "greatest.h"
#include "gui/settings_dialog.h"
#include "gui/settings.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static char *make_tmp_dir(void) {
  char tmpl[] = "/tmp/test_sdlg_XXXXXX";
  char *d = mkdtemp(tmpl);
  return d ? strdup(d) : NULL;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Lifecycle: new / free
TEST test_lifecycle(void) {
  char *tmp = make_tmp_dir();
  ASSERT(tmp != NULL);
  setenv("XDG_CONFIG_HOME", tmp, 1);

  settings *prefs = settings_open(NULL);
  ASSERT(prefs != NULL);

  settings_dialog *d = settings_dialog_new(prefs);
  ASSERT(d != NULL);

  settings_dialog_free(d);
  settings_close(prefs);
  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// Initially not visible
TEST test_initially_hidden(void) {
  char *tmp = make_tmp_dir();
  setenv("XDG_CONFIG_HOME", tmp, 1);

  settings *prefs = settings_open(NULL);
  settings_dialog *d = settings_dialog_new(prefs);
  ASSERT(d != NULL);

  ASSERT_EQ(false, settings_dialog_is_visible(d));

  settings_dialog_free(d);
  settings_close(prefs);
  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// Show makes it visible
TEST test_show_sets_visible(void) {
  char *tmp = make_tmp_dir();
  setenv("XDG_CONFIG_HOME", tmp, 1);

  settings *prefs = settings_open(NULL);
  settings_dialog *d = settings_dialog_new(prefs);
  ASSERT(d != NULL);

  settings_dialog_show(d);
  ASSERT_EQ(true, settings_dialog_is_visible(d));

  settings_dialog_free(d);
  settings_close(prefs);
  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// render returns false when not visible (no ctx needed)
TEST test_render_hidden_returns_false(void) {
  char *tmp = make_tmp_dir();
  setenv("XDG_CONFIG_HOME", tmp, 1);

  settings *prefs = settings_open(NULL);
  settings_dialog *d = settings_dialog_new(prefs);
  ASSERT(d != NULL);

  // dialog is hidden — render must return false immediately without dereferencing ctx
  bool changed = settings_dialog_render(d, NULL);
  ASSERT_EQ(false, changed);

  settings_dialog_free(d);
  settings_close(prefs);
  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// Settings values loaded from prefs on construction
TEST test_loads_saved_prefs(void) {
  char *tmp = make_tmp_dir();
  setenv("XDG_CONFIG_HOME", tmp, 1);

  // Persist some values first
  settings *prefs = settings_open(NULL);
  settings_set_int(prefs, "pre.cmap", 3);
  settings_set_int(prefs, "perf.ram_cache_gb", 8);
  settings_save(prefs);
  settings_close(prefs);

  // Re-open and create dialog — it should read saved values
  settings *prefs2 = settings_open(NULL);
  ASSERT(prefs2 != NULL);
  ASSERT_EQ(3, settings_get_int(prefs2, "pre.cmap", 0));
  ASSERT_EQ(8, settings_get_int(prefs2, "perf.ram_cache_gb", 4));

  settings_dialog *d = settings_dialog_new(prefs2);
  ASSERT(d != NULL);

  settings_dialog_free(d);
  settings_close(prefs2);
  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// Multiple new/free cycles — no leaks or crashes
TEST test_multiple_lifecycle(void) {
  char *tmp = make_tmp_dir();
  setenv("XDG_CONFIG_HOME", tmp, 1);

  for (int i = 0; i < 5; i++) {
    settings *prefs = settings_open(NULL);
    ASSERT(prefs != NULL);
    settings_dialog *d = settings_dialog_new(prefs);
    ASSERT(d != NULL);
    settings_dialog_show(d);
    ASSERT_EQ(true, settings_dialog_is_visible(d));
    settings_dialog_free(d);
    settings_close(prefs);
  }

  free(tmp);
  unsetenv("XDG_CONFIG_HOME");
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(settings_dialog_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_initially_hidden);
  RUN_TEST(test_show_sets_visible);
  RUN_TEST(test_render_hidden_returns_false);
  RUN_TEST(test_loads_saved_prefs);
  RUN_TEST(test_multiple_lifecycle);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(settings_dialog_suite);
  GREATEST_MAIN_END();
}
