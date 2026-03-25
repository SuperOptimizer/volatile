#include "greatest.h"
#include "gui/settings.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers — create a temp dir tree and set XDG_CONFIG_HOME to isolate tests
// from the real user config.
// ---------------------------------------------------------------------------

// Returns a heap-allocated path to a freshly created tmp directory.
// Caller must free.
static char *make_tmp_dir(void) {
  char tmpl[] = "/tmp/test_settings_XXXXXX";
  char *d = mkdtemp(tmpl);
  if (!d) return NULL;
  return strdup(d);
}

static void set_xdg(const char *base) {
  setenv("XDG_CONFIG_HOME", base, 1);
}

static void unset_xdg(void) {
  unsetenv("XDG_CONFIG_HOME");
}

// Write a string to a file, creating parent dirs if needed.
static void write_file(const char *dir, const char *name, const char *content) {
  mkdir(dir, 0755);
  char path[4096];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "w");
  if (f) { fputs(content, f); fclose(f); }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_open_close_no_files(void) {
  char *tmp = make_tmp_dir();
  ASSERT(tmp != NULL);
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  ASSERT(s != NULL);
  settings_close(s);

  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_get_defaults(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  ASSERT(s != NULL);

  ASSERT_STR_EQ("hello", settings_get_str(s, "missing", "hello"));
  ASSERT_EQ(42,           settings_get_int(s, "missing", 42));
  ASSERT(settings_get_float(s, "missing", 1.5f) == 1.5f);
  ASSERT_EQ(true,         settings_get_bool(s, "missing", true));

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_set_get_str(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  ASSERT(s != NULL);

  settings_set_str(s, "color", "red");
  ASSERT_STR_EQ("red", settings_get_str(s, "color", "blue"));

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_set_get_int(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  settings_set_int(s, "threads", 8);
  ASSERT_EQ(8, settings_get_int(s, "threads", 1));

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_save_reload(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  // Write and save.
  settings *s = settings_open(NULL);
  ASSERT(s != NULL);
  settings_set_str(s, "theme", "dark");
  settings_set_int(s, "zoom", 3);
  bool ok = settings_save(s);
  ASSERT(ok);
  settings_close(s);

  // Reload and verify.
  settings *s2 = settings_open(NULL);
  ASSERT(s2 != NULL);
  ASSERT_STR_EQ("dark", settings_get_str(s2, "theme", "light"));
  ASSERT_EQ(3, settings_get_int(s2, "zoom", 1));
  settings_close(s2);

  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_layering_project_overrides_global(void) {
  char *xdg = make_tmp_dir();
  char *proj = make_tmp_dir();
  set_xdg(xdg);

  // Seed global config.
  char global_dir[4096];
  snprintf(global_dir, sizeof(global_dir), "%s/volatile", xdg);
  write_file(global_dir, "config.json",
    "{\"key\":\"global_val\",\"global_only\":\"yes\"}");

  // Seed project config.
  char proj_dot[4096];
  snprintf(proj_dot, sizeof(proj_dot), "%s/.volatile", proj);
  write_file(proj_dot, "config.json",
    "{\"key\":\"project_val\"}");

  settings *s = settings_open(proj);
  ASSERT(s != NULL);

  // project overrides global for "key"
  ASSERT_STR_EQ("project_val", settings_get_str(s, "key", ""));
  // global-only key still visible
  ASSERT_STR_EQ("yes", settings_get_str(s, "global_only", ""));

  settings_close(s);
  free(xdg);
  free(proj);
  unset_xdg();
  PASS();
}

TEST test_set_writes_to_project_layer(void) {
  char *xdg  = make_tmp_dir();
  char *proj = make_tmp_dir();
  set_xdg(xdg);

  settings *s = settings_open(proj);
  ASSERT(s != NULL);

  settings_set_str(s, "scope", "project");
  // Value is in project layer; global must NOT see it after close/reopen.
  bool ok = settings_save(s);
  ASSERT(ok);
  settings_close(s);

  // Open global-only — should NOT see "scope".
  settings *g = settings_open(NULL);
  ASSERT_STR_EQ("default", settings_get_str(g, "scope", "default"));
  settings_close(g);

  // Open with project — should see "scope".
  settings *p = settings_open(proj);
  ASSERT_STR_EQ("project", settings_get_str(p, "scope", "default"));
  settings_close(p);

  free(xdg);
  free(proj);
  unset_xdg();
  PASS();
}

TEST test_get_bool_variants(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  settings_set_str(s, "flag_true",  "true");
  settings_set_str(s, "flag_false", "false");
  settings_set_str(s, "flag_one",   "1");
  settings_set_str(s, "flag_zero",  "0");

  ASSERT_EQ(true,  settings_get_bool(s, "flag_true",  false));
  ASSERT_EQ(false, settings_get_bool(s, "flag_false", true));
  ASSERT_EQ(true,  settings_get_bool(s, "flag_one",   false));
  ASSERT_EQ(false, settings_get_bool(s, "flag_zero",  true));

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_get_float(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  settings_set_str(s, "scale", "2.5");
  float f = settings_get_float(s, "scale", 1.0f);
  ASSERT(f > 2.49f && f < 2.51f);

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_overwrite_key(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  settings_set_str(s, "x", "first");
  settings_set_str(s, "x", "second");
  ASSERT_STR_EQ("second", settings_get_str(s, "x", ""));

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_dump_no_crash(void) {
  char *tmp = make_tmp_dir();
  set_xdg(tmp);

  settings *s = settings_open(NULL);
  settings_set_str(s, "a", "1");
  settings_set_str(s, "b", "2");

  FILE *f = tmpfile();
  ASSERT(f != NULL);
  settings_dump(s, f);
  fclose(f);

  settings_close(s);
  free(tmp);
  unset_xdg();
  PASS();
}

TEST test_load_json_number_and_bool(void) {
  char *xdg = make_tmp_dir();
  set_xdg(xdg);

  char global_dir[4096];
  snprintf(global_dir, sizeof(global_dir), "%s/volatile", xdg);
  write_file(global_dir, "config.json",
    "{\"count\":7,\"enabled\":true,\"ratio\":0.5}");

  settings *s = settings_open(NULL);
  ASSERT(s != NULL);
  ASSERT_EQ(7,    settings_get_int(s,  "count",   0));
  ASSERT_EQ(true, settings_get_bool(s, "enabled", false));
  float r = settings_get_float(s, "ratio", 0.0f);
  ASSERT(r > 0.49f && r < 0.51f);

  settings_close(s);
  free(xdg);
  unset_xdg();
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(settings_suite) {
  RUN_TEST(test_open_close_no_files);
  RUN_TEST(test_get_defaults);
  RUN_TEST(test_set_get_str);
  RUN_TEST(test_set_get_int);
  RUN_TEST(test_save_reload);
  RUN_TEST(test_layering_project_overrides_global);
  RUN_TEST(test_set_writes_to_project_layer);
  RUN_TEST(test_get_bool_variants);
  RUN_TEST(test_get_float);
  RUN_TEST(test_overwrite_key);
  RUN_TEST(test_dump_no_crash);
  RUN_TEST(test_load_json_number_and_bool);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(settings_suite);
  GREATEST_MAIN_END();
}
