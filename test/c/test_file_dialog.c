#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "gui/file_dialog.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static char g_tmpdir[256];

static void make_tmpdir(void) {
  snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/fdtest_%d", (int)getpid());
  mkdir(g_tmpdir, 0755);
}

static void touch(const char *dir, const char *name) {
  char p[512];
  snprintf(p, sizeof(p), "%s/%s", dir, name);
  FILE *f = fopen(p, "w");
  if (f) fclose(f);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_create_free(void) {
  file_dialog *d = file_dialog_new("Test", "*.zarr");
  ASSERT(d != NULL);
  file_dialog_free(d);
  PASS();
}

TEST test_free_null(void) {
  file_dialog_free(NULL);
  PASS();
}

TEST test_create_null_args(void) {
  file_dialog *d = file_dialog_new(NULL, NULL);
  ASSERT(d != NULL);
  file_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// Visibility
// ---------------------------------------------------------------------------

TEST test_not_visible_after_create(void) {
  file_dialog *d = file_dialog_new("T", "*.tiff");
  ASSERT(d != NULL);
  ASSERT_FALSE(file_dialog_is_visible(d));
  file_dialog_free(d);
  PASS();
}

TEST test_visible_after_show(void) {
  file_dialog *d = file_dialog_new("T", "*.tiff");
  ASSERT(d != NULL);
  file_dialog_show(d, "/tmp");
  ASSERT(file_dialog_is_visible(d));
  file_dialog_free(d);
  PASS();
}

TEST test_show_null_dir_uses_cwd(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  file_dialog_show(d, NULL);  // must not crash
  ASSERT(file_dialog_is_visible(d));
  file_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// Path accessors
// ---------------------------------------------------------------------------

TEST test_get_path_null_before_commit(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  // Before any commit, result is empty string (not NULL, not garbage)
  const char *p = file_dialog_get_path(d);
  ASSERT(p != NULL);
  ASSERT_EQ(0, (int)strlen(p));
  file_dialog_free(d);
  PASS();
}

TEST test_get_path_null_dialog(void) {
  ASSERT(file_dialog_get_path(NULL) == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Bookmarks
// ---------------------------------------------------------------------------

TEST test_add_bookmark(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  // create + show so the dialog is in a known state
  file_dialog_show(d, "/tmp");
  // add extra bookmark — must not crash
  file_dialog_add_bookmark(d, "Tmp", "/tmp");
  file_dialog_add_bookmark(d, "Data", "/data");
  file_dialog_free(d);
  PASS();
}

TEST test_bookmark_overflow_ignored(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  // Add 20 bookmarks — overflow silently ignored
  for (int i = 0; i < 20; i++) {
    char label[16], path[32];
    snprintf(label, sizeof(label), "B%d", i);
    snprintf(path, sizeof(path), "/tmp/%d", i);
    file_dialog_add_bookmark(d, label, path);  // must not crash
  }
  file_dialog_free(d);
  PASS();
}

TEST test_add_null_bookmark_ignored(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  file_dialog_add_bookmark(d, NULL, "/tmp");  // must not crash
  file_dialog_add_bookmark(d, "X", NULL);     // must not crash
  file_dialog_add_bookmark(NULL, "X", "/tmp");
  file_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// is_visible / render-less state
// ---------------------------------------------------------------------------

TEST test_is_visible_null(void) {
  ASSERT_FALSE(file_dialog_is_visible(NULL));
  PASS();
}

TEST test_render_null_ctx_returns_false(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  file_dialog_show(d, "/tmp");
  // render with NULL ctx must return false without crashing
  bool result = file_dialog_render(d, NULL);
  ASSERT_FALSE(result);
  file_dialog_free(d);
  PASS();
}

TEST test_render_when_not_visible_returns_false(void) {
  file_dialog *d = file_dialog_new("T", NULL);
  ASSERT(d != NULL);
  // never called show — not visible
  bool result = file_dialog_render(d, NULL);
  ASSERT_FALSE(result);
  file_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// show with a real temp directory (verifies no crash from scan_dir)
// ---------------------------------------------------------------------------

TEST test_show_real_directory(void) {
  make_tmpdir();
  touch(g_tmpdir, "volume.zarr");
  touch(g_tmpdir, "image.tiff");
  touch(g_tmpdir, "notes.txt");

  file_dialog *d = file_dialog_new("Open", "*.zarr;*.tiff");
  ASSERT(d != NULL);
  file_dialog_show(d, g_tmpdir);
  ASSERT(file_dialog_is_visible(d));
  // dialog scanned dir — no crash, still visible
  file_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(file_dialog_suite) {
  RUN_TEST(test_create_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_create_null_args);
  RUN_TEST(test_not_visible_after_create);
  RUN_TEST(test_visible_after_show);
  RUN_TEST(test_show_null_dir_uses_cwd);
  RUN_TEST(test_get_path_null_before_commit);
  RUN_TEST(test_get_path_null_dialog);
  RUN_TEST(test_add_bookmark);
  RUN_TEST(test_bookmark_overflow_ignored);
  RUN_TEST(test_add_null_bookmark_ignored);
  RUN_TEST(test_is_visible_null);
  RUN_TEST(test_render_null_ctx_returns_false);
  RUN_TEST(test_render_when_not_visible_returns_false);
  RUN_TEST(test_show_real_directory);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(file_dialog_suite);
  GREATEST_MAIN_END();
}
