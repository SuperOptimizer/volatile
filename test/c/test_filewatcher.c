#include "greatest.h"
#include "gui/filewatcher.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

typedef struct {
  char  last_path[256];
  int   count;
} callback_ctx;

static void on_changed(const char *path, void *ctx) {
  callback_ctx *c = ctx;
  strncpy(c->last_path, path, sizeof(c->last_path) - 1);
  c->count++;
}

// sleep milliseconds
static void sleep_ms(int ms) {
  struct timespec ts = { .tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000L };
  nanosleep(&ts, NULL);
}

// poll repeatedly for up to timeout_ms, returning when count >= expected
static int poll_until(file_watcher *w, callback_ctx *ctx, int expected, int timeout_ms) {
  int elapsed = 0;
  while (elapsed < timeout_ms) {
    file_watcher_poll(w);
    if (ctx->count >= expected) return ctx->count;
    sleep_ms(10);
    elapsed += 10;
  }
  return ctx->count;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  file_watcher_free(w);
  PASS();
}

TEST test_free_null(void) {
  file_watcher_free(NULL);
  PASS();
}

TEST test_add_nonexistent_path(void) {
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  // inotify_add_watch will fail; add should return false
  bool ok = file_watcher_add(w, "/tmp/filewatcher_no_such_path_xyz", on_changed, NULL);
  ASSERT(!ok);
  file_watcher_free(w);
  PASS();
}

TEST test_poll_no_events(void) {
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  int n = file_watcher_poll(w);
  ASSERT_EQ(0, n);
  file_watcher_free(w);
  PASS();
}

TEST test_watch_and_modify(void) {
  // create a temp file
  const char *path = "/tmp/test_fw_modify.txt";
  FILE *f = fopen(path, "w");
  ASSERT(f != NULL);
  fprintf(f, "init\n");
  fclose(f);

  callback_ctx ctx = {0};
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  ASSERT(file_watcher_add(w, path, on_changed, &ctx));

  // modify the file
  f = fopen(path, "w");
  ASSERT(f != NULL);
  fprintf(f, "modified\n");
  fclose(f);

  int n = poll_until(w, &ctx, 1, 500);
  ASSERT(n >= 1);
  ASSERT(strstr(ctx.last_path, "test_fw_modify.txt") != NULL);

  file_watcher_free(w);
  remove(path);
  PASS();
}

TEST test_watch_remove(void) {
  const char *path = "/tmp/test_fw_remove.txt";
  FILE *f = fopen(path, "w");
  ASSERT(f != NULL);
  fclose(f);

  callback_ctx ctx = {0};
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  ASSERT(file_watcher_add(w, path, on_changed, &ctx));
  ASSERT(file_watcher_remove(w, path));

  // modify after removal — callback should NOT fire
  f = fopen(path, "w");
  ASSERT(f != NULL);
  fprintf(f, "after remove\n");
  fclose(f);

  sleep_ms(50);
  file_watcher_poll(w);
  ASSERT_EQ(0, ctx.count);

  file_watcher_free(w);
  remove(path);
  PASS();
}

TEST test_remove_nonexistent(void) {
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  // removing a path that was never added should return false
  bool ok = file_watcher_remove(w, "/tmp/never_added.txt");
  ASSERT(!ok);
  file_watcher_free(w);
  PASS();
}

TEST test_debounce_suppresses_rapid_events(void) {
  const char *path = "/tmp/test_fw_debounce.txt";
  FILE *f = fopen(path, "w");
  ASSERT(f != NULL);
  fclose(f);

  callback_ctx ctx = {0};
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  // 200ms debounce window
  ASSERT(file_watcher_add_debounced(w, path, on_changed, &ctx, 200));

  // fire two rapid modifications
  for (int i = 0; i < 3; i++) {
    f = fopen(path, "w");
    if (f) { fprintf(f, "v%d\n", i); fclose(f); }
    sleep_ms(10);
    file_watcher_poll(w);
  }

  // should only have fired at most 1 time (debounced)
  ASSERT(ctx.count <= 1);

  file_watcher_free(w);
  remove(path);
  PASS();
}

TEST test_watch_directory(void) {
  const char *dir = "/tmp/test_fw_dir";
  mkdir(dir, 0755);

  callback_ctx ctx = {0};
  file_watcher *w = file_watcher_new();
  ASSERT(w != NULL);
  ASSERT(file_watcher_add(w, dir, on_changed, &ctx));

  // create a new file in the directory — should trigger IN_CREATE
  FILE *f = fopen("/tmp/test_fw_dir/newfile.txt", "w");
  ASSERT(f != NULL);
  fprintf(f, "hello\n");
  fclose(f);

  int n = poll_until(w, &ctx, 1, 500);
  ASSERT(n >= 1);

  file_watcher_free(w);
  remove("/tmp/test_fw_dir/newfile.txt");
  rmdir(dir);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(filewatcher_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_add_nonexistent_path);
  RUN_TEST(test_poll_no_events);
  RUN_TEST(test_watch_and_modify);
  RUN_TEST(test_watch_remove);
  RUN_TEST(test_remove_nonexistent);
  RUN_TEST(test_debounce_suppresses_rapid_events);
  RUN_TEST(test_watch_directory);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(filewatcher_suite);
  GREATEST_MAIN_END();
}
