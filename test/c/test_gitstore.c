#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "server/gitstore.h"
#include "core/geom.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Create a unique temp directory for the test repo.
static char *make_tmpdir(void) {
  char tmpl[] = "/tmp/test_gitstore_XXXXXX";
  char *d = mkdtemp(tmpl);
  if (!d) return NULL;
  char *out = malloc(64);
  snprintf(out, 64, "%s", d);
  return out;
}

// Recursively remove a temp directory via shell.
static void rm_tmpdir(const char *path) {
  if (!path) return;
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf '%s'", path);
  system(cmd);
}

// Build a simple quad_surface with all points set to (x, y, z).
static quad_surface *make_surface(int rows, int cols, float x, float y, float z) {
  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      quad_surface_set(s, r, c, (vec3f){x, y, z});
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// Per-test state
// ---------------------------------------------------------------------------

static char      *g_tmpdir = NULL;
static git_store *g_gs     = NULL;

static void setup(void *unused) {
  (void)unused;
  g_tmpdir = make_tmpdir();
  if (g_tmpdir)
    g_gs = git_store_open(g_tmpdir);
}

static void teardown(void *unused) {
  (void)unused;
  git_store_free(g_gs);
  rm_tmpdir(g_tmpdir);
  free(g_tmpdir);
  g_gs = NULL;
  g_tmpdir = NULL;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_open_creates_repo(void) {
  ASSERT(g_tmpdir != NULL);
  ASSERT(g_gs    != NULL);
  PASS();
}

TEST test_open_idempotent(void) {
  // Opening the same path twice should succeed.
  git_store *g2 = git_store_open(g_tmpdir);
  ASSERT(g2 != NULL);
  git_store_free(g2);
  PASS();
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

TEST test_write_read_file(void) {
  const char *data = "hello gitstore";
  bool ok = git_store_write_file(g_gs, "test.txt", data, strlen(data));
  ASSERT(ok);

  size_t len = 0;
  uint8_t *buf = git_store_read_file(g_gs, "test.txt", &len);
  ASSERT(buf != NULL);
  ASSERT_EQ(strlen(data), len);
  ASSERT(memcmp(buf, data, len) == 0);
  free(buf);
  PASS();
}

TEST test_read_missing_file_returns_null(void) {
  size_t len = 0;
  uint8_t *buf = git_store_read_file(g_gs, "no_such_file.bin", &len);
  ASSERT(buf == NULL);
  ASSERT_EQ(0u, len);
  PASS();
}

// ---------------------------------------------------------------------------
// Surface serialization
// ---------------------------------------------------------------------------

TEST test_write_surface_stages_file(void) {
  quad_surface *s = make_surface(2, 3, 1.0f, 2.0f, 3.0f);
  ASSERT(s != NULL);
  bool ok = git_store_write_surface(g_gs, "seg01", s);
  ASSERT(ok);

  // File should be readable now (staged but not committed).
  size_t len = 0;
  uint8_t *buf = git_store_read_file(g_gs, "segments/seg01/surface.json", &len);
  ASSERT(buf != NULL);
  ASSERT(len > 0);
  // Should contain JSON with rows/cols
  ASSERT(memchr(buf, '{', len) != NULL);
  free(buf);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Commit
// ---------------------------------------------------------------------------

TEST test_commit_after_write(void) {
  const char *data = "commit test data";
  git_store_write_file(g_gs, "data.bin", data, strlen(data));
  bool ok = git_store_commit(g_gs, "Test Author <test@localhost>", "add data.bin");
  ASSERT(ok);

  // Repo should be clean after commit.
  ASSERT(git_store_is_clean(g_gs));
  PASS();
}

TEST test_log_returns_entry(void) {
  const char *data = "log test";
  git_store_write_file(g_gs, "log.txt", data, strlen(data));
  git_store_commit(g_gs, "Log Author <log@localhost>", "add log.txt");

  git_log_entry entries[8];
  int n = git_store_log(g_gs, entries, 8);
  ASSERT(n >= 1);
  // Most recent entry should have our message.
  ASSERT(strstr(entries[0].message, "log.txt") != NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Branch workflow
// ---------------------------------------------------------------------------

TEST test_create_branch_and_checkout(void) {
  // Need at least one commit first so HEAD is valid.
  const char *data = "branch base";
  git_store_write_file(g_gs, "base.txt", data, strlen(data));
  git_store_commit(g_gs, "Author <a@localhost>", "base commit");

  bool ok = git_store_create_branch(g_gs, "user/alice/segment-1");
  ASSERT(ok);
  // Creating the branch also checks it out (git checkout -b).
  // Write a file on the branch.
  const char *branch_data = "branch file";
  ok = git_store_write_file(g_gs, "branch.txt", branch_data, strlen(branch_data));
  ASSERT(ok);
  ok = git_store_commit(g_gs, "Alice <alice@localhost>", "branch commit");
  ASSERT(ok);

  // Switch back to main.
  ok = git_store_checkout(g_gs, "main");
  ASSERT(ok);

  // branch.txt should not exist on main yet.
  size_t len = 0;
  uint8_t *buf = git_store_read_file(g_gs, "branch.txt", &len);
  ASSERT(buf == NULL);

  PASS();
}

TEST test_merge_branch_into_main(void) {
  // Base commit on main.
  const char *base = "merge base";
  git_store_write_file(g_gs, "base.txt", base, strlen(base));
  git_store_commit(g_gs, "Author <a@localhost>", "base");

  // Create branch, add file, commit.
  git_store_create_branch(g_gs, "user/bob/segment-2");
  const char *bdata = "from bob";
  git_store_write_file(g_gs, "bob.txt", bdata, strlen(bdata));
  git_store_commit(g_gs, "Bob <bob@localhost>", "bob commit");

  // Back to main, merge.
  git_store_checkout(g_gs, "main");
  bool ok = git_store_merge(g_gs, "user/bob/segment-2");
  ASSERT(ok);

  // bob.txt should now be present on main.
  size_t len = 0;
  uint8_t *buf = git_store_read_file(g_gs, "bob.txt", &len);
  ASSERT(buf != NULL);
  ASSERT(len > 0);
  free(buf);
  PASS();
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

TEST test_modified_count_after_write(void) {
  const char *data = "dirty";
  git_store_write_file(g_gs, "dirty.txt", data, strlen(data));
  // Staged but not committed — should appear in status.
  int n = git_store_modified_count(g_gs);
  ASSERT(n > 0);
  PASS();
}

TEST test_is_clean_after_commit(void) {
  const char *data = "clean";
  git_store_write_file(g_gs, "clean.txt", data, strlen(data));
  git_store_commit(g_gs, "Author <a@localhost>", "clean commit");
  ASSERT(git_store_is_clean(g_gs));
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(gitstore_suite) {
  SET_SETUP(setup, NULL);
  SET_TEARDOWN(teardown, NULL);

  RUN_TEST(test_open_creates_repo);
  RUN_TEST(test_open_idempotent);
  RUN_TEST(test_write_read_file);
  RUN_TEST(test_read_missing_file_returns_null);
  RUN_TEST(test_write_surface_stages_file);
  RUN_TEST(test_commit_after_write);
  RUN_TEST(test_log_returns_entry);
  RUN_TEST(test_create_branch_and_checkout);
  RUN_TEST(test_merge_branch_into_main);
  RUN_TEST(test_modified_count_after_write);
  RUN_TEST(test_is_clean_after_commit);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(gitstore_suite);
  GREATEST_MAIN_END();
}
