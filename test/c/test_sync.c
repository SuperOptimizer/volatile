#include "greatest.h"
#include "server/sync.h"
#include "server/gitstore.h"
#include "core/geom.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Mock git_store — lets us test sync.c without a real git repo.
// Tracks call counts and last arguments; defined here and linked in place of
// the real gitstore.c via the test-only link step.
// ---------------------------------------------------------------------------

struct git_store {
  int  write_surface_calls;
  int  write_annotation_calls;
  int  commit_calls;
  int  pull_calls;
  int  push_calls;
  int  merge_calls;
  bool next_pull_ok;      // controls return value of git_store_pull
  bool next_push_ok;
  bool dirty;             // simulates staged but uncommitted changes
  char last_author[64];
  char last_msg[256];
  char last_surface_name[64];
  char last_annot_name[64];
};

static struct git_store g_mock;

git_store *git_store_open(const char *path) { (void)path; return &g_mock; }
git_store *git_store_clone(const char *r, const char *l) { (void)r;(void)l; return &g_mock; }
void       git_store_free(git_store *g) { (void)g; }

bool git_store_write_file(git_store *g, const char *path,
                          const void *data, size_t len) {
  (void)path;(void)data;(void)len; g->dirty = true; return true;
}
bool git_store_write_surface(git_store *g, const char *name,
                              const quad_surface *s) {
  (void)s;
  g->write_surface_calls++;
  strncpy(g->last_surface_name, name, sizeof(g->last_surface_name) - 1);
  g->dirty = true;
  return true;
}
bool git_store_write_annotation(git_store *g, const char *name,
                                 const char *json) {
  (void)json;
  g->write_annotation_calls++;
  strncpy(g->last_annot_name, name, sizeof(g->last_annot_name) - 1);
  g->dirty = true;
  return true;
}
uint8_t *git_store_read_file(git_store *g, const char *path, size_t *len) {
  (void)g;(void)path; *len = 0; return NULL;
}
bool git_store_commit(git_store *g, const char *author, const char *message) {
  g->commit_calls++;
  strncpy(g->last_author, author,  sizeof(g->last_author)  - 1);
  strncpy(g->last_msg,    message, sizeof(g->last_msg)     - 1);
  g->dirty = false;
  return true;
}
bool git_store_create_branch(git_store *g, const char *b) { (void)g;(void)b; return true; }
bool git_store_checkout(git_store *g, const char *b) { (void)g;(void)b; return true; }
bool git_store_merge(git_store *g, const char *b) {
  (void)b; g->merge_calls++; g->dirty = false; return true;
}
int  git_store_log(git_store *g, git_log_entry *o, int m) { (void)g;(void)o;(void)m; return 0; }
char *git_store_diff(git_store *g, const char *a, const char *b) { (void)g;(void)a;(void)b; return NULL; }
bool git_store_push(git_store *g, const char *r) {
  (void)r; g->push_calls++; return g->next_push_ok;
}
bool git_store_pull(git_store *g, const char *r) {
  (void)r; g->pull_calls++; return g->next_pull_ok;
}
bool git_store_lfs_track(git_store *g, const char *p) { (void)g;(void)p; return true; }
int  git_store_modified_count(git_store *g) { return g->dirty ? 1 : 0; }
bool git_store_is_clean(git_store *g) { return !g->dirty; }

// server_broadcast stub — sync.c guards with `if (s->srv)` so this is
// never called in these tests, but the linker still needs the symbol.
#include "server/srv.h"
void server_broadcast(vol_server *s, msg_type_t type,
                      const uint8_t *payload, uint32_t len) {
  (void)s; (void)type; (void)payload; (void)len;
}

// ---------------------------------------------------------------------------
// Reset mock state between tests
// ---------------------------------------------------------------------------

static void reset_mock(void) {
  memset(&g_mock, 0, sizeof(g_mock));
  g_mock.next_pull_ok = true;
  g_mock.next_push_ok = true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_sync_new_free(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  ASSERT(s != NULL);
  sync_free(s);
  PASS();
}

TEST test_sync_new_null_store(void) {
  sync_manager *s = sync_new(NULL, NULL);
  ASSERT(s == NULL);
  PASS();
}

TEST test_segment_edit_staged(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  quad_surface *surf = quad_surface_new(4, 4);
  sync_on_segment_edit(s, 1, 42, surf);
  ASSERT_EQ(1, g_mock.write_surface_calls);
  ASSERT(strstr(g_mock.last_surface_name, "42") != NULL);
  ASSERT(g_mock.dirty);
  quad_surface_free(surf);
  sync_free(s);
  PASS();
}

TEST test_annotation_edit_staged(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  sync_on_annotation_edit(s, 2, 7, "{\"type\":\"point\"}");
  ASSERT_EQ(1, g_mock.write_annotation_calls);
  ASSERT(strstr(g_mock.last_annot_name, "7") != NULL);
  ASSERT(g_mock.dirty);
  sync_free(s);
  PASS();
}

TEST test_commit_pending_creates_commit(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.dirty = true;  // simulate staged changes
  bool ok = sync_commit_pending(s, 5, "manual commit");
  ASSERT(ok);
  ASSERT_EQ(1, g_mock.commit_calls);
  ASSERT(strstr(g_mock.last_author, "5") != NULL);
  ASSERT(strstr(g_mock.last_msg, "manual commit") != NULL);
  ASSERT(!g_mock.dirty);
  sync_free(s);
  PASS();
}

TEST test_commit_pending_noop_when_clean(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.dirty = false;
  bool ok = sync_commit_pending(s, 1, NULL);
  ASSERT(ok);
  ASSERT_EQ(0, g_mock.commit_calls);  // no commit when nothing staged
  sync_free(s);
  PASS();
}

TEST test_edit_then_commit(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  quad_surface *surf = quad_surface_new(8, 8);
  sync_on_segment_edit(s, 3, 100, surf);
  sync_commit_pending(s, 3, "save seg 100");
  ASSERT_EQ(1, g_mock.write_surface_calls);
  ASSERT_EQ(1, g_mock.commit_calls);
  quad_surface_free(surf);
  sync_free(s);
  PASS();
}

TEST test_pull_success(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.next_pull_ok = true;
  bool ok = sync_pull_latest(s);
  ASSERT(ok);
  ASSERT_EQ(1, g_mock.pull_calls);
  sync_free(s);
  PASS();
}

TEST test_pull_conflict(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.next_pull_ok = false;
  bool ok = sync_pull_latest(s);
  ASSERT(!ok);
  sync_free(s);
  PASS();
}

TEST test_push(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.next_push_ok = true;
  bool ok = sync_push(s);
  ASSERT(ok);
  ASSERT_EQ(1, g_mock.push_calls);
  sync_free(s);
  PASS();
}

TEST test_resolve_theirs(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.dirty = true;
  bool ok = sync_resolve_conflicts(s, CONFLICT_THEIRS);
  ASSERT(ok);
  ASSERT_EQ(1, g_mock.merge_calls);
  sync_free(s);
  PASS();
}

TEST test_resolve_ours(void) {
  reset_mock();
  sync_manager *s = sync_new(&g_mock, NULL);
  g_mock.dirty = true;
  bool ok = sync_resolve_conflicts(s, CONFLICT_OURS);
  ASSERT(ok);
  ASSERT_EQ(1, g_mock.commit_calls);
  sync_free(s);
  PASS();
}

TEST test_null_safety(void) {
  sync_on_segment_edit(NULL, 0, 0, NULL);  // must not crash
  sync_on_annotation_edit(NULL, 0, 0, NULL);
  sync_commit_pending(NULL, 0, NULL);
  sync_pull_latest(NULL);
  sync_push(NULL);
  sync_resolve_conflicts(NULL, CONFLICT_MERGE);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(sync_suite) {
  RUN_TEST(test_sync_new_free);
  RUN_TEST(test_sync_new_null_store);
  RUN_TEST(test_segment_edit_staged);
  RUN_TEST(test_annotation_edit_staged);
  RUN_TEST(test_commit_pending_creates_commit);
  RUN_TEST(test_commit_pending_noop_when_clean);
  RUN_TEST(test_edit_then_commit);
  RUN_TEST(test_pull_success);
  RUN_TEST(test_pull_conflict);
  RUN_TEST(test_push);
  RUN_TEST(test_resolve_theirs);
  RUN_TEST(test_resolve_ours);
  RUN_TEST(test_null_safety);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(sync_suite);
  GREATEST_MAIN_END();
}
