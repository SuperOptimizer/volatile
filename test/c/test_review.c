#include "greatest.h"
#include "server/db.h"
#include "server/review.h"

#include <string.h>

static seg_db        *g_db = NULL;
static review_system *g_rs = NULL;

// Open fresh in-memory DB + review system before each test.
static void setup(void *unused) {
  (void)unused;
  g_db = seg_db_open(":memory:");
  g_rs = review_new(g_db);
}

static void teardown(void *unused) {
  (void)unused;
  review_free(g_rs);
  seg_db_close(g_db);
  g_db = NULL; g_rs = NULL;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  ASSERT(g_db != NULL);
  ASSERT(g_rs != NULL);
  PASS();
}

TEST test_review_new_null_db(void) {
  review_system *r = review_new(NULL);
  ASSERT(r == NULL);
  review_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

TEST test_submit_returns_id(void) {
  int64_t rid = review_submit(g_rs, 100, 1);
  ASSERT(rid > 0);
  PASS();
}

TEST test_submit_multiple_unique_ids(void) {
  int64_t r1 = review_submit(g_rs, 1, 1);
  int64_t r2 = review_submit(g_rs, 2, 1);
  int64_t r3 = review_submit(g_rs, 1, 2);
  ASSERT(r1 > 0); ASSERT(r2 > 0); ASSERT(r3 > 0);
  ASSERT(r1 != r2); ASSERT(r2 != r3);
  PASS();
}

TEST test_submit_null_system(void) {
  int64_t rid = review_submit(NULL, 1, 1);
  ASSERT_EQ(-1, rid);
  PASS();
}

// ---------------------------------------------------------------------------
// Initial status is PENDING
// ---------------------------------------------------------------------------

TEST test_initial_status_pending(void) {
  review_submit(g_rs, 42, 1);
  review_status st = review_get_status(g_rs, 42);
  ASSERT_EQ(REVIEW_PENDING, st);
  PASS();
}

TEST test_get_status_unknown_surface(void) {
  review_status st = review_get_status(g_rs, 9999);
  ASSERT_EQ((review_status)-1, st);
  PASS();
}

// ---------------------------------------------------------------------------
// Approve
// ---------------------------------------------------------------------------

TEST test_approve(void) {
  int64_t rid = review_submit(g_rs, 10, 1);
  bool ok = review_approve(g_rs, rid, 2, "looks good");
  ASSERT(ok);
  ASSERT_EQ(REVIEW_APPROVED, review_get_status(g_rs, 10));
  PASS();
}

// ---------------------------------------------------------------------------
// Reject
// ---------------------------------------------------------------------------

TEST test_reject(void) {
  int64_t rid = review_submit(g_rs, 20, 1);
  bool ok = review_reject(g_rs, rid, 3, "boundary issues");
  ASSERT(ok);
  ASSERT_EQ(REVIEW_REJECTED, review_get_status(g_rs, 20));
  PASS();
}

// ---------------------------------------------------------------------------
// Request changes
// ---------------------------------------------------------------------------

TEST test_request_changes(void) {
  int64_t rid = review_submit(g_rs, 30, 1);
  bool ok = review_request_changes(g_rs, rid, 4, "smooth the edge");
  ASSERT(ok);
  ASSERT_EQ(REVIEW_NEEDS_CHANGES, review_get_status(g_rs, 30));
  PASS();
}

// ---------------------------------------------------------------------------
// list_pending
// ---------------------------------------------------------------------------

TEST test_list_pending(void) {
  review_submit(g_rs, 1, 1);
  review_submit(g_rs, 2, 1);
  int64_t r3 = review_submit(g_rs, 3, 1);
  review_approve(g_rs, r3, 2, "");  // no longer pending

  review_entry out[10];
  int n = review_list_pending(g_rs, out, 10);
  ASSERT_EQ(2, n);
  // All returned entries should be PENDING.
  for (int i = 0; i < n; i++)
    ASSERT_EQ(REVIEW_PENDING, out[i].status);
  PASS();
}

TEST test_list_pending_empty(void) {
  review_entry out[4];
  int n = review_list_pending(g_rs, out, 4);
  ASSERT_EQ(0, n);
  PASS();
}

// ---------------------------------------------------------------------------
// list_for_surface
// ---------------------------------------------------------------------------

TEST test_list_for_surface(void) {
  int64_t r1 = review_submit(g_rs, 55, 1);
  int64_t r2 = review_submit(g_rs, 55, 2);  // second reviewer
  review_submit(g_rs, 99, 1);               // different surface

  review_approve(g_rs, r2, 3, "ok");

  review_entry out[10];
  int n = review_list_for_surface(g_rs, 55, out, 10);
  ASSERT_EQ(2, n);
  for (int i = 0; i < n; i++)
    ASSERT_EQ(55, out[i].surface_id);

  (void)r1;
  PASS();
}

// ---------------------------------------------------------------------------
// count_by_status
// ---------------------------------------------------------------------------

TEST test_count_by_status(void) {
  int64_t r1 = review_submit(g_rs, 1, 1);
  int64_t r2 = review_submit(g_rs, 2, 1);
  int64_t r3 = review_submit(g_rs, 3, 1);
  review_approve(g_rs, r1, 2, "");
  review_approve(g_rs, r2, 2, "");
  review_reject (g_rs, r3, 2, "bad");

  ASSERT_EQ(2, review_count_by_status(g_rs, REVIEW_APPROVED));
  ASSERT_EQ(1, review_count_by_status(g_rs, REVIEW_REJECTED));
  ASSERT_EQ(0, review_count_by_status(g_rs, REVIEW_PENDING));
  ASSERT_EQ(0, review_count_by_status(g_rs, REVIEW_NEEDS_CHANGES));
  PASS();
}

// ---------------------------------------------------------------------------
// Comment content preserved
// ---------------------------------------------------------------------------

TEST test_comment_stored(void) {
  int64_t rid = review_submit(g_rs, 77, 1);
  review_reject(g_rs, rid, 2, "fix the topology");

  review_entry out[1];
  int n = review_list_for_surface(g_rs, 77, out, 1);
  ASSERT_EQ(1, n);
  ASSERT(strstr(out[0].comment, "topology") != NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(review_suite) {
  SET_SETUP(setup, NULL);
  SET_TEARDOWN(teardown, NULL);

  RUN_TEST(test_new_free);
  RUN_TEST(test_review_new_null_db);
  RUN_TEST(test_submit_returns_id);
  RUN_TEST(test_submit_multiple_unique_ids);
  RUN_TEST(test_submit_null_system);
  RUN_TEST(test_initial_status_pending);
  RUN_TEST(test_get_status_unknown_surface);
  RUN_TEST(test_approve);
  RUN_TEST(test_reject);
  RUN_TEST(test_request_changes);
  RUN_TEST(test_list_pending);
  RUN_TEST(test_list_pending_empty);
  RUN_TEST(test_list_for_surface);
  RUN_TEST(test_count_by_status);
  RUN_TEST(test_comment_stored);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(review_suite);
  GREATEST_MAIN_END();
}
