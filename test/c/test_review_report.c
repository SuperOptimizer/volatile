#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "gui/review_report.h"
#include "server/db.h"
#include "server/review.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Read a file into a heap-allocated string. Caller must free.
static char *read_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) { fclose(f); return NULL; }
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  buf[n] = '\0';
  fclose(f);
  return buf;
}

static const char *g_tmpdir = "/tmp";

static char *tmppath(const char *name, char *buf, int bufsz) {
  snprintf(buf, (size_t)bufsz, "%s/%s", g_tmpdir, name);
  return buf;
}

// ---------------------------------------------------------------------------
// NULL / degenerate inputs
// ---------------------------------------------------------------------------

TEST test_null_proj(void) {
  ASSERT(!review_report_generate(NULL, "/tmp/rr_null.txt", false));
  PASS();
}

TEST test_null_db(void) {
  project p = { .db = NULL };
  ASSERT(!review_report_generate(&p, "/tmp/rr_nulldb.txt", false));
  PASS();
}

TEST test_null_path(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  project p = { .db = db, .reviews = NULL };
  ASSERT(!review_report_generate(&p, NULL, false));
  seg_db_close(db);
  PASS();
}

// ---------------------------------------------------------------------------
// Plain text output
// ---------------------------------------------------------------------------

TEST test_text_empty_db(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  project p = { .db = db, .name = "TestProj" };

  char path[256];
  tmppath("rr_empty.txt", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, false));

  char *content = read_file(path);
  ASSERT(content != NULL);
  // Should contain the header and a count of 0
  ASSERT(strstr(content, "TestProj") != NULL);
  ASSERT(strstr(content, "0 segment") != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

TEST test_text_with_segments(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  int64_t id1 = seg_db_insert_segment(db, "vol-1", "Alpha", "/data/alpha.obj");
  int64_t id2 = seg_db_insert_segment(db, "vol-1", "Beta",  "/data/beta.obj");
  ASSERT(id1 > 0); ASSERT(id2 > 0);

  project p = { .db = db, .name = "MyProject" };
  char path[256];
  tmppath("rr_segs.txt", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, false));

  char *content = read_file(path);
  ASSERT(content != NULL);
  ASSERT(strstr(content, "Alpha") != NULL);
  ASSERT(strstr(content, "Beta")  != NULL);
  ASSERT(strstr(content, "2 segment") != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

TEST test_text_review_status_appears(void) {
  seg_db        *db = seg_db_open(":memory:");
  review_system *rs = review_new(db);
  ASSERT(db); ASSERT(rs);

  int64_t seg_id = seg_db_insert_segment(db, "vol-1", "MySeg", "/data/s.obj");
  int64_t rev_id = review_submit(rs, seg_id, 1);
  ASSERT(rev_id > 0);
  ASSERT(review_approve(rs, rev_id, 2, "looks great"));

  project p = { .db = db, .reviews = rs, .name = "Proj" };
  char path[256];
  tmppath("rr_status.txt", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, false));

  char *content = read_file(path);
  ASSERT(content != NULL);
  ASSERT(strstr(content, "Approved") != NULL);
  free(content);
  remove(path);

  review_free(rs);
  seg_db_close(db);
  PASS();
}

// ---------------------------------------------------------------------------
// HTML output
// ---------------------------------------------------------------------------

TEST test_html_structure(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  seg_db_insert_segment(db, "vol-1", "Gamma", "/data/gamma.obj");

  project p = { .db = db, .name = "HTMLTest" };
  char path[256];
  tmppath("rr_html.html", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, true));

  char *content = read_file(path);
  ASSERT(content != NULL);
  // Must be valid-looking HTML
  ASSERT(strstr(content, "<!DOCTYPE html>") != NULL);
  ASSERT(strstr(content, "<table>")         != NULL);
  ASSERT(strstr(content, "</table>")        != NULL);
  ASSERT(strstr(content, "HTMLTest")        != NULL);
  ASSERT(strstr(content, "Gamma")           != NULL);
  ASSERT(strstr(content, "1 segment")       != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

TEST test_html_escapes_special_chars(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  seg_db_insert_segment(db, "vol-1", "Seg<>&\"name", "/data/x.obj");

  project p = { .db = db, .name = "<HTML> & \"Test\"" };
  char path[256];
  tmppath("rr_escape.html", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, true));

  char *content = read_file(path);
  ASSERT(content != NULL);
  // Raw < must not appear in data fields (only in tags)
  // We can't easily distinguish tag < from data <, so check that entity forms appear
  ASSERT(strstr(content, "&lt;") != NULL);
  ASSERT(strstr(content, "&amp;") != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

TEST test_html_no_reviews_shows_pending(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  seg_db_insert_segment(db, "vol-1", "UnreviewedSeg", "/data/u.obj");

  project p = { .db = db, .reviews = NULL, .name = "P" };
  char path[256];
  tmppath("rr_noreview.html", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, true));

  char *content = read_file(path);
  ASSERT(content != NULL);
  ASSERT(strstr(content, "Pending") != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

TEST test_text_no_reviews_shows_pending(void) {
  seg_db *db = seg_db_open(":memory:");
  ASSERT(db);
  seg_db_insert_segment(db, "vol-1", "X", "/data/x.obj");

  project p = { .db = db, .reviews = NULL };
  char path[256];
  tmppath("rr_noreview.txt", path, sizeof(path));
  ASSERT(review_report_generate(&p, path, false));

  char *content = read_file(path);
  ASSERT(content != NULL);
  ASSERT(strstr(content, "Pending") != NULL);
  free(content);
  remove(path);

  seg_db_close(db);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(review_report_suite) {
  RUN_TEST(test_null_proj);
  RUN_TEST(test_null_db);
  RUN_TEST(test_null_path);
  RUN_TEST(test_text_empty_db);
  RUN_TEST(test_text_with_segments);
  RUN_TEST(test_text_review_status_appears);
  RUN_TEST(test_html_structure);
  RUN_TEST(test_html_escapes_special_chars);
  RUN_TEST(test_html_no_reviews_shows_pending);
  RUN_TEST(test_text_no_reviews_shows_pending);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(review_report_suite);
  GREATEST_MAIN_END();
}
