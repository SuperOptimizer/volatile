// test_s3_browser.c — unit tests for s3_browser lifecycle, state, and navigation.
// No real S3 calls are made; s3_list_objects is not invoked (no credentials set).

#include "greatest.h"

struct nk_context;  // stub

#include "gui/s3_browser.h"
#include "core/net.h"

#include <string.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);
  s3_browser_free(b);
  PASS();
}

TEST test_free_null(void) {
  s3_browser_free(NULL);  // must not crash
  PASS();
}

TEST test_initial_state(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);
  ASSERT(!s3_browser_is_visible(b));
  const char *url = s3_browser_get_url(b);
  // URL is empty or NULL initially
  ASSERT(url == NULL || url[0] == '\0');
  s3_browser_free(b);
  PASS();
}

TEST test_show_hide(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  ASSERT(!s3_browser_is_visible(b));
  s3_browser_show(b);
  ASSERT(s3_browser_is_visible(b));

  // render with NULL ctx → returns false, closes
  bool result = s3_browser_render(b, NULL);
  ASSERT(!result);

  s3_browser_free(b);
  PASS();
}

TEST test_set_creds(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  s3_credentials creds = {0};
  snprintf(creds.access_key, sizeof(creds.access_key), "AKIATEST");
  snprintf(creds.secret_key, sizeof(creds.secret_key), "secretkey");
  snprintf(creds.region,     sizeof(creds.region),     "us-east-1");

  s3_browser_set_creds(b, &creds);
  // no crash; credentials stored internally

  s3_browser_set_creds(NULL, &creds);  // null-safe
  s3_browser_set_creds(b, NULL);       // null-safe

  s3_browser_free(b);
  PASS();
}

TEST test_set_bucket(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  s3_browser_set_bucket(b, "my-scroll-data");

  // navigate into a prefix
  s3_browser_navigate(b, "volumes/");

  // go up should return to root
  s3_browser_go_up(b);

  // go up from root should be a no-op (not crash)
  s3_browser_go_up(b);

  s3_browser_free(b);
  PASS();
}

TEST test_navigate_go_up(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  s3_browser_set_bucket(b, "bucket");
  s3_browser_navigate(b, "a/b/c/");
  s3_browser_go_up(b);   // -> "a/b/"
  s3_browser_go_up(b);   // -> "a/"
  s3_browser_go_up(b);   // -> ""
  s3_browser_go_up(b);   // no-op at root

  s3_browser_free(b);
  PASS();
}

TEST test_null_api(void) {
  // All public functions must handle NULL browser without crashing
  ASSERT(!s3_browser_is_visible(NULL));
  ASSERT(s3_browser_get_url(NULL) == NULL);
  s3_browser_show(NULL);
  s3_browser_set_bucket(NULL, "bucket");
  s3_browser_navigate(NULL, "prefix/");
  s3_browser_go_up(NULL);
  ASSERT(!s3_browser_render(NULL, NULL));
  s3_browser_add_bookmark(NULL, "name", "s3://b/p/");
  s3_browser_add_recent(NULL, "s3://b/p/");
  PASS();
}

TEST test_bookmarks(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  s3_browser_add_bookmark(b, "Scroll1", "s3://scroll-data/volumes/scroll1/");
  s3_browser_add_bookmark(b, "Scroll2", "s3://scroll-data/volumes/scroll2/");
  // Duplicate URL should be ignored
  s3_browser_add_bookmark(b, "Dup",     "s3://scroll-data/volumes/scroll1/");

  // Null-safe
  s3_browser_add_bookmark(b, NULL, "s3://b/p/");
  s3_browser_add_bookmark(b, "name", NULL);

  s3_browser_free(b);
  PASS();
}

TEST test_recent(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);

  s3_browser_add_recent(b, "s3://bucket/vol1.zarr/");
  s3_browser_add_recent(b, "s3://bucket/vol2.zarr/");
  s3_browser_add_recent(b, "s3://bucket/vol3.zarr/");
  // Re-add existing: should move to front, not duplicate
  s3_browser_add_recent(b, "s3://bucket/vol1.zarr/");

  // Null / empty: safe
  s3_browser_add_recent(b, NULL);
  s3_browser_add_recent(b, "");

  s3_browser_free(b);
  PASS();
}

TEST test_recent_overflow(void) {
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);
  // Add 15 entries (RECENT_MAX=12) — oldest should be evicted silently
  for (int i = 0; i < 15; i++) {
    char url[64];
    snprintf(url, sizeof(url), "s3://bucket/vol%d.zarr/", i);
    s3_browser_add_recent(b, url);
  }
  s3_browser_free(b);
  PASS();
}

TEST test_filter_state(void) {
  // Filter is internal; test that navigate resets it and no crash
  s3_browser *b = s3_browser_new();
  ASSERT(b != NULL);
  s3_browser_set_bucket(b, "bucket");
  s3_browser_navigate(b, "a/");
  s3_browser_navigate(b, "a/b/");  // second navigate also fine
  s3_browser_go_up(b);
  s3_browser_free(b);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(s3_browser_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_initial_state);
  RUN_TEST(test_show_hide);
  RUN_TEST(test_set_creds);
  RUN_TEST(test_set_bucket);
  RUN_TEST(test_navigate_go_up);
  RUN_TEST(test_null_api);
  RUN_TEST(test_bookmarks);
  RUN_TEST(test_recent);
  RUN_TEST(test_recent_overflow);
  RUN_TEST(test_filter_state);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(s3_browser_suite);
  GREATEST_MAIN_END();
}
