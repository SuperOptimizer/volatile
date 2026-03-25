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
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(s3_browser_suite);
  GREATEST_MAIN_END();
}
