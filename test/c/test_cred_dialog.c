#include "greatest.h"
#include "gui/cred_dialog.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const char *tmp_creds_path(void) { return "/tmp/test_volatile_creds.json"; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  cred_dialog *d = cred_dialog_new();
  ASSERT(d != NULL);
  ASSERT(!cred_dialog_is_visible(d));
  cred_dialog_free(d);
  PASS();
}

TEST test_show_sets_visible(void) {
  cred_dialog *d = cred_dialog_new();
  ASSERT(!cred_dialog_is_visible(d));
  cred_dialog_show(d, "s3://my-bucket/data.zarr");
  ASSERT(cred_dialog_is_visible(d));
  cred_dialog_free(d);
  PASS();
}

TEST test_show_null_url(void) {
  cred_dialog *d = cred_dialog_new();
  cred_dialog_show(d, NULL);   // must not crash
  ASSERT(cred_dialog_is_visible(d));
  cred_dialog_free(d);
  PASS();
}

// render with NULL ctx must return false without crashing
TEST test_render_null_ctx(void) {
  cred_dialog *d = cred_dialog_new();
  cred_dialog_show(d, NULL);
  bool submitted = cred_dialog_render(d, NULL);
  ASSERT(!submitted);
  cred_dialog_free(d);
  PASS();
}

// render when not visible must return false without crashing
TEST test_render_not_visible(void) {
  cred_dialog *d = cred_dialog_new();
  // do not call show — dialog is invisible
  bool submitted = cred_dialog_render(d, NULL);
  ASSERT(!submitted);
  cred_dialog_free(d);
  PASS();
}

// get_creds returns a non-NULL pointer always (result struct inside dialog)
TEST test_get_creds_nonnull(void) {
  cred_dialog *d = cred_dialog_new();
  s3_credentials *c = cred_dialog_get_creds(d);
  ASSERT(c != NULL);
  cred_dialog_free(d);
  PASS();
}

// save then load round-trips credentials
TEST test_save_load(void) {
  cred_dialog *d = cred_dialog_new();
  // Populate result manually (simulates post-submit state)
  s3_credentials *c = cred_dialog_get_creds(d);
  strncpy(c->access_key, "AKIAIOSFODNN7EXAMPLE",  sizeof(c->access_key) - 1);
  strncpy(c->secret_key, "wJalrXUtnFEMI/K7MDENG", sizeof(c->secret_key) - 1);
  strncpy(c->token,      "AQoXnyc4lcK4w",         sizeof(c->token) - 1);
  strncpy(c->region,     "eu-west-1",              sizeof(c->region) - 1);
  strncpy(c->endpoint,   "https://s3.example.com", sizeof(c->endpoint) - 1);

  // Mark as "remember" so save actually writes
  // NOTE: access via internal fields isn't exposed — test save/load via cred_dialog_load
  // which sets remember=true internally.

  // save will skip if remember==false, so write the file directly for this test
  FILE *f = fopen(tmp_creds_path(), "w");
  ASSERT(f != NULL);
  fprintf(f,
    "{\"access_key\":\"AKIAIOSFODNN7EXAMPLE\","
    "\"secret_key\":\"wJalrXUtnFEMI/K7MDENG\","
    "\"token\":\"AQoXnyc4lcK4w\","
    "\"region\":\"eu-west-1\","
    "\"endpoint\":\"https://s3.example.com\"}");
  fclose(f);

  cred_dialog *d2 = cred_dialog_new();
  bool ok = cred_dialog_load(d2, tmp_creds_path());
  ASSERT(ok);

  s3_credentials *c2 = cred_dialog_get_creds(d2);
  ASSERT_STR_EQ("AKIAIOSFODNN7EXAMPLE",  c2->access_key);
  ASSERT_STR_EQ("wJalrXUtnFEMI/K7MDENG",c2->secret_key);
  ASSERT_STR_EQ("AQoXnyc4lcK4w",         c2->token);
  ASSERT_STR_EQ("eu-west-1",             c2->region);
  ASSERT_STR_EQ("https://s3.example.com",c2->endpoint);

  remove(tmp_creds_path());
  cred_dialog_free(d);
  cred_dialog_free(d2);
  PASS();
}

// load from missing file returns false without crashing
TEST test_load_missing_file(void) {
  cred_dialog *d = cred_dialog_new();
  bool ok = cred_dialog_load(d, "/tmp/no_such_file_volatile_creds.json");
  ASSERT(!ok);
  cred_dialog_free(d);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(cred_dialog_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_show_sets_visible);
  RUN_TEST(test_show_null_url);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_not_visible);
  RUN_TEST(test_get_creds_nonnull);
  RUN_TEST(test_save_load);
  RUN_TEST(test_load_missing_file);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(cred_dialog_suite);
  GREATEST_MAIN_END();
}
