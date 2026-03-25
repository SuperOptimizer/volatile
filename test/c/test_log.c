#include "greatest.h"
#include "core/log.h"

#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_version(void) {
  const char *v = volatile_version();
  ASSERT_STR_EQ("0.1.0", v);
  PASS();
}

TEST test_log_level_set_get(void) {
  log_set_level(LOG_DEBUG);
  ASSERT_EQ(LOG_DEBUG, log_get_level());

  log_set_level(LOG_WARN);
  ASSERT_EQ(LOG_WARN, log_get_level());

  log_set_level(LOG_ERROR);
  ASSERT_EQ(LOG_ERROR, log_get_level());

  // restore default
  log_set_level(LOG_INFO);
  ASSERT_EQ(LOG_INFO, log_get_level());
  PASS();
}

TEST test_log_output_no_crash(void) {
  // Exercise every level at LOG_DEBUG threshold — must not crash or abort.
  log_set_level(LOG_DEBUG);

  LOG_DEBUG("debug message %d", 1);
  LOG_INFO("info message %s", "hello");
  LOG_WARN("warn message");
  LOG_ERROR("error message");

  log_set_level(LOG_INFO);
  PASS();
}

TEST test_log_to_file(void) {
  // Write a log line to a tmpfile and verify it contains expected JSON fields.
  FILE *f = tmpfile();
  ASSERT(f != NULL);

  log_set_file(f);
  log_set_level(LOG_DEBUG);

  LOG_INFO("structured log test");

  log_set_file(NULL);
  log_set_level(LOG_INFO);

  rewind(f);
  char buf[512] = {0};
  size_t n = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);

  ASSERT(n > 0);
  ASSERT(strstr(buf, "\"level\"") != NULL);
  ASSERT(strstr(buf, "\"msg\"") != NULL);
  ASSERT(strstr(buf, "structured log test") != NULL);
  PASS();
}

TEST test_assert_macro_exists(void) {
  // In debug mode ASSERT evaluates the expression; in release it is a no-op.
  // Either way it must compile and the non-firing path must not abort.
  ASSERT(1 == 1);        // must not fire
  ASSERT(1 == 1, "ok");  // with optional message
  PASS();
}

TEST test_require_macro_no_fire(void) {
  // REQUIRE is always active — passing expression must not abort.
  REQUIRE(1 == 1);
  REQUIRE(2 > 1, "two is greater than one");
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(log_suite) {
  RUN_TEST(test_version);
  RUN_TEST(test_log_level_set_get);
  RUN_TEST(test_log_output_no_crash);
  RUN_TEST(test_log_to_file);
  RUN_TEST(test_assert_macro_exists);
  RUN_TEST(test_require_macro_no_fire);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(log_suite);
  GREATEST_MAIN_END();
}
