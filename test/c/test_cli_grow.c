#include "greatest.h"
#include "cli/cli_grow.h"

// ---------------------------------------------------------------------------
// --help: must print usage and return 0 without crashing
// ---------------------------------------------------------------------------

TEST test_grow_help(void) {
  char *argv[] = { "--help" };
  int rc = cmd_grow(1, argv);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_grow_help_short(void) {
  char *argv[] = { "--help", "-h" };
  int rc = cmd_grow(2, argv);
  ASSERT_EQ(0, rc);
  PASS();
}

// No volume path → non-zero exit (usage error).
TEST test_grow_no_args(void) {
  int rc = cmd_grow(0, NULL);
  ASSERT(rc != 0);
  PASS();
}

// Missing --seed → non-zero exit.
TEST test_grow_missing_seed(void) {
  char *argv[] = { "/nonexistent/volume.zarr" };
  int rc = cmd_grow(1, argv);
  ASSERT(rc != 0);
  PASS();
}

// Bad --seed format → non-zero exit.
TEST test_grow_bad_seed_format(void) {
  char *argv[] = { "/nonexistent/volume.zarr", "--seed", "notacoord" };
  int rc = cmd_grow(3, argv);
  ASSERT(rc != 0);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(cli_grow_suite) {
  RUN_TEST(test_grow_help);
  RUN_TEST(test_grow_help_short);
  RUN_TEST(test_grow_no_args);
  RUN_TEST(test_grow_missing_seed);
  RUN_TEST(test_grow_bad_seed_format);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(cli_grow_suite);
  GREATEST_MAIN_END();
}
