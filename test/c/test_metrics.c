#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "cli/cli_metrics.h"

#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_metrics_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_metrics(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_metrics_no_args(void) {
  int rc = cmd_metrics(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

TEST test_metrics_plane(void) {
  char *args[] = { "plane:z=5" };
  int rc = cmd_metrics(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_metrics_plane_z0(void) {
  char *args[] = { "plane:z=0" };
  int rc = cmd_metrics(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_metrics_unknown_surface(void) {
  // Unknown surface falls back to plane:z=0 with a warning — still returns 0.
  char *args[] = { "/nonexistent/surface.obj" };
  int rc = cmd_metrics(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(metrics_suite) {
  RUN_TEST(test_metrics_help);
  RUN_TEST(test_metrics_no_args);
  RUN_TEST(test_metrics_plane);
  RUN_TEST(test_metrics_plane_z0);
  RUN_TEST(test_metrics_unknown_surface);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(metrics_suite);
  GREATEST_MAIN_END();
}
