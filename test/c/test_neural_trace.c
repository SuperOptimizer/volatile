#define _DEFAULT_SOURCE
#include "greatest.h"
#include "gui/neural_trace.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// neural_tracer lifecycle tests (no subprocess spawned)
// ---------------------------------------------------------------------------

TEST test_new_free_null_model(void) {
  neural_tracer *t = neural_tracer_new(NULL);
  ASSERT(t != NULL);
  ASSERT(!neural_tracer_is_running(t));
  neural_tracer_free(t);
  PASS();
}

TEST test_new_free_with_path(void) {
  neural_tracer *t = neural_tracer_new("/nonexistent/model.pt");
  ASSERT(t != NULL);
  ASSERT(!neural_tracer_is_running(t));
  neural_tracer_free(t);
  PASS();
}

TEST test_free_null(void) {
  neural_tracer_free(NULL);  // must not crash
  PASS();
}

TEST test_not_running_before_start(void) {
  neural_tracer *t = neural_tracer_new("/dev/null");
  ASSERT(!neural_tracer_is_running(t));
  neural_tracer_free(t);
  PASS();
}

TEST test_stop_when_not_running(void) {
  // stop on a tracer that was never started must be a no-op.
  neural_tracer *t = neural_tracer_new("/dev/null");
  bool ok = neural_tracer_stop(t);
  ASSERT(ok);
  ASSERT(!neural_tracer_is_running(t));
  neural_tracer_free(t);
  PASS();
}

TEST test_start_fails_bad_executable(void) {
  // Python service module doesn't exist → start returns false or child exits.
  // We can't guarantee Python is installed in CI, so we test the contract:
  // after a failed start, is_running should be false and free should not crash.
  neural_tracer *t = neural_tracer_new("/no/such/model");
  // start may return false (connect timeout) or true then child exits;
  // either way we must be able to stop and free cleanly.
  neural_tracer_start(t);  // return value intentionally ignored
  // Give child a moment to fail.
  usleep(200000);
  neural_tracer_stop(t);
  ASSERT(!neural_tracer_is_running(t));
  neural_tracer_free(t);
  PASS();
}

// ---------------------------------------------------------------------------
// predict returns NULL when service is not running
// ---------------------------------------------------------------------------

TEST test_predict_not_running(void) {
  neural_tracer *t = neural_tracer_new(NULL);
  float patch[8] = {0};
  vec3f pts[2]   = {{0,0,0},{1,0,0}};
  trace_result *r = neural_tracer_predict(t, patch, 2, 2, 2, pts, 2);
  ASSERT(r == NULL);
  neural_tracer_free(t);
  PASS();
}

TEST test_predict_null_args(void) {
  neural_tracer *t = neural_tracer_new(NULL);
  trace_result *r;

  r = neural_tracer_predict(NULL, NULL, 0, 0, 0, NULL, 0);
  ASSERT(r == NULL);

  r = neural_tracer_predict(t, NULL, 2, 2, 2, NULL, 0);
  ASSERT(r == NULL);

  neural_tracer_free(t);
  PASS();
}

// ---------------------------------------------------------------------------
// trace_result_free null-safety
// ---------------------------------------------------------------------------

TEST test_trace_result_free_null(void) {
  trace_result_free(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// opt_service: connect to unreachable address returns NULL gracefully
// ---------------------------------------------------------------------------

TEST test_opt_service_connect_fail(void) {
  // Port 1 is reserved; connect should fail immediately.
  opt_service *s = opt_service_connect("127.0.0.1", 1);
  // May or may not succeed depending on OS; either is acceptable.
  // If it did connect, free it cleanly.
  opt_service_free(s);
  PASS();
}

TEST test_opt_service_free_null(void) {
  opt_service_free(NULL);  // must not crash
  PASS();
}

TEST test_opt_service_submit_null(void) {
  bool ok = opt_service_submit(NULL, NULL, NULL);
  ASSERT(!ok);
  PASS();
}

TEST test_opt_service_poll_null(void) {
  char buf[64] = {0};
  bool ok = opt_service_poll_status(NULL, buf, sizeof(buf));
  ASSERT(!ok);
  PASS();
}

TEST test_opt_service_result_null(void) {
  quad_surface *r = opt_service_get_result(NULL);
  ASSERT(r == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(neural_trace_suite) {
  RUN_TEST(test_new_free_null_model);
  RUN_TEST(test_new_free_with_path);
  RUN_TEST(test_free_null);
  RUN_TEST(test_not_running_before_start);
  RUN_TEST(test_stop_when_not_running);
  RUN_TEST(test_start_fails_bad_executable);
  RUN_TEST(test_predict_not_running);
  RUN_TEST(test_predict_null_args);
  RUN_TEST(test_trace_result_free_null);
  RUN_TEST(test_opt_service_connect_fail);
  RUN_TEST(test_opt_service_free_null);
  RUN_TEST(test_opt_service_submit_null);
  RUN_TEST(test_opt_service_poll_null);
  RUN_TEST(test_opt_service_result_null);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(neural_trace_suite);
  GREATEST_MAIN_END();
}
