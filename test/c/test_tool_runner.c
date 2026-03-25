#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include "greatest.h"
#include "gui/tool_runner.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

typedef struct {
  char lines[16][256];
  int  count;
} capture_t;

static void capture_cb(const char *line, void *ctx) {
  capture_t *c = ctx;
  if (c->count < 16) {
    strncpy(c->lines[c->count], line, 255);
    c->count++;
  }
}

// Drain the runner until the job finishes (with a spin limit for safety).
static void drain(tool_runner *r, int job_id) {
  for (int i = 0; i < 10000 && tool_runner_is_running(r, job_id); i++) {
    tool_runner_poll(r);
    if (tool_runner_is_running(r, job_id)) usleep(1000);
  }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_echo_output(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  capture_t cap = {0};
  int job = tool_runner_exec(r, "echo hello", capture_cb, &cap);
  ASSERT(job >= 0);

  drain(r, job);

  ASSERT_EQ(0, tool_runner_exit_code(r, job));
  ASSERT_EQ(1, cap.count);
  ASSERT_STR_EQ("hello", cap.lines[0]);

  tool_runner_free(r);
  PASS();
}

TEST test_multiline_output(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  capture_t cap = {0};
  int job = tool_runner_exec(r, "printf 'line1\\nline2\\nline3\\n'", capture_cb, &cap);
  ASSERT(job >= 0);

  drain(r, job);

  ASSERT_EQ(0, tool_runner_exit_code(r, job));
  ASSERT_EQ(3, cap.count);
  ASSERT_STR_EQ("line1", cap.lines[0]);
  ASSERT_STR_EQ("line2", cap.lines[1]);
  ASSERT_STR_EQ("line3", cap.lines[2]);

  tool_runner_free(r);
  PASS();
}

TEST test_exit_code(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  int job = tool_runner_exec(r, "exit 42", NULL, NULL);
  ASSERT(job >= 0);
  drain(r, job);
  ASSERT_EQ(42, tool_runner_exit_code(r, job));

  tool_runner_free(r);
  PASS();
}

TEST test_is_running_then_done(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  int job = tool_runner_exec(r, "sleep 0.1", NULL, NULL);
  ASSERT(job >= 0);

  // Immediately after exec, job should be running
  ASSERT(tool_runner_is_running(r, job));

  drain(r, job);

  ASSERT(!tool_runner_is_running(r, job));
  ASSERT_EQ(0, tool_runner_exit_code(r, job));

  tool_runner_free(r);
  PASS();
}

TEST test_cancel(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  int job = tool_runner_exec(r, "sleep 10", NULL, NULL);
  ASSERT(job >= 0);
  ASSERT(tool_runner_is_running(r, job));

  tool_runner_cancel(r, job);
  drain(r, job);

  // After cancel the job must have finished (non-zero exit)
  ASSERT(!tool_runner_is_running(r, job));

  tool_runner_free(r);
  PASS();
}

TEST test_active_count(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  ASSERT_EQ(0, tool_runner_active_count(r));

  int j1 = tool_runner_exec(r, "sleep 10", NULL, NULL);
  int j2 = tool_runner_exec(r, "sleep 10", NULL, NULL);
  ASSERT(j1 >= 0 && j2 >= 0);
  ASSERT_EQ(2, tool_runner_active_count(r));

  tool_runner_cancel(r, j1);
  tool_runner_cancel(r, j2);
  drain(r, j1);
  drain(r, j2);
  ASSERT_EQ(0, tool_runner_active_count(r));

  tool_runner_free(r);
  PASS();
}

TEST test_stderr_captured(void) {
  tool_runner *r = tool_runner_new();
  ASSERT_NEQ(NULL, r);

  capture_t cap = {0};
  int job = tool_runner_exec(r, "echo err_line >&2", capture_cb, &cap);
  ASSERT(job >= 0);
  drain(r, job);

  ASSERT_EQ(1, cap.count);
  ASSERT_STR_EQ("err_line", cap.lines[0]);

  tool_runner_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(tool_runner_suite) {
  RUN_TEST(test_echo_output);
  RUN_TEST(test_multiline_output);
  RUN_TEST(test_exit_code);
  RUN_TEST(test_is_running_then_done);
  RUN_TEST(test_cancel);
  RUN_TEST(test_active_count);
  RUN_TEST(test_stderr_captured);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(tool_runner_suite);
  GREATEST_MAIN_END();
}
