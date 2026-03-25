#pragma once
#include <stdbool.h>

// ---------------------------------------------------------------------------
// tool_runner — launch CLI tools from the GUI, capture output line-by-line.
//
// Usage:
//   tool_runner *r = tool_runner_new();
//   int job = tool_runner_exec(r, "volatile convert ...", my_cb, ctx);
//   while (tool_runner_is_running(r, job))
//     tool_runner_poll(r);   // call from main loop
//   int rc = tool_runner_exit_code(r, job);
//   tool_runner_free(r);
//
// Implementation: fork+exec via /bin/sh, pipe stdout+stderr merged, read in
// poll loop (non-blocking).  Output lines stored in a ring buffer per job.
// ---------------------------------------------------------------------------

typedef struct tool_runner tool_runner;

// Called once per output line (null-terminated, no trailing newline).
// Invoked from tool_runner_poll() in the caller's thread.
typedef void (*tool_output_fn)(const char *line, void *ctx);

tool_runner *tool_runner_new(void);
void         tool_runner_free(tool_runner *r);

// Launch a shell command asynchronously. Returns a job_id >= 0 on success,
// or -1 on fork/pipe failure.
int  tool_runner_exec(tool_runner *r, const char *command,
                      tool_output_fn on_output, void *ctx);

// Returns true if the job is still running.
bool tool_runner_is_running(const tool_runner *r, int job_id);

// Returns exit code once job finishes; -1 if still running or invalid id.
int  tool_runner_exit_code(const tool_runner *r, int job_id);

// Send SIGTERM to a running job.
void tool_runner_cancel(tool_runner *r, int job_id);

// Read pending output from all running jobs; invoke callbacks for each line.
// Call from the main/render loop. Returns total lines dispatched.
int  tool_runner_poll(tool_runner *r);

// Number of jobs currently running.
int  tool_runner_active_count(const tool_runner *r);
