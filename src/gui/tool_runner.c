#define _POSIX_C_SOURCE 200809L
#include "gui/tool_runner.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_JOBS     16
#define LINE_BUF_CAP 4096   // per-job partial-line accumulator

// ---------------------------------------------------------------------------
// Job state
// ---------------------------------------------------------------------------

typedef enum { JOB_FREE, JOB_RUNNING, JOB_DONE } job_state_t;

typedef struct {
  job_state_t     state;
  int             id;
  pid_t           pid;
  int             fd;          // read end of stdout+stderr pipe
  int             exit_code;
  tool_output_fn  on_output;
  void           *ctx;
  char            linebuf[LINE_BUF_CAP]; // partial line accumulator
  int             linelen;
} job_t;

struct tool_runner {
  job_t jobs[MAX_JOBS];
  int   next_id;   // monotonically increasing job id
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static job_t *find_job(tool_runner *r, int job_id) {
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].state != JOB_FREE && r->jobs[i].id == job_id)
      return &r->jobs[i];
  }
  return NULL;
}

static job_t *alloc_job(tool_runner *r) {
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].state == JOB_FREE) return &r->jobs[i];
  }
  return NULL;
}

// Flush accumulated line (even if no trailing newline — used at EOF).
static void flush_line(job_t *j) {
  if (j->linelen == 0) return;
  j->linebuf[j->linelen] = '\0';
  if (j->on_output) j->on_output(j->linebuf, j->ctx);
  j->linelen = 0;
}

// Feed raw bytes into the line accumulator; dispatch complete lines.
// Returns number of lines dispatched.
static int feed_bytes(job_t *j, const char *buf, int n) {
  int dispatched = 0;
  for (int i = 0; i < n; i++) {
    char c = buf[i];
    if (c == '\n' || c == '\r') {
      flush_line(j);
      dispatched++;
    } else {
      if (j->linelen < LINE_BUF_CAP - 1)
        j->linebuf[j->linelen++] = c;
    }
  }
  return dispatched;
}

// Reap a finished job: collect exit code, flush remaining output, mark done.
static void reap_job(job_t *j) {
  // Drain any remaining bytes from the pipe before closing.
  char tmp[256];
  int n;
  while ((n = (int)read(j->fd, tmp, sizeof(tmp))) > 0)
    feed_bytes(j, tmp, n);
  flush_line(j);  // partial last line without newline

  close(j->fd);
  j->fd = -1;

  int status = 0;
  if (waitpid(j->pid, &status, WNOHANG) > 0) {
    j->exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  } else {
    // Process may not have exited yet; do a blocking wait.
    waitpid(j->pid, &status, 0);
    j->exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
  }
  j->state = JOB_DONE;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

tool_runner *tool_runner_new(void) {
  tool_runner *r = calloc(1, sizeof(*r));
  if (!r) return NULL;
  for (int i = 0; i < MAX_JOBS; i++) r->jobs[i].fd = -1;
  r->next_id = 1;
  return r;
}

void tool_runner_free(tool_runner *r) {
  if (!r) return;
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].state == JOB_RUNNING) {
      kill(r->jobs[i].pid, SIGTERM);
      reap_job(&r->jobs[i]);
    } else if (r->jobs[i].fd >= 0) {
      close(r->jobs[i].fd);
    }
  }
  free(r);
}

int tool_runner_exec(tool_runner *r, const char *command,
                     tool_output_fn on_output, void *ctx) {
  if (!r || !command) return -1;

  job_t *j = alloc_job(r);
  if (!j) return -1;

  int pipefd[2];
  if (pipe(pipefd) != 0) return -1;

  // Make read end non-blocking so poll() doesn't stall.
  int flags = fcntl(pipefd[0], F_GETFL, 0);
  fcntl(pipefd[0], F_SETFL, flags | O_NONBLOCK);

  pid_t pid = fork();
  if (pid < 0) {
    close(pipefd[0]);
    close(pipefd[1]);
    return -1;
  }

  if (pid == 0) {
    // Child: redirect stdout + stderr to write end of pipe.
    close(pipefd[0]);
    dup2(pipefd[1], STDOUT_FILENO);
    dup2(pipefd[1], STDERR_FILENO);
    close(pipefd[1]);
    execl("/bin/sh", "sh", "-c", command, (char *)NULL);
    _exit(127);
  }

  // Parent
  close(pipefd[1]);

  j->state     = JOB_RUNNING;
  j->id        = r->next_id++;
  j->pid       = pid;
  j->fd        = pipefd[0];
  j->exit_code = -1;
  j->on_output = on_output;
  j->ctx       = ctx;
  j->linelen   = 0;
  return j->id;
}

bool tool_runner_is_running(const tool_runner *r, int job_id) {
  if (!r) return false;
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].state == JOB_RUNNING && r->jobs[i].id == job_id)
      return true;
  }
  return false;
}

int tool_runner_exit_code(const tool_runner *r, int job_id) {
  if (!r) return -1;
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].id == job_id && r->jobs[i].state == JOB_DONE)
      return r->jobs[i].exit_code;
  }
  return -1;
}

void tool_runner_cancel(tool_runner *r, int job_id) {
  if (!r) return;
  job_t *j = find_job(r, job_id);
  if (j && j->state == JOB_RUNNING)
    kill(j->pid, SIGTERM);
}

int tool_runner_poll(tool_runner *r) {
  if (!r) return 0;
  int dispatched = 0;

  for (int i = 0; i < MAX_JOBS; i++) {
    job_t *j = &r->jobs[i];
    if (j->state != JOB_RUNNING) continue;

    // Read all available bytes (non-blocking).
    char buf[512];
    int n;
    bool pipe_eof = false;
    while ((n = (int)read(j->fd, buf, sizeof(buf))) > 0) {
      dispatched += feed_bytes(j, buf, n);
    }
    if (n == 0) {
      pipe_eof = true;  // EOF: child closed write end
    } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
      pipe_eof = true;  // unexpected error, treat as EOF
    }

    if (pipe_eof) {
      reap_job(j);
    } else {
      // Check if child has already exited (pipe still open but no data left).
      int status = 0;
      pid_t ret = waitpid(j->pid, &status, WNOHANG);
      if (ret == j->pid) {
        // Child exited; drain pipe one more time then close.
        reap_job(j);
        // Override exit code from what we just waited for.
        j->exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
      }
    }
  }
  return dispatched;
}

int tool_runner_active_count(const tool_runner *r) {
  if (!r) return 0;
  int count = 0;
  for (int i = 0; i < MAX_JOBS; i++) {
    if (r->jobs[i].state == JOB_RUNNING) count++;
  }
  return count;
}
