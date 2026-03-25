// Request POSIX.1-2008 interfaces (clock_gettime, gmtime_r, flockfile, etc.)
#define _POSIX_C_SOURCE 200809L

#include "core/log.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

static _Atomic(int) g_level = LOG_INFO;
static FILE *g_log_file = NULL;
static log_callback_fn g_callback = NULL;
static void *g_callback_ctx = NULL;

// ---------------------------------------------------------------------------
// ANSI color codes (used when stderr is a tty)
// ---------------------------------------------------------------------------

#define COLOR_RESET  "\x1b[0m"
#define COLOR_GREY   "\x1b[90m"
#define COLOR_CYAN   "\x1b[36m"
#define COLOR_GREEN  "\x1b[32m"
#define COLOR_YELLOW "\x1b[33m"
#define COLOR_RED    "\x1b[31m"
#define COLOR_BOLD   "\x1b[1m"

static const char *level_color(log_level_t level) {
  switch (level) {
    case LOG_DEBUG: return COLOR_GREY;
    case LOG_INFO:  return COLOR_GREEN;
    case LOG_WARN:  return COLOR_YELLOW;
    case LOG_ERROR: return COLOR_RED;
    case LOG_FATAL: return COLOR_BOLD COLOR_RED;
    default:        return COLOR_RESET;
  }
}

static const char *level_str(log_level_t level) {
  switch (level) {
    case LOG_DEBUG: return "DEBUG";
    case LOG_INFO:  return "INFO ";
    case LOG_WARN:  return "WARN ";
    case LOG_ERROR: return "ERROR";
    case LOG_FATAL: return "FATAL";
    default:        return "?????";
  }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void log_set_level(log_level_t level) {
  atomic_store(&g_level, (int)level);
}

log_level_t log_get_level(void) {
  return (log_level_t)atomic_load(&g_level);
}

void log_set_file(FILE *f) {
  g_log_file = f;
}

void log_set_callback(log_callback_fn fn, void *ctx) {
  g_callback = fn;
  g_callback_ctx = ctx;
}

void log_msg(log_level_t level, const char *file, int line, const char *fmt, ...) {
  if ((int)level < atomic_load(&g_level)) return;

  // --- build timestamp ---
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tm_buf;
  gmtime_r(&ts.tv_sec, &tm_buf);

  char ts_str[32];
  strftime(ts_str, sizeof(ts_str), "%Y-%m-%dT%H:%M:%S", &tm_buf);

  // --- format caller message ---
  char msg[1024];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(msg, sizeof(msg), fmt, ap);
  va_end(ap);

  // --- strip path prefix from file for readability ---
  const char *short_file = strrchr(file, '/');
  short_file = short_file ? short_file + 1 : file;

  // --- human-readable stderr output ---
  bool use_color = isatty(fileno(stderr));

  flockfile(stderr);
  if (use_color) {
    fprintf(stderr, "%s%s.%03ldZ%s %s%s%s %s:%d: %s\n",
      COLOR_GREY, ts_str, ts.tv_nsec / 1000000L, COLOR_RESET,
      level_color(level), level_str(level), COLOR_RESET,
      short_file, line, msg);
  } else {
    fprintf(stderr, "%s.%03ldZ %s %s:%d: %s\n",
      ts_str, ts.tv_nsec / 1000000L,
      level_str(level), short_file, line, msg);
  }
  funlockfile(stderr);

  // --- invoke GUI callback (e.g. log_console) ---
  if (g_callback) {
    g_callback(g_callback_ctx, level, short_file, line, msg);
  }

  // --- structured JSON output to log file ---
  if (g_log_file) {
    // escape backslashes and double-quotes in msg
    char escaped[2048];
    size_t j = 0;
    for (size_t i = 0; msg[i] && j < sizeof(escaped) - 2; i++) {
      if (msg[i] == '"' || msg[i] == '\\') escaped[j++] = '\\';
      escaped[j++] = msg[i];
    }
    escaped[j] = '\0';

    flockfile(g_log_file);
    fprintf(g_log_file,
      "{\"ts\":\"%s.%03ldZ\",\"level\":\"%s\",\"file\":\"%s\",\"line\":%d,\"msg\":\"%s\"}\n",
      ts_str, ts.tv_nsec / 1000000L,
      level_str(level), short_file, line, escaped);
    fflush(g_log_file);
    funlockfile(g_log_file);
  }
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

const char *volatile_version(void) {
  return "0.1.0";
}
