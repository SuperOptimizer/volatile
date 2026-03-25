#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <time.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Log levels
// ---------------------------------------------------------------------------
typedef enum {
  LOG_DEBUG = 0, LOG_INFO, LOG_WARN, LOG_ERROR, LOG_FATAL
} log_level_t;

// global minimum log level (set at startup; default LOG_INFO)
void log_set_level(log_level_t level);
log_level_t log_get_level(void);

// optional JSON log file (NULL to disable)
void log_set_file(FILE *f);

// core logging function — prefer the macros below
void log_msg(log_level_t level, const char *file, int line, const char *fmt, ...);

// ---------------------------------------------------------------------------
// Convenience macros
// ---------------------------------------------------------------------------
#define LOG_DEBUG(...) log_msg(LOG_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  log_msg(LOG_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  log_msg(LOG_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) log_msg(LOG_ERROR, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_FATAL(...) do { log_msg(LOG_FATAL, __FILE__, __LINE__, __VA_ARGS__); abort(); } while(0)

// ---------------------------------------------------------------------------
// Assertions
// ---------------------------------------------------------------------------

// ASSERT — active in debug builds, compiled out in release (-DNDEBUG)
#ifndef NDEBUG
#define ASSERT(expr, ...) do { \
  if (!(expr)) { \
    LOG_FATAL("assertion failed: %s " __VA_OPT__(,) __VA_ARGS__, #expr); \
  } \
} while(0)
#else
#define ASSERT(expr, ...) ((void)0)
#endif

// REQUIRE — always-on assertion (fires in both debug and release)
#define REQUIRE(expr, ...) do { \
  if (!(expr)) { \
    LOG_FATAL("requirement failed: %s " __VA_OPT__(,) __VA_ARGS__, #expr); \
  } \
} while(0)

// ---------------------------------------------------------------------------
// Log callback — called after every log_msg that passes the level filter.
// fn receives the same arguments as log_msg plus the user-supplied ctx pointer.
// Pass fn=NULL to remove the callback.
// ---------------------------------------------------------------------------
typedef void (*log_callback_fn)(void *ctx, log_level_t level, const char *file, int line, const char *msg);
void log_set_callback(log_callback_fn fn, void *ctx);

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------
#define VOLATILE_VERSION_MAJOR 0
#define VOLATILE_VERSION_MINOR 1
#define VOLATILE_VERSION_PATCH 0

const char *volatile_version(void);
