#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

// init/shutdown
void profile_init(void);
void profile_shutdown(void);
void profile_enable(bool on);   // runtime toggle
bool profile_enabled(void);

// scoped timing (nest-safe, thread-safe)
void profile_begin(const char *name);  // start a named scope
void profile_end(void);                // end most recent scope

// convenience macro
#define PROFILE_SCOPE(name) \
  for (int _p = (profile_begin(name), 1); _p; _p = (profile_end(), 0))

// counters (atomic, always-on even when profiling disabled)
void    profile_counter_inc(const char *name);
void    profile_counter_add(const char *name, int64_t delta);
int64_t profile_counter_get(const char *name);

// frame boundary marker
void profile_frame_begin(void);
void profile_frame_end(void);

// query recent data
typedef struct {
  const char *name;
  double      total_ms;
  double      avg_ms;
  double      max_ms;
  int64_t     call_count;
} profile_entry;

// get top N hotspots sorted by total_ms
int profile_top_entries(profile_entry *out, int max_entries);

// frame stats
typedef struct {
  double  frame_time_ms;
  double  render_time_ms;
  int64_t cache_hits, cache_misses;
  int64_t chunks_loaded;
} profile_frame_stats;

bool profile_last_frame_stats(profile_frame_stats *out);

// export
bool profile_export_json(const char *path);  // Chrome trace format
void profile_print_summary(FILE *out);       // human-readable
void profile_reset(void);                    // clear all data
