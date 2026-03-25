#define _POSIX_C_SOURCE 200809L
#include "greatest.h"
#include "core/profile.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void sleep_ms(int ms) {
  struct timespec ts = { .tv_sec = 0, .tv_nsec = (long)ms * 1000000L };
  nanosleep(&ts, NULL);
}

// ---------------------------------------------------------------------------
// Enable / disable
// ---------------------------------------------------------------------------

TEST test_enable_disable(void) {
  profile_init();

  profile_enable(false);
  ASSERT_EQ(false, profile_enabled());

  profile_enable(true);
  ASSERT_EQ(true, profile_enabled());

  profile_enable(false);
  profile_shutdown();
  PASS();
}

TEST test_begin_end_when_disabled_no_crash(void) {
  profile_init();
  profile_enable(false);

  // must not crash even when disabled
  for (int i = 0; i < 100; i++) {
    profile_begin("noop");
    profile_end();
  }

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// Scoped timing
// ---------------------------------------------------------------------------

TEST test_timing_records_duration(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  profile_begin("sleep_scope");
  sleep_ms(5);
  profile_end();

  profile_entry entries[8];
  int n = profile_top_entries(entries, 8);
  ASSERT(n >= 1);

  // Find our scope
  int found = -1;
  for (int i = 0; i < n; i++) {
    if (strcmp(entries[i].name, "sleep_scope") == 0) { found = i; break; }
  }
  ASSERT(found >= 0);
  ASSERT(entries[found].total_ms >= 4.0);   // slept at least 5 ms, allow 4 for jitter
  ASSERT(entries[found].call_count == 1);

  profile_shutdown();
  PASS();
}

TEST test_timing_multiple_calls(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  for (int i = 0; i < 5; i++) {
    profile_begin("repeat");
    sleep_ms(2);
    profile_end();
  }

  profile_entry entries[8];
  int n = profile_top_entries(entries, 8);
  int found = -1;
  for (int i = 0; i < n; i++) {
    if (strcmp(entries[i].name, "repeat") == 0) { found = i; break; }
  }
  ASSERT(found >= 0);
  ASSERT_EQ(5, entries[found].call_count);
  ASSERT(entries[found].avg_ms >= 1.0);

  profile_shutdown();
  PASS();
}

TEST test_nested_scopes(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  profile_begin("outer");
    profile_begin("inner");
    sleep_ms(2);
    profile_end();
  sleep_ms(2);
  profile_end();

  profile_entry entries[8];
  int n = profile_top_entries(entries, 8);

  bool has_outer = false, has_inner = false;
  for (int i = 0; i < n; i++) {
    if (strcmp(entries[i].name, "outer") == 0) has_outer = true;
    if (strcmp(entries[i].name, "inner") == 0) has_inner = true;
  }
  ASSERT(has_outer);
  ASSERT(has_inner);

  profile_shutdown();
  PASS();
}

TEST test_profile_scope_macro(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  PROFILE_SCOPE("macro_scope") {
    sleep_ms(2);
  }

  profile_entry entries[8];
  int n = profile_top_entries(entries, 8);
  bool found = false;
  for (int i = 0; i < n; i++) {
    if (strcmp(entries[i].name, "macro_scope") == 0) { found = true; break; }
  }
  ASSERT(found);

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

TEST test_counter_inc(void) {
  profile_init();

  profile_counter_inc("hits");
  profile_counter_inc("hits");
  profile_counter_inc("hits");

  ASSERT_EQ(3, profile_counter_get("hits"));

  profile_shutdown();
  PASS();
}

TEST test_counter_add(void) {
  profile_init();

  profile_counter_add("bytes", 1024);
  profile_counter_add("bytes", 512);

  ASSERT_EQ(1536, profile_counter_get("bytes"));

  profile_shutdown();
  PASS();
}

TEST test_counter_missing_returns_zero(void) {
  profile_init();
  ASSERT_EQ(0, profile_counter_get("nonexistent_xyzzy"));
  profile_shutdown();
  PASS();
}

TEST test_counter_independent(void) {
  profile_init();
  profile_counter_inc("a");
  profile_counter_inc("a");
  profile_counter_inc("b");

  ASSERT_EQ(2, profile_counter_get("a"));
  ASSERT_EQ(1, profile_counter_get("b"));

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// Frame stats
// ---------------------------------------------------------------------------

TEST test_frame_stats(void) {
  profile_init();
  profile_enable(true);

  profile_frame_begin();
  sleep_ms(5);
  profile_frame_end();

  profile_frame_stats fs;
  bool ok = profile_last_frame_stats(&fs);
  ASSERT(ok);
  ASSERT(fs.frame_time_ms >= 4.0);

  profile_shutdown();
  PASS();
}

TEST test_frame_stats_no_frame_returns_false(void) {
  profile_init();
  profile_frame_stats fs;
  bool ok = profile_last_frame_stats(&fs);
  ASSERT_EQ(false, ok);
  profile_shutdown();
  PASS();
}

TEST test_frame_stats_cache_counters(void) {
  profile_init();
  profile_enable(true);

  profile_counter_add("cache_hits",   10);
  profile_counter_add("cache_misses",  3);
  profile_counter_add("chunks_loaded", 7);

  profile_frame_begin();
  profile_frame_end();

  profile_frame_stats fs;
  ASSERT(profile_last_frame_stats(&fs));
  ASSERT_EQ(10, fs.cache_hits);
  ASSERT_EQ(3,  fs.cache_misses);
  ASSERT_EQ(7,  fs.chunks_loaded);

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// top_entries sorted by total_ms
// ---------------------------------------------------------------------------

TEST test_top_entries_sorted(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  // fast scope
  profile_begin("fast");
  sleep_ms(1);
  profile_end();

  // slow scope
  profile_begin("slow");
  sleep_ms(10);
  profile_end();

  profile_entry entries[4];
  int n = profile_top_entries(entries, 4);
  ASSERT(n >= 2);
  // First entry should have more total_ms than second
  ASSERT(entries[0].total_ms >= entries[1].total_ms);

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

TEST test_reset_clears_data(void) {
  profile_init();
  profile_enable(true);

  profile_begin("before_reset");
  sleep_ms(2);
  profile_end();
  profile_counter_inc("rc");

  profile_reset();

  profile_entry entries[4];
  int n = profile_top_entries(entries, 4);
  // after reset, ring is empty
  ASSERT_EQ(0, n);
  ASSERT_EQ(0, profile_counter_get("rc"));

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

TEST test_export_json(void) {
  profile_init();
  profile_enable(true);
  profile_reset();

  profile_begin("json_scope");
  sleep_ms(1);
  profile_end();
  profile_counter_inc("json_counter");

  const char *path = "/tmp/test_profile_export.json";
  ASSERT(profile_export_json(path));

  FILE *f = fopen(path, "r");
  ASSERT(f != NULL);
  char buf[4096] = {0};
  size_t nr = fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);
  ASSERT(nr > 0);
  ASSERT(strstr(buf, "traceEvents") != NULL);
  ASSERT(strstr(buf, "json_scope") != NULL);
  ASSERT(strstr(buf, "json_counter") != NULL);

  profile_shutdown();
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_profile) {
  RUN_TEST(test_enable_disable);
  RUN_TEST(test_begin_end_when_disabled_no_crash);
  RUN_TEST(test_timing_records_duration);
  RUN_TEST(test_timing_multiple_calls);
  RUN_TEST(test_nested_scopes);
  RUN_TEST(test_profile_scope_macro);
  RUN_TEST(test_counter_inc);
  RUN_TEST(test_counter_add);
  RUN_TEST(test_counter_missing_returns_zero);
  RUN_TEST(test_counter_independent);
  RUN_TEST(test_frame_stats);
  RUN_TEST(test_frame_stats_no_frame_returns_false);
  RUN_TEST(test_frame_stats_cache_counters);
  RUN_TEST(test_top_entries_sorted);
  RUN_TEST(test_reset_clears_data);
  RUN_TEST(test_export_json);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_profile);
  GREATEST_MAIN_END();
}
