#include "greatest.h"
#include "gui/window_range.h"

#include <math.h>
#include <stdint.h>

static int floats_close(float a, float b, float eps) {
  float d = a - b;
  return (d < 0 ? -d : d) <= eps;
}

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

TEST test_init_defaults(void) {
  window_range_state s;
  window_range_init(&s);
  ASSERT(floats_close(s.low,    0.0f, 1e-6f));
  ASSERT(floats_close(s.high,   1.0f, 1e-6f));
  ASSERT(floats_close(s.window, 1.0f, 1e-6f));
  ASSERT(floats_close(s.level,  0.5f, 1e-6f));
  ASSERT(!s.auto_range);
  ASSERT_EQ(0, s.cmap_id);  // CMAP_GRAYSCALE
  PASS();
}

TEST test_init_null(void) {
  window_range_init(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// set
// ---------------------------------------------------------------------------

TEST test_set_basic(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.2f, 0.8f);
  ASSERT(floats_close(s.low,    0.2f, 1e-5f));
  ASSERT(floats_close(s.high,   0.8f, 1e-5f));
  ASSERT(floats_close(s.window, 0.6f, 1e-5f));
  ASSERT(floats_close(s.level,  0.5f, 1e-5f));
  PASS();
}

TEST test_set_swaps_inverted(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.9f, 0.1f);
  ASSERT(s.low < s.high);
  PASS();
}

TEST test_set_clamps_to_zero_one(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, -0.5f, 1.5f);
  ASSERT(s.low  >= 0.0f);
  ASSERT(s.high <= 1.0f);
  PASS();
}

TEST test_set_derives_window_level(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.3f, 0.7f);
  ASSERT(floats_close(s.window, s.high - s.low,           1e-6f));
  ASSERT(floats_close(s.level,  (s.low + s.high) * 0.5f, 1e-6f));
  PASS();
}

TEST test_set_minimum_separation(void) {
  window_range_state s;
  window_range_init(&s);
  // Equal low and high should produce a tiny but positive window.
  window_range_set(&s, 0.5f, 0.5f);
  ASSERT(s.high > s.low);
  ASSERT(s.window > 0.0f);
  PASS();
}

TEST test_set_null(void) {
  window_range_set(NULL, 0.2f, 0.8f);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// auto
// ---------------------------------------------------------------------------

TEST test_auto_sets_auto_range_flag(void) {
  window_range_state s;
  window_range_init(&s);
  ASSERT(!s.auto_range);
  window_range_auto(&s, 0.0f, 1.0f, 0.05f, 0.95f);
  ASSERT(s.auto_range);
  PASS();
}

TEST test_auto_uses_percentile_range(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_auto(&s, 0.0f, 1.0f, 0.1f, 0.9f);
  ASSERT(floats_close(s.low,  0.1f, 1e-5f));
  ASSERT(floats_close(s.high, 0.9f, 1e-5f));
  PASS();
}

TEST test_auto_fallback_to_data_range(void) {
  window_range_state s;
  window_range_init(&s);
  // Degenerate percentile range -> falls back to data_min/data_max.
  window_range_auto(&s, 0.2f, 0.7f, 0.5f, 0.5f);
  ASSERT(floats_close(s.low,  0.2f, 1e-5f));
  ASSERT(floats_close(s.high, 0.7f, 1e-5f));
  PASS();
}

TEST test_auto_fallback_full_range(void) {
  window_range_state s;
  window_range_init(&s);
  // Both percentile and data range degenerate -> [0,1].
  window_range_auto(&s, 0.5f, 0.5f, 0.5f, 0.5f);
  ASSERT(s.high > s.low);
  PASS();
}

TEST test_auto_null(void) {
  window_range_auto(NULL, 0.0f, 1.0f, 0.05f, 0.95f);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

TEST test_apply_full_range(void) {
  window_range_state s;
  window_range_init(&s);
  // With [0,1] window, output == input.
  ASSERT_EQ(  0, window_range_apply(&s,   0));
  ASSERT_EQ(128, window_range_apply(&s, 128));
  ASSERT_EQ(255, window_range_apply(&s, 255));
  PASS();
}

TEST test_apply_clamps_below_window(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.5f, 1.0f);
  // value 0 is below the window -> maps to 0
  ASSERT_EQ(0, window_range_apply(&s, 0));
  PASS();
}

TEST test_apply_clamps_above_window(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.0f, 0.5f);
  // value 255 is above the window -> maps to 255
  ASSERT_EQ(255, window_range_apply(&s, 255));
  PASS();
}

TEST test_apply_midpoint_mapping(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.0f, 0.5f);
  // input 64 (~= 0.25 normalised) is midpoint of [0,0.5] -> should map ~= 128
  uint8_t out = window_range_apply(&s, 64);
  ASSERT(out >= 125 && out <= 131);
  PASS();
}

TEST test_apply_null_returns_passthrough(void) {
  // With NULL state, returns value unchanged.
  ASSERT_EQ(42, window_range_apply(NULL, 42));
  PASS();
}

TEST test_apply_monotone(void) {
  window_range_state s;
  window_range_init(&s);
  window_range_set(&s, 0.1f, 0.9f);
  uint8_t prev = window_range_apply(&s, 0);
  for (int v = 1; v <= 255; v++) {
    uint8_t cur = window_range_apply(&s, (uint8_t)v);
    ASSERT(cur >= prev);
    prev = cur;
  }
  PASS();
}

// ---------------------------------------------------------------------------
// render — null ctx must not crash
// ---------------------------------------------------------------------------

TEST test_render_null_ctx(void) {
  window_range_state s;
  window_range_init(&s);
  bool changed = window_range_render(&s, NULL);
  ASSERT(!changed);
  PASS();
}

TEST test_render_null_state(void) {
  bool changed = window_range_render(NULL, NULL);
  ASSERT(!changed);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(window_range_suite) {
  RUN_TEST(test_init_defaults);
  RUN_TEST(test_init_null);
  RUN_TEST(test_set_basic);
  RUN_TEST(test_set_swaps_inverted);
  RUN_TEST(test_set_clamps_to_zero_one);
  RUN_TEST(test_set_derives_window_level);
  RUN_TEST(test_set_minimum_separation);
  RUN_TEST(test_set_null);
  RUN_TEST(test_auto_sets_auto_range_flag);
  RUN_TEST(test_auto_uses_percentile_range);
  RUN_TEST(test_auto_fallback_to_data_range);
  RUN_TEST(test_auto_fallback_full_range);
  RUN_TEST(test_auto_null);
  RUN_TEST(test_apply_full_range);
  RUN_TEST(test_apply_clamps_below_window);
  RUN_TEST(test_apply_clamps_above_window);
  RUN_TEST(test_apply_midpoint_mapping);
  RUN_TEST(test_apply_null_returns_passthrough);
  RUN_TEST(test_apply_monotone);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_null_state);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(window_range_suite);
  GREATEST_MAIN_END();
}
