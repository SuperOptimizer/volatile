#include "greatest.h"
#include "render/composite.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static float s0[4] = {0.1f, 0.5f, 0.9f, 0.3f};
static float s1[4] = {0.8f, 0.2f, 0.4f, 0.7f};
static float s2[4] = {0.3f, 0.6f, 0.1f, 0.5f};
static const float *slices3[3];

static void init_slices(void) {
  slices3[0] = s0;
  slices3[1] = s1;
  slices3[2] = s2;
}

// ---------------------------------------------------------------------------
// composite_params_default
// ---------------------------------------------------------------------------

TEST test_defaults(void) {
  composite_params p;
  composite_params_default(&p);
  ASSERT_EQ(COMPOSITE_MAX, p.mode);
  ASSERT_IN_RANGE(1.0f, p.alpha_opacity, 1e-6f);
  ASSERT_IN_RANGE(1.0f, p.extinction, 1e-6f);
  PASS();
}

// ---------------------------------------------------------------------------
// MAX
// ---------------------------------------------------------------------------

TEST test_max_pixel(void) {
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MAX;
  float vals[] = {0.1f, 0.9f, 0.5f};
  ASSERT_IN_RANGE(0.9f, composite_pixel(vals, 3, &p), 1e-6f);
  PASS();
}

TEST test_max_slices(void) {
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MAX;
  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);
  // expected: max across rows: [0.8, 0.6, 0.9, 0.7]
  ASSERT_IN_RANGE(0.8f, out[0], 1e-6f);
  ASSERT_IN_RANGE(0.6f, out[1], 1e-6f);
  ASSERT_IN_RANGE(0.9f, out[2], 1e-6f);
  ASSERT_IN_RANGE(0.7f, out[3], 1e-6f);
  PASS();
}

// ---------------------------------------------------------------------------
// MIN
// ---------------------------------------------------------------------------

TEST test_min_pixel(void) {
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MIN;
  float vals[] = {0.1f, 0.9f, 0.5f};
  ASSERT_IN_RANGE(0.1f, composite_pixel(vals, 3, &p), 1e-6f);
  PASS();
}

TEST test_min_slices(void) {
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MIN;
  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);
  // expected: min across rows: [0.1, 0.2, 0.1, 0.3]
  ASSERT_IN_RANGE(0.1f, out[0], 1e-6f);
  ASSERT_IN_RANGE(0.2f, out[1], 1e-6f);
  ASSERT_IN_RANGE(0.1f, out[2], 1e-6f);
  ASSERT_IN_RANGE(0.3f, out[3], 1e-6f);
  PASS();
}

// ---------------------------------------------------------------------------
// MEAN
// ---------------------------------------------------------------------------

TEST test_mean_pixel(void) {
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MEAN;
  float vals[] = {1.0f, 2.0f, 3.0f};
  ASSERT_IN_RANGE(2.0f, composite_pixel(vals, 3, &p), 1e-6f);
  PASS();
}

TEST test_mean_slices(void) {
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_MEAN;
  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);
  // expected means: [(0.1+0.8+0.3)/3, (0.5+0.2+0.6)/3, (0.9+0.4+0.1)/3, (0.3+0.7+0.5)/3]
  ASSERT_IN_RANGE((0.1f+0.8f+0.3f)/3.0f, out[0], 1e-5f);
  ASSERT_IN_RANGE((0.5f+0.2f+0.6f)/3.0f, out[1], 1e-5f);
  ASSERT_IN_RANGE((0.9f+0.4f+0.1f)/3.0f, out[2], 1e-5f);
  ASSERT_IN_RANGE((0.3f+0.7f+0.5f)/3.0f, out[3], 1e-5f);
  PASS();
}

// ---------------------------------------------------------------------------
// SUM
// ---------------------------------------------------------------------------

TEST test_sum_pixel(void) {
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_SUM;
  float vals[] = {1.0f, 2.0f, 3.0f};
  ASSERT_IN_RANGE(6.0f, composite_pixel(vals, 3, &p), 1e-6f);
  PASS();
}

TEST test_sum_slices(void) {
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode = COMPOSITE_SUM;
  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);
  ASSERT_IN_RANGE(0.1f+0.8f+0.3f, out[0], 1e-5f);
  ASSERT_IN_RANGE(0.5f+0.2f+0.6f, out[1], 1e-5f);
  PASS();
}

// ---------------------------------------------------------------------------
// ALPHA
// ---------------------------------------------------------------------------

TEST test_alpha_single_opaque(void) {
  // With alpha_opacity=1 and value=alpha_max, first sample is fully opaque
  composite_params p; composite_params_default(&p);
  p.mode         = COMPOSITE_ALPHA;
  p.alpha_min    = 0.0f;
  p.alpha_max    = 1.0f;
  p.alpha_opacity = 1.0f;

  // Single fully-opaque sample: opacity=1, output = (1-0)*1*v = v
  float vals[] = {1.0f};
  float result = composite_pixel(vals, 1, &p);
  ASSERT_IN_RANGE(1.0f, result, 1e-5f);
  PASS();
}

TEST test_alpha_accumulates_front_to_back(void) {
  composite_params p; composite_params_default(&p);
  p.mode          = COMPOSITE_ALPHA;
  p.alpha_min     = 0.0f;
  p.alpha_max     = 1.0f;
  p.alpha_opacity = 0.5f;

  // Two samples, both v=1.0:
  // k=0: opacity=0.5, transmit=1.0, acc_color += 0.5*1.0=0.5, acc_alpha=0.5
  // k=1: opacity=0.5, transmit=0.5, acc_color += 0.25*1.0=0.25, acc_alpha=0.75
  // total acc_color = 0.75
  float vals[] = {1.0f, 1.0f};
  float result = composite_pixel(vals, 2, &p);
  ASSERT_IN_RANGE(0.75f, result, 1e-5f);
  PASS();
}

TEST test_alpha_transparent_zero(void) {
  // v=0 maps to opacity=0, no contribution
  composite_params p; composite_params_default(&p);
  p.mode          = COMPOSITE_ALPHA;
  p.alpha_min     = 0.0f;
  p.alpha_max     = 1.0f;
  p.alpha_opacity = 1.0f;

  float vals[] = {0.0f, 0.0f, 0.0f};
  float result = composite_pixel(vals, 3, &p);
  ASSERT_IN_RANGE(0.0f, result, 1e-6f);
  PASS();
}

TEST test_alpha_slices_consistency(void) {
  // composite_slices and composite_pixel should agree
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode          = COMPOSITE_ALPHA;
  p.alpha_min     = 0.0f;
  p.alpha_max     = 1.0f;
  p.alpha_opacity = 0.4f;

  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);

  for (int i = 0; i < 4; i++) {
    float vals[3] = { s0[i], s1[i], s2[i] };
    float expected = composite_pixel(vals, 3, &p);
    ASSERT_IN_RANGE(expected, out[i], 1e-5f);
  }
  PASS();
}

// ---------------------------------------------------------------------------
// BEER-LAMBERT
// ---------------------------------------------------------------------------

TEST test_beer_lambert_pixel(void) {
  composite_params p; composite_params_default(&p);
  p.mode       = COMPOSITE_BEER_LAMBERT;
  p.extinction = 2.0f;

  float vals[] = {0.5f, 0.3f};  // sum = 0.8
  float expected = expf(-2.0f * 0.8f);
  float result   = composite_pixel(vals, 2, &p);
  ASSERT_IN_RANGE(expected, result, 1e-5f);
  PASS();
}

TEST test_beer_lambert_zero_extinction(void) {
  composite_params p; composite_params_default(&p);
  p.mode       = COMPOSITE_BEER_LAMBERT;
  p.extinction = 0.0f;

  float vals[] = {1.0f, 1.0f, 1.0f};
  // exp(0) = 1 regardless of values
  ASSERT_IN_RANGE(1.0f, composite_pixel(vals, 3, &p), 1e-6f);
  PASS();
}

TEST test_beer_lambert_slices(void) {
  init_slices();
  composite_params p; composite_params_default(&p);
  p.mode       = COMPOSITE_BEER_LAMBERT;
  p.extinction = 1.5f;

  float out[4];
  composite_slices(slices3, 3, out, 4, 1, &p);

  for (int i = 0; i < 4; i++) {
    float sum = s0[i] + s1[i] + s2[i];
    float expected = expf(-1.5f * sum);
    ASSERT_IN_RANGE(expected, out[i], 1e-5f);
  }
  PASS();
}

// ---------------------------------------------------------------------------
// Mode name
// ---------------------------------------------------------------------------

TEST test_mode_names(void) {
  ASSERT_STR_EQ("max",          composite_mode_name(COMPOSITE_MAX));
  ASSERT_STR_EQ("min",          composite_mode_name(COMPOSITE_MIN));
  ASSERT_STR_EQ("mean",         composite_mode_name(COMPOSITE_MEAN));
  ASSERT_STR_EQ("alpha",        composite_mode_name(COMPOSITE_ALPHA));
  ASSERT_STR_EQ("beer_lambert", composite_mode_name(COMPOSITE_BEER_LAMBERT));
  ASSERT_STR_EQ("sum",          composite_mode_name(COMPOSITE_SUM));
  PASS();
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST test_single_slice(void) {
  composite_params p; composite_params_default(&p);
  float data[] = {0.3f, 0.7f};
  const float *s[] = { data };
  float out[2];

  p.mode = COMPOSITE_MAX;
  composite_slices(s, 1, out, 2, 1, &p);
  ASSERT_IN_RANGE(0.3f, out[0], 1e-6f);
  ASSERT_IN_RANGE(0.7f, out[1], 1e-6f);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_composite) {
  RUN_TEST(test_defaults);
  RUN_TEST(test_max_pixel);
  RUN_TEST(test_max_slices);
  RUN_TEST(test_min_pixel);
  RUN_TEST(test_min_slices);
  RUN_TEST(test_mean_pixel);
  RUN_TEST(test_mean_slices);
  RUN_TEST(test_sum_pixel);
  RUN_TEST(test_sum_slices);
  RUN_TEST(test_alpha_single_opaque);
  RUN_TEST(test_alpha_accumulates_front_to_back);
  RUN_TEST(test_alpha_transparent_zero);
  RUN_TEST(test_alpha_slices_consistency);
  RUN_TEST(test_beer_lambert_pixel);
  RUN_TEST(test_beer_lambert_zero_extinction);
  RUN_TEST(test_beer_lambert_slices);
  RUN_TEST(test_mode_names);
  RUN_TEST(test_single_slice);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_composite);
  GREATEST_MAIN_END();
}
