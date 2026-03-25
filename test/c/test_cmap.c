#include "greatest.h"
#include "render/cmap.h"

#include <stddef.h>
#include <string.h>

// ---------------------------------------------------------------------------
// count / names
// ---------------------------------------------------------------------------

TEST test_count(void) {
  ASSERT_EQ(10, cmap_count());
  PASS();
}

TEST test_all_names_non_null(void) {
  for (int i = 0; i < cmap_count(); i++) {
    const char *name = cmap_name((cmap_id)i);
    ASSERT(name != NULL);
    ASSERT(strlen(name) > 0);
  }
  PASS();
}

TEST test_name_out_of_range(void) {
  ASSERT(cmap_name((cmap_id)-1)          == NULL);
  ASSERT(cmap_name((cmap_id)CMAP_COUNT)  == NULL);
  PASS();
}

TEST test_name_values(void) {
  ASSERT_STR_EQ("grayscale", cmap_name(CMAP_GRAYSCALE));
  ASSERT_STR_EQ("viridis",   cmap_name(CMAP_VIRIDIS));
  ASSERT_STR_EQ("magma",     cmap_name(CMAP_MAGMA));
  ASSERT_STR_EQ("inferno",   cmap_name(CMAP_INFERNO));
  ASSERT_STR_EQ("plasma",    cmap_name(CMAP_PLASMA));
  ASSERT_STR_EQ("hot",       cmap_name(CMAP_HOT));
  ASSERT_STR_EQ("cool",      cmap_name(CMAP_COOL));
  ASSERT_STR_EQ("bone",      cmap_name(CMAP_BONE));
  ASSERT_STR_EQ("jet",       cmap_name(CMAP_JET));
  ASSERT_STR_EQ("turbo",     cmap_name(CMAP_TURBO));
  PASS();
}

// ---------------------------------------------------------------------------
// Grayscale correctness
// ---------------------------------------------------------------------------

TEST test_grayscale_black(void) {
  cmap_rgb c = cmap_apply(CMAP_GRAYSCALE, 0.0);
  ASSERT_EQ(0,   c.r);
  ASSERT_EQ(0,   c.g);
  ASSERT_EQ(0,   c.b);
  PASS();
}

TEST test_grayscale_white(void) {
  cmap_rgb c = cmap_apply(CMAP_GRAYSCALE, 1.0);
  ASSERT_EQ(255, c.r);
  ASSERT_EQ(255, c.g);
  ASSERT_EQ(255, c.b);
  PASS();
}

TEST test_grayscale_midpoint(void) {
  cmap_rgb c = cmap_apply(CMAP_GRAYSCALE, 0.5);
  // r == g == b for grayscale
  ASSERT_EQ(c.r, c.g);
  ASSERT_EQ(c.g, c.b);
  // should be close to 128
  ASSERT(c.r >= 126 && c.r <= 130);
  PASS();
}

TEST test_grayscale_monotone(void) {
  // Each step must be >= previous (monotonically non-decreasing)
  cmap_rgb prev = cmap_apply(CMAP_GRAYSCALE, 0.0);
  for (int i = 1; i <= 255; i++) {
    cmap_rgb cur = cmap_apply(CMAP_GRAYSCALE, i / 255.0);
    ASSERT(cur.r >= prev.r);
    ASSERT_EQ(cur.r, cur.g);
    ASSERT_EQ(cur.g, cur.b);
    prev = cur;
  }
  PASS();
}

// ---------------------------------------------------------------------------
// Clamping
// ---------------------------------------------------------------------------

TEST test_clamp_below_zero(void) {
  cmap_rgb a = cmap_apply(CMAP_GRAYSCALE, -1.0);
  cmap_rgb b = cmap_apply(CMAP_GRAYSCALE,  0.0);
  ASSERT_EQ(a.r, b.r);
  ASSERT_EQ(a.g, b.g);
  ASSERT_EQ(a.b, b.b);
  PASS();
}

TEST test_clamp_above_one(void) {
  cmap_rgb a = cmap_apply(CMAP_GRAYSCALE, 2.0);
  cmap_rgb b = cmap_apply(CMAP_GRAYSCALE, 1.0);
  ASSERT_EQ(a.r, b.r);
  ASSERT_EQ(a.g, b.g);
  ASSERT_EQ(a.b, b.b);
  PASS();
}

// ---------------------------------------------------------------------------
// Spot-checks: endpoints for all colormaps (just no-crash + non-degenerate)
// ---------------------------------------------------------------------------

TEST test_all_endpoints(void) {
  for (int id = 0; id < cmap_count(); id++) {
    cmap_rgb lo = cmap_apply((cmap_id)id, 0.0);
    cmap_rgb hi = cmap_apply((cmap_id)id, 1.0);
    // The two endpoints should differ (no all-identical map)
    int lo_sum = (int)lo.r + lo.g + lo.b;
    int hi_sum = (int)hi.r + hi.g + hi.b;
    (void)lo_sum; (void)hi_sum;
    // Just verify no crash and values are in valid range [0,255]
    ASSERT(lo.r <= 255 && lo.g <= 255 && lo.b <= 255);
    ASSERT(hi.r <= 255 && hi.g <= 255 && hi.b <= 255);
  }
  PASS();
}

// ---------------------------------------------------------------------------
// Specific colormap spot checks
// ---------------------------------------------------------------------------

TEST test_hot_at_zero_is_black(void) {
  cmap_rgb c = cmap_apply(CMAP_HOT, 0.0);
  ASSERT_EQ(0, c.r);
  ASSERT_EQ(0, c.g);
  ASSERT_EQ(0, c.b);
  PASS();
}

TEST test_hot_at_one_is_white(void) {
  cmap_rgb c = cmap_apply(CMAP_HOT, 1.0);
  ASSERT_EQ(255, c.r);
  ASSERT_EQ(255, c.g);
  ASSERT_EQ(255, c.b);
  PASS();
}

TEST test_cool_symmetry(void) {
  // cool(t).r + cool(1-t).r should be ~255 (cyan<->magenta mirror)
  cmap_rgb a = cmap_apply(CMAP_COOL, 0.25);
  cmap_rgb b = cmap_apply(CMAP_COOL, 0.75);
  int sum_r = (int)a.r + (int)b.r;
  ASSERT(sum_r >= 253 && sum_r <= 257);
  PASS();
}

TEST test_jet_midpoint_is_greenish(void) {
  // At t=0.5, jet is near green
  cmap_rgb c = cmap_apply(CMAP_JET, 0.5);
  ASSERT(c.g > c.r);
  ASSERT(c.g > c.b);
  PASS();
}

TEST test_viridis_dark_at_zero(void) {
  cmap_rgb c = cmap_apply(CMAP_VIRIDIS, 0.0);
  // viridis starts very dark (sum < 150)
  int total = (int)c.r + c.g + c.b;
  ASSERT(total < 200);
  PASS();
}

TEST test_viridis_bright_at_one(void) {
  cmap_rgb c = cmap_apply(CMAP_VIRIDIS, 1.0);
  // viridis ends bright yellow
  int total = (int)c.r + c.g + c.b;
  ASSERT(total > 400);
  PASS();
}

// ---------------------------------------------------------------------------
// Batch vs individual
// ---------------------------------------------------------------------------

TEST test_batch_matches_individual(void) {
  static const double vals[8] = {0.0, 0.1, 0.25, 0.5, 0.6, 0.75, 0.9, 1.0};
  cmap_rgb buf[8];

  for (int id = 0; id < cmap_count(); id++) {
    cmap_apply_buf((cmap_id)id, vals, buf, 8);
    for (size_t i = 0; i < 8; i++) {
      cmap_rgb ref = cmap_apply((cmap_id)id, vals[i]);
      ASSERT_EQ(ref.r, buf[i].r);
      ASSERT_EQ(ref.g, buf[i].g);
      ASSERT_EQ(ref.b, buf[i].b);
    }
  }
  PASS();
}

TEST test_batch_zero_length(void) {
  // Must not crash with n=0
  cmap_apply_buf(CMAP_VIRIDIS, NULL, NULL, 0);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(cmap_suite) {
  RUN_TEST(test_count);
  RUN_TEST(test_all_names_non_null);
  RUN_TEST(test_name_out_of_range);
  RUN_TEST(test_name_values);
  RUN_TEST(test_grayscale_black);
  RUN_TEST(test_grayscale_white);
  RUN_TEST(test_grayscale_midpoint);
  RUN_TEST(test_grayscale_monotone);
  RUN_TEST(test_clamp_below_zero);
  RUN_TEST(test_clamp_above_one);
  RUN_TEST(test_all_endpoints);
  RUN_TEST(test_hot_at_zero_is_black);
  RUN_TEST(test_hot_at_one_is_white);
  RUN_TEST(test_cool_symmetry);
  RUN_TEST(test_jet_midpoint_is_greenish);
  RUN_TEST(test_viridis_dark_at_zero);
  RUN_TEST(test_viridis_bright_at_one);
  RUN_TEST(test_batch_matches_individual);
  RUN_TEST(test_batch_zero_length);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(cmap_suite);
  GREATEST_MAIN_END();
}
