#include "greatest.h"
#include "core/lineseg.h"

#include <math.h>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  line_seg_list *l = lineseg_new();
  ASSERT(l != NULL);
  ASSERT_EQ(0, l->count);
  lineseg_free(l);
  PASS();
}

TEST test_free_null_no_crash(void) {
  lineseg_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Add / count
// ---------------------------------------------------------------------------

TEST test_add_grows_count(void) {
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){1,0,0});
  lineseg_add(l, (vec3f){2,0,0});
  ASSERT_EQ(3, l->count);
  lineseg_free(l);
  PASS();
}

TEST test_add_many_triggers_realloc(void) {
  line_seg_list *l = lineseg_new();
  for (int i = 0; i < 64; i++)
    lineseg_add(l, (vec3f){(float)i, 0, 0});
  ASSERT_EQ(64, l->count);
  ASSERT(l->capacity >= 64);
  lineseg_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Length
// ---------------------------------------------------------------------------

TEST test_length_empty(void) {
  line_seg_list *l = lineseg_new();
  ASSERT_IN_RANGE(-0.01f, lineseg_length(l), 0.01f);
  lineseg_free(l);
  PASS();
}

TEST test_length_single_point(void) {
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){5,5,5});
  ASSERT_IN_RANGE(-0.01f, lineseg_length(l), 0.01f);
  lineseg_free(l);
  PASS();
}

TEST test_length_axis_aligned(void) {
  // 3 points along x: (0,0,0) -> (3,0,0) -> (6,0,0), total length = 6
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){3,0,0});
  lineseg_add(l, (vec3f){6,0,0});
  ASSERT_IN_RANGE(5.99f, lineseg_length(l), 6.01f);
  lineseg_free(l);
  PASS();
}

TEST test_length_3d_segment(void) {
  // (0,0,0) -> (1,2,2): sqrt(1+4+4) = 3
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){1,2,2});
  ASSERT_IN_RANGE(2.99f, lineseg_length(l), 3.01f);
  lineseg_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Sample
// ---------------------------------------------------------------------------

TEST test_sample_empty_returns_zero(void) {
  line_seg_list *l = lineseg_new();
  vec3f p = lineseg_sample(l, 0.5f);
  ASSERT_IN_RANGE(-0.01f, p.x, 0.01f);
  ASSERT_IN_RANGE(-0.01f, p.y, 0.01f);
  ASSERT_IN_RANGE(-0.01f, p.z, 0.01f);
  lineseg_free(l);
  PASS();
}

TEST test_sample_t0_returns_first(void) {
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){1,2,3});
  lineseg_add(l, (vec3f){4,5,6});
  vec3f p = lineseg_sample(l, 0.0f);
  ASSERT_IN_RANGE(0.99f, p.x, 1.01f);
  lineseg_free(l);
  PASS();
}

TEST test_sample_t1_returns_last(void) {
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){10,0,0});
  vec3f p = lineseg_sample(l, 1.0f);
  ASSERT_IN_RANGE(9.99f, p.x, 10.01f);
  lineseg_free(l);
  PASS();
}

TEST test_sample_midpoint(void) {
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){10,0,0});
  vec3f p = lineseg_sample(l, 0.5f);
  ASSERT_IN_RANGE(4.99f, p.x, 5.01f);
  lineseg_free(l);
  PASS();
}

TEST test_sample_across_segment_boundary(void) {
  // 2 equal segments: (0,0,0)-(5,0,0)-(10,0,0)
  // t=0.75 should land at x=7.5
  line_seg_list *l = lineseg_new();
  lineseg_add(l, (vec3f){0,0,0});
  lineseg_add(l, (vec3f){5,0,0});
  lineseg_add(l, (vec3f){10,0,0});
  vec3f p = lineseg_sample(l, 0.75f);
  ASSERT_IN_RANGE(7.49f, p.x, 7.51f);
  lineseg_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(lineseg_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null_no_crash);
  RUN_TEST(test_add_grows_count);
  RUN_TEST(test_add_many_triggers_realloc);
  RUN_TEST(test_length_empty);
  RUN_TEST(test_length_single_point);
  RUN_TEST(test_length_axis_aligned);
  RUN_TEST(test_length_3d_segment);
  RUN_TEST(test_sample_empty_returns_zero);
  RUN_TEST(test_sample_t0_returns_first);
  RUN_TEST(test_sample_t1_returns_last);
  RUN_TEST(test_sample_midpoint);
  RUN_TEST(test_sample_across_segment_boundary);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(lineseg_suite);
  GREATEST_MAIN_END();
}
