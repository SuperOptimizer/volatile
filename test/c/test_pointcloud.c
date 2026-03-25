#include "greatest.h"
#include "core/pointcloud.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  pointcloud *pc = pointcloud_new();
  ASSERT(pc != NULL);
  ASSERT_EQ(0, pointcloud_count(pc));
  pointcloud_free(pc);
  PASS();
}

TEST test_add_count(void) {
  pointcloud *pc = pointcloud_new();
  pointcloud_add(pc, (vec3f){1.0f, 2.0f, 3.0f});
  pointcloud_add(pc, (vec3f){4.0f, 5.0f, 6.0f});
  ASSERT_EQ(2, pointcloud_count(pc));
  pointcloud_free(pc);
  PASS();
}

TEST test_get_values(void) {
  pointcloud *pc = pointcloud_new();
  pointcloud_add(pc, (vec3f){10.0f, 20.0f, 30.0f});
  vec3f p = pointcloud_get(pc, 0);
  ASSERT_IN_RANGE(10.0f, p.x, 1e-5f);
  ASSERT_IN_RANGE(20.0f, p.y, 1e-5f);
  ASSERT_IN_RANGE(30.0f, p.z, 1e-5f);
  pointcloud_free(pc);
  PASS();
}

TEST test_add_batch(void) {
  pointcloud *pc = pointcloud_new();
  vec3f pts[5] = {
    {0,0,0},{1,0,0},{2,0,0},{3,0,0},{4,0,0},
  };
  pointcloud_add_batch(pc, pts, 5);
  ASSERT_EQ(5, pointcloud_count(pc));
  for (int i = 0; i < 5; i++) {
    vec3f p = pointcloud_get(pc, i);
    ASSERT_IN_RANGE((float)i, p.x, 1e-5f);
  }
  pointcloud_free(pc);
  PASS();
}

TEST test_batch_empty(void) {
  pointcloud *pc = pointcloud_new();
  pointcloud_add_batch(pc, NULL, 0);
  ASSERT_EQ(0, pointcloud_count(pc));
  pointcloud_free(pc);
  PASS();
}

TEST test_grows_beyond_initial_cap(void) {
  pointcloud *pc = pointcloud_new();
  // Initial cap is 256; add 1000 points
  for (int i = 0; i < 1000; i++) {
    pointcloud_add(pc, (vec3f){(float)i, 0.0f, 0.0f});
  }
  ASSERT_EQ(1000, pointcloud_count(pc));
  // Spot-check a few
  ASSERT_IN_RANGE(0.0f,   pointcloud_get(pc,   0).x, 1e-5f);
  ASSERT_IN_RANGE(999.0f, pointcloud_get(pc, 999).x, 1e-5f);
  pointcloud_free(pc);
  PASS();
}

// ---------------------------------------------------------------------------
// parallel_for: sum all x-coordinates into an atomic accumulator
// ---------------------------------------------------------------------------

typedef struct { _Atomic(int) sum; } _sum_ctx;

static void _sum_fn(vec3f pt, int index, void *ctx) {
  _sum_ctx *s = (_sum_ctx *)ctx;
  atomic_fetch_add(&s->sum, (int)pt.x);
  (void)index;
}

TEST test_parallel_for_sums_correctly(void) {
  pointcloud *pc = pointcloud_new();
  int n = 200;
  for (int i = 0; i < n; i++) {
    pointcloud_add(pc, (vec3f){(float)i, 0.0f, 0.0f});
  }

  _sum_ctx ctx = { .sum = 0 };
  pointcloud_parallel_for(pc, _sum_fn, &ctx, 4);

  int expected = n * (n - 1) / 2;  // sum 0..n-1
  ASSERT_EQ(expected, (int)atomic_load(&ctx.sum));
  pointcloud_free(pc);
  PASS();
}

TEST test_parallel_for_single_thread(void) {
  pointcloud *pc = pointcloud_new();
  for (int i = 0; i < 50; i++) {
    pointcloud_add(pc, (vec3f){(float)i, 0.0f, 0.0f});
  }

  _sum_ctx ctx = { .sum = 0 };
  pointcloud_parallel_for(pc, _sum_fn, &ctx, 1);

  ASSERT_EQ(50 * 49 / 2, (int)atomic_load(&ctx.sum));
  pointcloud_free(pc);
  PASS();
}

TEST test_parallel_for_empty(void) {
  pointcloud *pc = pointcloud_new();
  _sum_ctx ctx = { .sum = 0 };
  pointcloud_parallel_for(pc, _sum_fn, &ctx, 4);
  ASSERT_EQ(0, (int)atomic_load(&ctx.sum));
  pointcloud_free(pc);
  PASS();
}

TEST test_parallel_for_default_threads(void) {
  pointcloud *pc = pointcloud_new();
  for (int i = 0; i < 100; i++) {
    pointcloud_add(pc, (vec3f){1.0f, 0.0f, 0.0f});
  }
  _sum_ctx ctx = { .sum = 0 };
  pointcloud_parallel_for(pc, _sum_fn, &ctx, 0);  // 0 = auto
  ASSERT_EQ(100, (int)atomic_load(&ctx.sum));
  pointcloud_free(pc);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(pointcloud_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_add_count);
  RUN_TEST(test_get_values);
  RUN_TEST(test_add_batch);
  RUN_TEST(test_batch_empty);
  RUN_TEST(test_grows_beyond_initial_cap);
  RUN_TEST(test_parallel_for_sums_correctly);
  RUN_TEST(test_parallel_for_single_thread);
  RUN_TEST(test_parallel_for_empty);
  RUN_TEST(test_parallel_for_default_threads);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(pointcloud_suite);
  GREATEST_MAIN_END();
}
