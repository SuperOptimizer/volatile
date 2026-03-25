/* test_binary_pyramid.c — tests for binary_pyramid.c */
#include "greatest.h"
#include "core/binary_pyramid.h"

#include <stdlib.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------
 * 1. lifecycle: new returns non-NULL, free doesn't crash
 * --------------------------------------------------------------------- */

TEST test_lifecycle(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);
  binary_pyramid_free(p);
  PASS();
}

TEST test_free_null(void) {
  binary_pyramid_free(NULL);
  PASS();
}

TEST test_new_invalid(void) {
  ASSERT_EQ(binary_pyramid_new(0, 8, 8), NULL);
  ASSERT_EQ(binary_pyramid_new(8, 0, 8), NULL);
  ASSERT_EQ(binary_pyramid_new(8, 8, 0), NULL);
  PASS();
}

/* -------------------------------------------------------------------------
 * 2. get/set basic
 * --------------------------------------------------------------------- */

TEST test_get_initially_false(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);
  for (int z = 0; z < 4; z++)
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++)
        ASSERT(!binary_pyramid_get(p, z, y, x));
  binary_pyramid_free(p);
  PASS();
}

TEST test_set_and_get(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);

  binary_pyramid_set(p, 3, 2, 1, true);
  ASSERT(binary_pyramid_get(p, 3, 2, 1));
  ASSERT(!binary_pyramid_get(p, 3, 2, 0));
  ASSERT(!binary_pyramid_get(p, 0, 0, 0));

  binary_pyramid_free(p);
  PASS();
}

TEST test_set_and_clear(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);

  binary_pyramid_set(p, 1, 1, 1, true);
  ASSERT(binary_pyramid_get(p, 1, 1, 1));
  binary_pyramid_set(p, 1, 1, 1, false);
  ASSERT(!binary_pyramid_get(p, 1, 1, 1));

  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 3. out-of-bounds: get returns false, set is a no-op
 * --------------------------------------------------------------------- */

TEST test_oob_get(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);
  ASSERT(!binary_pyramid_get(p, -1, 0, 0));
  ASSERT(!binary_pyramid_get(p, 4, 0, 0));
  ASSERT(!binary_pyramid_get(p, 0, 4, 0));
  ASSERT(!binary_pyramid_get(p, 0, 0, 4));
  binary_pyramid_free(p);
  PASS();
}

TEST test_oob_set(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);
  binary_pyramid_set(p, -1, 0, 0, true);  // should not crash
  binary_pyramid_set(p,  4, 0, 0, true);
  ASSERT_EQ(binary_pyramid_count(p), 0);
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 4. count
 * --------------------------------------------------------------------- */

TEST test_count_empty(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);
  ASSERT_EQ(binary_pyramid_count(p), 0);
  binary_pyramid_free(p);
  PASS();
}

TEST test_count_after_sets(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);
  binary_pyramid_set(p, 0, 0, 0, true);
  binary_pyramid_set(p, 1, 1, 1, true);
  binary_pyramid_set(p, 7, 7, 7, true);
  ASSERT_EQ(binary_pyramid_count(p), 3);
  binary_pyramid_set(p, 1, 1, 1, false);
  ASSERT_EQ(binary_pyramid_count(p), 2);
  binary_pyramid_free(p);
  PASS();
}

TEST test_count_null(void) {
  ASSERT_EQ(binary_pyramid_count(NULL), 0);
  PASS();
}

/* -------------------------------------------------------------------------
 * 5. any_in_region: basic checks
 * --------------------------------------------------------------------- */

TEST test_any_empty_region(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);
  // empty pyramid → no region has anything
  ASSERT(!binary_pyramid_any_in_region(p, 0, 0, 0, 8, 8, 8));
  binary_pyramid_free(p);
  PASS();
}

TEST test_any_single_voxel_in_region(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);
  binary_pyramid_set(p, 3, 4, 5, true);
  ASSERT(binary_pyramid_any_in_region(p, 0, 0, 0, 8, 8, 8));  // whole volume
  ASSERT(binary_pyramid_any_in_region(p, 3, 4, 5, 4, 5, 6));  // exact cell
  ASSERT(!binary_pyramid_any_in_region(p, 0, 0, 0, 3, 4, 5)); // just outside
  binary_pyramid_free(p);
  PASS();
}

TEST test_any_region_excludes_outside(void) {
  binary_pyramid *p = binary_pyramid_new(16, 16, 16);
  ASSERT(p != NULL);
  binary_pyramid_set(p, 10, 10, 10, true);
  // Region that doesn't include (10,10,10)
  ASSERT(!binary_pyramid_any_in_region(p, 0, 0, 0, 8, 8, 8));
  ASSERT(!binary_pyramid_any_in_region(p, 11, 11, 11, 16, 16, 16));
  // Region that includes it
  ASSERT(binary_pyramid_any_in_region(p, 8, 8, 8, 12, 12, 12));
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 6. non-power-of-two dimensions
 * --------------------------------------------------------------------- */

TEST test_non_pow2_dims(void) {
  binary_pyramid *p = binary_pyramid_new(5, 7, 3);
  ASSERT(p != NULL);
  binary_pyramid_set(p, 4, 6, 2, true);
  ASSERT(binary_pyramid_get(p, 4, 6, 2));
  ASSERT_EQ(binary_pyramid_count(p), 1);
  ASSERT(binary_pyramid_any_in_region(p, 3, 5, 1, 5, 7, 3));
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 7. fill entire volume then check all-true
 * --------------------------------------------------------------------- */

TEST test_all_set(void) {
  binary_pyramid *p = binary_pyramid_new(4, 4, 4);
  ASSERT(p != NULL);
  for (int z = 0; z < 4; z++)
    for (int y = 0; y < 4; y++)
      for (int x = 0; x < 4; x++)
        binary_pyramid_set(p, z, y, x, true);
  ASSERT_EQ(binary_pyramid_count(p), 64);
  ASSERT(binary_pyramid_any_in_region(p, 0, 0, 0, 4, 4, 4));
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 8. set + clear preserves pyramid consistency
 * --------------------------------------------------------------------- */

TEST test_pyramid_consistency_after_clear(void) {
  binary_pyramid *p = binary_pyramid_new(8, 8, 8);
  ASSERT(p != NULL);
  // Set 2 voxels in the same parent region
  binary_pyramid_set(p, 0, 0, 0, true);
  binary_pyramid_set(p, 0, 0, 1, true);
  ASSERT(binary_pyramid_any_in_region(p, 0, 0, 0, 2, 2, 2));
  // Clear first voxel — region still has (0,0,1)
  binary_pyramid_set(p, 0, 0, 0, false);
  ASSERT(binary_pyramid_any_in_region(p, 0, 0, 0, 2, 2, 2));
  // Clear second — region now empty
  binary_pyramid_set(p, 0, 0, 1, false);
  ASSERT(!binary_pyramid_any_in_region(p, 0, 0, 0, 2, 2, 2));
  ASSERT_EQ(binary_pyramid_count(p), 0);
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 9. 1×1×1 degenerate pyramid
 * --------------------------------------------------------------------- */

TEST test_single_voxel_pyramid(void) {
  binary_pyramid *p = binary_pyramid_new(1, 1, 1);
  ASSERT(p != NULL);
  ASSERT(!binary_pyramid_get(p, 0, 0, 0));
  binary_pyramid_set(p, 0, 0, 0, true);
  ASSERT(binary_pyramid_get(p, 0, 0, 0));
  ASSERT_EQ(binary_pyramid_count(p), 1);
  ASSERT(binary_pyramid_any_in_region(p, 0, 0, 0, 1, 1, 1));
  binary_pyramid_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * Suite
 * --------------------------------------------------------------------- */

SUITE(binary_pyramid_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_free_null);
  RUN_TEST(test_new_invalid);
  RUN_TEST(test_get_initially_false);
  RUN_TEST(test_set_and_get);
  RUN_TEST(test_set_and_clear);
  RUN_TEST(test_oob_get);
  RUN_TEST(test_oob_set);
  RUN_TEST(test_count_empty);
  RUN_TEST(test_count_after_sets);
  RUN_TEST(test_count_null);
  RUN_TEST(test_any_empty_region);
  RUN_TEST(test_any_single_voxel_in_region);
  RUN_TEST(test_any_region_excludes_outside);
  RUN_TEST(test_non_pow2_dims);
  RUN_TEST(test_all_set);
  RUN_TEST(test_pyramid_consistency_after_clear);
  RUN_TEST(test_single_voxel_pyramid);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(binary_pyramid_suite);
  GREATEST_MAIN_END();
}
