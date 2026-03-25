#include "greatest.h"
#include "gui/seeding.h"
#include "render/overlay.h"
#include "core/math.h"

#include <math.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_add_count(void) {
  seed_manager *m = seed_mgr_new();
  ASSERT(m != NULL);
  ASSERT_EQ(0, seed_mgr_count(m));

  int id1 = seed_mgr_add(m, (vec3f){1, 2, 3});
  int id2 = seed_mgr_add(m, (vec3f){4, 5, 6});
  ASSERT(id1 >= 0);
  ASSERT(id2 >= 0);
  ASSERT(id1 != id2);
  ASSERT_EQ(2, seed_mgr_count(m));

  seed_mgr_free(m);
  PASS();
}

TEST test_get(void) {
  seed_manager *m = seed_mgr_new();
  int id = seed_mgr_add(m, (vec3f){10, 20, 30});
  vec3f p = seed_mgr_get(m, id);
  ASSERT(fabsf(p.x - 10) < 1e-5f);
  ASSERT(fabsf(p.y - 20) < 1e-5f);
  ASSERT(fabsf(p.z - 30) < 1e-5f);
  seed_mgr_free(m);
  PASS();
}

TEST test_get_missing_returns_zero(void) {
  seed_manager *m = seed_mgr_new();
  vec3f p = seed_mgr_get(m, 999);
  ASSERT_EQ(0, (int)p.x);
  ASSERT_EQ(0, (int)p.y);
  ASSERT_EQ(0, (int)p.z);
  seed_mgr_free(m);
  PASS();
}

TEST test_remove(void) {
  seed_manager *m = seed_mgr_new();
  int id1 = seed_mgr_add(m, (vec3f){1, 1, 1});
  int id2 = seed_mgr_add(m, (vec3f){2, 2, 2});
  ASSERT_EQ(2, seed_mgr_count(m));

  ASSERT(seed_mgr_remove(m, id1));
  ASSERT_EQ(1, seed_mgr_count(m));

  // Double-remove returns false.
  ASSERT(!seed_mgr_remove(m, id1));

  // id2 still accessible.
  vec3f p = seed_mgr_get(m, id2);
  ASSERT(fabsf(p.x - 2) < 1e-5f);

  seed_mgr_free(m);
  PASS();
}

TEST test_remove_invalid_id(void) {
  seed_manager *m = seed_mgr_new();
  ASSERT(!seed_mgr_remove(m, -1));
  ASSERT(!seed_mgr_remove(m, 0));
  ASSERT(!seed_mgr_remove(m, 999));
  seed_mgr_free(m);
  PASS();
}

TEST test_many_seeds(void) {
  seed_manager *m = seed_mgr_new();
  int ids[64];
  for (int i = 0; i < 64; i++) {
    ids[i] = seed_mgr_add(m, (vec3f){(float)i, 0, 0});
    ASSERT(ids[i] >= 0);
  }
  ASSERT_EQ(64, seed_mgr_count(m));
  // Verify all retrievable.
  for (int i = 0; i < 64; i++) {
    vec3f p = seed_mgr_get(m, ids[i]);
    ASSERT(fabsf(p.x - (float)i) < 1e-5f);
  }
  // Remove even ones.
  for (int i = 0; i < 64; i += 2) seed_mgr_remove(m, ids[i]);
  ASSERT_EQ(32, seed_mgr_count(m));
  // Re-add and slot reuse.
  int new_id = seed_mgr_add(m, (vec3f){99, 99, 99});
  ASSERT(new_id >= 0);
  ASSERT_EQ(33, seed_mgr_count(m));
  seed_mgr_free(m);
  PASS();
}

TEST test_overlay_populated(void) {
  seed_manager *m = seed_mgr_new();
  seed_mgr_add(m, (vec3f){10, 20, 0});
  seed_mgr_add(m, (vec3f){30, 40, 0});

  overlay_list *ol = overlay_list_new();
  ASSERT(ol != NULL);
  seed_mgr_to_overlay(m, ol, 8.0f);
  // Each seed emits a circle + 2 crosshair lines = 3 entries.
  ASSERT(overlay_count(ol) >= 2);

  overlay_list_free(ol);
  seed_mgr_free(m);
  PASS();
}

TEST test_overlay_empty(void) {
  seed_manager *m = seed_mgr_new();
  overlay_list *ol = overlay_list_new();
  seed_mgr_to_overlay(m, ol, 5.0f);
  ASSERT_EQ(0, overlay_count(ol));
  overlay_list_free(ol);
  seed_mgr_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(seeding_suite) {
  RUN_TEST(test_add_count);
  RUN_TEST(test_get);
  RUN_TEST(test_get_missing_returns_zero);
  RUN_TEST(test_remove);
  RUN_TEST(test_remove_invalid_id);
  RUN_TEST(test_many_seeds);
  RUN_TEST(test_overlay_populated);
  RUN_TEST(test_overlay_empty);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(seeding_suite);
  GREATEST_MAIN_END();
}
