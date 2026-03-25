#include "greatest.h"
#include "gui/vol_info_panel.h"

// vol_info_panel has no Nuklear-free logic paths we can unit-test without a
// GPU context, so we test the lifecycle and NULL-safety of the panel object.

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  vol_info_panel *p = vol_info_panel_new();
  ASSERT(p != NULL);
  vol_info_panel_free(p);
  PASS();
}

TEST test_free_null(void) {
  vol_info_panel_free(NULL);  // must not crash
  PASS();
}

TEST test_new_returns_distinct_objects(void) {
  vol_info_panel *a = vol_info_panel_new();
  vol_info_panel *b = vol_info_panel_new();
  ASSERT(a != NULL);
  ASSERT(b != NULL);
  ASSERT(a != b);
  vol_info_panel_free(a);
  vol_info_panel_free(b);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(vol_info_panel_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_new_returns_distinct_objects);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vol_info_panel_suite);
  GREATEST_MAIN_END();
}
