#include "greatest.h"
#include "gui/vol_selector.h"

#include <string.h>

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  vol_selector *s = vol_selector_new();
  ASSERT(s != NULL);
  ASSERT_EQ(0, vol_selector_count(s));
  ASSERT_EQ(-1, vol_selector_selected(s));
  ASSERT(vol_selector_selected_path(s) == NULL);
  vol_selector_free(s);
  PASS();
}

TEST test_free_null(void) {
  vol_selector_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// add
// ---------------------------------------------------------------------------

TEST test_add_single(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(s, "my_vol", "/data/my_vol.zarr");
  ASSERT_EQ(1, vol_selector_count(s));
  ASSERT_EQ(0, vol_selector_selected(s));
  ASSERT_STR_EQ("/data/my_vol.zarr", vol_selector_selected_path(s));
  vol_selector_free(s);
  PASS();
}

TEST test_add_multiple(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(s, "vol_a", "/data/a.zarr");
  vol_selector_add(s, "vol_b", "/data/b.zarr");
  vol_selector_add(s, "vol_c", "/data/c.zarr");
  ASSERT_EQ(3, vol_selector_count(s));
  // selection stays at first-added entry
  ASSERT_EQ(0, vol_selector_selected(s));
  ASSERT_STR_EQ("/data/a.zarr", vol_selector_selected_path(s));
  vol_selector_free(s);
  PASS();
}

TEST test_add_null_ignored(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(NULL, "x", "/x");  // must not crash
  vol_selector_add(s, NULL, "/x");    // must not crash
  vol_selector_add(s, "x", NULL);     // must not crash
  ASSERT_EQ(0, vol_selector_count(s));
  vol_selector_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// clear
// ---------------------------------------------------------------------------

TEST test_clear(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(s, "v", "/v.zarr");
  ASSERT_EQ(1, vol_selector_count(s));
  vol_selector_clear(s);
  ASSERT_EQ(0, vol_selector_count(s));
  ASSERT_EQ(-1, vol_selector_selected(s));
  ASSERT(vol_selector_selected_path(s) == NULL);
  vol_selector_free(s);
  PASS();
}

TEST test_clear_then_add(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(s, "old", "/old.zarr");
  vol_selector_clear(s);
  vol_selector_add(s, "new", "/new.zarr");
  ASSERT_EQ(1, vol_selector_count(s));
  ASSERT_EQ(0, vol_selector_selected(s));
  ASSERT_STR_EQ("/new.zarr", vol_selector_selected_path(s));
  vol_selector_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

TEST test_selected_path_out_of_range(void) {
  vol_selector *s = vol_selector_new();
  // empty
  ASSERT(vol_selector_selected_path(s) == NULL);
  vol_selector_free(s);
  PASS();
}

TEST test_count_null(void) {
  ASSERT_EQ(0, vol_selector_count(NULL));
  PASS();
}

TEST test_selected_null(void) {
  ASSERT_EQ(-1, vol_selector_selected(NULL));
  PASS();
}

TEST test_selected_path_null(void) {
  ASSERT(vol_selector_selected_path(NULL) == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// render — null ctx must not crash
// ---------------------------------------------------------------------------

TEST test_render_null_ctx(void) {
  vol_selector *s = vol_selector_new();
  vol_selector_add(s, "v", "/v.zarr");
  // passing NULL ctx returns false without crashing
  bool changed = vol_selector_render(s, NULL);
  ASSERT(!changed);
  vol_selector_free(s);
  PASS();
}

TEST test_render_empty_null_ctx(void) {
  vol_selector *s = vol_selector_new();
  bool changed = vol_selector_render(s, NULL);
  ASSERT(!changed);
  vol_selector_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(vol_selector_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_add_single);
  RUN_TEST(test_add_multiple);
  RUN_TEST(test_add_null_ignored);
  RUN_TEST(test_clear);
  RUN_TEST(test_clear_then_add);
  RUN_TEST(test_selected_path_out_of_range);
  RUN_TEST(test_count_null);
  RUN_TEST(test_selected_null);
  RUN_TEST(test_selected_path_null);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_empty_null_ctx);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vol_selector_suite);
  GREATEST_MAIN_END();
}
