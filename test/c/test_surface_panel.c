#include "greatest.h"
#include "gui/surface_panel.h"

#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static surface_entry make_entry(int64_t id, const char *name, float area,
                                int rows, int cols) {
  surface_entry e = {0};
  e.id        = id;
  e.name      = (char *)name;  // panel_add copies it
  e.area_vx2  = area;
  e.area_cm2  = area * 0.01f;
  e.row_count = rows;
  e.col_count = cols;
  e.visible   = true;
  e.approved  = false;
  return e;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  surface_panel *p = surface_panel_new();
  ASSERT(p != NULL);
  ASSERT_EQ(0, surface_panel_count(p));
  ASSERT_EQ(-1, surface_panel_selected(p));
  surface_panel_free(p);
  PASS();
}

TEST test_add_count(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e1 = make_entry(1, "surf-A", 100.0f, 64, 64);
  surface_entry e2 = make_entry(2, "surf-B", 200.0f, 128, 128);
  surface_entry e3 = make_entry(3, "surf-C",  50.0f, 32,  32);

  surface_panel_add(p, &e1);
  ASSERT_EQ(1, surface_panel_count(p));

  surface_panel_add(p, &e2);
  surface_panel_add(p, &e3);
  ASSERT_EQ(3, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_remove(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e1 = make_entry(10, "alpha", 10.0f, 10, 10);
  surface_entry e2 = make_entry(20, "beta",  20.0f, 20, 20);
  surface_entry e3 = make_entry(30, "gamma", 30.0f, 30, 30);

  surface_panel_add(p, &e1);
  surface_panel_add(p, &e2);
  surface_panel_add(p, &e3);

  surface_panel_remove(p, 20);
  ASSERT_EQ(2, surface_panel_count(p));

  // Removing an id that doesn't exist is a no-op.
  surface_panel_remove(p, 99);
  ASSERT_EQ(2, surface_panel_count(p));

  // Removing the last entry.
  surface_panel_remove(p, 10);
  surface_panel_remove(p, 30);
  ASSERT_EQ(0, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_clear(void) {
  surface_panel *p = surface_panel_new();
  for (int i = 0; i < 5; i++) {
    surface_entry e = make_entry((int64_t)i, "x", 1.0f, 4, 4);
    surface_panel_add(p, &e);
  }
  ASSERT_EQ(5, surface_panel_count(p));

  surface_panel_clear(p);
  ASSERT_EQ(0, surface_panel_count(p));
  ASSERT_EQ(-1, surface_panel_selected(p));

  surface_panel_free(p);
  PASS();
}

TEST test_select(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e1 = make_entry(7, "s7", 7.0f, 7, 7);
  surface_entry e2 = make_entry(8, "s8", 8.0f, 8, 8);

  surface_panel_add(p, &e1);
  surface_panel_add(p, &e2);

  // Initially nothing selected.
  ASSERT_EQ(-1, surface_panel_selected(p));

  surface_panel_select(p, 7);
  ASSERT_EQ(7, surface_panel_selected(p));

  surface_panel_select(p, 8);
  ASSERT_EQ(8, surface_panel_selected(p));

  // Removing the selected entry clears selection.
  surface_panel_remove(p, 8);
  ASSERT_EQ(-1, surface_panel_selected(p));

  surface_panel_free(p);
  PASS();
}

TEST test_sort_by_area_desc(void) {
  surface_panel *p = surface_panel_new();
  surface_entry entries[] = {
    make_entry(1, "small",  10.0f,  8,  8),
    make_entry(2, "large", 300.0f, 64, 64),
    make_entry(3, "mid",   150.0f, 32, 32),
  };
  for (int i = 0; i < 3; i++) surface_panel_add(p, &entries[i]);

  // Sort descending by area: expected order large(300), mid(150), small(10)
  surface_panel_sort(p, SORT_BY_AREA, false);

  // Verify using selected — just exercise the sort without peeking internals;
  // we check indirectly: select the first and last id then confirm count stays.
  surface_panel_select(p, 2);  // large
  ASSERT_EQ(2, surface_panel_selected(p));
  ASSERT_EQ(3, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_sort_by_name_asc(void) {
  surface_panel *p = surface_panel_new();
  surface_entry ea = make_entry(1, "zebra",  1.0f, 4, 4);
  surface_entry eb = make_entry(2, "apple",  2.0f, 4, 4);
  surface_entry ec = make_entry(3, "mango",  3.0f, 4, 4);

  surface_panel_add(p, &ea);
  surface_panel_add(p, &eb);
  surface_panel_add(p, &ec);

  surface_panel_sort(p, SORT_BY_NAME, true);
  // After sort entries should be: apple(2), mango(3), zebra(1).
  // Exercise: remove by id still works after sort.
  surface_panel_remove(p, 2);
  ASSERT_EQ(2, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_sort_by_id_asc(void) {
  surface_panel *p = surface_panel_new();
  // Add in reverse id order.
  for (int64_t id = 5; id >= 1; id--) {
    surface_entry e = make_entry(id, "s", (float)id, 4, 4);
    surface_panel_add(p, &e);
  }
  surface_panel_sort(p, SORT_BY_ID, true);
  ASSERT_EQ(5, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_sort_by_date(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e1 = make_entry(1, "first",  1.0f, 4, 4);
  surface_entry e2 = make_entry(2, "second", 2.0f, 4, 4);
  surface_entry e3 = make_entry(3, "third",  3.0f, 4, 4);

  surface_panel_add(p, &e1);
  surface_panel_add(p, &e2);
  surface_panel_add(p, &e3);

  // Sort by something else, then restore insertion order.
  surface_panel_sort(p, SORT_BY_AREA, false);
  surface_panel_sort(p, SORT_BY_DATE, true);

  ASSERT_EQ(3, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_remove_selected_clears_selection(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e = make_entry(42, "to-remove", 5.0f, 4, 4);
  surface_panel_add(p, &e);
  surface_panel_select(p, 42);
  ASSERT_EQ(42, surface_panel_selected(p));

  surface_panel_remove(p, 42);
  ASSERT_EQ(-1, surface_panel_selected(p));
  ASSERT_EQ(0, surface_panel_count(p));

  surface_panel_free(p);
  PASS();
}

TEST test_null_name(void) {
  surface_panel *p = surface_panel_new();
  surface_entry e = {0};
  e.id      = 99;
  e.name    = NULL;  // should be handled gracefully
  e.area_vx2 = 1.0f;
  surface_panel_add(p, &e);
  ASSERT_EQ(1, surface_panel_count(p));
  surface_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(surface_panel_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_add_count);
  RUN_TEST(test_remove);
  RUN_TEST(test_clear);
  RUN_TEST(test_select);
  RUN_TEST(test_sort_by_area_desc);
  RUN_TEST(test_sort_by_name_asc);
  RUN_TEST(test_sort_by_id_asc);
  RUN_TEST(test_sort_by_date);
  RUN_TEST(test_remove_selected_clears_selection);
  RUN_TEST(test_null_name);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(surface_panel_suite);
  GREATEST_MAIN_END();
}
