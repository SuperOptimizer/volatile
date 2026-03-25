#include "greatest.h"
#include "gui/toolbar.h"

#include <stdbool.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_clicked = 0;
static void on_click(void *ctx) { (void)ctx; g_clicked++; }

// ---------------------------------------------------------------------------
// Toolbar tests
// ---------------------------------------------------------------------------

TEST test_toolbar_new_empty(void) {
  toolbar *t = toolbar_new();
  ASSERT(t != NULL);
  ASSERT_EQ(0, toolbar_button_count(t));
  toolbar_free(t);
  PASS();
}

TEST test_toolbar_add_buttons(void) {
  toolbar *t = toolbar_new();

  toolbar_add_button(t, "Open",  NULL,  on_click, NULL);
  toolbar_add_button(t, "Save",  NULL,  on_click, NULL);
  toolbar_add_button(t, "Close", "\xE2\x9C\x95", on_click, NULL);

  ASSERT_EQ(3, toolbar_button_count(t));
  toolbar_free(t);
  PASS();
}

TEST test_toolbar_separator_not_counted(void) {
  toolbar *t = toolbar_new();

  toolbar_add_button(t, "A", NULL, NULL, NULL);
  toolbar_add_separator(t);
  toolbar_add_button(t, "B", NULL, NULL, NULL);
  toolbar_add_separator(t);
  toolbar_add_button(t, "C", NULL, NULL, NULL);

  // 3 buttons, 2 separators — only buttons counted
  ASSERT_EQ(3, toolbar_button_count(t));
  toolbar_free(t);
  PASS();
}

TEST test_toolbar_toggle_counted(void) {
  toolbar *t = toolbar_new();
  bool flag = false;

  toolbar_add_button(t, "Btn", NULL, NULL, NULL);
  toolbar_add_toggle(t, "Grid", &flag);

  ASSERT_EQ(2, toolbar_button_count(t));
  toolbar_free(t);
  PASS();
}

TEST test_toolbar_many_buttons_grows(void) {
  toolbar *t = toolbar_new();

  // Add more than the initial capacity (8) to exercise realloc path
  for (int i = 0; i < 20; i++)
    toolbar_add_button(t, "X", NULL, on_click, NULL);

  ASSERT_EQ(20, toolbar_button_count(t));
  toolbar_free(t);
  PASS();
}

// ---------------------------------------------------------------------------
// Context menu tests
// ---------------------------------------------------------------------------

TEST test_context_menu_new_empty(void) {
  context_menu *m = context_menu_new();
  ASSERT(m != NULL);
  ASSERT_EQ(0, context_menu_item_count(m));
  context_menu_free(m);
  PASS();
}

TEST test_context_menu_add_items(void) {
  context_menu *m = context_menu_new();

  context_menu_add(m, "Cut",   on_click, NULL);
  context_menu_add(m, "Copy",  on_click, NULL);
  context_menu_add(m, "Paste", on_click, NULL);

  ASSERT_EQ(3, context_menu_item_count(m));
  context_menu_free(m);
  PASS();
}

TEST test_context_menu_separator_not_counted(void) {
  context_menu *m = context_menu_new();

  context_menu_add(m, "Undo", on_click, NULL);
  context_menu_add_separator(m);
  context_menu_add(m, "Redo", on_click, NULL);

  ASSERT_EQ(2, context_menu_item_count(m));
  context_menu_free(m);
  PASS();
}

TEST test_context_menu_grows(void) {
  context_menu *m = context_menu_new();

  for (int i = 0; i < 20; i++)
    context_menu_add(m, "Item", NULL, NULL);

  ASSERT_EQ(20, context_menu_item_count(m));
  context_menu_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(toolbar_suite) {
  RUN_TEST(test_toolbar_new_empty);
  RUN_TEST(test_toolbar_add_buttons);
  RUN_TEST(test_toolbar_separator_not_counted);
  RUN_TEST(test_toolbar_toggle_counted);
  RUN_TEST(test_toolbar_many_buttons_grows);
  RUN_TEST(test_context_menu_new_empty);
  RUN_TEST(test_context_menu_add_items);
  RUN_TEST(test_context_menu_separator_not_counted);
  RUN_TEST(test_context_menu_grows);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(toolbar_suite);
  GREATEST_MAIN_END();
}
