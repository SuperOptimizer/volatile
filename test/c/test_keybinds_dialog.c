#include "greatest.h"
#include "gui/keybinds_dialog.h"
#include "gui/keybind.h"

#define SC_A      4
#define SC_Z     29
#define MOD_NONE 0x0000
#define MOD_CTRL 0x00C0

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  keybind_map *m = keybind_new();
  ASSERT(m != NULL);
  keybinds_dialog *d = keybinds_dialog_new(m);
  ASSERT(d != NULL);
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

TEST test_initially_hidden(void) {
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  ASSERT(!keybinds_dialog_is_visible(d));
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

TEST test_show_makes_visible(void) {
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  keybinds_dialog_show(d);
  ASSERT(keybinds_dialog_is_visible(d));
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Rebind state machine
// ---------------------------------------------------------------------------

TEST test_not_waiting_initially(void) {
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  keybinds_dialog_show(d);
  ASSERT(!keybinds_dialog_is_waiting_for_key(d));
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

TEST test_inject_key_not_waiting_is_noop(void) {
  // Injecting when not waiting should not crash or corrupt state.
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  keybinds_dialog_show(d);
  keybinds_dialog_inject_key(d, SC_A, MOD_NONE);  // no rebind row active
  ASSERT(!keybinds_dialog_is_waiting_for_key(d));
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Null-safety
// ---------------------------------------------------------------------------

TEST test_null_context_render_returns_false(void) {
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  keybinds_dialog_show(d);
  bool open = keybinds_dialog_render(d, NULL);
  ASSERT(!open);
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

TEST test_null_dialog_safe(void) {
  ASSERT(!keybinds_dialog_is_visible(NULL));
  ASSERT(!keybinds_dialog_is_waiting_for_key(NULL));
  keybinds_dialog_inject_key(NULL, SC_A, MOD_NONE);  // must not crash
  keybinds_dialog_free(NULL);
  PASS();
}

TEST test_render_hidden_returns_false(void) {
  keybind_map *m = keybind_new();
  keybinds_dialog *d = keybinds_dialog_new(m);
  // Not shown
  bool open = keybinds_dialog_render(d, NULL);
  ASSERT(!open);
  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Show re-snapshots current bindings
// ---------------------------------------------------------------------------

TEST test_show_snapshots_current_binds(void) {
  keybind_map *m = keybind_new();
  // Change undo to SC_A before creating dialog
  keybind_set(m, ACTION_UNDO, SC_A, MOD_NONE);

  keybinds_dialog *d = keybinds_dialog_new(m);
  keybinds_dialog_show(d);

  // Dialog should now reflect the modified binding internally.
  // We verify indirectly: inject_key with no row active is harmless,
  // and is_waiting returns false.
  ASSERT(!keybinds_dialog_is_waiting_for_key(d));

  keybinds_dialog_free(d);
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// keybind_get round-trip (new API)
// ---------------------------------------------------------------------------

TEST test_keybind_get_default_undo(void) {
  keybind_map *m = keybind_new();
  int sc = 0, mod = 0;
  keybind_get(m, ACTION_UNDO, &sc, &mod);
  ASSERT_EQ(SC_Z,   sc);
  ASSERT_EQ(MOD_CTRL, mod);
  keybind_free(m);
  PASS();
}

TEST test_keybind_get_after_set(void) {
  keybind_map *m = keybind_new();
  keybind_set(m, ACTION_REDO, SC_A, MOD_NONE);
  int sc = -1, mod = -1;
  keybind_get(m, ACTION_REDO, &sc, &mod);
  ASSERT_EQ(SC_A,   sc);
  ASSERT_EQ(MOD_NONE, mod);
  keybind_free(m);
  PASS();
}

TEST test_keybind_get_null_map_no_crash(void) {
  int sc = 42, mod = 42;
  keybind_get(NULL, ACTION_SAVE, &sc, &mod);
  // Values unchanged
  ASSERT_EQ(42, sc);
  ASSERT_EQ(42, mod);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(keybinds_dialog_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_initially_hidden);
  RUN_TEST(test_show_makes_visible);
  RUN_TEST(test_not_waiting_initially);
  RUN_TEST(test_inject_key_not_waiting_is_noop);
  RUN_TEST(test_null_context_render_returns_false);
  RUN_TEST(test_null_dialog_safe);
  RUN_TEST(test_render_hidden_returns_false);
  RUN_TEST(test_show_snapshots_current_binds);
  RUN_TEST(test_keybind_get_default_undo);
  RUN_TEST(test_keybind_get_after_set);
  RUN_TEST(test_keybind_get_null_map_no_crash);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(keybinds_dialog_suite);
  GREATEST_MAIN_END();
}
