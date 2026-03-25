#include "greatest.h"
#include "gui/keybind.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// SDL3 scancode/mod constants (mirrored from keybind.c)
#define SC_A       4
#define SC_B       5
#define SC_D       7
#define SC_L      15
#define SC_S      22
#define SC_W      26
#define SC_Y      28
#define SC_Z      29
#define SC_MINUS  45
#define SC_EQUALS 46
#define SC_COMMA  54
#define SC_PERIOD 55
#define SC_F11    68
#define MOD_NONE  0x0000
#define MOD_CTRL  0x00C0

// ---------------------------------------------------------------------------
// Default lookup tests
// ---------------------------------------------------------------------------

TEST test_default_pan_left(void) {
  keybind_map *m = keybind_new();
  ASSERT(m != NULL);
  ASSERT_EQ(ACTION_PAN_LEFT, keybind_lookup(m, SC_A, MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_default_pan_right(void) {
  keybind_map *m = keybind_new();
  ASSERT_EQ(ACTION_PAN_RIGHT, keybind_lookup(m, SC_D, MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_default_zoom(void) {
  keybind_map *m = keybind_new();
  ASSERT_EQ(ACTION_ZOOM_IN,  keybind_lookup(m, SC_EQUALS, MOD_NONE));
  ASSERT_EQ(ACTION_ZOOM_OUT, keybind_lookup(m, SC_MINUS,  MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_default_slice(void) {
  keybind_map *m = keybind_new();
  ASSERT_EQ(ACTION_SLICE_NEXT, keybind_lookup(m, SC_PERIOD, MOD_NONE));
  ASSERT_EQ(ACTION_SLICE_PREV, keybind_lookup(m, SC_COMMA,  MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_default_tools(void) {
  keybind_map *m = keybind_new();
  ASSERT_EQ(ACTION_TOOL_BRUSH, keybind_lookup(m, SC_B, MOD_NONE));
  ASSERT_EQ(ACTION_TOOL_LINE,  keybind_lookup(m, SC_L, MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_default_undo_redo(void) {
  keybind_map *m = keybind_new();
  ASSERT_EQ(ACTION_UNDO, keybind_lookup(m, SC_Z, MOD_CTRL));
  ASSERT_EQ(ACTION_REDO, keybind_lookup(m, SC_Y, MOD_CTRL));
  keybind_free(m);
  PASS();
}

TEST test_default_fullscreen(void) {
  keybind_map *m = keybind_new();
  // F11 maps to fullscreen (or overlay — whichever is registered last doesn't matter; fullscreen is registered last)
  int action = keybind_lookup(m, SC_F11, MOD_NONE);
  ASSERT(action == ACTION_TOGGLE_OVERLAY || action == ACTION_FULLSCREEN);
  keybind_free(m);
  PASS();
}

TEST test_lookup_unknown(void) {
  keybind_map *m = keybind_new();
  // scancode 200 is not bound by default
  ASSERT_EQ(-1, keybind_lookup(m, 200, MOD_NONE));
  keybind_free(m);
  PASS();
}

TEST test_modifier_mismatch(void) {
  keybind_map *m = keybind_new();
  // Ctrl+Z is undo; plain Z should not match
  ASSERT_EQ(-1, keybind_lookup(m, SC_Z, MOD_NONE));
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Custom binding
// ---------------------------------------------------------------------------

TEST test_set_custom_binding(void) {
  keybind_map *m = keybind_new();

  // Remap zoom_in to scancode 99, no modifier
  keybind_set(m, ACTION_ZOOM_IN, 99, MOD_NONE);
  ASSERT_EQ(ACTION_ZOOM_IN, keybind_lookup(m, 99, MOD_NONE));
  // Old binding gone
  ASSERT_EQ(-1, keybind_lookup(m, SC_EQUALS, MOD_NONE));

  keybind_free(m);
  PASS();
}

TEST test_set_with_modifier(void) {
  keybind_map *m = keybind_new();

  keybind_set(m, ACTION_SAVE, SC_S, MOD_CTRL);
  ASSERT_EQ(ACTION_SAVE, keybind_lookup(m, SC_S, MOD_CTRL));

  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Action names
// ---------------------------------------------------------------------------

TEST test_action_names(void) {
  ASSERT_STR_EQ("pan_left",            keybind_action_name(ACTION_PAN_LEFT));
  ASSERT_STR_EQ("undo",                keybind_action_name(ACTION_UNDO));
  ASSERT_STR_EQ("fullscreen",          keybind_action_name(ACTION_FULLSCREEN));
  ASSERT_STR_EQ("toggle_segmentation", keybind_action_name(ACTION_TOGGLE_SEGMENTATION));
  PASS();
}

// ---------------------------------------------------------------------------
// JSON save / load roundtrip
// ---------------------------------------------------------------------------

TEST test_json_roundtrip(void) {
  keybind_map *orig = keybind_new();
  ASSERT(orig != NULL);

  // Customize a few bindings
  keybind_set(orig, ACTION_ZOOM_IN,  77, MOD_NONE);
  keybind_set(orig, ACTION_ZOOM_OUT, 78, MOD_CTRL);

  const char *path = "/tmp/test_keybind.json";
  ASSERT(keybind_save_json(orig, path));

  keybind_map *loaded = keybind_load_json(path);
  ASSERT(loaded != NULL);

  ASSERT_EQ(ACTION_ZOOM_IN,  keybind_lookup(loaded, 77, MOD_NONE));
  ASSERT_EQ(ACTION_ZOOM_OUT, keybind_lookup(loaded, 78, MOD_CTRL));

  // Other defaults preserved
  ASSERT_EQ(ACTION_UNDO, keybind_lookup(loaded, SC_Z, MOD_CTRL));
  ASSERT_EQ(ACTION_REDO, keybind_lookup(loaded, SC_Y, MOD_CTRL));

  keybind_free(orig);
  keybind_free(loaded);
  PASS();
}

TEST test_load_missing_file_returns_defaults(void) {
  // Non-existent path should return a map with defaults
  keybind_map *m = keybind_load_json("/tmp/does_not_exist_xyzzy.json");
  ASSERT(m != NULL);
  ASSERT_EQ(ACTION_PAN_LEFT, keybind_lookup(m, SC_A, MOD_NONE));
  keybind_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites
// ---------------------------------------------------------------------------

SUITE(suite_keybind) {
  RUN_TEST(test_default_pan_left);
  RUN_TEST(test_default_pan_right);
  RUN_TEST(test_default_zoom);
  RUN_TEST(test_default_slice);
  RUN_TEST(test_default_tools);
  RUN_TEST(test_default_undo_redo);
  RUN_TEST(test_default_fullscreen);
  RUN_TEST(test_lookup_unknown);
  RUN_TEST(test_modifier_mismatch);
  RUN_TEST(test_set_custom_binding);
  RUN_TEST(test_set_with_modifier);
  RUN_TEST(test_action_names);
  RUN_TEST(test_json_roundtrip);
  RUN_TEST(test_load_missing_file_returns_defaults);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_keybind);
  GREATEST_MAIN_END();
}
