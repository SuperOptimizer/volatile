#pragma once
#include <stdbool.h>

typedef enum {
  ACTION_PAN_LEFT,  ACTION_PAN_RIGHT,  ACTION_PAN_UP,  ACTION_PAN_DOWN,
  ACTION_ZOOM_IN,   ACTION_ZOOM_OUT,
  ACTION_SLICE_NEXT, ACTION_SLICE_PREV,
  ACTION_TOOL_BRUSH, ACTION_TOOL_LINE, ACTION_TOOL_PUSHPULL,
  ACTION_UNDO, ACTION_REDO,
  ACTION_FOCUS_NEXT, ACTION_FOCUS_PREV,
  ACTION_TOGGLE_OVERLAY, ACTION_TOGGLE_SEGMENTATION,
  ACTION_SAVE, ACTION_FULLSCREEN,
  ACTION_COUNT
} action_id;

typedef struct keybind_map keybind_map;

keybind_map       *keybind_new(void);    // creates with VC3D-compatible defaults
void               keybind_free(keybind_map *m);
void               keybind_set(keybind_map *m, action_id action, int sdl_scancode, int modifiers);
int                keybind_lookup(const keybind_map *m, int sdl_scancode, int modifiers); // action_id or -1
const char        *keybind_action_name(action_id a);
bool               keybind_save_json(const keybind_map *m, const char *path);
keybind_map       *keybind_load_json(const char *path);  // falls back to defaults on failure
