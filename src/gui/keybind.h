#pragma once
#include <stdbool.h>

typedef enum {
  // Pan (WASD — tested)
  ACTION_PAN_LEFT,  ACTION_PAN_RIGHT,  ACTION_PAN_UP,  ACTION_PAN_DOWN,

  // Zoom (tested)
  ACTION_ZOOM_IN,   ACTION_ZOOM_OUT,

  // Slice (tested)
  ACTION_SLICE_NEXT, ACTION_SLICE_PREV,

  // Tools (tested)
  ACTION_TOOL_BRUSH, ACTION_TOOL_LINE,  ACTION_TOOL_PUSHPULL,

  // Undo/Redo (tested)
  ACTION_UNDO, ACTION_REDO,

  // Focus (tested legacy names)
  ACTION_FOCUS_NEXT, ACTION_FOCUS_PREV,

  // Overlay / segmentation toggle (tested)
  ACTION_TOGGLE_OVERLAY, ACTION_TOGGLE_SEGMENTATION,

  // Save / fullscreen (tested)
  ACTION_SAVE, ACTION_FULLSCREEN,

  // -----------------------------------------------------------------------
  // VC3D full shortcut catalog
  // -----------------------------------------------------------------------

  // Navigation
  ACTION_CENTER_ON_CURSOR,   // R
  ACTION_FOCUS_BACK,         // F
  ACTION_FOCUS_FORWARD,      // Ctrl+F
  ACTION_RESET_VIEW,         // M

  // Segments
  ACTION_SEG_NEXT,   // ]
  ACTION_SEG_PREV,   // [

  // Editing
  ACTION_LINE_DRAW,        // S (hold)
  ACTION_CORRECTION_MODE,  // T
  ACTION_EDIT_MODE,        // Shift+T
  ACTION_CANCEL,           // Esc

  // Approval
  ACTION_PAINT,      // B
  ACTION_UNPAINT,    // N
  ACTION_UNDO_MASK,  // Ctrl+B

  // Growth
  ACTION_GROW_ALL,     // Ctrl+G
  ACTION_GROW_LEFT,    // 1
  ACTION_GROW_UP,      // 2
  ACTION_GROW_DOWN,    // 3
  ACTION_GROW_RIGHT,   // 4
  ACTION_GROW_ALL_DIR, // 5
  ACTION_GROW_ONE,     // 6

  // Push/Pull
  ACTION_PUSH,           // A (hold) — also pan_left; push gets secondary binding
  ACTION_PULL,           // D (hold) — also pan_right; pull gets secondary binding
  ACTION_RADIUS_SMALLER, // Q
  ACTION_RADIUS_BIGGER,  // E

  // View
  ACTION_COMPOSITE,      // C
  ACTION_RAW_POINTS,     // P
  ACTION_SLICE_PLANES,   // Ctrl+J
  ACTION_TOGGLE_NORMALS, // Ctrl+N

  ACTION_COUNT
} action_id;

typedef struct keybind_map keybind_map;

keybind_map       *keybind_new(void);    // creates with VC3D-compatible defaults
void               keybind_free(keybind_map *m);
void               keybind_set(keybind_map *m, action_id action, int sdl_scancode, int modifiers);
int                keybind_lookup(const keybind_map *m, int sdl_scancode, int modifiers); // action_id or -1
void               keybind_get(const keybind_map *m, action_id action, int *sdl_scancode_out, int *modifiers_out);
const char        *keybind_action_name(action_id a);
bool               keybind_save_json(const keybind_map *m, const char *path);
keybind_map       *keybind_load_json(const char *path);  // falls back to defaults on failure
