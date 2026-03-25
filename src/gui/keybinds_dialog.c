// NOTE: NK_IMPLEMENTATION is owned by app.c — include nuklear declaration-only here.
#include "nuklear.h"
#include "gui/keybinds_dialog.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Scancode -> display name table (SDL3 numeric values, subset used by keybind.c)
// ---------------------------------------------------------------------------

typedef struct { int sc; const char *name; } sc_name;

static const sc_name k_sc_names[] = {
  {  4, "A"      }, {  5, "B"      }, {  6, "C"      }, {  7, "D"      },
  {  8, "E"      }, {  9, "F"      }, { 10, "G"      }, { 11, "H"      },
  { 12, "I"      }, { 13, "J"      }, { 14, "K"      }, { 15, "L"      },
  { 16, "M"      }, { 17, "N"      }, { 18, "O"      }, { 19, "P"      },
  { 20, "Q"      }, { 21, "R"      }, { 22, "S"      }, { 23, "T"      },
  { 24, "U"      }, { 25, "V"      }, { 26, "W"      }, { 27, "X"      },
  { 28, "Y"      }, { 29, "Z"      },
  { 45, "-"      }, { 46, "="      }, { 54, ","      }, { 55, "."      },
  { 58, "F1"     }, { 59, "F2"     }, { 60, "F3"     }, { 61, "F4"     },
  { 62, "F5"     }, { 63, "F6"     }, { 64, "F7"     }, { 65, "F8"     },
  { 66, "F9"     }, { 67, "F10"    }, { 68, "F11"    }, { 69, "F12"    },
  { 79, "Right"  }, { 80, "Left"   }, { 81, "Down"   }, { 82, "Up"     },
  { 41, "Esc"    }, { 40, "Enter"  }, { 43, "Tab"    }, { 44, "Space"  },
  {  0, NULL     },
};

static const char *sc_to_name(int sc) {
  for (int i = 0; k_sc_names[i].name; i++)
    if (k_sc_names[i].sc == sc) return k_sc_names[i].name;
  return "?";
}

// Format a scancode+modifier pair into buf.
static void format_binding(int sc, int mod, char *buf, size_t bufsz) {
  buf[0] = '\0';
  if (sc == 0) { snprintf(buf, bufsz, "(none)"); return; }
  char tmp[32] = "";
  // mod: 0x00C0 = Ctrl, 0x0003 = Shift (SDL3 kmod bit patterns from keybind.c)
  if (mod & 0x00C0) strncat(tmp, "Ctrl+",  sizeof(tmp) - strlen(tmp) - 1);
  if (mod & 0x0003) strncat(tmp, "Shift+", sizeof(tmp) - strlen(tmp) - 1);
  strncat(tmp, sc_to_name(sc), sizeof(tmp) - strlen(tmp) - 1);
  snprintf(buf, bufsz, "%s", tmp);
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

// Working copy of bindings that the dialog edits before Save/Cancel.
typedef struct { int sc; int mod; } wbind;

struct keybinds_dialog {
  bool         visible;
  keybind_map *map;          // external map to write back on Save

  wbind  work[ACTION_COUNT]; // editable copy
  wbind  defs[ACTION_COUNT]; // defaults snapshot (for "Reset")

  int    rebind_row;         // action currently awaiting key (-1 = none)
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

keybinds_dialog *keybinds_dialog_new(keybind_map *binds) {
  REQUIRE(binds, "keybinds_dialog_new: null map");
  keybinds_dialog *d = calloc(1, sizeof(*d));
  REQUIRE(d, "keybinds_dialog_new: calloc failed");
  d->map        = binds;
  d->rebind_row = -1;

  // Snapshot current bindings into working copy
  for (int a = 0; a < ACTION_COUNT; a++)
    keybind_get(binds, (action_id)a, &d->work[a].sc, &d->work[a].mod);

  // Snapshot defaults by creating a throwaway keybind_map
  keybind_map *def = keybind_new();
  if (def) {
    for (int a = 0; a < ACTION_COUNT; a++)
      keybind_get(def, (action_id)a, &d->defs[a].sc, &d->defs[a].mod);
    keybind_free(def);
  }

  return d;
}

void keybinds_dialog_free(keybinds_dialog *d) {
  free(d);
}

void keybinds_dialog_show(keybinds_dialog *d) {
  REQUIRE(d, "keybinds_dialog_show: null");
  d->visible = true;
  d->rebind_row = -1;
  // Re-snapshot current bindings each time dialog is opened
  for (int a = 0; a < ACTION_COUNT; a++)
    keybind_get(d->map, (action_id)a, &d->work[a].sc, &d->work[a].mod);
}

bool keybinds_dialog_is_visible(const keybinds_dialog *d) {
  return d && d->visible;
}

bool keybinds_dialog_is_waiting_for_key(const keybinds_dialog *d) {
  return d && d->visible && d->rebind_row >= 0;
}

void keybinds_dialog_inject_key(keybinds_dialog *d, int scancode, int modifiers) {
  if (!d || d->rebind_row < 0 || d->rebind_row >= ACTION_COUNT) return;
  d->work[d->rebind_row].sc  = scancode;
  d->work[d->rebind_row].mod = modifiers;
  d->rebind_row = -1;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

bool keybinds_dialog_render(keybinds_dialog *d, struct nk_context *ctx) {
  if (!d || !ctx || !d->visible) return false;

  static const float ROW_H  = 20.0f;
  static const float BTN_H  = 26.0f;
  static const float COL_W0 = 180.0f;  // Action
  static const float COL_W1 = 110.0f;  // Current Key
  static const float COL_W2 = 110.0f;  // Default Key

  if (nk_begin(ctx, "Keyboard Shortcuts",
               nk_rect(80, 60, 450, 440),
               NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_MOVABLE |
               NK_WINDOW_SCALABLE)) {

    // Column headers
    nk_layout_row_begin(ctx, NK_STATIC, ROW_H, 3);
    nk_layout_row_push(ctx, COL_W0);
    nk_label(ctx, "Action",      NK_TEXT_LEFT);
    nk_layout_row_push(ctx, COL_W1);
    nk_label(ctx, "Current Key", NK_TEXT_LEFT);
    nk_layout_row_push(ctx, COL_W2);
    nk_label(ctx, "Default Key", NK_TEXT_LEFT);
    nk_layout_row_end(ctx);

    // Scrollable list
    nk_layout_row_dynamic(ctx, 280.0f, 1);
    if (nk_group_begin(ctx, "binds_list", NK_WINDOW_BORDER)) {
      char cur_buf[32], def_buf[32];
      for (int a = 0; a < ACTION_COUNT; a++) {
        bool awaiting = (d->rebind_row == a);
        format_binding(d->work[a].sc, d->work[a].mod, cur_buf, sizeof(cur_buf));
        format_binding(d->defs[a].sc, d->defs[a].mod, def_buf, sizeof(def_buf));

        nk_layout_row_begin(ctx, NK_STATIC, ROW_H, 3);

        // Action name
        nk_layout_row_push(ctx, COL_W0);
        nk_label(ctx, keybind_action_name((action_id)a), NK_TEXT_LEFT);

        // Current key — button to trigger rebind
        nk_layout_row_push(ctx, COL_W1);
        const char *btn_label = awaiting ? "[ press key... ]" : cur_buf;
        if (nk_button_label(ctx, btn_label)) {
          d->rebind_row = awaiting ? -1 : a;
        }

        // Default key (read-only label)
        nk_layout_row_push(ctx, COL_W2);
        nk_label(ctx, def_buf, NK_TEXT_LEFT);

        nk_layout_row_end(ctx);
      }
      nk_group_end(ctx);
    }

    // Bottom buttons
    nk_layout_row_dynamic(ctx, 8.0f, 1);
    nk_spacing(ctx, 1);

    nk_layout_row_begin(ctx, NK_STATIC, BTN_H, 3);

    nk_layout_row_push(ctx, 130.0f);
    if (nk_button_label(ctx, "Reset to Defaults")) {
      for (int a = 0; a < ACTION_COUNT; a++)
        d->work[a] = d->defs[a];
      d->rebind_row = -1;
    }

    nk_layout_row_push(ctx, 130.0f);
    if (nk_button_label(ctx, "Save")) {
      for (int a = 0; a < ACTION_COUNT; a++)
        keybind_set(d->map, (action_id)a, d->work[a].sc, d->work[a].mod);
      d->visible    = false;
      d->rebind_row = -1;
    }

    nk_layout_row_push(ctx, 130.0f);
    if (nk_button_label(ctx, "Cancel")) {
      d->visible    = false;
      d->rebind_row = -1;
    }

    nk_layout_row_end(ctx);
  }
  nk_end(ctx);

  return d->visible;
}
