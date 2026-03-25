// WIDGET TYPE: WINDOW — renders its own nk_begin/nk_end window, call OUTSIDE any nk_begin block.
#pragma once
#include "gui/keybind.h"
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// keybinds_dialog — scrollable list of all keyboard shortcuts.
//
// Columns: Action | Current Key | Default Key
// Click a row to enter rebind mode (next key press replaces the binding).
// "Reset to Defaults" reverts all bindings.
// "Save" commits changes back to the keybind_map; "Cancel" discards them.
// ---------------------------------------------------------------------------

typedef struct keybinds_dialog keybinds_dialog;

// Create dialog bound to the given keybind_map (does not take ownership).
keybinds_dialog *keybinds_dialog_new(keybind_map *binds);
void             keybinds_dialog_free(keybinds_dialog *d);

// Make the dialog visible.
void keybinds_dialog_show(keybinds_dialog *d);

bool keybinds_dialog_is_visible(const keybinds_dialog *d);

// Render one frame.  Returns true while the dialog is still open.
// ctx may be NULL — returns false without crashing.
bool keybinds_dialog_render(keybinds_dialog *d, struct nk_context *ctx);

// Programmatically inject a key press for the row currently awaiting rebind.
// scancode and modifiers use the same numeric values as SDL3 scancodes/kmod.
// Call this from the SDL event loop when keybinds_dialog_is_waiting_for_key.
void keybinds_dialog_inject_key(keybinds_dialog *d, int scancode, int modifiers);

bool keybinds_dialog_is_waiting_for_key(const keybinds_dialog *d);
