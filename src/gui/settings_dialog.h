// WIDGET TYPE: WINDOW — renders its own nk_begin/nk_end window, call OUTSIDE any nk_begin block.
#pragma once
#include <stdbool.h>

// Forward declarations
struct nk_context;
typedef struct settings settings;

// ---------------------------------------------------------------------------
// Settings dialog: Nuklear-based UI for all app preferences.
// ---------------------------------------------------------------------------

typedef struct settings_dialog settings_dialog;

// Create dialog bound to the given settings object (does not take ownership).
settings_dialog *settings_dialog_new(settings *prefs);
void             settings_dialog_free(settings_dialog *d);

// Show / hide the dialog window.
void settings_dialog_show(settings_dialog *d);
bool settings_dialog_is_visible(const settings_dialog *d);

// Render one frame.  Returns true if any setting changed this frame.
// Call each frame while the dialog may be visible.
bool settings_dialog_render(settings_dialog *d, struct nk_context *ctx);
