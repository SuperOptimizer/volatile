#pragma once
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// about_dialog — shows app name, version, build info, and credits.
// ---------------------------------------------------------------------------

typedef struct about_dialog about_dialog;

about_dialog *about_dialog_new(void);
void          about_dialog_free(about_dialog *d);

// Make the dialog visible.
void about_dialog_show(about_dialog *d);

bool about_dialog_is_visible(const about_dialog *d);

// Render one frame.  Returns true while the dialog is still open,
// false once the user dismisses it.
bool about_dialog_render(about_dialog *d, struct nk_context *ctx);
