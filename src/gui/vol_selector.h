// WIDGET TYPE: CONTENT — call inside an nk_begin/nk_end block.
#pragma once
#include <stdbool.h>

// Forward-declare nk_context to avoid pulling nuklear.h into consumers.
struct nk_context;

// ---------------------------------------------------------------------------
// vol_selector — combo-box volume picker (port of VC3D VolumeSelector)
// ---------------------------------------------------------------------------

typedef struct vol_selector vol_selector;

vol_selector *vol_selector_new(void);
void          vol_selector_free(vol_selector *s);

// Add a named volume entry. name is a display label; path is the local path or URL.
void vol_selector_add(vol_selector *s, const char *name, const char *path);
void vol_selector_clear(vol_selector *s);

// Render as a Nuklear combo box. Returns true if the selection changed this frame.
bool vol_selector_render(vol_selector *s, struct nk_context *ctx);

// Query current state.
int         vol_selector_selected(const vol_selector *s);      // -1 if empty
const char *vol_selector_selected_path(const vol_selector *s); // NULL if empty
int         vol_selector_count(const vol_selector *s);
