#pragma once
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// welcome_panel — home screen shown when no volume is loaded
// ---------------------------------------------------------------------------

typedef struct welcome_panel welcome_panel;

typedef enum {
  WELCOME_NONE = 0,
  WELCOME_OPEN_ZARR,
  WELCOME_OPEN_VOLPKG,
  WELCOME_OPEN_S3,
  WELCOME_OPEN_URL,
  WELCOME_OPEN_PROJECT,
  WELCOME_OPEN_RECENT,
} welcome_action;

typedef struct {
  welcome_action action;
  char           url[2048];  // populated for WELCOME_OPEN_URL or WELCOME_OPEN_RECENT
} welcome_result;

welcome_panel *welcome_panel_new(void);
void           welcome_panel_free(welcome_panel *w);

// Add a recent file entry. path is the URL/path used to open; name is a
// short display label. Up to 8 recent entries are kept.
void welcome_panel_add_recent(welcome_panel *w, const char *path,
                               const char *name);

// Render the panel filling the given pixel dimensions.
// Must be called inside an open nk_begin / nk_end block.
// Returns a result with action != WELCOME_NONE when the user clicks something.
welcome_result welcome_panel_render(welcome_panel *w, struct nk_context *ctx,
                                    int panel_w, int panel_h);
