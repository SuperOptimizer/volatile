// WIDGET TYPE: CONTENT — call inside an nk_begin/nk_end block.
#pragma once
#include <stdbool.h>

// Forward declarations
struct nk_context;
typedef struct volume volume;
typedef struct chunk_cache chunk_cache;

// ---------------------------------------------------------------------------
// vol_info_panel — Nuklear panel showing volume metadata and cache stats.
// ---------------------------------------------------------------------------

typedef struct vol_info_panel vol_info_panel;

vol_info_panel *vol_info_panel_new(void);
void            vol_info_panel_free(vol_info_panel *p);

// Render the panel. vol and cache may be NULL (shows "No volume loaded").
// Call inside an nk_begin/nk_end pair, or pass title != NULL to open its own window.
// Returns true if the user clicked Refresh.
bool vol_info_panel_render(vol_info_panel *p, struct nk_context *ctx,
                           volume *vol, chunk_cache *cache);
