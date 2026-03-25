#pragma once
#include <stdbool.h>
#include "render/cmap.h"

struct nk_context;

// ---------------------------------------------------------------------------
// dt_panel — Distance Transform overlay panel.
//
// Controls visibility of the EDT overlay, threshold slider, colormap
// selection, and auto-compute on surface change.
//
// Render loop usage:
//   dt_panel_render(p, ctx, "Distance Transform");
//   if (dt_panel_dirty(p)) {
//     recompute_edt(dt_panel_threshold(p), dt_panel_cmap(p));
//     dt_panel_clear_dirty(p);
//   }
// ---------------------------------------------------------------------------

typedef struct dt_panel dt_panel;

// Callback invoked when user requests recomputation.
typedef void (*dt_recompute_fn)(float threshold, cmap_id cmap, void *ctx);

dt_panel *dt_panel_new(void);
void      dt_panel_free(dt_panel *p);

// Draw the panel inside the current Nuklear window.
void dt_panel_render(dt_panel *p, struct nk_context *ctx, const char *title);

// Wire a recompute callback (called when auto_compute is on or user clicks).
void dt_panel_set_recompute_cb(dt_panel *p, dt_recompute_fn fn, void *ctx);

// Notify the panel that the surface has changed (triggers auto-recompute).
void dt_panel_on_surface_changed(dt_panel *p);

// State queries
bool    dt_panel_visible(const dt_panel *p);     // overlay enabled
float   dt_panel_threshold(const dt_panel *p);   // EDT threshold (voxels)
cmap_id dt_panel_cmap(const dt_panel *p);        // colormap for distance field
bool    dt_panel_auto_compute(const dt_panel *p);// auto recompute on change
bool    dt_panel_dirty(const dt_panel *p);       // needs recompute
void    dt_panel_clear_dirty(dt_panel *p);
