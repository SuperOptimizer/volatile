#include "gui/dt_panel.h"

#include <stdlib.h>
#include <string.h>

// Nuklear is included only when compiling as part of volatile_gui.
// Tests stub this out.
#ifndef NK_INCLUDE_DEFAULT_ALLOCATOR
struct nk_context;
#define NK_STUB
#endif

#ifndef NK_STUB
#define NK_IMPLEMENTATION
#include "nuklear.h"
#endif

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct dt_panel {
  bool    visible;
  float   threshold;     // EDT threshold in voxels [0..512]
  int     cmap_sel;      // index into cmap_id enum
  bool    auto_compute;
  bool    dirty;

  dt_recompute_fn recompute_fn;
  void           *recompute_ctx;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

dt_panel *dt_panel_new(void) {
  dt_panel *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->visible      = true;
  p->threshold    = 32.0f;
  p->cmap_sel     = CMAP_VIRIDIS;
  p->auto_compute = true;
  p->dirty        = false;
  return p;
}

void dt_panel_free(dt_panel *p) {
  free(p);
}

// ---------------------------------------------------------------------------
// Callbacks
// ---------------------------------------------------------------------------

void dt_panel_set_recompute_cb(dt_panel *p, dt_recompute_fn fn, void *ctx) {
  if (!p) return;
  p->recompute_fn  = fn;
  p->recompute_ctx = ctx;
}

void dt_panel_on_surface_changed(dt_panel *p) {
  if (!p) return;
  if (p->auto_compute) {
    p->dirty = true;
    if (p->recompute_fn)
      p->recompute_fn(p->threshold, (cmap_id)p->cmap_sel, p->recompute_ctx);
  }
}

// ---------------------------------------------------------------------------
// State queries
// ---------------------------------------------------------------------------

bool    dt_panel_visible(const dt_panel *p)      { return p && p->visible; }
float   dt_panel_threshold(const dt_panel *p)    { return p ? p->threshold : 0.0f; }
cmap_id dt_panel_cmap(const dt_panel *p)         { return p ? (cmap_id)p->cmap_sel : CMAP_VIRIDIS; }
bool    dt_panel_auto_compute(const dt_panel *p) { return p && p->auto_compute; }
bool    dt_panel_dirty(const dt_panel *p)        { return p && p->dirty; }
void    dt_panel_clear_dirty(dt_panel *p)        { if (p) p->dirty = false; }

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

static const char *cmap_labels[] = {
  "grayscale", "viridis", "magma", "inferno", "plasma",
  "hot", "cool", "bone", "jet", "turbo",
};
#define N_CMAPS 10

void dt_panel_render(dt_panel *p, struct nk_context *ctx, const char *title) {
#ifdef NK_STUB
  (void)p; (void)ctx; (void)title;
  return;
#else
  if (!p || !ctx) return;
  (void)title;  // panel rendered inside caller's nk window

  // --- Visibility toggle ---
  nk_layout_row_dynamic(ctx, 22, 2);
  nk_label(ctx, "Show overlay", NK_TEXT_LEFT);
  if (nk_checkbox_label(ctx, "", &(int){p->visible ? 1 : 0}))
    p->visible = !p->visible;

  // --- Threshold slider ---
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "EDT threshold (voxels)", NK_TEXT_LEFT);
  nk_layout_row_dynamic(ctx, 22, 1);
  float prev_thresh = p->threshold;
  p->threshold = nk_slide_float(ctx, 0.0f, p->threshold, 512.0f, 1.0f);
  if (p->threshold != prev_thresh) p->dirty = true;

  // --- Colormap ---
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Colormap", NK_TEXT_LEFT);
  nk_layout_row_dynamic(ctx, 22, 1);
  int prev_cmap = p->cmap_sel;
  p->cmap_sel = nk_combo(ctx, cmap_labels, N_CMAPS, p->cmap_sel, 22,
                          nk_vec2(200, 160));
  if (p->cmap_sel != prev_cmap) p->dirty = true;

  // --- Auto-compute checkbox ---
  nk_layout_row_dynamic(ctx, 22, 2);
  nk_label(ctx, "Auto-compute", NK_TEXT_LEFT);
  int auto_val = p->auto_compute ? 1 : 0;
  nk_checkbox_label(ctx, "", &auto_val);
  p->auto_compute = (auto_val != 0);

  // --- Manual recompute button ---
  nk_layout_row_dynamic(ctx, 28, 1);
  if (nk_button_label(ctx, "Recompute Now")) {
    p->dirty = true;
    if (p->recompute_fn)
      p->recompute_fn(p->threshold, (cmap_id)p->cmap_sel, p->recompute_ctx);
  }
#endif
}
