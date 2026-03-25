#pragma once
#include "gui/drawing.h"
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// draw_panel — Nuklear UI panel wrapping a drawing_canvas
//
// Provides tool selector, brush-size slider, preset colour palette,
// path-ID layer management, clear/undo/redo buttons, and mask export.
// ---------------------------------------------------------------------------

#define DRAW_PANEL_MAX_LAYERS 16

typedef struct draw_panel draw_panel;

// Create a panel backed by a canvas of the given dimensions.
draw_panel *draw_panel_new(int canvas_w, int canvas_h);
void        draw_panel_free(draw_panel *p);

// Render the panel UI.  Call inside an open nk_begin / nk_end block.
void draw_panel_render(draw_panel *p, struct nk_context *ctx);

// Mouse events forwarded from the host window (canvas-local coordinates).
void draw_panel_mouse_down(draw_panel *p, float x, float y);
void draw_panel_mouse_drag(draw_panel *p, float x, float y);
void draw_panel_mouse_up(draw_panel *p);

// Read-back for rendering or export.
const uint8_t *draw_panel_get_pixels(const draw_panel *p);

// Export the current layer as a single-channel mask.
// `out` must be canvas_w * canvas_h bytes.
void draw_panel_export_mask(const draw_panel *p, uint8_t *out);

// Active layer / path ID management.
int  draw_panel_active_layer(const draw_panel *p);
void draw_panel_set_layer(draw_panel *p, int layer);
int  draw_panel_layer_count(const draw_panel *p);
bool draw_panel_add_layer(draw_panel *p);    // returns false if at max
bool draw_panel_remove_layer(draw_panel *p, int layer);
