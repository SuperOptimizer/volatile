#include "gui/draw_panel.h"
#include "gui/drawing.h"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Preset palette (10 colours + eraser-white)
// ---------------------------------------------------------------------------

#define N_PALETTE 10

static const uint8_t g_palette[N_PALETTE][4] = {
  {220,  50,  50, 255},  // red
  {255, 140,   0, 255},  // orange
  {255, 215,   0, 255},  // yellow
  { 50, 200,  50, 255},  // green
  { 30, 144, 255, 255},  // blue
  {148,   0, 211, 255},  // purple
  {255, 105, 180, 255},  // pink
  { 64, 224, 208, 255},  // teal
  {255, 255, 255, 255},  // white
  {  0,   0,   0, 255},  // black
};

// ---------------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------------

typedef struct {
  drawing_canvas *canvas;
  char            name[32];
} layer;

// ---------------------------------------------------------------------------
// draw_panel
// ---------------------------------------------------------------------------

struct draw_panel {
  int       canvas_w, canvas_h;

  layer     layers[DRAW_PANEL_MAX_LAYERS];
  int       n_layers;
  int       active;          // index into layers[]

  draw_params params;        // current tool settings
  int         palette_sel;   // selected palette entry

  bool        mouse_down;
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static layer *active_layer(draw_panel *p) {
  return &p->layers[p->active];
}

static void init_layer(layer *l, int canvas_w, int canvas_h, int idx) {
  l->canvas = drawing_canvas_new(canvas_w, canvas_h);
  snprintf(l->name, sizeof(l->name), "Layer %d", idx + 1);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

draw_panel *draw_panel_new(int canvas_w, int canvas_h) {
  draw_panel *p = calloc(1, sizeof(*p));
  if (!p) return NULL;

  p->canvas_w = canvas_w;
  p->canvas_h = canvas_h;

  // Start with one layer
  init_layer(&p->layers[0], canvas_w, canvas_h, 0);
  p->n_layers = 1;
  p->active   = 0;

  // Default tool params
  p->params.tool         = DRAW_FREEHAND;
  p->params.brush_radius = 8.0f;
  p->params.line_width   = 2.0f;
  memcpy(p->params.color, g_palette[0], 4);
  p->palette_sel = 0;

  return p;
}

void draw_panel_free(draw_panel *p) {
  if (!p) return;
  for (int i = 0; i < p->n_layers; i++)
    drawing_canvas_free(p->layers[i].canvas);
  free(p);
}

// ---------------------------------------------------------------------------
// Nuklear UI
// ---------------------------------------------------------------------------

void draw_panel_render(draw_panel *p, struct nk_context *ctx) {
  if (!p || !ctx) return;

  // ---- Tool selector ----
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Tool", NK_TEXT_LEFT);

  nk_layout_row_dynamic(ctx, 28, 5);
  static const char *tool_labels[] = {"Brush","Erase","Line","Rect","Circle"};
  static const draw_tool tool_vals[] = {
    DRAW_FREEHAND, DRAW_ERASER, DRAW_LINE, DRAW_RECT, DRAW_CIRCLE
  };
  for (int i = 0; i < 5; i++) {
    bool active = (p->params.tool == tool_vals[i]);
    if (nk_button_label(ctx, tool_labels[i])) {
      p->params.tool = tool_vals[i];
    }
    (void)active;
  }

  // ---- Brush size ----
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Brush size", NK_TEXT_LEFT);
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_slider_float(ctx, 1.0f, &p->params.brush_radius, 64.0f, 1.0f);

  // ---- Colour palette ----
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Colour", NK_TEXT_LEFT);
  nk_layout_row_dynamic(ctx, 28, N_PALETTE);
  for (int i = 0; i < N_PALETTE; i++) {
    struct nk_color col = {
      g_palette[i][0], g_palette[i][1], g_palette[i][2], g_palette[i][3]
    };
    if (nk_button_color(ctx, col)) {
      p->palette_sel = i;
      memcpy(p->params.color, g_palette[i], 4);
    }
  }

  // ---- Undo / Redo / Clear ----
  nk_layout_row_dynamic(ctx, 28, 3);
  if (nk_button_label(ctx, "Undo"))  drawing_undo(active_layer(p)->canvas);
  if (nk_button_label(ctx, "Redo"))  drawing_redo(active_layer(p)->canvas);
  if (nk_button_label(ctx, "Clear")) drawing_clear(active_layer(p)->canvas);

  // ---- Export mask (stub — real export goes via a file dialog) ----
  nk_layout_row_dynamic(ctx, 28, 1);
  if (nk_button_label(ctx, "Export mask")) {
    // Caller should hook this; for now no-op (no filesystem access from UI)
  }

  // ---- Layer management ----
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Layers", NK_TEXT_LEFT);

  for (int i = 0; i < p->n_layers; i++) {
    nk_layout_row_dynamic(ctx, 22, 2);
    bool is_active = (i == p->active);
    char lbl[48];
    snprintf(lbl, sizeof(lbl), "%s%s", p->layers[i].name, is_active ? " *" : "");
    if (nk_button_label(ctx, lbl)) p->active = i;
    if (nk_button_label(ctx, "X") && p->n_layers > 1)
      draw_panel_remove_layer(p, i);
  }

  nk_layout_row_dynamic(ctx, 28, 1);
  if (p->n_layers < DRAW_PANEL_MAX_LAYERS) {
    if (nk_button_label(ctx, "+ Layer")) draw_panel_add_layer(p);
  }
}

// ---------------------------------------------------------------------------
// Mouse forwarding
// ---------------------------------------------------------------------------

void draw_panel_mouse_down(draw_panel *p, float x, float y) {
  if (!p) return;
  p->mouse_down = true;
  drawing_canvas *c = active_layer(p)->canvas;
  if (p->params.tool == DRAW_FREEHAND || p->params.tool == DRAW_ERASER) {
    drawing_begin_stroke(c, x, y, &p->params);
  } else {
    drawing_begin_shape(c, x, y, p->params.tool, &p->params);
  }
}

void draw_panel_mouse_drag(draw_panel *p, float x, float y) {
  if (!p || !p->mouse_down) return;
  drawing_canvas *c = active_layer(p)->canvas;
  if (p->params.tool == DRAW_FREEHAND || p->params.tool == DRAW_ERASER) {
    drawing_continue_stroke(c, x, y);
  } else {
    drawing_update_shape(c, x, y);
  }
}

void draw_panel_mouse_up(draw_panel *p) {
  if (!p) return;
  p->mouse_down = false;
  drawing_canvas *c = active_layer(p)->canvas;
  if (p->params.tool == DRAW_FREEHAND || p->params.tool == DRAW_ERASER) {
    drawing_end_stroke(c);
  } else {
    drawing_finish_shape(c);
  }
}

// ---------------------------------------------------------------------------
// Read-back / export
// ---------------------------------------------------------------------------

const uint8_t *draw_panel_get_pixels(const draw_panel *p) {
  if (!p) return NULL;
  return drawing_get_pixels(p->layers[p->active].canvas);
}

void draw_panel_export_mask(const draw_panel *p, uint8_t *out) {
  if (!p || !out) return;
  drawing_export_mask(p->layers[p->active].canvas, out);
}

// ---------------------------------------------------------------------------
// Layer management
// ---------------------------------------------------------------------------

int draw_panel_active_layer(const draw_panel *p) {
  return p ? p->active : 0;
}

void draw_panel_set_layer(draw_panel *p, int layer) {
  if (!p || layer < 0 || layer >= p->n_layers) return;
  p->active = layer;
}

int draw_panel_layer_count(const draw_panel *p) {
  return p ? p->n_layers : 0;
}

bool draw_panel_add_layer(draw_panel *p) {
  if (!p || p->n_layers >= DRAW_PANEL_MAX_LAYERS) return false;
  init_layer(&p->layers[p->n_layers], p->canvas_w, p->canvas_h, p->n_layers);
  p->active = p->n_layers;
  p->n_layers++;
  return true;
}

bool draw_panel_remove_layer(draw_panel *p, int layer) {
  if (!p || p->n_layers <= 1 || layer < 0 || layer >= p->n_layers) return false;
  drawing_canvas_free(p->layers[layer].canvas);
  // shift remaining layers down
  for (int i = layer; i < p->n_layers - 1; i++)
    p->layers[i] = p->layers[i + 1];
  p->n_layers--;
  if (p->active >= p->n_layers) p->active = p->n_layers - 1;
  return true;
}
