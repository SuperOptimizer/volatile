// nk_widgets.c — custom Nuklear widgets for volatile
//
// NK_IMPLEMENTATION is owned by app.c (exactly once in the link unit).
// Here we include nuklear.h declaration-only by defining the feature flags
// but NOT NK_IMPLEMENTATION.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "nk_widgets.h"
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <alloca.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define ROW_H   22.0f   // default widget row height
#define LABEL_W 0.40f   // fraction of row given to the label column

// snprintf into a fixed-size buffer; always NUL-terminates.
static void fmt_buf(char *dst, int dstn, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(dst, (size_t)dstn, fmt, ap);
  va_end(ap);
  dst[dstn - 1] = '\0';
}

// ---------------------------------------------------------------------------
// nk_widget_range_slider
// Two nk_slider_float widgets stacked in a single row.
// The lower handle is clamped to <= upper and vice-versa.
// ---------------------------------------------------------------------------

bool nk_widget_range_slider(struct nk_context *ctx,
                            float *min_val, float *max_val,
                            float lo, float hi, float step) {
  bool changed = false;
  float prev_min = *min_val;
  float prev_max = *max_val;

  nk_layout_row_dynamic(ctx, ROW_H, 2);

  // Lower handle
  if (nk_slider_float(ctx, lo, min_val, *max_val, step)) changed = true;
  // Upper handle
  if (nk_slider_float(ctx, *min_val, max_val, hi, step)) changed = true;

  // Enforce ordering in case of simultaneous drag (shouldn't normally happen).
  if (*min_val > *max_val) { float t = *min_val; *min_val = *max_val; *max_val = t; }

  (void)prev_min; (void)prev_max;
  return changed;
}

// ---------------------------------------------------------------------------
// nk_widget_collapsible_begin / _end
// Wraps nk_tree_push / nk_tree_pop using NK_TREE_TAB style.
// *expanded carries the collapse state across frames.
// ---------------------------------------------------------------------------

bool nk_widget_collapsible_begin(struct nk_context *ctx, const char *title, bool *expanded) {
  enum nk_collapse_states state = *expanded ? NK_MAXIMIZED : NK_MINIMIZED;
  // nk_tree_push_hashed uses caller-provided hash for stable state across re-orders.
  bool open = (bool)nk_tree_push_hashed(ctx, NK_TREE_TAB, title, state,
                                        title, (int)strlen(title), 0);
  *expanded = open;
  return open;
}

void nk_widget_collapsible_end(struct nk_context *ctx) {
  nk_tree_pop(ctx);
}

// ---------------------------------------------------------------------------
// nk_widget_labeled_float / _int / _str
// Two-column row: left = label (LABEL_W fraction), right = value.
// ---------------------------------------------------------------------------

void nk_widget_labeled_float(struct nk_context *ctx, const char *label, float value, const char *fmt) {
  char vbuf[32];
  fmt_buf(vbuf, sizeof(vbuf), fmt ? fmt : "%.4g", value);
  nk_layout_row_begin(ctx, NK_DYNAMIC, ROW_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, label, NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 1.0f - LABEL_W);
    nk_label(ctx, vbuf, NK_TEXT_RIGHT);
  nk_layout_row_end(ctx);
}

void nk_widget_labeled_int(struct nk_context *ctx, const char *label, int value) {
  char vbuf[32];
  fmt_buf(vbuf, sizeof(vbuf), "%d", value);
  nk_layout_row_begin(ctx, NK_DYNAMIC, ROW_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, label, NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 1.0f - LABEL_W);
    nk_label(ctx, vbuf, NK_TEXT_RIGHT);
  nk_layout_row_end(ctx);
}

void nk_widget_labeled_str(struct nk_context *ctx, const char *label, const char *value) {
  nk_layout_row_begin(ctx, NK_DYNAMIC, ROW_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, label, NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 1.0f - LABEL_W);
    nk_label(ctx, value ? value : "", NK_TEXT_RIGHT);
  nk_layout_row_end(ctx);
}

// ---------------------------------------------------------------------------
// nk_widget_color_swatch
// Allocates a widget region of (width x height) and fills it with the color.
// ---------------------------------------------------------------------------

void nk_widget_color_swatch(struct nk_context *ctx,
                             uint8_t r, uint8_t g, uint8_t b,
                             float width, float height) {
  nk_layout_row_static(ctx, height, (int)width, 1);
  struct nk_rect bounds = nk_widget_bounds(ctx);
  struct nk_command_buffer *canvas = nk_window_get_canvas(ctx);
  // Advance the widget cursor without drawing a control.
  nk_spacing(ctx, 1);
  struct nk_color col = {r, g, b, 255};
  nk_fill_rect(canvas, bounds, 0.0f, col);
  // Draw a thin border so the swatch is visible on matching backgrounds.
  struct nk_color border = {(uint8_t)(r/2), (uint8_t)(g/2), (uint8_t)(b/2), 255};
  nk_stroke_rect(canvas, bounds, 0.0f, 1.0f, border);
}

// ---------------------------------------------------------------------------
// nk_widget_searchable_combo
// Shows an nk_edit_string filter box followed by a filtered combo list.
// Returns true if *selected changed.
// ---------------------------------------------------------------------------

bool nk_widget_searchable_combo(struct nk_context *ctx,
                                const char **items, int count, int *selected,
                                char *filter_buf, int filter_len) {
  // Edit filter field.
  nk_layout_row_dynamic(ctx, ROW_H, 1);
  int filter_text_len = (int)strlen(filter_buf);
  nk_edit_string(ctx, NK_EDIT_SIMPLE, filter_buf, &filter_text_len, filter_len - 1, nk_filter_default);
  filter_buf[filter_text_len] = '\0';

  // Build filtered item list. We pass matching item indices to nk_combo.
  // nk_combo takes a const char *const * so we need a temporary array.
  // Use a VLA bounded by count (typically small — colormaps, layer names, etc.)
  const char **filtered = (const char **)alloca((size_t)count * sizeof(char *));
  int  filtered_idx[count];   // original index of each filtered entry
  int  fcount = 0;
  int  selected_in_filtered = 0;

  for (int i = 0; i < count; i++) {
    // Simple case-insensitive substring search.
    if (filter_buf[0] == '\0' || strstr(items[i], filter_buf) != NULL) {
      if (i == *selected) selected_in_filtered = fcount;
      filtered[fcount] = items[i];
      filtered_idx[fcount] = i;
      fcount++;
    }
  }

  if (fcount == 0) return false;

  nk_layout_row_dynamic(ctx, ROW_H, 1);
  struct nk_vec2 combo_size = {200, 200};
  int new_sel = nk_combo(ctx, filtered, fcount, selected_in_filtered,
                         (int)ROW_H, combo_size);

  if (new_sel != selected_in_filtered) {
    *selected = filtered_idx[new_sel];
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// nk_widget_progress_labeled
// Draws a progress bar then overlays a centered text label using a canvas rect.
// ---------------------------------------------------------------------------

void nk_widget_progress_labeled(struct nk_context *ctx, const char *label, float progress) {
  if (progress < 0.0f) progress = 0.0f;
  if (progress > 1.0f) progress = 1.0f;

  nk_layout_row_dynamic(ctx, ROW_H, 1);
  struct nk_rect bounds = nk_widget_bounds(ctx);

  nk_size cur = (nk_size)(progress * 1000.0f);
  nk_prog(ctx, cur, 1000, nk_false);

  // Draw label on top via canvas (non-interactive).
  struct nk_command_buffer *canvas = nk_window_get_canvas(ctx);
  char buf[64];
  fmt_buf(buf, sizeof(buf), "%s  %.0f%%", label ? label : "", progress * 100.0f);
  struct nk_color white = {255, 255, 255, 200};
  nk_draw_text(canvas, bounds, buf, (int)strlen(buf),
               ctx->style.font, (struct nk_color){0,0,0,0}, white);
}

// ---------------------------------------------------------------------------
// nk_widget_coord_display
// Shows "X: %fmt  Y: %fmt  Z: %fmt" in a single row.
// ---------------------------------------------------------------------------

void nk_widget_coord_display(struct nk_context *ctx, float x, float y, float z, const char *fmt) {
  if (!fmt) fmt = "%.1f";
  char xbuf[20], ybuf[20], zbuf[20];
  fmt_buf(xbuf, sizeof(xbuf), fmt, x);
  fmt_buf(ybuf, sizeof(ybuf), fmt, y);
  fmt_buf(zbuf, sizeof(zbuf), fmt, z);

  nk_layout_row_begin(ctx, NK_DYNAMIC, ROW_H, 6);
    nk_layout_row_push(ctx, 0.05f); nk_label(ctx, "X:", NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.28f); nk_label(ctx, xbuf, NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 0.05f); nk_label(ctx, "Y:", NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.28f); nk_label(ctx, ybuf, NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 0.05f); nk_label(ctx, "Z:", NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.29f); nk_label(ctx, zbuf, NK_TEXT_RIGHT);
  nk_layout_row_end(ctx);
}
