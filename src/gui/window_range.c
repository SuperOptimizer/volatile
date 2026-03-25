// window_range.c — window/level contrast control
// Port of VC3D WindowRangeWidget (Qt) -> plain C + Nuklear.
//
// NK_IMPLEMENTATION is owned by app.c; include nuklear.h declaration-only.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/window_range.h"
#include "core/math.h"
#include "render/cmap.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// Recompute derived fields after low/high change.
static void derive(window_range_state *s) {
  s->window = s->high - s->low;
  s->level  = (s->low + s->high) * 0.5f;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

void window_range_init(window_range_state *s) {
  if (!s) return;
  s->low        = 0.0f;
  s->high       = 1.0f;
  s->auto_range = false;
  s->cmap_id    = CMAP_GRAYSCALE;
  derive(s);
}

void window_range_set(window_range_state *s, float low, float high) {
  if (!s) return;
  if (low > high) { float t = low; low = high; high = t; }
  s->low  = clampf(low,  0.0f, 1.0f);
  s->high = clampf(high, 0.0f, 1.0f);
  // Enforce minimum separation of 1/255 so the window is never zero.
  if (s->high - s->low < 1.0f / 255.0f) s->high = s->low + 1.0f / 255.0f;
  if (s->high > 1.0f) { s->high = 1.0f; s->low = s->high - 1.0f / 255.0f; }
  derive(s);
}

void window_range_auto(window_range_state *s,
                       float data_min, float data_max,
                       float p2, float p98) {
  if (!s) return;
  // Prefer the percentile range; fall back to full data range if degenerate.
  float lo = p2, hi = p98;
  if (hi - lo < 1e-6f) { lo = data_min; hi = data_max; }
  if (hi - lo < 1e-6f) { lo = 0.0f; hi = 1.0f; }
  s->auto_range = true;
  window_range_set(s, lo, hi);
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

bool window_range_render(window_range_state *s, struct nk_context *ctx) {
  if (!s || !ctx) return false;
  bool changed = false;

  // --- dual range slider (low / high) ---
  float prev_low  = s->low;
  float prev_high = s->high;

  nk_layout_row_dynamic(ctx, 22, 2);
  if (nk_slider_float(ctx, 0.0f, &s->low, s->high, 0.001f)) {
    s->auto_range = false;
    changed = true;
  }
  if (nk_slider_float(ctx, s->low, &s->high, 1.0f, 0.001f)) {
    s->auto_range = false;
    changed = true;
  }

  if (s->low > s->high) s->low = s->high;  // guard against simultaneous drag
  if (changed) derive(s);

  // Show W/L values.
  char wl_buf[48];
  snprintf(wl_buf, sizeof(wl_buf), "W %.3f  L %.3f", s->window, s->level);
  nk_layout_row_dynamic(ctx, 18, 1);
  nk_label(ctx, wl_buf, NK_TEXT_LEFT);

  // --- auto toggle ---
  nk_layout_row_dynamic(ctx, 22, 2);
  bool prev_auto = s->auto_range;
  int auto_val = s->auto_range ? 1 : 0;
  nk_checkbox_label(ctx, "Auto", &auto_val);
  s->auto_range = (auto_val != 0);
  if (s->auto_range != prev_auto) changed = true;

  // --- colormap selector ---
  int cmap_count = cmap_count();
  const char *cmap_names[CMAP_COUNT];
  for (int i = 0; i < cmap_count; i++) cmap_names[i] = cmap_name((cmap_id)i);

  int prev_cmap = s->cmap_id;
  struct nk_vec2 combo_size = {200, 200};
  s->cmap_id = nk_combo(ctx, cmap_names, cmap_count, s->cmap_id, 22, combo_size);
  if (s->cmap_id != prev_cmap) changed = true;

  (void)prev_low; (void)prev_high;
  return changed;
}

// ---------------------------------------------------------------------------
// apply
// ---------------------------------------------------------------------------

uint8_t window_range_apply(const window_range_state *s, uint8_t value) {
  if (!s) return value;
  float norm = (float)value / 255.0f;
  // Map [low, high] -> [0, 1].
  float win = s->high - s->low;
  float out;
  if (win < 1e-9f) {
    out = norm >= s->high ? 1.0f : 0.0f;
  } else {
    out = (norm - s->low) / win;
  }
  out = clampf(out, 0.0f, 1.0f);
  return (uint8_t)(out * 255.0f + 0.5f);
}
