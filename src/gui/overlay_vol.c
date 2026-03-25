#include "gui/overlay_vol.h"
#include "core/log.h"
#include "render/cmap.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// Nuklear included only for render_controls; guard with NK_IMPLEMENTATION check.
#ifdef NK_INCLUDE_DEFAULT_ALLOCATOR
#  include "nuklear.h"
#endif

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct overlay_volume {
  volume            *vol;
  float              opacity;      // [0, 1]
  int                cmap_id;
  float              threshold;    // [0, 1] normalised — skip pixels below
  bool               visible;
  overlay_blend_mode blend;
  float              win_center;   // window/level center [0, 255]
  float              win_width;    // window/level width  [1, 255]
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

overlay_volume *overlay_volume_new(void) {
  overlay_volume *v = calloc(1, sizeof(*v));
  if (!v) return NULL;
  v->opacity    = 0.5f;
  v->cmap_id    = CMAP_HOT;
  v->threshold  = 0.0f;
  v->visible    = true;
  v->blend      = OVERLAY_BLEND_ALPHA;
  v->win_center = 127.5f;
  v->win_width  = 255.0f;
  return v;
}

void overlay_volume_free(overlay_volume *v) {
  free(v);
}

// ---------------------------------------------------------------------------
// Setters
// ---------------------------------------------------------------------------

void overlay_volume_set_volume(overlay_volume *v, volume *vol) {
  if (v) v->vol = vol;
}

void overlay_volume_set_opacity(overlay_volume *v, float opacity) {
  if (!v) return;
  if (opacity < 0.0f) opacity = 0.0f;
  if (opacity > 1.0f) opacity = 1.0f;
  v->opacity = opacity;
}

void overlay_volume_set_cmap(overlay_volume *v, int cmap_id) {
  if (!v) return;
  if (cmap_id < 0 || cmap_id >= CMAP_COUNT) return;
  v->cmap_id = cmap_id;
}

void overlay_volume_set_threshold(overlay_volume *v, float threshold) {
  if (!v) return;
  if (threshold < 0.0f) threshold = 0.0f;
  if (threshold > 1.0f) threshold = 1.0f;
  v->threshold = threshold;
}

void overlay_volume_set_visible(overlay_volume *v, bool visible) {
  if (v) v->visible = visible;
}

void overlay_volume_set_blend(overlay_volume *v, overlay_blend_mode mode) {
  if (v) v->blend = mode;
}

void overlay_volume_set_window(overlay_volume *v, float center, float width) {
  if (!v) return;
  v->win_center = center;
  v->win_width  = (width < 1.0f) ? 1.0f : width;
}

// ---------------------------------------------------------------------------
// Getters
// ---------------------------------------------------------------------------

bool               overlay_volume_visible(const overlay_volume *v)   { return v && v->visible; }
float              overlay_volume_opacity(const overlay_volume *v)    { return v ? v->opacity : 0.0f; }
int                overlay_volume_cmap(const overlay_volume *v)       { return v ? v->cmap_id : 0; }
float              overlay_volume_threshold(const overlay_volume *v)  { return v ? v->threshold : 0.0f; }
overlay_blend_mode overlay_volume_blend(const overlay_volume *v)      { return v ? v->blend : OVERLAY_BLEND_ALPHA; }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Apply window/level: map raw byte value to [0,1].
static inline float apply_window(float val, float center, float width) {
  float lo = center - width * 0.5f;
  float hi = center + width * 0.5f;
  if (val <= lo) return 0.0f;
  if (val >= hi) return 1.0f;
  return (val - lo) / width;
}

// Blend a colormapped overlay pixel onto a destination RGBA8 pixel.
static inline void blend_pixel(uint8_t *dst, cmap_rgb col, float alpha,
                                overlay_blend_mode mode) {
  float da = dst[3] / 255.0f;
  float dr = dst[0] / 255.0f;
  float dg = dst[1] / 255.0f;
  float db = dst[2] / 255.0f;
  float sr = col.r / 255.0f;
  float sg = col.g / 255.0f;
  float sb = col.b / 255.0f;
  float out_r, out_g, out_b, out_a;

  switch (mode) {
    case OVERLAY_BLEND_ADDITIVE:
      out_r = fminf(dr + sr * alpha, 1.0f);
      out_g = fminf(dg + sg * alpha, 1.0f);
      out_b = fminf(db + sb * alpha, 1.0f);
      out_a = fminf(da + alpha, 1.0f);
      break;
    case OVERLAY_BLEND_MULTIPLY:
      out_r = dr * (1.0f - alpha + sr * alpha);
      out_g = dg * (1.0f - alpha + sg * alpha);
      out_b = db * (1.0f - alpha + sb * alpha);
      out_a = da;
      break;
    default: // OVERLAY_BLEND_ALPHA
      out_r = sr * alpha + dr * (1.0f - alpha);
      out_g = sg * alpha + dg * (1.0f - alpha);
      out_b = sb * alpha + db * (1.0f - alpha);
      out_a = alpha + da * (1.0f - alpha);
      break;
  }
  dst[0] = (uint8_t)(out_r * 255.0f + 0.5f);
  dst[1] = (uint8_t)(out_g * 255.0f + 0.5f);
  dst[2] = (uint8_t)(out_b * 255.0f + 0.5f);
  dst[3] = (uint8_t)(out_a * 255.0f + 0.5f);
}

// ---------------------------------------------------------------------------
// Composite
// ---------------------------------------------------------------------------

void overlay_volume_composite_tile(const overlay_volume *v,
                                   uint8_t *tile_rgba, int w, int h,
                                   float z, float y0, float x0,
                                   float scale, int axis) {
  if (!v || !v->visible || !v->vol || !tile_rgba) return;
  if (v->opacity <= 0.0f) return;

  for (int py = 0; py < h; py++) {
    for (int px = 0; px < w; px++) {
      // Map pixel to voxel coords depending on slice axis
      float vz, vy, vx;
      switch (axis) {
        case 1: // Y-plane: z=row, y=constant(z param), x=col
          vz = y0 + py * scale;
          vy = z;
          vx = x0 + px * scale;
          break;
        case 2: // X-plane: z=row, y=col, x=constant
          vz = y0 + py * scale;
          vy = x0 + px * scale;
          vx = z;
          break;
        default: // axis=0: Z-plane
          vz = z;
          vy = y0 + py * scale;
          vx = x0 + px * scale;
          break;
      }

      float raw = vol_sample(v->vol, 0, vz, vy, vx);  // [0, 255]
      float norm = apply_window(raw, v->win_center, v->win_width);

      if (norm < v->threshold) continue;

      cmap_rgb col = cmap_apply((cmap_id)v->cmap_id, (double)norm);
      uint8_t *dst = tile_rgba + (py * w + px) * 4;
      blend_pixel(dst, col, v->opacity, v->blend);
    }
  }
}

// ---------------------------------------------------------------------------
// Nuklear controls panel
// ---------------------------------------------------------------------------

bool overlay_volume_render_controls(overlay_volume *v, struct nk_context *ctx) {
  if (!v || !ctx) return false;
#ifndef NK_INCLUDE_DEFAULT_ALLOCATOR
  (void)v; (void)ctx;
  LOG_WARN("overlay_volume_render_controls: Nuklear not available");
  return false;
#else
  bool changed = false;
  static const char *blend_names[] = { "Alpha", "Additive", "Multiply" };
  static const int   blend_count   = 3;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Overlay", NK_TEXT_LEFT);
  bool new_vis = nk_check_label(ctx, "Visible", v->visible);
  if (new_vis != v->visible) { v->visible = new_vis; changed = true; }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Opacity", NK_TEXT_LEFT);
  float new_op = nk_slide_float(ctx, 0.0f, v->opacity, 1.0f, 0.01f);
  if (new_op != v->opacity) { v->opacity = new_op; changed = true; }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Threshold", NK_TEXT_LEFT);
  float new_thr = nk_slide_float(ctx, 0.0f, v->threshold, 1.0f, 0.01f);
  if (new_thr != v->threshold) { v->threshold = new_thr; changed = true; }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "W/L center", NK_TEXT_LEFT);
  float new_c = nk_slide_float(ctx, 0.0f, v->win_center, 255.0f, 1.0f);
  if (new_c != v->win_center) { v->win_center = new_c; changed = true; }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "W/L width", NK_TEXT_LEFT);
  float new_w = nk_slide_float(ctx, 1.0f, v->win_width, 255.0f, 1.0f);
  if (new_w != v->win_width) { v->win_width = new_w; changed = true; }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Blend", NK_TEXT_LEFT);
  int new_blend = nk_combo(ctx, blend_names, blend_count, (int)v->blend,
                            20, nk_vec2(120, 80));
  if (new_blend != (int)v->blend) { v->blend = (overlay_blend_mode)new_blend; changed = true; }

  // Colormap combo
  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Colormap", NK_TEXT_LEFT);
  const char *cmap_names[CMAP_COUNT];
  for (int i = 0; i < CMAP_COUNT; i++) cmap_names[i] = cmap_name((cmap_id)i);
  int new_cmap = nk_combo(ctx, cmap_names, CMAP_COUNT, v->cmap_id,
                           20, nk_vec2(120, 160));
  if (new_cmap != v->cmap_id) { v->cmap_id = new_cmap; changed = true; }

  return changed;
#endif
}
