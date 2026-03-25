// viewer_controls.c — viewer controls dock panel
// NK_IMPLEMENTATION is owned by app.c; include nuklear.h declaration-only.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/viewer_controls.h"
#include "gui/viewer.h"
#include "gui/window_range.h"
#include "gui/nk_widgets.h"
#include "render/cmap.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Composite mode labels (same order as composite.h enum)
// ---------------------------------------------------------------------------

static const char *COMPOSITE_LABELS[] = {
  "Max (MIP)", "Min", "Mean", "Alpha", "Beer-Lambert", "Sum"
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct viewer_controls {
  // Slice position
  float pos_z, pos_y, pos_x;

  // Zoom
  float zoom;

  // Pyramid level
  int   pyr_level;
  int   pyr_level_override; // -1 = auto

  // Composite
  composite_params composite;
  int              composite_sel;   // index into COMPOSITE_LABELS

  // Colormap
  int  cmap_sel;

  // Window / Level (reuse window_range widget state)
  window_range_state wr;

  // Overlay volume
  bool  overlay_enabled;
  float overlay_opacity;
  int   overlay_cmap;

  // Normal overlay
  bool  normals_visible;
  int   normal_count;
  float normal_length;

  // Intersection lines
  float isect_opacity;
  float isect_thickness;

  // Scale bar
  bool  scalebar_visible;
  float voxel_size_um;     // voxel physical size in micrometres

  // Collapsible section open/closed state
  bool sec_position;
  bool sec_zoom;
  bool sec_pyramid;
  bool sec_composite;
  bool sec_colormap;
  bool sec_window;
  bool sec_overlay;
  bool sec_normals;
  bool sec_isect;
  bool sec_scalebar;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

viewer_controls *viewer_controls_new(void) {
  viewer_controls *c = calloc(1, sizeof *c);
  if (!c) return NULL;

  composite_params_default(&c->composite);
  c->composite_sel = (int)c->composite.mode;

  window_range_init(&c->wr);

  c->pyr_level_override = -1;
  c->zoom               = 1.0f;
  c->overlay_opacity    = 0.5f;
  c->overlay_cmap       = CMAP_VIRIDIS;
  c->normal_count       = 64;
  c->normal_length      = 4.0f;
  c->isect_opacity      = 0.8f;
  c->isect_thickness    = 1.0f;
  c->voxel_size_um      = 1.0f;

  // All sections open by default.
  c->sec_position = c->sec_zoom    = c->sec_pyramid  = true;
  c->sec_composite= c->sec_colormap= c->sec_window   = true;
  c->sec_overlay  = c->sec_normals = c->sec_isect    = false;
  c->sec_scalebar = false;

  return c;
}

void viewer_controls_free(viewer_controls *c) {
  free(c);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

int viewer_controls_get_cmap(const viewer_controls *c) {
  return c->cmap_sel;
}

composite_params viewer_controls_get_composite(const viewer_controls *c) {
  return c->composite;
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

// Sync readable state FROM the active viewer (if any).
static void sync_from_viewer(viewer_controls *c, slice_viewer *v) {
  if (!v) return;
  c->pos_z    = viewer_current_slice(v);
  c->pyr_level= viewer_current_level(v);
}

static bool render_section_position(viewer_controls *c, struct nk_context *ctx,
                                    slice_viewer *v) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Slice Position", &c->sec_position))
    return false;

  char buf[32];
  snprintf(buf, sizeof buf, "%.1f", (double)c->pos_z);
  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Z:", NK_TEXT_LEFT);
  nk_label(ctx, buf, NK_TEXT_RIGHT);

  snprintf(buf, sizeof buf, "%.1f", (double)c->pos_y);
  nk_label(ctx, "Y:", NK_TEXT_LEFT);
  nk_label(ctx, buf, NK_TEXT_RIGHT);

  snprintf(buf, sizeof buf, "%.1f", (double)c->pos_x);
  nk_label(ctx, "X:", NK_TEXT_LEFT);
  nk_label(ctx, buf, NK_TEXT_RIGHT);

  (void)v;
  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_zoom(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Zoom", &c->sec_zoom))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  char buf[32];
  snprintf(buf, sizeof buf, "%.2fx", (double)c->zoom);
  nk_label(ctx, "Zoom:", NK_TEXT_LEFT);
  nk_label(ctx, buf, NK_TEXT_RIGHT);

  nk_layout_row_dynamic(ctx, 24, 1);
  if (nk_button_label(ctx, "Reset Zoom")) {
    c->zoom = 1.0f;
    changed = true;
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_pyramid(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Pyramid Level", &c->sec_pyramid))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Current LOD:", NK_TEXT_LEFT);
  char buf[16];
  snprintf(buf, sizeof buf, "%d", c->pyr_level);
  nk_label(ctx, buf, NK_TEXT_RIGHT);

  static const char *pyr_opts[] = { "Auto", "0", "1", "2", "3", "4", "5" };
  nk_layout_row_dynamic(ctx, 24, 2);
  nk_label(ctx, "Override:", NK_TEXT_LEFT);
  int prev = c->pyr_level_override + 1; // map -1 -> 0 (Auto)
  int sel  = nk_combo(ctx, pyr_opts, 7, prev, 24, nk_vec2(100, 140));
  if (sel != prev) {
    c->pyr_level_override = sel - 1;
    changed = true;
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_composite(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Composite", &c->sec_composite))
    return false;

  nk_layout_row_dynamic(ctx, 24, 2);
  nk_label(ctx, "Mode:", NK_TEXT_LEFT);
  int sel = nk_combo(ctx, COMPOSITE_LABELS, COMPOSITE_COUNT,
                     c->composite_sel, 24, nk_vec2(130, 150));
  if (sel != c->composite_sel) {
    c->composite_sel  = sel;
    c->composite.mode = (composite_mode)sel;
    changed = true;
  }

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Front layers:", NK_TEXT_LEFT);
  int front = nk_propertyi(ctx, "#", 0, c->composite.num_layers_front, 32, 1, 1);
  if (front != c->composite.num_layers_front) {
    c->composite.num_layers_front = front;
    changed = true;
  }

  nk_label(ctx, "Behind layers:", NK_TEXT_LEFT);
  int behind = nk_propertyi(ctx, "#", 0, c->composite.num_layers_behind, 32, 1, 1);
  if (behind != c->composite.num_layers_behind) {
    c->composite.num_layers_behind = behind;
    changed = true;
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_colormap(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Colormap", &c->sec_colormap))
    return false;

  // Build cmap name list from cmap_name() API.
  const char *names[CMAP_COUNT];
  for (int i = 0; i < CMAP_COUNT; i++)
    names[i] = cmap_name((cmap_id)i);

  nk_layout_row_dynamic(ctx, 24, 1);
  int sel = nk_combo(ctx, names, CMAP_COUNT, c->cmap_sel, 24, nk_vec2(160, 200));
  if (sel != c->cmap_sel) {
    c->cmap_sel = sel;
    c->wr.cmap_id = sel;
    changed = true;
  }

  // Preview swatch: show the middle color of the selected cmap.
  cmap_rgb mid = cmap_apply((cmap_id)c->cmap_sel, 0.5);
  nk_layout_row_dynamic(ctx, 14, 1);
  nk_widget_color_swatch(ctx, mid.r, mid.g, mid.b, 0.0f, 14.0f);

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_window(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Window / Level", &c->sec_window))
    return false;

  nk_layout_row_dynamic(ctx, 24, 1);
  changed = window_range_render(&c->wr, ctx);

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_overlay(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Overlay Volume", &c->sec_overlay))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Enable:", NK_TEXT_LEFT);
  bool prev = c->overlay_enabled;
  nk_checkbox_label(ctx, "", (nk_bool *)&c->overlay_enabled);
  if (c->overlay_enabled != prev) changed = true;

  if (c->overlay_enabled) {
    nk_label(ctx, "Opacity:", NK_TEXT_LEFT);
    float op = nk_slide_float(ctx, 0.0f, c->overlay_opacity, 1.0f, 0.01f);
    if (fabsf(op - c->overlay_opacity) > 1e-4f) {
      c->overlay_opacity = op;
      changed = true;
    }

    const char *names[CMAP_COUNT];
    for (int i = 0; i < CMAP_COUNT; i++) names[i] = cmap_name((cmap_id)i);
    nk_layout_row_dynamic(ctx, 24, 2);
    nk_label(ctx, "Colormap:", NK_TEXT_LEFT);
    int sel = nk_combo(ctx, names, CMAP_COUNT, c->overlay_cmap, 24, nk_vec2(120, 180));
    if (sel != c->overlay_cmap) { c->overlay_cmap = sel; changed = true; }
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_normals(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Normal Overlay", &c->sec_normals))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Visible:", NK_TEXT_LEFT);
  bool prev = c->normals_visible;
  nk_checkbox_label(ctx, "", (nk_bool *)&c->normals_visible);
  if (c->normals_visible != prev) changed = true;

  if (c->normals_visible) {
    nk_label(ctx, "Count:", NK_TEXT_LEFT);
    int cnt = nk_propertyi(ctx, "#", 4, c->normal_count, 256, 4, 4);
    if (cnt != c->normal_count) { c->normal_count = cnt; changed = true; }

    nk_label(ctx, "Length:", NK_TEXT_LEFT);
    float len = nk_propertyf(ctx, "#", 0.5f, c->normal_length, 20.0f, 0.5f, 0.5f);
    if (fabsf(len - c->normal_length) > 1e-4f) {
      c->normal_length = len; changed = true;
    }
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_isect(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Intersection Lines", &c->sec_isect))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Opacity:", NK_TEXT_LEFT);
  float op = nk_slide_float(ctx, 0.0f, c->isect_opacity, 1.0f, 0.01f);
  if (fabsf(op - c->isect_opacity) > 1e-4f) {
    c->isect_opacity = op; changed = true;
  }

  nk_label(ctx, "Thickness:", NK_TEXT_LEFT);
  float th = nk_slide_float(ctx, 0.5f, c->isect_thickness, 5.0f, 0.25f);
  if (fabsf(th - c->isect_thickness) > 1e-4f) {
    c->isect_thickness = th; changed = true;
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

static bool render_section_scalebar(viewer_controls *c, struct nk_context *ctx) {
  bool changed = false;
  if (!nk_widget_collapsible_begin(ctx, "Scale Bar", &c->sec_scalebar))
    return false;

  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, "Visible:", NK_TEXT_LEFT);
  bool prev = c->scalebar_visible;
  nk_checkbox_label(ctx, "", (nk_bool *)&c->scalebar_visible);
  if (c->scalebar_visible != prev) changed = true;

  nk_label(ctx, "Voxel (µm):", NK_TEXT_LEFT);
  float vsz = nk_propertyf(ctx, "#", 0.01f, c->voxel_size_um, 1000.0f, 0.01f, 0.1f);
  if (fabsf(vsz - c->voxel_size_um) > 1e-5f) {
    c->voxel_size_um = vsz; changed = true;
  }

  nk_widget_collapsible_end(ctx);
  return changed;
}

// ---------------------------------------------------------------------------
// Public: viewer_controls_render
// ---------------------------------------------------------------------------

bool viewer_controls_render(viewer_controls *c, struct nk_context *ctx,
                            slice_viewer *active_viewer) {
  if (!c || !ctx) return false;

  sync_from_viewer(c, active_viewer);

  bool changed = false;
  changed |= render_section_position(c, ctx, active_viewer);
  changed |= render_section_zoom(c, ctx);
  changed |= render_section_pyramid(c, ctx);
  changed |= render_section_composite(c, ctx);
  changed |= render_section_colormap(c, ctx);
  changed |= render_section_window(c, ctx);
  changed |= render_section_overlay(c, ctx);
  changed |= render_section_normals(c, ctx);
  changed |= render_section_isect(c, ctx);
  changed |= render_section_scalebar(c, ctx);

  return changed;
}
