#include "gui/settings_dialog.h"
#include "gui/settings.h"
#include "gui/nk_widgets.h"

// NK_IMPLEMENTATION is owned by app.c (exactly once per link unit).
// Include nuklear.h declaration-only here.
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Composite method dropdown items
// ---------------------------------------------------------------------------

static const char *k_composite_methods[] = {
  "max", "alpha", "beerLambert", "mean", "min",
};
static const int k_composite_method_count = 5;

// Downscale override items
static const char *k_downscale_items[] = {
  "Auto", "1x", "2x", "4x", "8x",
};
static const int k_downscale_count = 5;

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct settings_dialog {
  settings *prefs;
  bool      visible;

  // section expand state
  bool sec_preproc;
  bool sec_normal;
  bool sec_view;
  bool sec_overlay;
  bool sec_render;
  bool sec_composite;
  bool sec_postproc;
  bool sec_transforms;
  bool sec_perf;

  // --- Preprocessing ---
  float  pre_win_lo, pre_win_hi;
  int    pre_cmap;         // index into cmap list
  float  pre_stretch;

  // --- Normal Visualization ---
  bool  norm_show_hints;
  bool  norm_show_normals;
  float norm_arrow_count;  // stored as float for slider, integer semantics
  float norm_arrow_len;

  // --- View ---
  float zoom_sensitivity;
  bool  reset_on_surface_change;

  // --- Overlay Volume ---
  float  ov_opacity;
  int    ov_cmap;
  float  ov_threshold;
  float  ov_win_lo, ov_win_hi;

  // --- Render Settings ---
  float  rs_intersect_opacity;
  float  rs_intersect_thickness;
  float  rs_sampling_stride;

  // --- Composite Rendering ---
  int    comp_layers_front;
  int    comp_layers_behind;
  int    comp_method;
  float  comp_extinction;
  float  comp_emission;
  float  comp_ambient;
  float  comp_alpha_min;
  float  comp_alpha_max;
  float  comp_alpha_opacity;
  float  comp_alpha_cutoff;

  // --- Post-Processing ---
  float  pp_stretch;
  int    pp_small_comp_remove;

  // --- Transforms ---
  float  xf_affine[16];  // 4x4 affine preview (read-only display)

  // --- Performance ---
  float  perf_ram_cache_gb;  // float for slider
  int    perf_downscale;
  bool   perf_fast_interp;

  // searchable combo filter buffers (small, just enough for short names)
  char   pre_cmap_filter[32];
  char   ov_cmap_filter[32];
};

// ---------------------------------------------------------------------------
// Cmap list (matches cmap.h order)
// ---------------------------------------------------------------------------

static const char *k_cmap_names[] = {
  "grayscale", "viridis", "magma", "plasma", "inferno",
  "cividis", "turbo", "coolwarm", "rdbu", "spectral",
};
static const int k_cmap_count = 10;

// ---------------------------------------------------------------------------
// Key names used for settings persistence
// ---------------------------------------------------------------------------

#define KEY_PRE_WIN_LO         "pre.win_lo"
#define KEY_PRE_WIN_HI         "pre.win_hi"
#define KEY_PRE_CMAP           "pre.cmap"
#define KEY_PRE_STRETCH        "pre.stretch"
#define KEY_NORM_HINTS         "norm.show_hints"
#define KEY_NORM_NORMALS       "norm.show_normals"
#define KEY_NORM_ARROW_COUNT   "norm.arrow_count"
#define KEY_NORM_ARROW_LEN     "norm.arrow_len"
#define KEY_VIEW_ZOOM_SENS     "view.zoom_sensitivity"
#define KEY_VIEW_RESET_SURF    "view.reset_on_surface_change"
#define KEY_OV_OPACITY         "ov.opacity"
#define KEY_OV_CMAP            "ov.cmap"
#define KEY_OV_THRESHOLD       "ov.threshold"
#define KEY_OV_WIN_LO          "ov.win_lo"
#define KEY_OV_WIN_HI          "ov.win_hi"
#define KEY_RS_INT_OPACITY     "rs.intersect_opacity"
#define KEY_RS_INT_THICK       "rs.intersect_thickness"
#define KEY_RS_STRIDE          "rs.sampling_stride"
#define KEY_COMP_FRONT         "comp.layers_front"
#define KEY_COMP_BEHIND        "comp.layers_behind"
#define KEY_COMP_METHOD        "comp.method"
#define KEY_COMP_EXTINCTION    "comp.extinction"
#define KEY_COMP_EMISSION      "comp.emission"
#define KEY_COMP_AMBIENT       "comp.ambient"
#define KEY_COMP_ALPHA_MIN     "comp.alpha_min"
#define KEY_COMP_ALPHA_MAX     "comp.alpha_max"
#define KEY_COMP_ALPHA_OP      "comp.alpha_opacity"
#define KEY_COMP_ALPHA_CUT     "comp.alpha_cutoff"
#define KEY_PP_STRETCH         "pp.stretch"
#define KEY_PP_SMALL_COMP      "pp.small_comp_remove"
#define KEY_PERF_RAM_GB        "perf.ram_cache_gb"
#define KEY_PERF_DOWNSCALE     "perf.downscale"
#define KEY_PERF_FAST_INTERP   "perf.fast_interpolation"

// ---------------------------------------------------------------------------
// Load settings into dialog state
// ---------------------------------------------------------------------------

static void load_from_settings(settings_dialog *d) {
  settings *s = d->prefs;
  d->pre_win_lo           = settings_get_float(s, KEY_PRE_WIN_LO,         0.0f);
  d->pre_win_hi           = settings_get_float(s, KEY_PRE_WIN_HI,         1.0f);
  d->pre_cmap             = settings_get_int  (s, KEY_PRE_CMAP,            0);
  d->pre_stretch          = settings_get_float(s, KEY_PRE_STRETCH,         1.0f);
  d->norm_show_hints      = settings_get_bool (s, KEY_NORM_HINTS,          false);
  d->norm_show_normals    = settings_get_bool (s, KEY_NORM_NORMALS,        false);
  d->norm_arrow_count     = (float)settings_get_int(s, KEY_NORM_ARROW_COUNT, 64);
  d->norm_arrow_len       = settings_get_float(s, KEY_NORM_ARROW_LEN,      5.0f);
  d->zoom_sensitivity     = settings_get_float(s, KEY_VIEW_ZOOM_SENS,      1.0f);
  d->reset_on_surface_change = settings_get_bool(s, KEY_VIEW_RESET_SURF,   true);
  d->ov_opacity           = settings_get_float(s, KEY_OV_OPACITY,          0.5f);
  d->ov_cmap              = settings_get_int  (s, KEY_OV_CMAP,             0);
  d->ov_threshold         = settings_get_float(s, KEY_OV_THRESHOLD,        0.0f);
  d->ov_win_lo            = settings_get_float(s, KEY_OV_WIN_LO,           0.0f);
  d->ov_win_hi            = settings_get_float(s, KEY_OV_WIN_HI,           1.0f);
  d->rs_intersect_opacity = settings_get_float(s, KEY_RS_INT_OPACITY,      0.8f);
  d->rs_intersect_thickness = settings_get_float(s, KEY_RS_INT_THICK,      2.0f);
  d->rs_sampling_stride   = settings_get_float(s, KEY_RS_STRIDE,           1.0f);
  d->comp_layers_front    = settings_get_int  (s, KEY_COMP_FRONT,          8);
  d->comp_layers_behind   = settings_get_int  (s, KEY_COMP_BEHIND,         8);
  d->comp_method          = settings_get_int  (s, KEY_COMP_METHOD,         0);
  d->comp_extinction      = settings_get_float(s, KEY_COMP_EXTINCTION,     0.1f);
  d->comp_emission        = settings_get_float(s, KEY_COMP_EMISSION,       0.1f);
  d->comp_ambient         = settings_get_float(s, KEY_COMP_AMBIENT,        0.05f);
  d->comp_alpha_min       = settings_get_float(s, KEY_COMP_ALPHA_MIN,      0.0f);
  d->comp_alpha_max       = settings_get_float(s, KEY_COMP_ALPHA_MAX,      1.0f);
  d->comp_alpha_opacity   = settings_get_float(s, KEY_COMP_ALPHA_OP,       1.0f);
  d->comp_alpha_cutoff    = settings_get_float(s, KEY_COMP_ALPHA_CUT,      0.01f);
  d->pp_stretch           = settings_get_float(s, KEY_PP_STRETCH,          1.0f);
  d->pp_small_comp_remove = settings_get_int  (s, KEY_PP_SMALL_COMP,       0);
  d->perf_ram_cache_gb    = (float)settings_get_int(s, KEY_PERF_RAM_GB,    4);
  d->perf_downscale       = settings_get_int  (s, KEY_PERF_DOWNSCALE,      0);
  d->perf_fast_interp     = settings_get_bool (s, KEY_PERF_FAST_INTERP,    false);
  // identity affine
  for (int i = 0; i < 16; i++) d->xf_affine[i] = (i % 5 == 0) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

settings_dialog *settings_dialog_new(settings *prefs) {
  settings_dialog *d = calloc(1, sizeof(*d));
  if (!d) return NULL;
  d->prefs = prefs;
  d->visible = false;
  d->sec_preproc   = true;
  d->sec_normal    = false;
  d->sec_view      = false;
  d->sec_overlay   = false;
  d->sec_render    = false;
  d->sec_composite = false;
  d->sec_postproc  = false;
  d->sec_transforms = false;
  d->sec_perf      = false;
  load_from_settings(d);
  return d;
}

void settings_dialog_free(settings_dialog *d) {
  free(d);
}

void settings_dialog_show(settings_dialog *d) {
  d->visible = true;
}

bool settings_dialog_is_visible(const settings_dialog *d) {
  return d->visible;
}

// ---------------------------------------------------------------------------
// Helper: float slider row with label; returns true if changed
// ---------------------------------------------------------------------------

static bool row_slider_f(struct nk_context *ctx, const char *label,
                          float *val, float lo, float hi, float step) {
  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, label, NK_TEXT_LEFT);
  float prev = *val;
  *val = nk_slide_float(ctx, lo, *val, hi, step);
  return *val != prev;
}

// Helper: checkbox row; returns true if changed
static bool row_check(struct nk_context *ctx, const char *label, bool *val) {
  nk_layout_row_dynamic(ctx, 20, 1);
  int iv = *val ? 1 : 0;
  int prev = iv;
  nk_checkbox_label(ctx, label, &iv);
  *val = iv != 0;
  return iv != prev;
}

// Helper: int spinbox row; returns true if changed
static bool row_spin_i(struct nk_context *ctx, const char *label,
                        int *val, int lo, int hi) {
  nk_layout_row_dynamic(ctx, 20, 2);
  nk_label(ctx, label, NK_TEXT_LEFT);
  int prev = *val;
  *val = nk_propertyi(ctx, "#", lo, *val, hi, 1, 1.0f);
  return *val != prev;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

bool settings_dialog_render(settings_dialog *d, struct nk_context *ctx) {
  if (!d->visible) return false;

  settings *prefs = d->prefs;
  bool changed = false;

  nk_flags flags = NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE
                 | NK_WINDOW_CLOSABLE | NK_WINDOW_TITLE;

  if (nk_begin(ctx, "Settings", nk_rect(60, 60, 420, 600), flags)) {

    // --- Preprocessing ---
    if (nk_widget_collapsible_begin(ctx, "Preprocessing", &d->sec_preproc)) {
      if (nk_widget_range_slider(ctx, &d->pre_win_lo, &d->pre_win_hi, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_PRE_WIN_LO, (int)(d->pre_win_lo * 1000));
        settings_set_int(prefs, KEY_PRE_WIN_HI, (int)(d->pre_win_hi * 1000));
        changed = true;
      }
      nk_layout_row_dynamic(ctx, 20, 2);
      nk_label(ctx, "Colormap", NK_TEXT_LEFT);
      if (nk_widget_searchable_combo(ctx, k_cmap_names, k_cmap_count,
                                      &d->pre_cmap,
                                      d->pre_cmap_filter, (int)sizeof(d->pre_cmap_filter))) {
        settings_set_int(prefs, KEY_PRE_CMAP, d->pre_cmap);
        changed = true;
      }
      if (row_slider_f(ctx, "Stretch", &d->pre_stretch, 0.1f, 4.0f, 0.05f)) {
        settings_set_int(prefs, KEY_PRE_STRETCH, (int)(d->pre_stretch * 1000));
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Normal Visualization ---
    if (nk_widget_collapsible_begin(ctx, "Normal Visualization", &d->sec_normal)) {
      if (row_check(ctx, "Show direction hints", &d->norm_show_hints)) {
        settings_set_int(prefs, KEY_NORM_HINTS, d->norm_show_hints ? 1 : 0);
        changed = true;
      }
      if (row_check(ctx, "Show surface normals", &d->norm_show_normals)) {
        settings_set_int(prefs, KEY_NORM_NORMALS, d->norm_show_normals ? 1 : 0);
        changed = true;
      }
      if (row_slider_f(ctx, "Arrow count", &d->norm_arrow_count, 4.0f, 512.0f, 4.0f)) {
        settings_set_int(prefs, KEY_NORM_ARROW_COUNT, (int)d->norm_arrow_count);
        changed = true;
      }
      if (row_slider_f(ctx, "Arrow length", &d->norm_arrow_len, 1.0f, 50.0f, 0.5f)) {
        settings_set_int(prefs, KEY_NORM_ARROW_LEN, (int)(d->norm_arrow_len * 100));
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- View ---
    if (nk_widget_collapsible_begin(ctx, "View", &d->sec_view)) {
      if (row_slider_f(ctx, "Zoom sensitivity", &d->zoom_sensitivity, 0.1f, 5.0f, 0.1f)) {
        settings_set_int(prefs, KEY_VIEW_ZOOM_SENS, (int)(d->zoom_sensitivity * 1000));
        changed = true;
      }
      if (row_check(ctx, "Reset view on surface change", &d->reset_on_surface_change)) {
        settings_set_int(prefs, KEY_VIEW_RESET_SURF, d->reset_on_surface_change ? 1 : 0);
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Overlay Volume ---
    if (nk_widget_collapsible_begin(ctx, "Overlay Volume", &d->sec_overlay)) {
      if (row_slider_f(ctx, "Opacity", &d->ov_opacity, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_OV_OPACITY, (int)(d->ov_opacity * 1000));
        changed = true;
      }
      nk_layout_row_dynamic(ctx, 20, 2);
      nk_label(ctx, "Colormap", NK_TEXT_LEFT);
      if (nk_widget_searchable_combo(ctx, k_cmap_names, k_cmap_count,
                                      &d->ov_cmap,
                                      d->ov_cmap_filter, (int)sizeof(d->ov_cmap_filter))) {
        settings_set_int(prefs, KEY_OV_CMAP, d->ov_cmap);
        changed = true;
      }
      if (row_slider_f(ctx, "Threshold", &d->ov_threshold, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_OV_THRESHOLD, (int)(d->ov_threshold * 1000));
        changed = true;
      }
      if (nk_widget_range_slider(ctx, &d->ov_win_lo, &d->ov_win_hi, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_OV_WIN_LO, (int)(d->ov_win_lo * 1000));
        settings_set_int(prefs, KEY_OV_WIN_HI, (int)(d->ov_win_hi * 1000));
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Render Settings ---
    if (nk_widget_collapsible_begin(ctx, "Render Settings", &d->sec_render)) {
      if (row_slider_f(ctx, "Intersection opacity", &d->rs_intersect_opacity, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_RS_INT_OPACITY, (int)(d->rs_intersect_opacity * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "Intersection thickness", &d->rs_intersect_thickness, 0.5f, 20.0f, 0.5f)) {
        settings_set_int(prefs, KEY_RS_INT_THICK, (int)(d->rs_intersect_thickness * 100));
        changed = true;
      }
      if (row_slider_f(ctx, "Sampling stride", &d->rs_sampling_stride, 0.25f, 4.0f, 0.25f)) {
        settings_set_int(prefs, KEY_RS_STRIDE, (int)(d->rs_sampling_stride * 1000));
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Composite Rendering ---
    if (nk_widget_collapsible_begin(ctx, "Composite Rendering", &d->sec_composite)) {
      if (row_spin_i(ctx, "Layers front", &d->comp_layers_front, 0, 64)) {
        settings_set_int(prefs, KEY_COMP_FRONT, d->comp_layers_front);
        changed = true;
      }
      if (row_spin_i(ctx, "Layers behind", &d->comp_layers_behind, 0, 64)) {
        settings_set_int(prefs, KEY_COMP_BEHIND, d->comp_layers_behind);
        changed = true;
      }
      nk_layout_row_dynamic(ctx, 20, 2);
      nk_label(ctx, "Method", NK_TEXT_LEFT);
      int prev_method = d->comp_method;
      d->comp_method = nk_combo(ctx, k_composite_methods, k_composite_method_count,
                                 d->comp_method, 20, nk_vec2(160, 200));
      if (d->comp_method != prev_method) {
        settings_set_int(prefs, KEY_COMP_METHOD, d->comp_method);
        changed = true;
      }
      // Beer-Lambert params
      nk_layout_row_dynamic(ctx, 14, 1);
      nk_label(ctx, "Beer-Lambert:", NK_TEXT_LEFT);
      if (row_slider_f(ctx, "  Extinction", &d->comp_extinction, 0.0f, 2.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_EXTINCTION, (int)(d->comp_extinction * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "  Emission", &d->comp_emission, 0.0f, 2.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_EMISSION, (int)(d->comp_emission * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "  Ambient", &d->comp_ambient, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_AMBIENT, (int)(d->comp_ambient * 1000));
        changed = true;
      }
      // Alpha params
      nk_layout_row_dynamic(ctx, 14, 1);
      nk_label(ctx, "Alpha:", NK_TEXT_LEFT);
      if (row_slider_f(ctx, "  Min", &d->comp_alpha_min, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_ALPHA_MIN, (int)(d->comp_alpha_min * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "  Max", &d->comp_alpha_max, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_ALPHA_MAX, (int)(d->comp_alpha_max * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "  Opacity", &d->comp_alpha_opacity, 0.0f, 1.0f, 0.01f)) {
        settings_set_int(prefs, KEY_COMP_ALPHA_OP, (int)(d->comp_alpha_opacity * 1000));
        changed = true;
      }
      if (row_slider_f(ctx, "  Cutoff", &d->comp_alpha_cutoff, 0.0f, 0.5f, 0.001f)) {
        settings_set_int(prefs, KEY_COMP_ALPHA_CUT, (int)(d->comp_alpha_cutoff * 10000));
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Post-Processing ---
    if (nk_widget_collapsible_begin(ctx, "Post-Processing", &d->sec_postproc)) {
      if (row_slider_f(ctx, "Stretch", &d->pp_stretch, 0.1f, 4.0f, 0.05f)) {
        settings_set_int(prefs, KEY_PP_STRETCH, (int)(d->pp_stretch * 1000));
        changed = true;
      }
      if (row_spin_i(ctx, "Remove small components (px)", &d->pp_small_comp_remove, 0, 10000)) {
        settings_set_int(prefs, KEY_PP_SMALL_COMP, d->pp_small_comp_remove);
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Transforms ---
    if (nk_widget_collapsible_begin(ctx, "Transforms", &d->sec_transforms)) {
      nk_layout_row_dynamic(ctx, 14, 1);
      nk_label(ctx, "Affine preview (read-only):", NK_TEXT_LEFT);
      for (int row = 0; row < 4; row++) {
        nk_layout_row_dynamic(ctx, 16, 4);
        for (int col = 0; col < 4; col++) {
          char buf[16];
          snprintf(buf, sizeof(buf), "%.2f", (double)d->xf_affine[row * 4 + col]);
          nk_label(ctx, buf, NK_TEXT_CENTERED);
        }
      }
      nk_widget_collapsible_end(ctx);
    }

    // --- Performance ---
    if (nk_widget_collapsible_begin(ctx, "Performance", &d->sec_perf)) {
      if (row_slider_f(ctx, "RAM cache (GB)", &d->perf_ram_cache_gb, 1.0f, 64.0f, 1.0f)) {
        settings_set_int(prefs, KEY_PERF_RAM_GB, (int)d->perf_ram_cache_gb);
        changed = true;
      }
      nk_layout_row_dynamic(ctx, 20, 2);
      nk_label(ctx, "Downscale override", NK_TEXT_LEFT);
      int prev_ds = d->perf_downscale;
      d->perf_downscale = nk_combo(ctx, k_downscale_items, k_downscale_count,
                                    d->perf_downscale, 20, nk_vec2(140, 160));
      if (d->perf_downscale != prev_ds) {
        settings_set_int(prefs, KEY_PERF_DOWNSCALE, d->perf_downscale);
        changed = true;
      }
      if (row_check(ctx, "Fast interpolation", &d->perf_fast_interp)) {
        settings_set_int(prefs, KEY_PERF_FAST_INTERP, d->perf_fast_interp ? 1 : 0);
        changed = true;
      }
      nk_widget_collapsible_end(ctx);
    }

  } else {
    // window was closed via X button
    d->visible = false;
  }
  nk_end(ctx);
  return changed;
}
