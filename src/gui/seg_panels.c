// seg_panels.c — VC3D segmentation panel hierarchy (right dock)
//
// NK_IMPLEMENTATION is owned by app.c; include nuklear.h declarations only.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/seg_panels.h"
#include "gui/nk_widgets.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define ROW_H     22.0f
#define MAX_CORRECTIONS 64
#define PARAMS_JSON_LEN 1024

// ---------------------------------------------------------------------------
// Correction point list entry
// ---------------------------------------------------------------------------

typedef struct {
  float u, v;
  float tx, ty, tz;  // 3D target position
} correction_entry;

// ---------------------------------------------------------------------------
// Lasagna / neural tracer status strings
// ---------------------------------------------------------------------------

typedef enum {
  SVC_IDLE,
  SVC_CONNECTING,
  SVC_RUNNING,
  SVC_ERROR,
} svc_status;

static const char *svc_status_str(svc_status s) {
  switch (s) {
    case SVC_IDLE:       return "Idle";
    case SVC_CONNECTING: return "Connecting";
    case SVC_RUNNING:    return "Running";
    case SVC_ERROR:      return "Error";
  }
  return "Unknown";
}

// ---------------------------------------------------------------------------
// seg_panels internals
// ---------------------------------------------------------------------------

struct seg_panels {
  // --- panel open/collapsed state ---
  bool header_open;
  bool editing_open;
  bool growth_open;
  bool corrections_open;
  bool approval_open;
  bool custom_open;
  bool reopt_open;
  bool dirfield_open;
  bool lasagna_open;
  bool neural_open;

  // --- panel 1: header row ---
  char seg_name[64];
  int  seg_id;
  // status is derived from active_surface on render

  // --- panel 2: editing ---
  seg_tool_params tool;
  // edit mode: index into mode_names below
  int edit_mode_idx;  // 0=brush 1=line 2=push-pull 3=eraser

  // --- panel 3: growth ---
  growth_params growth;
  // method/dir as combo indices
  int growth_method_idx;
  int growth_dir_idx;

  // --- panel 4: corrections ---
  correction_entry corrections[MAX_CORRECTIONS];
  int  n_corrections;
  int  selected_correction;  // -1 = none

  // --- panel 5: approval mask ---
  float brush_size;
  bool  paint_mode;  // true=paint approved, false=erase
  float coverage;    // updated each frame from active surface

  // --- panel 6: custom params ---
  char params_json[PARAMS_JSON_LEN];
  bool params_dirty;

  // --- panel 7: cell reopt ---
  float reopt_learning_rate;
  int   reopt_iterations;
  bool  reopt_use_gpu;

  // --- panel 8: direction field ---
  bool  dirfield_enabled;
  float dirfield_alpha;  // overlay opacity

  // --- panel 9: lasagna ---
  char  lasagna_host[128];
  int   lasagna_port;
  svc_status lasagna_status;
  char  lasagna_job_id[64];

  // --- panel 10: neural tracer ---
  char  neural_model_path[256];
  svc_status neural_status;
  float neural_confidence;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

seg_panels *seg_panels_new(void) {
  seg_panels *p = calloc(1, sizeof(*p));
  if (!p) return NULL;

  // sensible defaults
  p->header_open     = true;
  p->editing_open    = true;
  p->growth_open     = false;
  p->corrections_open= false;
  p->approval_open   = false;
  p->custom_open     = false;
  p->reopt_open      = false;
  p->dirfield_open   = false;
  p->lasagna_open    = false;
  p->neural_open     = false;

  strncpy(p->seg_name, "Unnamed", sizeof(p->seg_name) - 1);
  p->seg_id   = -1;

  p->tool.tool         = SEG_TOOL_BRUSH;
  p->tool.radius       = 5.0f;
  p->tool.sigma        = 2.0f;
  p->tool.push_amount  = 1.0f;
  p->edit_mode_idx     = 0;

  p->growth.method      = GROWTH_TRACER;
  p->growth.direction   = GROWTH_DIR_ALL;
  p->growth.generations = 1;
  p->growth.step_size   = 1.0f;
  p->growth.straightness_weight = 0.5f;
  p->growth.distance_weight     = 0.5f;
  p->growth_method_idx  = 0;
  p->growth_dir_idx     = 0;

  p->selected_correction = -1;
  p->brush_size   = 10.0f;
  p->paint_mode   = true;
  p->coverage     = 0.0f;

  strncpy(p->params_json, "{}", sizeof(p->params_json) - 1);

  p->reopt_learning_rate = 0.001f;
  p->reopt_iterations    = 100;
  p->reopt_use_gpu       = true;

  p->dirfield_enabled = false;
  p->dirfield_alpha   = 0.5f;

  strncpy(p->lasagna_host, "localhost", sizeof(p->lasagna_host) - 1);
  p->lasagna_port   = 8765;
  p->lasagna_status = SVC_IDLE;

  strncpy(p->neural_model_path, "models/tracer.ckpt", sizeof(p->neural_model_path) - 1);
  p->neural_status    = SVC_IDLE;
  p->neural_confidence = 0.0f;

  return p;
}

void seg_panels_free(seg_panels *p) {
  free(p);
}

// ---------------------------------------------------------------------------
// Panel renderers (each returns immediately if collapsed)
// ---------------------------------------------------------------------------

static void render_header(seg_panels *p, struct nk_context *ctx,
                           quad_surface *surf) {
  if (!nk_widget_collapsible_begin(ctx, "Segment Info", &p->header_open)) return;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Name:", NK_TEXT_LEFT);
  nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, p->seg_name,
                                  (int)sizeof(p->seg_name), nk_filter_default);

  char id_buf[32];
  snprintf(id_buf, sizeof(id_buf), "%d", p->seg_id);
  nk_widget_labeled_str(ctx, "ID:", id_buf);

  const char *status = surf ? "Active" : "No surface";
  nk_widget_labeled_str(ctx, "Status:", status);

  if (surf) {
    char dim_buf[64];
    snprintf(dim_buf, sizeof(dim_buf), "%d x %d",
             surf->rows, surf->cols);
    nk_widget_labeled_str(ctx, "Dims:", dim_buf);
  }

  nk_widget_collapsible_end(ctx);
}

static const char *k_edit_modes[] = { "Brush", "Line", "Push-Pull", "Eraser" };
static const seg_tool_id k_tool_ids[] = {
  SEG_TOOL_BRUSH, SEG_TOOL_LINE, SEG_TOOL_PUSHPULL, SEG_TOOL_ERASER
};

static void render_editing(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Editing", &p->editing_open)) return;

  // Edit mode selector
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Mode:", NK_TEXT_LEFT);
  struct nk_vec2 combo_sz = {160, 100};
  p->edit_mode_idx = nk_combo(ctx, k_edit_modes, 4, p->edit_mode_idx, (int)ROW_H, combo_sz);
  p->tool.tool = k_tool_ids[p->edit_mode_idx];

  // Radius slider
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Radius:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 0.5f, &p->tool.radius, 50.0f, 0.5f);

  char rbuf[16]; snprintf(rbuf, sizeof(rbuf), "%.1f", p->tool.radius);
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "", NK_TEXT_LEFT);
  nk_label(ctx, rbuf, NK_TEXT_RIGHT);

  // Sigma slider
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Sigma:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 0.1f, &p->tool.sigma, 20.0f, 0.1f);

  // Push amount (only relevant for push-pull)
  if (p->tool.tool == SEG_TOOL_PUSHPULL) {
    nk_layout_row_dynamic(ctx, ROW_H, 2);
    nk_label(ctx, "Push amount:", NK_TEXT_LEFT);
    nk_slider_float(ctx, 0.1f, &p->tool.push_amount, 10.0f, 0.1f);
  }

  nk_widget_collapsible_end(ctx);
}

static const char *k_growth_methods[] = { "Tracer", "Extrapolation", "Corrections" };
static const growth_method k_growth_method_vals[] = {
  GROWTH_TRACER, GROWTH_EXTRAPOLATION, GROWTH_CORRECTIONS
};
static const char *k_growth_dirs[] = { "All", "Up", "Down", "Left", "Right" };
static const growth_direction k_growth_dir_vals[] = {
  GROWTH_DIR_ALL, GROWTH_DIR_UP, GROWTH_DIR_DOWN, GROWTH_DIR_LEFT, GROWTH_DIR_RIGHT
};

static void render_growth(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Growth", &p->growth_open)) return;

  struct nk_vec2 combo_sz = {160, 120};

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Method:", NK_TEXT_LEFT);
  p->growth_method_idx = nk_combo(ctx, k_growth_methods, 3,
                                   p->growth_method_idx, (int)ROW_H, combo_sz);
  p->growth.method = k_growth_method_vals[p->growth_method_idx];

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Direction:", NK_TEXT_LEFT);
  p->growth_dir_idx = nk_combo(ctx, k_growth_dirs, 5,
                                p->growth_dir_idx, (int)ROW_H, combo_sz);
  p->growth.direction = k_growth_dir_vals[p->growth_dir_idx];

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Generations:", NK_TEXT_LEFT);
  nk_slider_int(ctx, 1, &p->growth.generations, 32, 1);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Step size:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 0.1f, &p->growth.step_size, 10.0f, 0.1f);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Straightness:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 0.0f, &p->growth.straightness_weight, 1.0f, 0.01f);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Distance wt:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 0.0f, &p->growth.distance_weight, 1.0f, 0.01f);

  nk_layout_row_dynamic(ctx, ROW_H, 1);
  if (nk_button_label(ctx, "Grow"))
    LOG_INFO("seg_panels: grow requested (method=%d gen=%d)", p->growth.method,
             p->growth.generations);

  nk_widget_collapsible_end(ctx);
}

static void render_corrections(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Corrections", &p->corrections_open)) return;

  // List existing corrections
  char buf[64];
  for (int i = 0; i < p->n_corrections; i++) {
    correction_entry *c = &p->corrections[i];
    nk_layout_row_dynamic(ctx, ROW_H, 2);
    snprintf(buf, sizeof(buf), "  (%.1f,%.1f) -> (%.1f,%.1f,%.1f)",
             c->u, c->v, c->tx, c->ty, c->tz);
    bool selected = (p->selected_correction == i);
    int sel_val = selected ? 1 : 0;
    nk_checkbox_label(ctx, buf, &sel_val);
    if (sel_val) p->selected_correction = i;

    if (nk_button_label(ctx, "X")) {
      // remove entry i (shift down)
      memmove(&p->corrections[i], &p->corrections[i + 1],
              (size_t)(p->n_corrections - i - 1) * sizeof(correction_entry));
      p->n_corrections--;
      if (p->selected_correction >= p->n_corrections)
        p->selected_correction = p->n_corrections - 1;
      break;
    }
  }

  // Add button (adds a placeholder; real coordinates set interactively)
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  if (nk_button_label(ctx, "Add") && p->n_corrections < MAX_CORRECTIONS) {
    correction_entry *c = &p->corrections[p->n_corrections++];
    memset(c, 0, sizeof(*c));
    LOG_INFO("seg_panels: correction point added (%d total)", p->n_corrections);
  }
  if (nk_button_label(ctx, "Clear all")) {
    p->n_corrections = 0;
    p->selected_correction = -1;
  }

  nk_widget_collapsible_end(ctx);
}

static void render_approval(seg_panels *p, struct nk_context *ctx,
                             quad_surface *surf) {
  if (!nk_widget_collapsible_begin(ctx, "Approval Mask", &p->approval_open)) return;

  // Paint / Erase toggle
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  int paint_val = p->paint_mode ? 1 : 0;
  nk_checkbox_label(ctx, "Paint (approve)", &paint_val);
  p->paint_mode = (paint_val != 0);

  // Brush size
  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Brush size:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 1.0f, &p->brush_size, 100.0f, 1.0f);

  // Coverage display
  if (surf) {
    char cov_buf[32];
    snprintf(cov_buf, sizeof(cov_buf), "%.1f%%", p->coverage * 100.0f);
    nk_widget_labeled_str(ctx, "Coverage:", cov_buf);
    nk_widget_progress_labeled(ctx, "Coverage", p->coverage);
  } else {
    nk_layout_row_dynamic(ctx, ROW_H, 1);
    nk_label(ctx, "No active surface", NK_TEXT_CENTERED);
  }

  nk_widget_collapsible_end(ctx);
}

static void render_custom_params(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Custom Params (JSON)", &p->custom_open)) return;

  nk_layout_row_dynamic(ctx, 80, 1);
  nk_flags ev = nk_edit_string_zero_terminated(ctx,
    NK_EDIT_BOX | NK_EDIT_MULTILINE,
    p->params_json, PARAMS_JSON_LEN, nk_filter_default);
  if (ev & NK_EDIT_COMMITED) p->params_dirty = true;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  if (nk_button_label(ctx, "Apply")) {
    p->params_dirty = false;
    LOG_INFO("seg_panels: custom params applied: %.64s", p->params_json);
  }
  if (nk_button_label(ctx, "Reset"))
    strncpy(p->params_json, "{}", sizeof(p->params_json) - 1);

  nk_widget_collapsible_end(ctx);
}

static void render_cell_reopt(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Cell Reoptimization", &p->reopt_open)) return;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Learning rate:", NK_TEXT_LEFT);
  nk_slider_float(ctx, 1e-5f, &p->reopt_learning_rate, 0.1f, 1e-5f);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Iterations:", NK_TEXT_LEFT);
  nk_slider_int(ctx, 1, &p->reopt_iterations, 1000, 10);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  int gpu_val = p->reopt_use_gpu ? 1 : 0;
  nk_checkbox_label(ctx, "Use GPU", &gpu_val);
  p->reopt_use_gpu = (gpu_val != 0);

  char lr_buf[32];
  snprintf(lr_buf, sizeof(lr_buf), "lr=%.5f  iter=%d", p->reopt_learning_rate,
           p->reopt_iterations);
  nk_layout_row_dynamic(ctx, ROW_H, 1);
  nk_label(ctx, lr_buf, NK_TEXT_LEFT);

  nk_layout_row_dynamic(ctx, ROW_H, 1);
  if (nk_button_label(ctx, "Run Cell Reopt"))
    LOG_INFO("seg_panels: cell reopt requested lr=%.5f iter=%d gpu=%d",
             p->reopt_learning_rate, p->reopt_iterations, p->reopt_use_gpu);

  nk_widget_collapsible_end(ctx);
}

static void render_direction_field(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Direction Field", &p->dirfield_open)) return;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  int en_val = p->dirfield_enabled ? 1 : 0;
  nk_checkbox_label(ctx, "Enable overlay", &en_val);
  p->dirfield_enabled = (en_val != 0);

  if (p->dirfield_enabled) {
    nk_layout_row_dynamic(ctx, ROW_H, 2);
    nk_label(ctx, "Opacity:", NK_TEXT_LEFT);
    nk_slider_float(ctx, 0.0f, &p->dirfield_alpha, 1.0f, 0.01f);
  }

  nk_widget_collapsible_end(ctx);
}

static void render_lasagna(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Lasagna Service", &p->lasagna_open)) return;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Host:", NK_TEXT_LEFT);
  nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, p->lasagna_host,
                                  (int)sizeof(p->lasagna_host), nk_filter_default);

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Port:", NK_TEXT_LEFT);
  nk_property_int(ctx, "#", 1, &p->lasagna_port, 65535, 1, 1.0f);

  nk_widget_labeled_str(ctx, "Status:", svc_status_str(p->lasagna_status));
  if (p->lasagna_job_id[0])
    nk_widget_labeled_str(ctx, "Job ID:", p->lasagna_job_id);

  nk_layout_row_dynamic(ctx, ROW_H, 3);
  if (nk_button_label(ctx, "Connect"))
    LOG_INFO("seg_panels: lasagna connect %s:%d", p->lasagna_host, p->lasagna_port);
  if (nk_button_label(ctx, "Submit"))
    LOG_INFO("seg_panels: lasagna submit");
  if (nk_button_label(ctx, "Cancel"))
    LOG_INFO("seg_panels: lasagna cancel");

  nk_widget_collapsible_end(ctx);
}

static void render_neural_tracer(seg_panels *p, struct nk_context *ctx) {
  if (!nk_widget_collapsible_begin(ctx, "Neural Tracer", &p->neural_open)) return;

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  nk_label(ctx, "Model:", NK_TEXT_LEFT);
  nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, p->neural_model_path,
                                  (int)sizeof(p->neural_model_path), nk_filter_default);

  nk_widget_labeled_str(ctx, "Status:", svc_status_str(p->neural_status));

  if (p->neural_status == SVC_RUNNING) {
    char conf_buf[32];
    snprintf(conf_buf, sizeof(conf_buf), "%.2f", p->neural_confidence);
    nk_widget_labeled_str(ctx, "Confidence:", conf_buf);
    nk_widget_progress_labeled(ctx, "Inference", p->neural_confidence);
  }

  nk_layout_row_dynamic(ctx, ROW_H, 2);
  if (nk_button_label(ctx, "Start"))
    LOG_INFO("seg_panels: neural tracer start model=%s", p->neural_model_path);
  if (nk_button_label(ctx, "Stop"))
    LOG_INFO("seg_panels: neural tracer stop");

  nk_widget_collapsible_end(ctx);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void seg_panels_render(seg_panels *p, struct nk_context *ctx,
                       quad_surface *active_surface) {
  if (!p || !ctx) return;
  render_header(p, ctx, active_surface);
  render_editing(p, ctx);
  render_growth(p, ctx);
  render_corrections(p, ctx);
  render_approval(p, ctx, active_surface);
  render_custom_params(p, ctx);
  render_cell_reopt(p, ctx);
  render_direction_field(p, ctx);
  render_lasagna(p, ctx);
  render_neural_tracer(p, ctx);
}

seg_tool_params seg_panels_get_tool_params(const seg_panels *p) {
  if (!p) { seg_tool_params z = {0}; return z; }
  return p->tool;
}

growth_params seg_panels_get_growth_params(const seg_panels *p) {
  if (!p) { growth_params z = {0}; return z; }
  return p->growth;
}
