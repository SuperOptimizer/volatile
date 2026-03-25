// ---------------------------------------------------------------------------
// menubar.c — VC3D-style menu bar (File / Edit / View / Selection / Help)
// Uses nk_menubar_begin/end + nk_menu_begin_label/end + nk_menu_item_label.
// NK_IMPLEMENTATION is owned by app.c; include nuklear in declaration mode.
// ---------------------------------------------------------------------------

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA
#include <nuklear.h>
#pragma GCC diagnostic pop

#include "gui/menubar.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MENU_HEIGHT   25     // px, menu bar row height
#define MENU_BTN_W    80     // px, each top-level menu button width
#define ITEM_H        22     // px, per-item row height
#define DROPDOWN_W    220    // px, dropdown panel width
#define RECENT_MAX    16     // max recent file entries
#define PATH_MAX_DISP 64     // chars to display for recent paths

// ---------------------------------------------------------------------------
// Action slot: label + callback
// ---------------------------------------------------------------------------

typedef struct {
  char          label[64];
  menu_action_fn fn;
  void          *fn_ctx;
} action_slot;

static void slot_fire(const action_slot *s) {
  if (s && s->fn) s->fn(s->fn_ctx);
}

// ---------------------------------------------------------------------------
// menubar struct
// ---------------------------------------------------------------------------

struct menubar {
  // File
  action_slot open_volpkg;
  action_slot open_zarr;
  action_slot open_remote;
  action_slot attach_remote_zarr;
  action_slot generate_report;
  action_slot settings;
  action_slot import_obj;
  action_slot exit_app;

  // View
  action_slot toggle_volumes;
  action_slot toggle_segmentation;
  action_slot toggle_distance_transform;
  action_slot toggle_drawing;
  action_slot toggle_viewer_controls;
  action_slot toggle_point_collection;
  action_slot sync_cursor;
  action_slot reset_seg_views;
  action_slot show_console;

  // Selection
  action_slot draw_bbox;
  action_slot surface_from_selection;
  action_slot clear_selection;
  action_slot inpaint_rebuild;

  // Help
  action_slot keybinds;
  action_slot about;

  // Recent files
  char recent[RECENT_MAX][256];
  int  recent_count;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

menubar *menubar_new(void) {
  menubar *m = calloc(1, sizeof(*m));
  if (!m) { LOG_ERROR("menubar_new: calloc failed"); return NULL; }

  // Pre-fill display labels (used during render for nk_menu_item_label)
  snprintf(m->open_volpkg.label,          sizeof(m->open_volpkg.label),          "Open volpkg...");
  snprintf(m->open_zarr.label,            sizeof(m->open_zarr.label),            "Open Local Zarr...");
  snprintf(m->open_remote.label,          sizeof(m->open_remote.label),          "Open Remote Volume...");
  snprintf(m->attach_remote_zarr.label,   sizeof(m->attach_remote_zarr.label),   "Attach Remote Zarr...");
  snprintf(m->generate_report.label,      sizeof(m->generate_report.label),      "Generate Review Report...");
  snprintf(m->settings.label,             sizeof(m->settings.label),             "Settings");
  snprintf(m->import_obj.label,           sizeof(m->import_obj.label),           "Import OBJ as Patch...");
  snprintf(m->exit_app.label,             sizeof(m->exit_app.label),             "Exit");

  snprintf(m->toggle_volumes.label,           sizeof(m->toggle_volumes.label),           "Toggle Volumes panel");
  snprintf(m->toggle_segmentation.label,      sizeof(m->toggle_segmentation.label),      "Toggle Segmentation panel");
  snprintf(m->toggle_distance_transform.label,sizeof(m->toggle_distance_transform.label),"Toggle Distance Transform panel");
  snprintf(m->toggle_drawing.label,           sizeof(m->toggle_drawing.label),           "Toggle Drawing panel");
  snprintf(m->toggle_viewer_controls.label,   sizeof(m->toggle_viewer_controls.label),   "Toggle Viewer Controls panel");
  snprintf(m->toggle_point_collection.label,  sizeof(m->toggle_point_collection.label),  "Toggle Point Collection panel");
  snprintf(m->sync_cursor.label,              sizeof(m->sync_cursor.label),              "Sync cursor to Surface view");
  snprintf(m->reset_seg_views.label,          sizeof(m->reset_seg_views.label),          "Reset Segmentation Views");
  snprintf(m->show_console.label,             sizeof(m->show_console.label),             "Show Console Output");

  snprintf(m->draw_bbox.label,              sizeof(m->draw_bbox.label),              "Draw BBox");
  snprintf(m->surface_from_selection.label, sizeof(m->surface_from_selection.label), "Surface from Selection");
  snprintf(m->clear_selection.label,        sizeof(m->clear_selection.label),        "Clear Selection");
  snprintf(m->inpaint_rebuild.label,        sizeof(m->inpaint_rebuild.label),        "Inpaint (Telea) & Rebuild Segment");

  snprintf(m->keybinds.label, sizeof(m->keybinds.label), "Keybinds");
  snprintf(m->about.label,    sizeof(m->about.label),    "About...");

  return m;
}

void menubar_free(menubar *m) {
  free(m);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Render a single menu item; fires its callback if clicked.
static void render_item(struct nk_context *ctx, action_slot *s) {
  if (nk_menu_item_label(ctx, s->label, NK_TEXT_LEFT))
    slot_fire(s);
}

// Compute dropdown height as N items * ITEM_H + optional separator rows.
static float item_height(int n_items, int n_separators) {
  return (float)(n_items * ITEM_H + n_separators * 6);
}

// ---------------------------------------------------------------------------
// menubar_render
// ---------------------------------------------------------------------------

void menubar_render(menubar *m, struct nk_context *ctx) {
  if (!m || !ctx) return;

  nk_menubar_begin(ctx);

  // -------------------------------------------------------------------------
  // File
  // -------------------------------------------------------------------------
  nk_layout_row_begin(ctx, NK_STATIC, MENU_HEIGHT, 5);
  nk_layout_row_push(ctx, (float)MENU_BTN_W);
  {
    // 8 items + 3 separators (after open group, after report, after import)
    float h = item_height(8, 3);
    if (nk_menu_begin_label(ctx, "File", NK_TEXT_LEFT,
                            nk_vec2((float)DROPDOWN_W, h))) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->open_volpkg);
      render_item(ctx, &m->open_zarr);
      render_item(ctx, &m->open_remote);
      render_item(ctx, &m->attach_remote_zarr);

      // Recent volpkg sub-label (non-interactive header)
      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      nk_label(ctx, "  Open recent volpkg", NK_TEXT_LEFT);
      if (m->recent_count == 0) {
        nk_label(ctx, "    (none)", NK_TEXT_LEFT);
      } else {
        for (int i = 0; i < m->recent_count; i++) {
          char label[PATH_MAX_DISP + 4];
          snprintf(label, sizeof(label), "    %.*s", PATH_MAX_DISP, m->recent[i]);
          // Recent items fire open_volpkg with no dedicated slot per path;
          // callers can override open_volpkg callback to inspect the path via
          // their own state. This keeps the API surface small.
          if (nk_menu_item_label(ctx, label, NK_TEXT_LEFT))
            slot_fire(&m->open_volpkg);
        }
      }

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      render_item(ctx, &m->generate_report);

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      render_item(ctx, &m->settings);
      render_item(ctx, &m->import_obj);

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      render_item(ctx, &m->exit_app);

      nk_menu_end(ctx);
    }
  }

  // -------------------------------------------------------------------------
  // Edit  (reserved — segmentation editing actions added later)
  // -------------------------------------------------------------------------
  nk_layout_row_push(ctx, (float)MENU_BTN_W);
  {
    float h = item_height(1, 0);
    if (nk_menu_begin_label(ctx, "Edit", NK_TEXT_LEFT,
                            nk_vec2((float)DROPDOWN_W, h))) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      nk_label(ctx, "(no actions)", NK_TEXT_LEFT);
      nk_menu_end(ctx);
    }
  }

  // -------------------------------------------------------------------------
  // View
  // -------------------------------------------------------------------------
  nk_layout_row_push(ctx, (float)MENU_BTN_W);
  {
    float h = item_height(9, 2);
    if (nk_menu_begin_label(ctx, "View", NK_TEXT_LEFT,
                            nk_vec2((float)DROPDOWN_W, h))) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->toggle_volumes);
      render_item(ctx, &m->toggle_segmentation);
      render_item(ctx, &m->toggle_distance_transform);
      render_item(ctx, &m->toggle_drawing);
      render_item(ctx, &m->toggle_viewer_controls);
      render_item(ctx, &m->toggle_point_collection);

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->sync_cursor);
      render_item(ctx, &m->reset_seg_views);

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->show_console);

      nk_menu_end(ctx);
    }
  }

  // -------------------------------------------------------------------------
  // Selection
  // -------------------------------------------------------------------------
  nk_layout_row_push(ctx, (float)MENU_BTN_W);
  {
    float h = item_height(4, 0);
    if (nk_menu_begin_label(ctx, "Selection", NK_TEXT_LEFT,
                            nk_vec2((float)DROPDOWN_W, h))) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->draw_bbox);
      render_item(ctx, &m->surface_from_selection);
      render_item(ctx, &m->clear_selection);
      render_item(ctx, &m->inpaint_rebuild);

      nk_menu_end(ctx);
    }
  }

  // -------------------------------------------------------------------------
  // Help
  // -------------------------------------------------------------------------
  nk_layout_row_push(ctx, (float)MENU_BTN_W);
  {
    float h = item_height(2, 0);
    if (nk_menu_begin_label(ctx, "Help", NK_TEXT_LEFT,
                            nk_vec2((float)DROPDOWN_W, h))) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

      render_item(ctx, &m->keybinds);
      render_item(ctx, &m->about);

      nk_menu_end(ctx);
    }
  }

  nk_layout_row_end(ctx);
  nk_menubar_end(ctx);
}

// ---------------------------------------------------------------------------
// Callback registration — File
// ---------------------------------------------------------------------------

void menubar_on_open_volpkg(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->open_volpkg.fn = fn; m->open_volpkg.fn_ctx = ctx; }
}
void menubar_on_open_zarr(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->open_zarr.fn = fn; m->open_zarr.fn_ctx = ctx; }
}
void menubar_on_open_remote(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->open_remote.fn = fn; m->open_remote.fn_ctx = ctx; }
}
void menubar_on_attach_remote_zarr(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->attach_remote_zarr.fn = fn; m->attach_remote_zarr.fn_ctx = ctx; }
}
void menubar_on_generate_report(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->generate_report.fn = fn; m->generate_report.fn_ctx = ctx; }
}
void menubar_on_settings(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->settings.fn = fn; m->settings.fn_ctx = ctx; }
}
void menubar_on_import_obj(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->import_obj.fn = fn; m->import_obj.fn_ctx = ctx; }
}
void menubar_on_exit(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->exit_app.fn = fn; m->exit_app.fn_ctx = ctx; }
}

// ---------------------------------------------------------------------------
// Callback registration — View
// ---------------------------------------------------------------------------

void menubar_on_toggle_volumes(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_volumes.fn = fn; m->toggle_volumes.fn_ctx = ctx; }
}
void menubar_on_toggle_segmentation(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_segmentation.fn = fn; m->toggle_segmentation.fn_ctx = ctx; }
}
void menubar_on_toggle_distance_transform(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_distance_transform.fn = fn; m->toggle_distance_transform.fn_ctx = ctx; }
}
void menubar_on_toggle_drawing(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_drawing.fn = fn; m->toggle_drawing.fn_ctx = ctx; }
}
void menubar_on_toggle_viewer_controls(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_viewer_controls.fn = fn; m->toggle_viewer_controls.fn_ctx = ctx; }
}
void menubar_on_toggle_point_collection(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->toggle_point_collection.fn = fn; m->toggle_point_collection.fn_ctx = ctx; }
}
void menubar_on_sync_cursor(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->sync_cursor.fn = fn; m->sync_cursor.fn_ctx = ctx; }
}
void menubar_on_reset_seg_views(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->reset_seg_views.fn = fn; m->reset_seg_views.fn_ctx = ctx; }
}
void menubar_on_show_console(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->show_console.fn = fn; m->show_console.fn_ctx = ctx; }
}

// ---------------------------------------------------------------------------
// Callback registration — Selection
// ---------------------------------------------------------------------------

void menubar_on_draw_bbox(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->draw_bbox.fn = fn; m->draw_bbox.fn_ctx = ctx; }
}
void menubar_on_surface_from_selection(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->surface_from_selection.fn = fn; m->surface_from_selection.fn_ctx = ctx; }
}
void menubar_on_clear_selection(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->clear_selection.fn = fn; m->clear_selection.fn_ctx = ctx; }
}
void menubar_on_inpaint_rebuild(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->inpaint_rebuild.fn = fn; m->inpaint_rebuild.fn_ctx = ctx; }
}

// ---------------------------------------------------------------------------
// Callback registration — Help
// ---------------------------------------------------------------------------

void menubar_on_keybinds(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->keybinds.fn = fn; m->keybinds.fn_ctx = ctx; }
}
void menubar_on_about(menubar *m, menu_action_fn fn, void *ctx) {
  if (m) { m->about.fn = fn; m->about.fn_ctx = ctx; }
}

// ---------------------------------------------------------------------------
// Recent files
// ---------------------------------------------------------------------------

void menubar_add_recent(menubar *m, const char *path) {
  if (!m || !path || !path[0]) return;
  // Deduplicate: move to front if already present
  for (int i = 0; i < m->recent_count; i++) {
    if (strncmp(m->recent[i], path, sizeof(m->recent[i])) == 0) {
      // Shift down and place at front
      char tmp[256];
      snprintf(tmp, sizeof(tmp), "%s", m->recent[i]);
      for (int j = i; j > 0; j--)
        memcpy(m->recent[j], m->recent[j-1], sizeof(m->recent[0]));
      memcpy(m->recent[0], tmp, sizeof(m->recent[0]));
      return;
    }
  }
  // New entry: push front, evict oldest if full
  int n = m->recent_count < RECENT_MAX ? m->recent_count : RECENT_MAX - 1;
  for (int i = n; i > 0; i--)
    memcpy(m->recent[i], m->recent[i-1], sizeof(m->recent[0]));
  snprintf(m->recent[0], sizeof(m->recent[0]), "%s", path);
  if (m->recent_count < RECENT_MAX) m->recent_count++;
}
