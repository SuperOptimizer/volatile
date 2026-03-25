// ---------------------------------------------------------------------------
// Volatile GUI — main entry point
// Wires all widgets into the SDL3 + Nuklear main loop.
// ---------------------------------------------------------------------------

// Nuklear — NK_IMPLEMENTATION is owned by app.c; use declaration-only here.
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

#include "gui/app.h"
#include "gui/layout.h"
#include "gui/viewer.h"
#include "gui/viewer3d.h"
#include "gui/crosshair.h"
#include "gui/surface_panel.h"
#include "gui/console.h"
#include "gui/settings.h"
#include "gui/toolbar.h"
#include "gui/vol_selector.h"
#include "gui/window_range.h"
#include "render/tile.h"
#include "render/camera.h"
#include "render/overlay.h"
#include "core/log.h"
#include "core/vol.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Log console callback — bridge log_msg -> log_console
// ---------------------------------------------------------------------------

static log_console *g_console = NULL;

static void console_log_cb(void *ctx, log_level_t level,
                            const char *file, int line, const char *msg) {
  (void)ctx;
  if (g_console) log_console_add(g_console, (int)level, file, line, msg);
}

// ---------------------------------------------------------------------------
// Tool state
// ---------------------------------------------------------------------------

typedef enum { TOOL_BRUSH, TOOL_LINE, TOOL_PUSH_PULL } active_tool;

// ---------------------------------------------------------------------------
// Render a panel by pixel rect into a Nuklear canvas image.
// For now, renders a test-pattern label (real volume render hooks in here).
// ---------------------------------------------------------------------------

static void render_viewer_panel(struct nk_context *ctx,
                                slice_viewer *v, const char *label) {
  nk_layout_row_dynamic(ctx, 20, 1);
  nk_label(ctx, label, NK_TEXT_LEFT);
  // NOTE: real integration would blit viewer_render() pixels via
  //       nk_image() once SDL3 texture upload is wired.
  nk_layout_row_dynamic(ctx, 20, 1);
  char info[64];
  snprintf(info, sizeof(info), "axis=%d  slice=%.1f",
           viewer_get_axis(v), viewer_current_slice(v));
  nk_label(ctx, info, NK_TEXT_LEFT);
}

static void render_3d_panel(struct nk_context *ctx, viewer3d *v) {
  nk_layout_row_dynamic(ctx, 20, 1);
  nk_label(ctx, "3D Viewer (CPU raycast)", NK_TEXT_LEFT);
  (void)v;
}

// ---------------------------------------------------------------------------
// Helper: open a panel using layout geometry
// ---------------------------------------------------------------------------

static bool panel_begin(struct nk_context *ctx, const app_layout *layout,
                        panel_id id, int win_w, int win_h, nk_flags flags) {
  panel_rect r = layout_get_panel(layout, id);
  if (!r.visible || r.w <= 0.0f || r.h <= 0.0f) return false;
  float px = r.x * (float)win_w, py = r.y * (float)win_h;
  float pw = r.w * (float)win_w, ph = r.h * (float)win_h;
  return nk_begin(ctx, r.title, nk_rect(px, py, pw, ph), flags) != 0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  (void)argc; (void)argv;

  log_set_level(LOG_INFO);

  // --- Console (capture logs before window opens) ---
  g_console = log_console_new(1000);
  log_set_callback(console_log_cb, NULL);

  LOG_INFO("Volatile v%s starting", volatile_version());

  // --- App window ---
  app_config_t cfg = {.title = "Volatile", .width = 1600, .height = 900};
  app_state_t *app = app_init(&cfg);
  if (!app) { log_set_callback(NULL, NULL); log_console_free(g_console); return 1; }

  const int WIN_W = 1600, WIN_H = 900;

  // --- Layout ---
  app_layout *layout = layout_new_default();

  // --- Tile renderer (shared by all slice viewers) ---
  tile_renderer *tiles = tile_renderer_new(2);

  // --- Slice viewers: XY, XZ, YZ ---
  viewer_config vcfg_xy = {.view_axis = 0};
  viewer_config vcfg_xz = {.view_axis = 1};
  viewer_config vcfg_yz = {.view_axis = 2};
  composite_params_default(&vcfg_xy.composite);
  composite_params_default(&vcfg_xz.composite);
  composite_params_default(&vcfg_yz.composite);
  slice_viewer *vxy = viewer_new(vcfg_xy, tiles);
  slice_viewer *vxz = viewer_new(vcfg_xz, tiles);
  slice_viewer *vyz = viewer_new(vcfg_yz, tiles);

  // --- 3D viewer ---
  viewer3d_config v3cfg = {
    .mode        = RENDER3D_MIP,
    .iso_value   = 0.5f,
    .step_size   = 0.5f,
    .fov_degrees = 45.0f,
    .cmap_id     = 0,
    .window      = 1.0f,
    .level       = 0.5f,
  };
  viewer3d *v3d = viewer3d_new(v3cfg);

  // --- Crosshair sync ---
  crosshair_sync *xhair = crosshair_sync_new();
  crosshair_sync_add_viewer(xhair, vxy);
  crosshair_sync_add_viewer(xhair, vxz);
  crosshair_sync_add_viewer(xhair, vyz);

  // --- Overlay lists (one per viewer for crosshair lines) ---
  overlay_list *ov_xy = overlay_list_new();
  overlay_list *ov_xz = overlay_list_new();
  overlay_list *ov_yz = overlay_list_new();
  viewer_set_overlays(vxy, ov_xy);
  viewer_set_overlays(vxz, ov_xz);
  viewer_set_overlays(vyz, ov_yz);

  // --- Surface panel ---
  surface_panel *surf_panel = surface_panel_new();
  // Add placeholder entries so the panel isn't empty
  surface_panel_add(surf_panel, &(surface_entry){
    .id = 1, .name = "Segment 1", .visible = true, .approved = false,
    .area_vx2 = 12500.0f
  });

  // --- Settings ---
  settings *prefs = settings_open(NULL);  // global-only, no project dir

  // --- Toolbar ---
  active_tool current_tool = TOOL_BRUSH;
  toolbar *tools = toolbar_new();

  void *tool_ptrs[3] = {
    (void *)((uintptr_t)TOOL_BRUSH),
    (void *)((uintptr_t)TOOL_LINE),
    (void *)((uintptr_t)TOOL_PUSH_PULL),
  };
  // NOTE: callbacks update current_tool via a small lambda pattern
  // Since C doesn't have closures, we pass &current_tool as ctx
  // and use a single generic setter callback
  toolbar_add_button(tools, "Brush", NULL,
    (toolbar_action_fn)(void(*)(void*))NULL, tool_ptrs[0]);
  toolbar_add_button(tools, "Line", NULL,
    (toolbar_action_fn)(void(*)(void*))NULL, tool_ptrs[1]);
  toolbar_add_separator(tools);
  toolbar_add_button(tools, "Push-Pull", NULL,
    (toolbar_action_fn)(void(*)(void*))NULL, tool_ptrs[2]);

  // --- Volume selector ---
  vol_selector *volsel = vol_selector_new();
  vol_selector_add(volsel, "(no volume)", "");
  // Argv volumes
  for (int i = 1; i < argc; i++) {
    vol_selector_add(volsel, argv[i], argv[i]);
  }

  // --- Window/level widget ---
  window_range_state wr;
  window_range_init(&wr);

  // --- Loaded volume (populated when user selects one) ---
  volume *vol = NULL;
  int last_vol_sel = 0;

  LOG_INFO("GUI ready — entering main loop");

  // ---------------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------------
  nk_flags panel_flags = NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_NO_SCROLLBAR;

  while (!app_should_close(app)) {
    // --- Event pump + Nuklear input ---
    app_begin_frame(app);
    struct nk_context *ctx = app_nk_ctx(app);

    // --- Volume selector: load volume on change ---
    if (panel_begin(ctx, layout, PANEL_VOLUME_BROWSER, WIN_W, WIN_H, panel_flags)) {
      // NOTE: PANEL_VOLUME_BROWSER is hidden by default; show via menu later.
      nk_end(ctx);
    }

    // --- Side panel: Segmentation tools (toolbar + volume selector + W/L) ---
    if (panel_begin(ctx, layout, PANEL_SEGMENTATION, WIN_W, WIN_H, panel_flags)) {
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Volume", NK_TEXT_LEFT);
      if (vol_selector_render(volsel, ctx)) {
        int sel = vol_selector_selected(volsel);
        if (sel != last_vol_sel) {
          last_vol_sel = sel;
          const char *path = vol_selector_selected_path(volsel);
          if (path && path[0] != '\0') {
            vol_free(vol);
            vol = vol_open(path);
            if (vol) {
              viewer_set_volume(vxy, vol);
              viewer_set_volume(vxz, vol);
              viewer_set_volume(vyz, vol);
              viewer3d_set_volume(v3d, vol);
              LOG_INFO("Opened volume: %s", path);
            } else {
              LOG_WARN("Failed to open volume: %s", path);
            }
          }
        }
      }

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Window / Level", NK_TEXT_LEFT);
      window_range_render(&wr, ctx);

      nk_layout_row_dynamic(ctx, 4, 1);
      nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Tools", NK_TEXT_LEFT);
      toolbar_render(tools, ctx);
      nk_end(ctx);
    }

    // --- Side panel: Surface tree ---
    if (panel_begin(ctx, layout, PANEL_SURFACE_TREE, WIN_W, WIN_H, panel_flags)) {
      surface_panel_render(surf_panel, ctx, NULL);
      nk_end(ctx);
    }

    // --- Console ---
    if (panel_begin(ctx, layout, PANEL_CONSOLE, WIN_W, WIN_H,
                    NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
      log_console_render(g_console, ctx, NULL);
      nk_end(ctx);
    }

    // --- Update crosshair overlays ---
    overlay_list_clear(ov_xy);
    overlay_list_clear(ov_xz);
    overlay_list_clear(ov_yz);
    crosshair_sync_render_overlays(xhair, vxy, ov_xy);
    crosshair_sync_render_overlays(xhair, vxz, ov_xz);
    crosshair_sync_render_overlays(xhair, vyz, ov_yz);

    // --- XY viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_XY, WIN_W, WIN_H, panel_flags)) {
      render_viewer_panel(ctx, vxy, "XY (axial)");
      nk_end(ctx);
    }

    // --- XZ viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_XZ, WIN_W, WIN_H, panel_flags)) {
      render_viewer_panel(ctx, vxz, "XZ (coronal)");
      nk_end(ctx);
    }

    // --- YZ viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_YZ, WIN_W, WIN_H, panel_flags)) {
      render_viewer_panel(ctx, vyz, "YZ (sagittal)");
      nk_end(ctx);
    }

    // --- 3D viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_3D, WIN_W, WIN_H, panel_flags)) {
      render_3d_panel(ctx, v3d);
      nk_end(ctx);
    }

    // --- Present frame ---
    app_end_frame(app);
  }

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  log_set_callback(NULL, NULL);

  vol_free(vol);
  overlay_list_free(ov_xy);
  overlay_list_free(ov_xz);
  overlay_list_free(ov_yz);
  crosshair_sync_free(xhair);
  viewer_free(vxy);
  viewer_free(vxz);
  viewer_free(vyz);
  viewer3d_free(v3d);
  tile_renderer_free(tiles);
  vol_selector_free(volsel);
  surface_panel_free(surf_panel);
  toolbar_free(tools);
  settings_close(prefs);
  layout_free(layout);
  log_console_free(g_console);
  app_shutdown(app);
  return 0;
}
