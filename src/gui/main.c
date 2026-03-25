// ---------------------------------------------------------------------------
// Volatile GUI — main entry point
// Wires all widgets into the SDL3 + Nuklear main loop.
// ---------------------------------------------------------------------------

#define _POSIX_C_SOURCE 200809L

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
#include "gui/menubar.h"
#include "gui/settings_dialog.h"
#include "gui/about_dialog.h"
#include "gui/keybinds_dialog.h"
#include "gui/file_dialog.h"
#include "gui/s3_browser.h"
#include "gui/seg_panels.h"
#include "gui/viewer_controls.h"
#include "gui/draw_panel.h"
#include "gui/dt_panel.h"
#include "gui/point_panel.h"
#include "gui/statusbar.h"
#include "gui/scalebar.h"
#include "gui/cred_dialog.h"
#include "gui/overlay_vol.h"
#include "gui/vol_info_panel.h"
#include "gui/tool_runner.h"
#include "gui/slice_controller.h"
#include "gui/keybind.h"
#include "render/tile.h"
#include "render/camera.h"
#include "render/overlay.h"
#include "core/log.h"
#include "core/vol.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>

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
// Floating panel state — panels not in the named layout slots
// ---------------------------------------------------------------------------

typedef struct {
  bool draw_panel_open;
  bool dt_panel_open;
  bool point_panel_open;
  bool viewer_controls_open;
  bool vol_info_open;
} float_panels;

// ---------------------------------------------------------------------------
// Menubar callback context — bundles all dialog/state pointers
// ---------------------------------------------------------------------------

typedef struct {
  settings_dialog  *settings_dlg;
  about_dialog     *about_dlg;
  keybinds_dialog  *keybinds_dlg;
  file_dialog      *file_dlg_volpkg;
  file_dialog      *file_dlg_zarr;
  s3_browser       *s3_brow;
  app_state_t      *app;
  float_panels     *panels;
  app_layout       *layout;
} menu_ctx;

static void cb_settings(void *ctx)    { settings_dialog_show(((menu_ctx*)ctx)->settings_dlg); }
static void cb_about(void *ctx)       { about_dialog_show(((menu_ctx*)ctx)->about_dlg); }
static void cb_keybinds(void *ctx)    { keybinds_dialog_show(((menu_ctx*)ctx)->keybinds_dlg); }
static void cb_open_volpkg(void *ctx) { file_dialog_show(((menu_ctx*)ctx)->file_dlg_volpkg, NULL); }
static void cb_open_zarr(void *ctx)   { file_dialog_show(((menu_ctx*)ctx)->file_dlg_zarr, NULL); }
static void cb_open_remote(void *ctx) {
  menu_ctx *m = ctx;
  // If no credentials are set in the browser, show cred_dialog first
  if (!s3_browser_is_visible(m->s3_brow))
    s3_browser_show(m->s3_brow);
}

static void cb_toggle_volumes(void *ctx) {
  layout_toggle_panel(((menu_ctx*)ctx)->layout, PANEL_VOLUME_BROWSER);
}
static void cb_toggle_segmentation(void *ctx) {
  layout_toggle_panel(((menu_ctx*)ctx)->layout, PANEL_SEGMENTATION);
}
static void cb_toggle_drawing(void *ctx) {
  ((menu_ctx*)ctx)->panels->draw_panel_open ^= true;
}
static void cb_toggle_dt(void *ctx) {
  ((menu_ctx*)ctx)->panels->dt_panel_open ^= true;
}
static void cb_toggle_viewer_controls(void *ctx) {
  ((menu_ctx*)ctx)->panels->viewer_controls_open ^= true;
}
static void cb_toggle_point_collection(void *ctx) {
  ((menu_ctx*)ctx)->panels->point_panel_open ^= true;
}
static void cb_show_console(void *ctx) {
  layout_toggle_panel(((menu_ctx*)ctx)->layout, PANEL_CONSOLE);
}

// ---------------------------------------------------------------------------
// Keyboard dispatch
// ---------------------------------------------------------------------------

typedef struct {
  keybind_map  *binds;
  slice_viewer *vxy, *vxz, *vyz;
} key_dispatch_ctx;

static void on_key_event(int scancode, int modifiers, bool pressed, void *raw_ctx) {
  if (!pressed) return;  // only handle key-down
  key_dispatch_ctx *kc = raw_ctx;
  int action = keybind_lookup(kc->binds, scancode, modifiers);
  if (action < 0) return;

  switch ((action_id)action) {
    case ACTION_PAN_LEFT:   viewer_pan(kc->vxy, -5.0f,  0.0f); break;
    case ACTION_PAN_RIGHT:  viewer_pan(kc->vxy,  5.0f,  0.0f); break;
    case ACTION_PAN_UP:     viewer_pan(kc->vxy,  0.0f, -5.0f); break;
    case ACTION_PAN_DOWN:   viewer_pan(kc->vxy,  0.0f,  5.0f); break;
    case ACTION_ZOOM_IN:    viewer_zoom(kc->vxy,  1.1f, 0.0f, 0.0f); break;
    case ACTION_ZOOM_OUT:   viewer_zoom(kc->vxy,  0.9f, 0.0f, 0.0f); break;
    case ACTION_SLICE_NEXT: viewer_scroll_slice(kc->vxy,  1.0f); break;
    case ACTION_SLICE_PREV: viewer_scroll_slice(kc->vxy, -1.0f); break;
    case ACTION_GROW_LEFT:  LOG_INFO("keybind: grow left");  break;
    case ACTION_GROW_RIGHT: LOG_INFO("keybind: grow right"); break;
    case ACTION_GROW_UP:    LOG_INFO("keybind: grow up");    break;
    case ACTION_GROW_DOWN:  LOG_INFO("keybind: grow down");  break;
    case ACTION_GROW_ALL_DIR: LOG_INFO("keybind: grow all"); break;
    case ACTION_GROW_ONE:   LOG_INFO("keybind: grow one");   break;
    case ACTION_GROW_ALL:   LOG_INFO("keybind: grow all (Ctrl+G)"); break;
    default: break;
  }
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

static bool panel_begin(struct nk_context *ctx, const app_layout *layout,
                        panel_id id, int win_w, int win_h, nk_flags flags) {
  panel_rect r = layout_get_panel(layout, id);
  if (!r.visible || r.w <= 0.0f || r.h <= 0.0f) return false;
  float px = r.x * (float)win_w, py = r.y * (float)win_h;
  float pw = r.w * (float)win_w, ph = r.h * (float)win_h;
  return nk_begin(ctx, r.title, nk_rect(px, py, pw, ph), flags) != 0;
}

static void render_viewer_panel(struct nk_context *ctx,
                                slice_viewer *v, const char *label,
                                scalebar *sb) {
  nk_layout_row_dynamic(ctx, 20, 1);
  nk_label(ctx, label, NK_TEXT_LEFT);
  nk_layout_row_dynamic(ctx, 20, 1);
  char info[64];
  snprintf(info, sizeof(info), "axis=%d  slice=%.1f",
           viewer_get_axis(v), viewer_current_slice(v));
  nk_label(ctx, info, NK_TEXT_LEFT);
  // Scale bar (zoom=1 until real pixel pipeline is wired)
  nk_layout_row_dynamic(ctx, 20, 1);
  scalebar_render(sb, ctx, 1.0f, 120);
}

static void render_3d_panel(struct nk_context *ctx, viewer3d *v) {
  nk_layout_row_dynamic(ctx, 20, 1);
  nk_label(ctx, "3D Viewer (CPU raycast)", NK_TEXT_LEFT);
  (void)v;
}

// Simple elapsed-ms helper using CLOCK_MONOTONIC
static float elapsed_ms(struct timespec *prev) {
  struct timespec now;
  clock_gettime(CLOCK_MONOTONIC, &now);
  float dt = (float)(now.tv_sec  - prev->tv_sec)  * 1000.0f
           + (float)(now.tv_nsec - prev->tv_nsec) / 1e6f;
  *prev = now;
  return dt;
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

  int win_w = 1600, win_h = 900;
  const int MENUBAR_H   = 30;
  const int STATUSBAR_H = 24;

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

  // --- Slice controllers (debounce) ---
  slice_controller *sc_xy = slice_controller_new(vxy);
  slice_controller *sc_xz = slice_controller_new(vxz);
  slice_controller *sc_yz = slice_controller_new(vyz);

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

  // --- Scale bars (one per slice viewer; default voxel size 65 nm) ---
  scalebar *sb_xy = scalebar_new(0.065f);
  scalebar *sb_xz = scalebar_new(0.065f);
  scalebar *sb_yz = scalebar_new(0.065f);

  // --- Surface panel ---
  surface_panel *surf_panel = surface_panel_new();
  surface_panel_add(surf_panel, &(surface_entry){
    .id = 1, .name = "Segment 1", .visible = true, .approved = false,
    .area_vx2 = 12500.0f
  });

  // --- Segmentation panels (replaces basic toolbar in PANEL_SEGMENTATION) ---
  seg_panels *segp = seg_panels_new();

  // --- Viewer controls ---
  viewer_controls *vctrl = viewer_controls_new();

  // --- Overlay volume (composited in viewer panels) ---
  overlay_volume *ovol = overlay_volume_new();

  // --- Draw panel ---
  draw_panel *drawp = draw_panel_new(512, 512);

  // --- Distance transform panel ---
  dt_panel *dtp = dt_panel_new();

  // --- Point collection panel ---
  point_panel *pointp = point_panel_new();

  // --- Volume info panel ---
  vol_info_panel *vol_info = vol_info_panel_new();

  // --- Status bar ---
  statusbar *sbar = statusbar_new();

  // --- Settings ---
  settings *prefs = settings_open(NULL);

  // --- Keybinds ---
  keybind_map *binds = keybind_new();
  key_dispatch_ctx kdc = { .binds = binds, .vxy = vxy, .vxz = vxz, .vyz = vyz };
  app_set_key_handler(app, on_key_event, &kdc);

  // --- Dialogs ---
  settings_dialog  *settings_dlg = settings_dialog_new(prefs);
  about_dialog     *about_dlg    = about_dialog_new();
  keybinds_dialog  *keybinds_dlg = keybinds_dialog_new(binds);
  file_dialog      *fdlg_volpkg  = file_dialog_new("Open volpkg", "*.volpkg");
  file_dialog      *fdlg_zarr    = file_dialog_new("Open Local Zarr", "*.zarr");
  s3_browser       *s3_brow      = s3_browser_new();
  cred_dialog      *cred_dlg     = cred_dialog_new();

  file_dialog_add_bookmark(fdlg_volpkg, "Home",  getenv("HOME") ? getenv("HOME") : "/");
  file_dialog_add_bookmark(fdlg_volpkg, "Data",  "/data");
  file_dialog_add_bookmark(fdlg_zarr,   "Home",  getenv("HOME") ? getenv("HOME") : "/");

  // --- Volume selector ---
  vol_selector *volsel = vol_selector_new();
  vol_selector_add(volsel, "(no volume)", "");
  for (int i = 1; i < argc; i++)
    vol_selector_add(volsel, argv[i], argv[i]);

  // --- Window/level widget ---
  window_range_state wr;
  window_range_init(&wr);

  // --- Toolbar (kept for reference; seg_panels is primary) ---
  toolbar *tools = toolbar_new();
  toolbar_add_button(tools, "Brush",     NULL, NULL, NULL);
  toolbar_add_button(tools, "Line",      NULL, NULL, NULL);
  toolbar_add_separator(tools);
  toolbar_add_button(tools, "Push-Pull", NULL, NULL, NULL);

  // --- Tool runner (CLI integration) ---
  tool_runner *runner = tool_runner_new();

  // --- Menu bar ---
  menubar *mbar = menubar_new();
  for (int i = argc - 1; i >= 1; i--)
    menubar_add_recent(mbar, argv[i]);

  // --- Floating panel visibility state ---
  float_panels flt = {
    .draw_panel_open       = false,
    .dt_panel_open         = false,
    .point_panel_open      = false,
    .viewer_controls_open  = false,
    .vol_info_open         = false,
  };

  // --- Menu callback context ---
  menu_ctx mctx = {
    .settings_dlg    = settings_dlg,
    .about_dlg       = about_dlg,
    .keybinds_dlg    = keybinds_dlg,
    .file_dlg_volpkg = fdlg_volpkg,
    .file_dlg_zarr   = fdlg_zarr,
    .s3_brow         = s3_brow,
    .app             = app,
    .panels          = &flt,
    .layout          = layout,
  };

  // Wire menubar callbacks
  menubar_on_open_volpkg(mbar,                cb_open_volpkg,          &mctx);
  menubar_on_open_zarr(mbar,                  cb_open_zarr,            &mctx);
  menubar_on_open_remote(mbar,                cb_open_remote,          &mctx);
  menubar_on_settings(mbar,                   cb_settings,             &mctx);
  menubar_on_about(mbar,                      cb_about,                &mctx);
  menubar_on_keybinds(mbar,                   cb_keybinds,             &mctx);
  menubar_on_toggle_volumes(mbar,             cb_toggle_volumes,       &mctx);
  menubar_on_toggle_segmentation(mbar,        cb_toggle_segmentation,  &mctx);
  menubar_on_toggle_drawing(mbar,             cb_toggle_drawing,       &mctx);
  menubar_on_toggle_distance_transform(mbar,  cb_toggle_dt,            &mctx);
  menubar_on_toggle_viewer_controls(mbar,     cb_toggle_viewer_controls,&mctx);
  menubar_on_toggle_point_collection(mbar,    cb_toggle_point_collection,&mctx);
  menubar_on_show_console(mbar,               cb_show_console,         &mctx);

  // --- Loaded volume ---
  volume *vol = NULL;
  int last_vol_sel = 0;

  // --- Frame timing ---
  struct timespec frame_ts;
  clock_gettime(CLOCK_MONOTONIC, &frame_ts);

  LOG_INFO("GUI ready — entering main loop");

  // ---------------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------------
  nk_flags panel_flags = NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_NO_SCROLLBAR;
  nk_flags float_flags = NK_WINDOW_BORDER | NK_WINDOW_TITLE
                       | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE | NK_WINDOW_CLOSABLE;

  while (!app_should_close(app)) {
    float dt_ms = elapsed_ms(&frame_ts);

    // 1. Event pump + Nuklear input
    app_begin_frame(app);
    app_get_size(app, &win_w, &win_h);
    struct nk_context *ctx = app_nk_ctx(app);

    // 2. Menu bar (full-width, 30 px, y=0)
    if (nk_begin(ctx, "##menubar",
                 nk_rect(0, 0, (float)win_w, (float)MENUBAR_H),
                 NK_WINDOW_NO_SCROLLBAR)) {
      menubar_render(mbar, ctx);
    }
    nk_end(ctx);

    // 3. Named layout panels
    // --- Volume browser (hidden by default; toggled via View menu) ---
    if (panel_begin(ctx, layout, PANEL_VOLUME_BROWSER, win_w, win_h, panel_flags)) {
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
              menubar_add_recent(mbar, path);
              LOG_INFO("Opened volume: %s", path);
            } else {
              LOG_WARN("Failed to open volume: %s", path);
            }
          }
        }
      }
    }
    nk_end(ctx);

    // --- Segmentation panel ---
    if (panel_begin(ctx, layout, PANEL_SEGMENTATION, win_w, win_h, panel_flags)) {
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
              menubar_add_recent(mbar, path);
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
      seg_panels_render(segp, ctx, NULL);
    }
    nk_end(ctx);

    // --- Surface tree ---
    if (panel_begin(ctx, layout, PANEL_SURFACE_TREE, win_w, win_h, panel_flags)) {
      surface_panel_render(surf_panel, ctx, NULL);
    }
    nk_end(ctx);

    // --- Console ---
    if (panel_begin(ctx, layout, PANEL_CONSOLE, win_w, win_h,
                    NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
      log_console_render(g_console, ctx, NULL);
    }
    nk_end(ctx);

    // --- Update crosshair overlays ---
    overlay_list_clear(ov_xy);
    overlay_list_clear(ov_xz);
    overlay_list_clear(ov_yz);
    crosshair_sync_render_overlays(xhair, vxy, ov_xy);
    crosshair_sync_render_overlays(xhair, vxz, ov_xz);
    crosshair_sync_render_overlays(xhair, vyz, ov_yz);

    // --- XY viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_XY, win_w, win_h, panel_flags)) {
      render_viewer_panel(ctx, vxy, "XY (axial)", sb_xy);
    }
    nk_end(ctx);

    // --- XZ viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_XZ, win_w, win_h, panel_flags)) {
      render_viewer_panel(ctx, vxz, "XZ (coronal)", sb_xz);
    }
    nk_end(ctx);

    // --- YZ viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_YZ, win_w, win_h, panel_flags)) {
      render_viewer_panel(ctx, vyz, "YZ (sagittal)", sb_yz);
    }
    nk_end(ctx);

    // --- 3D viewer ---
    if (panel_begin(ctx, layout, PANEL_VIEWER_3D, win_w, win_h, panel_flags)) {
      render_3d_panel(ctx, v3d);
    }
    nk_end(ctx);

    // 4. Floating panels (movable, toggled via View menu)

    // --- Viewer controls ---
    if (flt.viewer_controls_open) {
      if (nk_begin(ctx, "Viewer Controls",
                   nk_rect(1200, 50, 380, 400), float_flags)) {
        flt.viewer_controls_open = viewer_controls_render(vctrl, ctx, vxy);
        // overlay volume controls in same panel
        nk_layout_row_dynamic(ctx, 4, 1);
        nk_rule_horizontal(ctx, ctx->style.window.border_color, false);
        overlay_volume_render_controls(ovol, ctx);
      } else {
        flt.viewer_controls_open = false;
      }
      nk_end(ctx);
    }

    // --- Volume info ---
    if (flt.vol_info_open) {
      if (nk_begin(ctx, "Volume Info",
                   nk_rect(20, 50, 320, 300), float_flags)) {
        vol_info_panel_render(vol_info, ctx, vol, NULL);
      } else {
        flt.vol_info_open = false;
      }
      nk_end(ctx);
    }

    // --- Drawing panel ---
    if (flt.draw_panel_open) {
      if (nk_begin(ctx, "Drawing",
                   nk_rect(20, 360, 320, 400), float_flags)) {
        draw_panel_render(drawp, ctx);
      } else {
        flt.draw_panel_open = false;
      }
      nk_end(ctx);
    }

    // --- Distance transform panel ---
    if (flt.dt_panel_open) {
      if (nk_begin(ctx, "Distance Transform",
                   nk_rect(350, 360, 300, 300), float_flags)) {
        dt_panel_render(dtp, ctx, "Distance Transform");
      } else {
        flt.dt_panel_open = false;
      }
      nk_end(ctx);
    }

    // --- Point collection panel ---
    if (flt.point_panel_open) {
      if (nk_begin(ctx, "Point Collections",
                   nk_rect(660, 360, 300, 300), float_flags)) {
        point_panel_render(pointp, ctx, "Points");
      } else {
        flt.point_panel_open = false;
      }
      nk_end(ctx);
    }

    // 5. Dialogs (floating popups)
    settings_dialog_render(settings_dlg, ctx);
    about_dialog_render(about_dlg, ctx);
    keybinds_dialog_render(keybinds_dlg, ctx);
    file_dialog_render(fdlg_volpkg, ctx);
    file_dialog_render(fdlg_zarr, ctx);
    // S3 browser — show cred_dialog first if not authenticated
    if (s3_browser_is_visible(s3_brow) && !cred_dialog_is_visible(cred_dlg)) {
      if (s3_browser_render(s3_brow, ctx)) {
        const char *url = s3_browser_get_url(s3_brow);
        if (url && url[0] != '\0') {
          vol_free(vol);
          vol = vol_open(url);
          if (vol) {
            viewer_set_volume(vxy, vol);
            viewer_set_volume(vxz, vol);
            viewer_set_volume(vyz, vol);
            viewer3d_set_volume(v3d, vol);
            menubar_add_recent(mbar, url);
            LOG_INFO("Opened remote volume: %s", url);
          } else {
            // Auth may have failed — show cred dialog
            cred_dialog_show(cred_dlg, url);
            LOG_WARN("Failed to open remote volume: %s", url);
          }
        }
      }
    }
    // If cred_dialog submits, push creds into s3_browser
    if (cred_dialog_render(cred_dlg, ctx)) {
      s3_credentials *creds = cred_dialog_get_creds(cred_dlg);
      if (creds) s3_browser_set_creds(s3_brow, creds);
    }

    // Handle file dialog commits
    if (!file_dialog_is_visible(fdlg_volpkg)) {
      const char *path = file_dialog_get_path(fdlg_volpkg);
      if (path && path[0] != '\0') {
        vol_free(vol);
        vol = vol_open(path);
        if (vol) {
          viewer_set_volume(vxy, vol);
          viewer_set_volume(vxz, vol);
          viewer_set_volume(vyz, vol);
          viewer3d_set_volume(v3d, vol);
          menubar_add_recent(mbar, path);
          LOG_INFO("Opened volpkg: %s", path);
        } else {
          LOG_WARN("Failed to open: %s", path);
        }
      }
    }

    // 6. Status bar (full-width strip at bottom)
    statusbar_update(sbar, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0,
                     dt_ms > 0.0f ? 1000.0f / dt_ms : 0.0f, 0);
    if (nk_begin(ctx, "##statusbar",
                 nk_rect(0, (float)(win_h - STATUSBAR_H),
                         (float)win_w, (float)STATUSBAR_H),
                 NK_WINDOW_NO_SCROLLBAR)) {
      statusbar_render(sbar, ctx, STATUSBAR_H);
    }
    nk_end(ctx);

    // 7. Tick slice controllers (debounce)
    slice_controller_tick(sc_xy, dt_ms);
    slice_controller_tick(sc_xz, dt_ms);
    slice_controller_tick(sc_yz, dt_ms);

    // 8. Poll tool runner (CLI output)
    tool_runner_poll(runner);

    // 9. Present frame
    app_end_frame(app);
  }

  // ---------------------------------------------------------------------------
  // Cleanup
  // ---------------------------------------------------------------------------
  log_set_callback(NULL, NULL);

  tool_runner_free(runner);
  vol_free(vol);
  overlay_list_free(ov_xy);
  overlay_list_free(ov_xz);
  overlay_list_free(ov_yz);
  crosshair_sync_free(xhair);
  slice_controller_free(sc_xy);
  slice_controller_free(sc_xz);
  slice_controller_free(sc_yz);
  viewer_free(vxy);
  viewer_free(vxz);
  viewer_free(vyz);
  viewer3d_free(v3d);
  tile_renderer_free(tiles);
  scalebar_free(sb_xy);
  scalebar_free(sb_xz);
  scalebar_free(sb_yz);
  vol_selector_free(volsel);
  surface_panel_free(surf_panel);
  seg_panels_free(segp);
  viewer_controls_free(vctrl);
  overlay_volume_free(ovol);
  draw_panel_free(drawp);
  dt_panel_free(dtp);
  point_panel_free(pointp);
  vol_info_panel_free(vol_info);
  statusbar_free(sbar);
  toolbar_free(tools);
  menubar_free(mbar);
  settings_dialog_free(settings_dlg);
  about_dialog_free(about_dlg);
  keybinds_dialog_free(keybinds_dlg);
  file_dialog_free(fdlg_volpkg);
  file_dialog_free(fdlg_zarr);
  s3_browser_free(s3_brow);
  cred_dialog_free(cred_dlg);
  keybind_free(binds);
  settings_close(prefs);
  layout_free(layout);
  log_console_free(g_console);
  app_shutdown(app);
  return 0;
}
