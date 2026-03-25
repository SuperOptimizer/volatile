// volatile GUI — full panel set.
//
// WIDGET TYPE REFERENCE (see individual .h files for authoritative annotation):
//   CONTENT widgets: emit layout commands, must be called INSIDE nk_begin/nk_end.
//     menubar_render, welcome_panel_render, log_console_render, vol_selector_render,
//     window_range_render, seg_panels_render, statusbar_render, viewer_controls_render,
//     draw_panel_render, dt_panel_render, point_panel_render, vol_info_panel_render
//
//   WINDOW widgets: own their nk_begin/nk_end, call OUTSIDE any nk_begin block.
//     surface_panel_render, file_dialog_render, s3_browser_render, cred_dialog_render,
//     settings_dialog_render, about_dialog_render, keybinds_dialog_render
//
// Main loop sections:
//   A) CONTENT windows — we open nk_begin, call CONTENT widgets inside, then nk_end
//   B) WINDOW widgets  — call standalone, they manage their own begin/end
#define _POSIX_C_SOURCE 200809L

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

// CONTENT widgets
#include "gui/app.h"
#include "gui/menubar.h"
#include "gui/welcome_panel.h"
#include "gui/console.h"
#include "gui/seg_panels.h"
#include "gui/vol_selector.h"
#include "gui/window_range.h"
#include "gui/statusbar.h"
#include "gui/viewer_controls.h"
#include "gui/draw_panel.h"
#include "gui/dt_panel.h"
#include "gui/point_panel.h"
#include "gui/vol_info_panel.h"
// WINDOW widgets
#include "gui/surface_panel.h"
#include "gui/settings.h"
#include "gui/settings_dialog.h"
#include "gui/about_dialog.h"
#include "gui/keybind.h"
#include "gui/keybinds_dialog.h"
#include "gui/file_dialog.h"
#include "gui/s3_browser.h"
#include "gui/cred_dialog.h"
#include "core/log.h"
#include "core/vol.h"

#include <stdlib.h>
#include <string.h>

#define MB   30   // menubar height
#define SB   24   // statusbar height
#define RW   280  // right dock width
#define CON  180  // console height

typedef struct {
  bool viewer_controls, draw_panel, dt_panel, point_panel, vol_info, console;
} panel_vis;

// Log console singleton
static log_console *g_console;
static void console_hook(void *u, log_level_t lv, const char *f, int ln,
                          const char *m) {
  (void)u;
  if (g_console) log_console_add(g_console, (int)lv, f, ln, m);
}

// Menu callback context
typedef struct {
  panel_vis *vis;
  settings_dialog *settings_dlg;
  about_dialog *about_dlg;
  keybinds_dialog *keybinds_dlg;
  file_dialog *fd_zarr, *fd_volpkg;
  s3_browser *s3;
} menu_ctx;

static void cb_open_zarr(void *c)   { file_dialog_show(((menu_ctx *)c)->fd_zarr, NULL); }
static void cb_open_volpkg(void *c) { file_dialog_show(((menu_ctx *)c)->fd_volpkg, NULL); }
static void cb_open_s3(void *c)     { s3_browser_show(((menu_ctx *)c)->s3); }
static void cb_settings(void *c)    { settings_dialog_show(((menu_ctx *)c)->settings_dlg); }
static void cb_keybinds(void *c)    { keybinds_dialog_show(((menu_ctx *)c)->keybinds_dlg); }
static void cb_about(void *c)       { about_dialog_show(((menu_ctx *)c)->about_dlg); }

#define TOGGLE(f) static void cb_toggle_##f(void *c) { \
  menu_ctx *m = (menu_ctx *)c; m->vis->f = !m->vis->f; }
TOGGLE(viewer_controls) TOGGLE(draw_panel) TOGGLE(dt_panel)
TOGGLE(point_panel) TOGGLE(vol_info) TOGGLE(console)
#undef TOGGLE

static volume *try_open(const char *path, cred_dialog *cred) {
  if (!path || !path[0]) return NULL;
  LOG_INFO("Opening: %s", path);
  volume *v = vol_open(path);
  if (v) {
    LOG_INFO("Opened %s (%d levels)", path, vol_num_levels(v));
  } else {
    LOG_WARN("Failed: %s", path);
    if (strncmp(path, "s3://", 5) == 0) cred_dialog_show(cred, path);
  }
  return v;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  g_console = log_console_new(500);
  log_set_callback(console_hook, NULL);
  log_set_level(LOG_INFO);
  LOG_INFO("Volatile v%s starting", volatile_version());

  app_config_t cfg = {.title = "Volatile", .width = 1600, .height = 900};
  app_state_t *app = app_init(&cfg);
  if (!app) { log_set_callback(NULL, NULL); log_console_free(g_console); return 1; }

  settings *prefs = settings_open(NULL);

  // CONTENT widgets
  welcome_panel   *welcome = welcome_panel_new();
  seg_panels      *seg     = seg_panels_new();
  vol_selector    *volsel  = vol_selector_new();
  statusbar       *sbar    = statusbar_new();
  viewer_controls *vc      = viewer_controls_new();
  draw_panel      *drawp   = draw_panel_new(512, 512);
  dt_panel        *dtp     = dt_panel_new();
  point_panel     *pts     = point_panel_new();
  vol_info_panel  *vip     = vol_info_panel_new();
  window_range_state wr;
  window_range_init(&wr);
  vol_selector_add(volsel, "(no volume)", "");

  // WINDOW widgets
  surface_panel   *surfp        = surface_panel_new();
  settings_dialog *settings_dlg = settings_dialog_new(prefs);
  about_dialog    *about_dlg    = about_dialog_new();
  keybind_map     *keybinds     = keybind_new();
  keybinds_dialog *keybinds_dlg = keybinds_dialog_new(keybinds);
  file_dialog     *fd_zarr      = file_dialog_new("Open Local Zarr", "*.zarr");
  file_dialog     *fd_volpkg    = file_dialog_new("Open volpkg", "*.volpkg");
  s3_browser      *s3           = s3_browser_new();
  cred_dialog     *cred         = cred_dialog_new();
  menubar         *mbar         = menubar_new();

  panel_vis vis  = { .console = true };
  menu_ctx  mctx = {
    .vis = &vis, .settings_dlg = settings_dlg, .about_dlg = about_dlg,
    .keybinds_dlg = keybinds_dlg, .fd_zarr = fd_zarr, .fd_volpkg = fd_volpkg,
    .s3 = s3,
  };
  menubar_on_open_zarr(mbar, cb_open_zarr, &mctx);
  menubar_on_open_volpkg(mbar, cb_open_volpkg, &mctx);
  menubar_on_open_remote(mbar, cb_open_s3, &mctx);
  menubar_on_settings(mbar, cb_settings, &mctx);
  menubar_on_toggle_viewer_controls(mbar, cb_toggle_viewer_controls, &mctx);
  menubar_on_toggle_drawing(mbar, cb_toggle_draw_panel, &mctx);
  menubar_on_toggle_distance_transform(mbar, cb_toggle_dt_panel, &mctx);
  menubar_on_toggle_point_collection(mbar, cb_toggle_point_panel, &mctx);
  menubar_on_toggle_volumes(mbar, cb_toggle_vol_info, &mctx);
  menubar_on_show_console(mbar, cb_toggle_console, &mctx);
  menubar_on_keybinds(mbar, cb_keybinds, &mctx);
  menubar_on_about(mbar, cb_about, &mctx);

  volume *vol = NULL;

  while (!app_should_close(app)) {
    app_begin_frame(app);
    int w, h;
    app_get_size(app, &w, &h);
    struct nk_context *ctx = app_nk_ctx(app);
    int mid_h  = h - MB - SB;
    int main_w = w - RW;
    int view_h = vis.console ? mid_h - CON : mid_h;
    int con_y  = MB + view_h;

    // ================================================================
    // A) CONTENT windows — we own nk_begin/nk_end, call CONTENT widgets
    // ================================================================

    // Menubar (CONTENT: menubar_render)
    if (nk_begin(ctx, "##menubar", nk_rect(0, 0, (float)w, MB),
                 NK_WINDOW_NO_SCROLLBAR)) {
      menubar_render(mbar, ctx);                         // CONTENT
    }
    nk_end(ctx);

    // Main area: welcome panel (no vol) or 4 viewer windows (vol loaded)
    if (!vol) {
      if (nk_begin(ctx, "##welcome",
                   nk_rect(0, MB, (float)main_w, (float)view_h),
                   NK_WINDOW_NO_SCROLLBAR)) {
        welcome_result r = welcome_panel_render(welcome, ctx, main_w, view_h); // CONTENT
        switch (r.action) {
          case WELCOME_OPEN_ZARR:   file_dialog_show(fd_zarr, NULL);   break;
          case WELCOME_OPEN_VOLPKG: file_dialog_show(fd_volpkg, NULL); break;
          case WELCOME_OPEN_S3:     s3_browser_show(s3);               break;
          case WELCOME_OPEN_URL:
          case WELCOME_OPEN_RECENT:
            vol = try_open(r.url, cred);
            if (vol) menubar_add_recent(mbar, r.url);
            break;
          default: break;
        }
      }
      nk_end(ctx);
    } else {
      int hw = main_w / 2, hh = view_h / 2;
      if (nk_begin(ctx, "XY (axial)",
                   nk_rect(0, MB, (float)hw, (float)hh),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "XY", NK_TEXT_LEFT);
      }
      nk_end(ctx);
      if (nk_begin(ctx, "XZ (coronal)",
                   nk_rect((float)hw, MB, (float)(main_w - hw), (float)hh),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "XZ", NK_TEXT_LEFT);
      }
      nk_end(ctx);
      if (nk_begin(ctx, "YZ (sagittal)",
                   nk_rect(0, (float)(MB + hh), (float)hw, (float)(view_h - hh)),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "YZ", NK_TEXT_LEFT);
      }
      nk_end(ctx);
      if (nk_begin(ctx, "3D Viewer",
                   nk_rect((float)hw, (float)(MB + hh),
                            (float)(main_w - hw), (float)(view_h - hh)),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "3D", NK_TEXT_LEFT);
      }
      nk_end(ctx);
    }

    // Console (CONTENT: log_console_render)
    if (vis.console) {
      if (nk_begin(ctx, "Console",
                   nk_rect(0, (float)con_y, (float)main_w, CON),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
        log_console_render(g_console, ctx, NULL);        // CONTENT
      }
      nk_end(ctx);
    }

    // Right dock (CONTENT: vol_selector, window_range, seg_panels)
    if (nk_begin(ctx, "Controls",
                 nk_rect((float)main_w, MB, RW, (float)mid_h),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_SCROLL_AUTO_HIDE)) {
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Volume", NK_TEXT_LEFT);
      if (vol_selector_render(volsel, ctx)) {             // CONTENT
        const char *p = vol_selector_selected_path(volsel);
        if (p && p[0] && !vol) {
          vol = try_open(p, cred);
          if (vol) menubar_add_recent(mbar, p);
        }
      }
      nk_layout_row_dynamic(ctx, 4, 1); nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Window / Level", NK_TEXT_LEFT);
      window_range_render(&wr, ctx);                     // CONTENT
      nk_layout_row_dynamic(ctx, 4, 1); nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Segmentation", NK_TEXT_LEFT);
      seg_panels_render(seg, ctx, NULL);                 // CONTENT
    }
    nk_end(ctx);

    // Statusbar (CONTENT: statusbar_render)
    if (nk_begin(ctx, "##status",
                 nk_rect(0, (float)(h - SB), (float)w, SB),
                 NK_WINDOW_NO_SCROLLBAR)) {
      statusbar_update(sbar, 0, 0, 0, 0, 1.0f, 0, 60.0f, 0);
      statusbar_render(sbar, ctx, SB - 2);               // CONTENT
    }
    nk_end(ctx);

    // Floating CONTENT panels (skip both begin+end when hidden)
    if (vis.viewer_controls) {
      if (nk_begin(ctx, "Viewer Controls", nk_rect(20, MB + 10, 260, 400),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE |
                   NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
        viewer_controls_render(vc, ctx, NULL);           // CONTENT
      }
      if (nk_window_is_closed(ctx, "Viewer Controls")) vis.viewer_controls = false;
      nk_end(ctx);
    }
    if (vis.draw_panel) {
      if (nk_begin(ctx, "Drawing", nk_rect(20, MB + 10, 240, 360),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE |
                   NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
        draw_panel_render(drawp, ctx);                   // CONTENT
      }
      if (nk_window_is_closed(ctx, "Drawing")) vis.draw_panel = false;
      nk_end(ctx);
    }
    if (vis.dt_panel) {
      if (nk_begin(ctx, "Distance Transform", nk_rect(20, MB + 10, 240, 280),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE |
                   NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
        dt_panel_render(dtp, ctx, NULL);                 // CONTENT
      }
      if (nk_window_is_closed(ctx, "Distance Transform")) vis.dt_panel = false;
      nk_end(ctx);
    }
    if (vis.point_panel) {
      if (nk_begin(ctx, "Point Collections", nk_rect(20, MB + 10, 240, 320),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE |
                   NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
        point_panel_render(pts, ctx, NULL);              // CONTENT
      }
      if (nk_window_is_closed(ctx, "Point Collections")) vis.point_panel = false;
      nk_end(ctx);
    }
    if (vis.vol_info) {
      if (nk_begin(ctx, "Volume Info", nk_rect(20, MB + 10, 260, 300),
                   NK_WINDOW_BORDER | NK_WINDOW_TITLE |
                   NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
        vol_info_panel_render(vip, ctx, vol, NULL);      // CONTENT
      }
      if (nk_window_is_closed(ctx, "Volume Info")) vis.vol_info = false;
      nk_end(ctx);
    }

    // ================================================================
    // B) WINDOW widgets — own their nk_begin/nk_end, call standalone
    // ================================================================

    surface_panel_render(surfp, ctx, "Surfaces");        // WINDOW

    if (settings_dialog_is_visible(settings_dlg))
      settings_dialog_render(settings_dlg, ctx);        // WINDOW

    if (about_dialog_is_visible(about_dlg))
      about_dialog_render(about_dlg, ctx);              // WINDOW

    if (keybinds_dialog_is_visible(keybinds_dlg))
      keybinds_dialog_render(keybinds_dlg, ctx);        // WINDOW

    if (file_dialog_is_visible(fd_zarr)) {              // WINDOW
      if (file_dialog_render(fd_zarr, ctx)) {
        const char *p = file_dialog_get_path(fd_zarr);
        vol = try_open(p, cred);
        if (vol) {
          menubar_add_recent(mbar, p);
          welcome_panel_add_recent(welcome, p, p);
          vol_selector_add(volsel, p, p);
        }
      }
    }
    if (file_dialog_is_visible(fd_volpkg)) {            // WINDOW
      if (file_dialog_render(fd_volpkg, ctx)) {
        const char *p = file_dialog_get_path(fd_volpkg);
        vol = try_open(p, cred);
        if (vol) {
          menubar_add_recent(mbar, p);
          welcome_panel_add_recent(welcome, p, p);
          vol_selector_add(volsel, p, p);
        }
      }
    }
    if (s3_browser_is_visible(s3)) {                    // WINDOW
      if (s3_browser_render(s3, ctx)) {
        const char *url = s3_browser_get_url(s3);
        vol = try_open(url, cred);
        if (vol) {
          s3_browser_add_recent(s3, url);
          menubar_add_recent(mbar, url);
          welcome_panel_add_recent(welcome, url, url);
          vol_selector_add(volsel, url, url);
        }
      }
    }
    if (cred_dialog_is_visible(cred)) {                 // WINDOW
      if (cred_dialog_render(cred, ctx)) {
        s3_credentials *creds = cred_dialog_get_creds(cred);
        s3_browser_set_creds(s3, creds);
        s3_browser_show(s3);
      }
    }

    app_end_frame(app);
  }

  log_set_callback(NULL, NULL);
  vol_free(vol);
  menubar_free(mbar);
  welcome_panel_free(welcome);
  surface_panel_free(surfp);
  seg_panels_free(seg);
  vol_selector_free(volsel);
  statusbar_free(sbar);
  viewer_controls_free(vc);
  draw_panel_free(drawp);
  dt_panel_free(dtp);
  point_panel_free(pts);
  vol_info_panel_free(vip);
  settings_dialog_free(settings_dlg);
  about_dialog_free(about_dlg);
  keybinds_dialog_free(keybinds_dlg);
  keybind_free(keybinds);
  file_dialog_free(fd_zarr);
  file_dialog_free(fd_volpkg);
  s3_browser_free(s3);
  cred_dialog_free(cred);
  settings_close(prefs);
  log_console_free(g_console);
  app_shutdown(app);
  return 0;
}
