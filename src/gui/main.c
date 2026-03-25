// volatile GUI — all panels, correct nk_begin/nk_end
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
#include "gui/app.h"
#include "gui/menubar.h"
#include "gui/console.h"
#include "gui/seg_panels.h"
#include "gui/vol_selector.h"
#include "gui/window_range.h"
#include "gui/statusbar.h"
#include "gui/file_dialog.h"
#include "gui/s3_browser.h"
#include "gui/cred_dialog.h"
#include "core/log.h"
#include "core/vol.h"
#include <stdlib.h>
#include <string.h>

static log_console *g_con;
static void log_hook(void *u, log_level_t lv, const char *f, int ln, const char *m) {
  (void)u; if (g_con) log_console_add(g_con, (int)lv, f, ln, m);
}
static volume *try_open(const char *path, cred_dialog *cd) {
  if (!path || !path[0]) return NULL;
  LOG_INFO("Opening: %s", path);
  volume *v = vol_open(path);
  if (v) LOG_INFO("Opened (%d levels)", vol_num_levels(v));
  else { LOG_WARN("Failed: %s", path); if (strncmp(path,"s3://",5)==0) cred_dialog_show(cd,path); }
  return v;
}

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  log_set_level(LOG_INFO);
  g_con = log_console_new(500);
  log_set_callback(log_hook, NULL);
  LOG_INFO("Volatile v%s starting", volatile_version());
  app_config_t cfg = {.title = "Volatile", .width = 1600, .height = 900};
  app_state_t *app = app_init(&cfg);
  if (!app) { log_set_callback(NULL,NULL); log_console_free(g_con); return 1; }
  menubar *mbar = menubar_new();
  seg_panels *seg = seg_panels_new();
  vol_selector *vs = vol_selector_new();
  window_range_state wr; window_range_init(&wr);
  statusbar *sbar = statusbar_new();
  file_dialog *fd_z = file_dialog_new("Open Zarr", "*.zarr");
  file_dialog *fd_v = file_dialog_new("Open volpkg", "*.volpkg");
  s3_browser *s3 = s3_browser_new();
  cred_dialog *cd = cred_dialog_new();
  vol_selector_add(vs, "(no volume)", "");
  volume *vol = NULL;
  char url_buf[2048] = {0};
  int show_url = 0;
  LOG_INFO("GUI ready");

  while (!app_should_close(app)) {
    app_begin_frame(app);
    int w, h; app_get_size(app, &w, &h);
    struct nk_context *ctx = app_nk_ctx(app);
    float mb = 30, sb = 24;

    // 1. Menubar
    if (nk_begin(ctx, "##mb", nk_rect(0,0,(float)w,mb), NK_WINDOW_NO_SCROLLBAR))
      menubar_render(mbar, ctx);
    nk_end(ctx);

    // 2. Main area
    if (!vol) {
      if (nk_begin(ctx, "Welcome", nk_rect(0,mb,(float)(w-280),(float)(h-mb-sb)),
                   NK_WINDOW_BORDER|NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(ctx, 40, 1);
        nk_label(ctx, "Welcome to Volatile", NK_TEXT_CENTERED);
        nk_layout_row_dynamic(ctx, 10, 1); nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 50, 3);
        if (nk_button_label(ctx, "Open Local Zarr")) file_dialog_show(fd_z, NULL);
        if (nk_button_label(ctx, "Open volpkg")) file_dialog_show(fd_v, NULL);
        if (nk_button_label(ctx, "Browse S3")) s3_browser_show(s3);
        nk_layout_row_dynamic(ctx, 10, 1); nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 50, 1);
        if (nk_button_label(ctx, "Open URL / Path")) show_url = !show_url;
        if (show_url) {
          nk_layout_row_dynamic(ctx, 22, 1);
          nk_label(ctx, "Enter path or URL:", NK_TEXT_LEFT);
          nk_layout_row(ctx, NK_DYNAMIC, 30, 2, (float[]){0.82f, 0.18f});
          nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, url_buf, sizeof(url_buf), nk_filter_default);
          if (nk_button_label(ctx, "Go") && url_buf[0]) {
            vol = try_open(url_buf, cd);
            if (vol) show_url = 0;
          }
        }
        nk_layout_row_dynamic(ctx, 10, 1); nk_spacing(ctx, 1);
        nk_layout_row_dynamic(ctx, 2, 1);
        nk_rule_horizontal(ctx, ctx->style.window.border_color, nk_false);
        nk_layout_row_dynamic(ctx, 18, 1);
        nk_label(ctx, "Console:", NK_TEXT_LEFT);
        log_console_render(g_con, ctx, NULL);
      }
      nk_end(ctx);
    } else {
      float vw = (float)(w-280)*0.5f, vh = (float)(h-mb-sb)*0.5f;
      if (nk_begin(ctx,"XY (axial)",nk_rect(0,mb,vw,vh),NK_WINDOW_BORDER|NK_WINDOW_TITLE)){
        nk_layout_row_dynamic(ctx,20,1); nk_label(ctx,"XY slice viewer",NK_TEXT_LEFT);
      } nk_end(ctx);
      if (nk_begin(ctx,"XZ (coronal)",nk_rect(vw,mb,vw,vh),NK_WINDOW_BORDER|NK_WINDOW_TITLE)){
        nk_layout_row_dynamic(ctx,20,1); nk_label(ctx,"XZ slice viewer",NK_TEXT_LEFT);
      } nk_end(ctx);
      if (nk_begin(ctx,"YZ (sagittal)",nk_rect(0,mb+vh,vw,vh),NK_WINDOW_BORDER|NK_WINDOW_TITLE)){
        nk_layout_row_dynamic(ctx,20,1); nk_label(ctx,"YZ slice viewer",NK_TEXT_LEFT);
      } nk_end(ctx);
      if (nk_begin(ctx,"3D Viewer",nk_rect(vw,mb+vh,vw,vh),NK_WINDOW_BORDER|NK_WINDOW_TITLE)){
        nk_layout_row_dynamic(ctx,20,1); nk_label(ctx,"3D renderer",NK_TEXT_LEFT);
      } nk_end(ctx);
    }

    // 3. Side panel (CONTENT widgets only - no nk_begin inside)
    if (nk_begin(ctx, "Controls", nk_rect((float)(w-280),mb,280,(float)(h-mb-sb)),
                 NK_WINDOW_BORDER|NK_WINDOW_TITLE)) {
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Volume", NK_TEXT_LEFT);
      vol_selector_render(vs, ctx);
      nk_layout_row_dynamic(ctx, 4, 1); nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Window / Level", NK_TEXT_LEFT);
      window_range_render(&wr, ctx);
      nk_layout_row_dynamic(ctx, 4, 1); nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Segmentation", NK_TEXT_LEFT);
      seg_panels_render(seg, ctx, NULL);
    }
    nk_end(ctx);

    // 4. Statusbar
    if (nk_begin(ctx, "##sb", nk_rect(0,(float)(h-sb),(float)w,sb), NK_WINDOW_NO_SCROLLBAR)) {
      statusbar_update(sbar, 0,0,0,0, 1.0f, 0, 60.0f, 0);
      statusbar_render(sbar, ctx, sb-2);
    }
    nk_end(ctx);

    // 5. WINDOW widgets (standalone - own nk_begin/end)
    if (file_dialog_is_visible(fd_z))
      if (file_dialog_render(fd_z, ctx))
        vol = try_open(file_dialog_get_path(fd_z), cd);
    if (file_dialog_is_visible(fd_v))
      if (file_dialog_render(fd_v, ctx))
        vol = try_open(file_dialog_get_path(fd_v), cd);
    if (s3_browser_is_visible(s3))
      if (s3_browser_render(s3, ctx))
        vol = try_open(s3_browser_get_url(s3), cd);
    if (cred_dialog_is_visible(cd))
      cred_dialog_render(cd, ctx);

    app_end_frame(app);
  }
  log_set_callback(NULL, NULL);
  vol_free(vol); menubar_free(mbar); seg_panels_free(seg);
  vol_selector_free(vs); statusbar_free(sbar);
  file_dialog_free(fd_z); file_dialog_free(fd_v);
  s3_browser_free(s3); cred_dialog_free(cd);
  log_console_free(g_con); app_shutdown(app);
  return 0;
}
