// Minimal volatile GUI - just menubar + welcome panel
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
#include "core/log.h"
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  log_set_level(LOG_INFO);
  LOG_INFO("Volatile v%s starting", volatile_version());

  app_config_t cfg = {.title = "Volatile", .width = 1600, .height = 900};
  app_state_t *app = app_init(&cfg);
  if (!app) return 1;

  menubar *mbar = menubar_new();

  while (!app_should_close(app)) {
    app_begin_frame(app);
    int w, h;
    app_get_size(app, &w, &h);
    struct nk_context *ctx = app_nk_ctx(app);

    // Menubar
    if (nk_begin(ctx, "##menubar", nk_rect(0, 0, (float)w, 30), NK_WINDOW_NO_SCROLLBAR)) {
      menubar_render(mbar, ctx);
    }
    nk_end(ctx);

    // Main area - simple welcome message
    if (nk_begin(ctx, "Welcome", nk_rect(0, 32, (float)w, (float)(h - 56)),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
      nk_layout_row_dynamic(ctx, 40, 1);
      nk_label(ctx, "Welcome to Volatile", NK_TEXT_CENTERED);
      nk_layout_row_dynamic(ctx, 30, 1);
      nk_label(ctx, "Use File menu to open a volume", NK_TEXT_CENTERED);
      nk_layout_row_dynamic(ctx, 60, 3);
      if (nk_button_label(ctx, "Open Local Zarr")) LOG_INFO("Open Zarr clicked");
      if (nk_button_label(ctx, "Open volpkg")) LOG_INFO("Open volpkg clicked");
      if (nk_button_label(ctx, "Browse S3")) LOG_INFO("Browse S3 clicked");
    }
    nk_end(ctx);

    // Statusbar
    if (nk_begin(ctx, "##status", nk_rect(0, (float)(h - 24), (float)w, 24), NK_WINDOW_NO_SCROLLBAR)) {
      nk_layout_row_dynamic(ctx, 18, 1);
      nk_label(ctx, "Ready | No volume loaded", NK_TEXT_LEFT);
    }
    nk_end(ctx);

    app_end_frame(app);
  }

  menubar_free(mbar);
  app_shutdown(app);
  return 0;
}
