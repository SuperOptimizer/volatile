// Volatile GUI - absolutely minimal
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
#include "core/log.h"
#include <stdlib.h>

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  log_set_level(LOG_INFO);
  LOG_INFO("Volatile v%s starting", volatile_version());

  app_config_t cfg = {.title = "Volatile", .width = 1280, .height = 720};
  app_state_t *app = app_init(&cfg);
  if (!app) return 1;

  while (!app_should_close(app)) {
    app_begin_frame(app);
    int w, h;
    app_get_size(app, &w, &h);
    struct nk_context *ctx = app_nk_ctx(app);

    if (nk_begin(ctx, "Volatile", nk_rect(0, 0, (float)w, (float)h), NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
      nk_layout_row_dynamic(ctx, 40, 1);
      nk_label(ctx, "Welcome to Volatile", NK_TEXT_CENTERED);
      nk_layout_row_dynamic(ctx, 40, 3);
      if (nk_button_label(ctx, "Open Zarr")) LOG_INFO("Zarr");
      if (nk_button_label(ctx, "Open volpkg")) LOG_INFO("volpkg");
      if (nk_button_label(ctx, "Browse S3")) LOG_INFO("S3");
    }
    nk_end(ctx);

    app_end_frame(app);
  }

  app_shutdown(app);
  return 0;
}
