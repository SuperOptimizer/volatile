// ---------------------------------------------------------------------------
// Volatile GUI — entry point
// ---------------------------------------------------------------------------
#include "gui/app.h"
#include "core/log.h"

// Nuklear must be included to use nk_* panel functions.
// Suppress pedantic warnings for the third-party header.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
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

int main(void) {
  log_set_level(LOG_DEBUG);

  app_config_t cfg = {
    .title  = "Volatile v0.1.0",
    .width  = 1280,
    .height = 720,
  };

  app_state_t *app = app_init(&cfg);
  if (!app) return 1;

  while (!app_should_close(app)) {
    app_begin_frame(app);

    struct nk_context *ctx = app_nk_ctx(app);

    // NOTE: basic info panel — confirms Nuklear + SDL3 are wired up
    if (nk_begin(ctx, "Volatile v0.1.0",
                 nk_rect(20, 20, 280, 120),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_NO_SCROLLBAR)) {
      nk_layout_row_dynamic(ctx, 30, 1);
      nk_label(ctx, "Volatile v0.1.0", NK_TEXT_CENTERED);
      nk_layout_row_dynamic(ctx, 30, 1);
      nk_label(ctx, "SDL3 + Nuklear running", NK_TEXT_CENTERED);
    }
    nk_end(ctx);

    app_end_frame(app);
  }

  app_shutdown(app);
  return 0;
}
