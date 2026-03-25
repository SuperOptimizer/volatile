// NOTE: NK_IMPLEMENTATION is owned by app.c — include nuklear declaration-only here.
#include "nuklear.h"
#include "gui/about_dialog.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct about_dialog {
  bool visible;
};

static const char *k_credits =
  "Volatile is an open-source volumetric segmentation tool.\n"
  "\n"
  "Libraries used:\n"
  "  Nuklear   — immediate-mode GUI (MIT)\n"
  "  SDL3      — window/input (zlib)\n"
  "  SQLite3   — embedded database (public domain)\n"
  "  blosc2    — compression (BSD-3)\n"
  "\n"
  "Copyright (c) 2024 the Volatile contributors.\n"
  "Released under the MIT License.";

static const char *k_repo = "https://github.com/volatile-project/volatile";

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

about_dialog *about_dialog_new(void) {
  about_dialog *d = calloc(1, sizeof(*d));
  REQUIRE(d, "about_dialog_new: calloc failed");
  return d;
}

void about_dialog_free(about_dialog *d) {
  free(d);
}

void about_dialog_show(about_dialog *d) {
  REQUIRE(d, "about_dialog_show: null");
  d->visible = true;
}

bool about_dialog_is_visible(const about_dialog *d) {
  return d && d->visible;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

bool about_dialog_render(about_dialog *d, struct nk_context *ctx) {
  if (!d || !ctx || !d->visible) return false;

  if (nk_begin(ctx, "About Volatile",
               nk_rect(120, 100, 400, 340),
               NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_MOVABLE |
               NK_WINDOW_NO_SCROLLBAR)) {

    // App name + version
    nk_layout_row_dynamic(ctx, 28, 1);
    nk_label(ctx, "Volatile", NK_TEXT_CENTERED);

    nk_layout_row_dynamic(ctx, 18, 1);
    {
      char ver_buf[64];
      snprintf(ver_buf, sizeof(ver_buf), "Version: %s", volatile_version());
      nk_label(ctx, ver_buf, NK_TEXT_CENTERED);
    }

    nk_layout_row_dynamic(ctx, 18, 1);
    nk_label(ctx, "Built: " __DATE__ "  " __TIME__, NK_TEXT_CENTERED);

    // Divider (thin spacer)
    nk_layout_row_dynamic(ctx, 8, 1);
    nk_spacing(ctx, 1);

    // Credits block — multi-line label via one label per line
    nk_layout_row_dynamic(ctx, 14, 1);
    const char *p = k_credits;
    char line[128];
    while (*p) {
      const char *nl = strchr(p, '\n');
      size_t len = nl ? (size_t)(nl - p) : strlen(p);
      if (len >= sizeof(line)) len = sizeof(line) - 1;
      memcpy(line, p, len);
      line[len] = '\0';
      nk_label(ctx, line[0] ? line : " ", NK_TEXT_LEFT);
      if (!nl) break;
      p = nl + 1;
    }

    // Divider
    nk_layout_row_dynamic(ctx, 8, 1);
    nk_spacing(ctx, 1);

    // Repo URL
    nk_layout_row_dynamic(ctx, 16, 1);
    nk_label(ctx, k_repo, NK_TEXT_CENTERED);

    nk_layout_row_dynamic(ctx, 8, 1);
    nk_spacing(ctx, 1);

    // OK button
    nk_layout_row_begin(ctx, NK_STATIC, 26, 3);
    nk_layout_row_push(ctx, 120);
    nk_spacing(ctx, 1);
    nk_layout_row_push(ctx, 160);
    if (nk_button_label(ctx, "OK"))
      d->visible = false;
    nk_layout_row_push(ctx, 120);
    nk_spacing(ctx, 1);
    nk_layout_row_end(ctx);
  }
  nk_end(ctx);

  return d->visible;
}
