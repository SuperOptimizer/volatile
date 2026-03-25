// ---------------------------------------------------------------------------
// welcome_panel.c — home screen rendered in the viewer area when no volume
// is loaded.  Uses Nuklear styled buttons (no SDL or font-load dependency).
// NK_IMPLEMENTATION is owned by app.c — declaration-only here.
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

#include "gui/welcome_panel.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define RECENT_MAX   8
#define BTN_H        64    // big button height
#define BTN_GAP      12    // gap between buttons
#define URL_BUF_MAX  2048

// ---------------------------------------------------------------------------
// Recent entry
// ---------------------------------------------------------------------------

typedef struct {
  char path[URL_BUF_MAX];
  char name[128];
} recent_entry;

// ---------------------------------------------------------------------------
// welcome_panel struct
// ---------------------------------------------------------------------------

struct welcome_panel {
  recent_entry recent[RECENT_MAX];
  int          recent_count;
  char         url_buf[URL_BUF_MAX];
  int          url_len;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

welcome_panel *welcome_panel_new(void) {
  welcome_panel *w = calloc(1, sizeof(*w));
  if (!w) { LOG_ERROR("welcome_panel_new: calloc failed"); return NULL; }
  return w;
}

void welcome_panel_free(welcome_panel *w) { free(w); }

void welcome_panel_add_recent(welcome_panel *w, const char *path,
                               const char *name) {
  if (!w || !path || !path[0]) return;
  // Dedup: move to front
  for (int i = 0; i < w->recent_count; i++) {
    if (!strncmp(w->recent[i].path, path, URL_BUF_MAX - 1)) {
      recent_entry tmp = w->recent[i];
      memmove(&w->recent[1], &w->recent[0],
              (size_t)i * sizeof(recent_entry));
      w->recent[0] = tmp;
      return;
    }
  }
  int n = w->recent_count < RECENT_MAX ? w->recent_count : RECENT_MAX - 1;
  memmove(&w->recent[1], &w->recent[0], (size_t)n * sizeof(recent_entry));
  snprintf(w->recent[0].path, URL_BUF_MAX, "%s", path);
  snprintf(w->recent[0].name, sizeof(w->recent[0].name), "%s",
           name && name[0] ? name : path);
  if (w->recent_count < RECENT_MAX) w->recent_count++;
}

// ---------------------------------------------------------------------------
// Style helpers
// ---------------------------------------------------------------------------

// Push a tinted button color for visual variety.
static void push_btn_color(struct nk_context *ctx, struct nk_color base) {
  struct nk_color hover  = { (nk_byte)NK_MIN(base.r + 20, 255),
                              (nk_byte)NK_MIN(base.g + 20, 255),
                              (nk_byte)NK_MIN(base.b + 20, 255), 255 };
  struct nk_color active = { (nk_byte)NK_MAX(base.r - 20, 0),
                              (nk_byte)NK_MAX(base.g - 20, 0),
                              (nk_byte)NK_MAX(base.b - 20, 0), 255 };
  nk_style_push_color(ctx, &ctx->style.button.normal.data.color,  base);
  nk_style_push_color(ctx, &ctx->style.button.hover.data.color,   hover);
  nk_style_push_color(ctx, &ctx->style.button.active.data.color,  active);
}

static void pop_btn_color(struct nk_context *ctx) {
  nk_style_pop_color(ctx);
  nk_style_pop_color(ctx);
  nk_style_pop_color(ctx);
}

// Big two-line button: first line is the title, second is a subtitle.
// Returns true when clicked.
static bool big_button(struct nk_context *ctx, const char *title,
                       const char *subtitle, struct nk_color color) {
  push_btn_color(ctx, color);
  // Compose a two-line label using a newline character
  char label[128];
  snprintf(label, sizeof(label), "%s\n%s", title, subtitle);
  bool clicked = nk_button_label(ctx, label) != 0;
  pop_btn_color(ctx);
  return clicked;
}

// ---------------------------------------------------------------------------
// welcome_panel_render
// ---------------------------------------------------------------------------

welcome_result welcome_panel_render(welcome_panel *w, struct nk_context *ctx,
                                    int panel_w, int panel_h) {
  welcome_result result = {WELCOME_NONE, {0}};
  if (!w || !ctx) return result;

  // Determine button width: two columns with a gap
  const float margin  = 24.0f;
  const float gap     = (float)BTN_GAP;
  const float btn_w   = ((float)panel_w - margin * 2.0f - gap) * 0.5f;
  const float btn_h   = (float)BTN_H;

  // Title
  nk_layout_row_dynamic(ctx, 12.0f, 1);  // top spacer
  nk_spacing(ctx, 1);
  nk_layout_row_dynamic(ctx, 32.0f, 1);
  nk_label(ctx, "Welcome to Volatile", NK_TEXT_CENTERED);
  nk_layout_row_dynamic(ctx, 8.0f, 1);
  nk_spacing(ctx, 1);

  // Helper macro: two-column row of big buttons
  // Each call to nk_layout_row_begin/end with two fixed-width cells.
#define TWO_BTN_ROW(h) \
  nk_layout_row_begin(ctx, NK_STATIC, (h), 4); \
  nk_layout_row_push(ctx, margin); nk_spacing(ctx, 1); \
  nk_layout_row_push(ctx, btn_w);
#define MID_BTN \
  nk_layout_row_push(ctx, gap); nk_spacing(ctx, 1); \
  nk_layout_row_push(ctx, btn_w);
#define END_BTN_ROW \
  nk_layout_row_end(ctx);

  // Row 1: Open Zarr + Open volpkg
  TWO_BTN_ROW(btn_h);
  if (big_button(ctx, "Open Local Zarr",
                 "Browse *.zarr directories",
                 nk_rgb(60, 90, 140)))
    result.action = WELCOME_OPEN_ZARR;
  MID_BTN;
  if (big_button(ctx, "Open volpkg",
                 "Open volume package",
                 nk_rgb(60, 110, 80)))
    result.action = WELCOME_OPEN_VOLPKG;
  END_BTN_ROW;

  nk_layout_row_dynamic(ctx, gap, 1); nk_spacing(ctx, 1);

  // Row 2: Open S3 + Open HTTP URL
  TWO_BTN_ROW(btn_h);
  if (big_button(ctx, "Open Remote S3",
                 "Browse S3 buckets",
                 nk_rgb(130, 80, 50)))
    result.action = WELCOME_OPEN_S3;
  MID_BTN;
  if (big_button(ctx, "Open HTTP URL",
                 "Paste zarr/HTTP URL",
                 nk_rgb(100, 70, 130)))
    result.action = WELCOME_OPEN_URL;
  END_BTN_ROW;

  nk_layout_row_dynamic(ctx, gap, 1); nk_spacing(ctx, 1);

  // Row 3: Open Project + Recent Files
  TWO_BTN_ROW(btn_h);
  if (big_button(ctx, "Open Project",
                 "Load .json project file",
                 nk_rgb(80, 100, 100)))
    result.action = WELCOME_OPEN_PROJECT;
  MID_BTN;
  if (big_button(ctx, "Recent Files",
                 w->recent_count > 0 ? "Click to expand" : "(none)",
                 nk_rgb(90, 90, 60)))
    ; // handled by recent list below — no action from the header button itself
  END_BTN_ROW;

#undef TWO_BTN_ROW
#undef MID_BTN
#undef END_BTN_ROW

  // Recent files list (compact, one per row)
  if (w->recent_count > 0) {
    nk_layout_row_dynamic(ctx, 6.0f, 1); nk_spacing(ctx, 1);
    for (int i = 0; i < w->recent_count; i++) {
      nk_layout_row_begin(ctx, NK_STATIC, 22.0f, 3);
      nk_layout_row_push(ctx, margin);
      nk_spacing(ctx, 1);
      nk_layout_row_push(ctx, (float)panel_w - margin * 2.0f - 60.0f);
      char label[160];
      snprintf(label, sizeof(label), "%.150s", w->recent[i].name);
      nk_label(ctx, label, NK_TEXT_LEFT);
      nk_layout_row_push(ctx, 52.0f);
      if (nk_button_label(ctx, "Open")) {
        result.action = WELCOME_OPEN_RECENT;
        snprintf(result.url, sizeof(result.url), "%s", w->recent[i].path);
      }
      nk_layout_row_end(ctx);
    }
  }

  // Separator before URL bar
  int used_h = (int)(32 + 8 + 12 + 3 * (BTN_H + BTN_GAP) + 40
               + w->recent_count * 22 + 16);
  int spacer_h = panel_h - used_h - 60;
  if (spacer_h > 8) {
    nk_layout_row_dynamic(ctx, (float)spacer_h, 1);
    nk_spacing(ctx, 1);
  }

  nk_layout_row_dynamic(ctx, 4.0f, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);
  nk_layout_row_dynamic(ctx, 8.0f, 1); nk_spacing(ctx, 1);

  // URL entry row
  nk_layout_row_begin(ctx, NK_STATIC, 26.0f, 4);
  nk_layout_row_push(ctx, margin);
  nk_spacing(ctx, 1);
  nk_layout_row_push(ctx, (float)panel_w - margin * 2.0f - 60.0f);
  nk_edit_string(ctx, NK_EDIT_SIMPLE, w->url_buf, &w->url_len,
                 URL_BUF_MAX - 1, nk_filter_default);
  nk_layout_row_push(ctx, 52.0f);
  if (nk_button_label(ctx, "Go") && w->url_len > 0) {
    w->url_buf[w->url_len] = '\0';
    result.action = WELCOME_OPEN_URL;
    snprintf(result.url, sizeof(result.url), "%s", w->url_buf);
  }
  nk_layout_row_end(ctx);

  nk_layout_row_dynamic(ctx, 8.0f, 1); nk_spacing(ctx, 1);

  return result;
}
