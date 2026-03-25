// console.c — ring-buffer log capture + Nuklear display widget
//
// NK_IMPLEMENTATION is owned by app.c. Include nuklear.h declaration-only here.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/console.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <ctype.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Ring-buffer entry
// ---------------------------------------------------------------------------

#define MSG_MAX   256
#define FILE_MAX   48
#define FILTER_MAX 128

typedef struct {
  int   level;
  long  timestamp;    // unix seconds
  char  file[FILE_MAX];
  int   line;
  char  msg[MSG_MAX];
} console_entry_t;

struct log_console {
  console_entry_t *entries;   // ring buffer
  int              cap;       // allocated size
  int              head;      // index of oldest entry (when full)
  int              count;     // number of valid entries (0..cap)

  int              min_level;
  char             filter[FILTER_MAX];
  bool             auto_scroll;
  bool             scroll_to_bottom;  // one-shot flag
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

log_console *log_console_new(int max_lines) {
  if (max_lines <= 0) max_lines = 1000;
  log_console *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->entries = calloc((size_t)max_lines, sizeof(console_entry_t));
  if (!c->entries) { free(c); return NULL; }
  c->cap         = max_lines;
  c->auto_scroll = true;
  return c;
}

void log_console_free(log_console *c) {
  if (!c) return;
  free(c->entries);
  free(c);
}

// ---------------------------------------------------------------------------
// Add entry
// ---------------------------------------------------------------------------

void log_console_add(log_console *c, int level, const char *file, int line, const char *msg) {
  if (!c) return;

  int idx;
  if (c->count < c->cap) {
    idx = c->count++;
  } else {
    // Overwrite oldest
    idx = c->head;
    c->head = (c->head + 1) % c->cap;
  }

  console_entry_t *e = &c->entries[idx];
  e->level     = level;
  e->timestamp = (long)time(NULL);
  e->line      = line;
  snprintf(e->file, FILE_MAX, "%s", file ? file : "");
  snprintf(e->msg,  MSG_MAX,  "%s", msg  ? msg  : "");

  if (c->auto_scroll) c->scroll_to_bottom = true;
}

// ---------------------------------------------------------------------------
// Filters / state
// ---------------------------------------------------------------------------

void log_console_set_min_level(log_console *c, int level) { if (c) c->min_level = level; }

void log_console_set_filter(log_console *c, const char *text) {
  if (!c) return;
  if (!text || text[0] == '\0') { c->filter[0] = '\0'; return; }
  snprintf(c->filter, FILTER_MAX, "%s", text);
}

void log_console_clear(log_console *c) {
  if (!c) return;
  c->count = 0;
  c->head  = 0;
}

int log_console_count(const log_console *c) { return c ? c->count : 0; }

bool log_console_auto_scroll(const log_console *c) { return c && c->auto_scroll; }
void log_console_set_auto_scroll(log_console *c, bool enable) { if (c) c->auto_scroll = enable; }

// ---------------------------------------------------------------------------
// Nuklear rendering helpers
// ---------------------------------------------------------------------------

// Per-level foreground color
static struct nk_color level_nk_color(int level) {
  switch (level) {
    case 0:  return nk_rgb(150, 150, 150);  // DEBUG  — gray
    case 1:  return nk_rgb(255, 255, 255);  // INFO   — white
    case 2:  return nk_rgb(255, 220,  50);  // WARN   — yellow
    case 3:  return nk_rgb(255,  80,  80);  // ERROR  — red
    case 4:  return nk_rgb(220,  80, 220);  // FATAL  — magenta
    default: return nk_rgb(200, 200, 200);
  }
}

static const char *level_tag(int level) {
  switch (level) {
    case 0: return "DBG";
    case 1: return "INF";
    case 2: return "WRN";
    case 3: return "ERR";
    case 4: return "FTL";
    default: return "???";
  }
}

// Case-insensitive substring search
static bool str_contains_icase(const char *haystack, const char *needle) {
  if (!needle || needle[0] == '\0') return true;
  size_t nl = strlen(needle);
  size_t hl = strlen(haystack);
  if (nl > hl) return false;
  for (size_t i = 0; i <= hl - nl; i++) {
    bool match = true;
    for (size_t j = 0; j < nl; j++) {
      if (tolower((unsigned char)haystack[i + j]) != tolower((unsigned char)needle[j])) {
        match = false; break;
      }
    }
    if (match) return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void log_console_render(log_console *c, struct nk_context *ctx, const char *title) {
  if (!c || !ctx) return;
  if (!title) title = "Console";

  // Toolbar row: filter box + Clear button + Auto-scroll toggle
  nk_layout_row_begin(ctx, NK_DYNAMIC, 22, 3);
  nk_layout_row_push(ctx, 0.60f);
  nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, c->filter, FILTER_MAX, nk_filter_default);
  nk_layout_row_push(ctx, 0.20f);
  if (nk_button_label(ctx, "Clear")) log_console_clear(c);
  nk_layout_row_push(ctx, 0.20f);
  {
    bool as = c->auto_scroll;
    if (nk_checkbox_label(ctx, "Auto", &(nk_bool){as})) {
      c->auto_scroll = !as;
    }
  }
  nk_layout_row_end(ctx);

  // Scrollable group for log lines
  nk_flags group_flags = NK_WINDOW_BORDER;
  if (nk_group_begin(ctx, "log_lines", group_flags)) {
    for (int i = 0; i < c->count; i++) {
      // Iterate in insertion order: (head + i) % cap
      int idx = (c->head + i) % c->cap;
      const console_entry_t *e = &c->entries[idx];

      if (e->level < c->min_level) continue;
      if (c->filter[0] && !str_contains_icase(e->msg, c->filter)) continue;

      // Format: "[WRN] file:line  message"
      char line_buf[MSG_MAX + FILE_MAX + 32];
      snprintf(line_buf, sizeof(line_buf), "[%s] %s:%d  %s",
               level_tag(e->level), e->file, e->line, e->msg);

      nk_layout_row_dynamic(ctx, 16, 1);
      nk_label_colored(ctx, line_buf, NK_TEXT_LEFT, level_nk_color(e->level));
    }

    // Scroll to bottom when auto_scroll fires
    if (c->scroll_to_bottom) {
      nk_group_set_scroll(ctx, "log_lines", 0, (nk_uint)0xFFFFFFFFu);
      c->scroll_to_bottom = false;
    }

    nk_group_end(ctx);
  }
}
