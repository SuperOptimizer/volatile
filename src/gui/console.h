// WIDGET TYPE: CONTENT — call inside an nk_begin/nk_end block.
#pragma once
#include <stdbool.h>

// Forward-declare nk_context so consumers need not include nuklear.h directly.
struct nk_context;

// ---------------------------------------------------------------------------
// log_console — ring-buffer log capture + Nuklear display widget
// ---------------------------------------------------------------------------

typedef struct log_console log_console;

// Allocate a console with a ring buffer of max_lines entries (default 1000).
// Returns NULL on allocation failure.
log_console *log_console_new(int max_lines);

// Free all resources. Caller is responsible for unregistering the log callback
// (via log_set_callback(NULL, NULL)) before calling this if still wired.
void log_console_free(log_console *c);

// Add an entry. Intended to be called from the log_set_callback hook.
void log_console_add(log_console *c, int level, const char *file, int line, const char *msg);

// Render the console as a Nuklear panel occupying whatever layout space the
// caller has set up. Call between nk_begin/nk_end or inside a group.
void log_console_render(log_console *c, struct nk_context *ctx, const char *title);

// ---------------------------------------------------------------------------
// Filtering
// ---------------------------------------------------------------------------

// Suppress entries below this level (default LOG_DEBUG = 0, i.e. show all).
void log_console_set_min_level(log_console *c, int level);

// Substring filter on the message text (case-insensitive). NULL or "" = no filter.
void log_console_set_filter(log_console *c, const char *text);

// ---------------------------------------------------------------------------
// State / actions
// ---------------------------------------------------------------------------

void log_console_clear(log_console *c);
int  log_console_count(const log_console *c);  // total entries currently stored

bool log_console_auto_scroll(const log_console *c);
void log_console_set_auto_scroll(log_console *c, bool enable);
