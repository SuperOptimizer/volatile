#pragma once
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// file_dialog — lightweight POSIX directory browser using Nuklear
//
// Usage:
//   file_dialog *d = file_dialog_new("Open Volume", "*.zarr;*.volpkg;*.tiff");
//   file_dialog_show(d, "/data");
//   // inside render loop:
//   if (file_dialog_render(d, ctx)) {
//     const char *path = file_dialog_get_path(d);
//     // ... open path ...
//   }
//   file_dialog_free(d);
// ---------------------------------------------------------------------------

typedef struct file_dialog file_dialog;

file_dialog *file_dialog_new(const char *title, const char *filter);
void         file_dialog_free(file_dialog *d);

// Show the dialog starting at start_dir (NULL = cwd).
void file_dialog_show(file_dialog *d, const char *start_dir);

// Render; returns true exactly once when the user commits a selection.
bool file_dialog_render(file_dialog *d, struct nk_context *ctx);

// Path of the last committed selection (valid until next show/free).
const char *file_dialog_get_path(const file_dialog *d);

bool file_dialog_is_visible(const file_dialog *d);

// Add a bookmark (label + absolute path).  Max 8 bookmarks.
void file_dialog_add_bookmark(file_dialog *d, const char *label, const char *path);
