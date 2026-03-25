#define _POSIX_C_SOURCE 200809L

#include "gui/file_dialog.h"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include <dirent.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_ENTRIES    512
#define MAX_BOOKMARKS    8
#define PATH_CAP      2048
#define FILTER_CAP      64
#define RECENT_MAX       8

// ---------------------------------------------------------------------------
// Entry (file/directory in the listing)
// ---------------------------------------------------------------------------

typedef struct {
  char name[256];
  bool is_dir;
} dir_entry;

// ---------------------------------------------------------------------------
// file_dialog
// ---------------------------------------------------------------------------

typedef struct {
  char label[64];
  char path[PATH_CAP];
} bookmark;

struct file_dialog {
  char    title[128];
  char    filter[FILTER_CAP];   // ";"-separated globs: "*.zarr;*.tiff"

  bool    visible;
  char    cwd[PATH_CAP];         // currently browsed directory
  char    selected[PATH_CAP];    // highlighted entry (may be dir or file)
  char    result[PATH_CAP];      // committed path (returned by get_path)
  bool    committed;             // true for one frame after selection

  char    path_buf[PATH_CAP];    // editable path text box
  int     path_buf_len;

  // Directory listing
  dir_entry entries[MAX_ENTRIES];
  int       n_entries;

  // Bookmarks
  bookmark  bookmarks[MAX_BOOKMARKS];
  int       n_bookmarks;

  // Recent files
  char      recent[RECENT_MAX][PATH_CAP];
  int       n_recent;
};

// ---------------------------------------------------------------------------
// Filter matching
// ---------------------------------------------------------------------------

// Returns true if `name` matches any of the ";"-separated glob patterns.
// Only supports prefix "*.ext" style matching.
static bool matches_filter(const char *filter, const char *name) {
  if (!filter || filter[0] == '\0') return true;

  char buf[FILTER_CAP];
  snprintf(buf, sizeof(buf), "%s", filter);

  char *tok = strtok(buf, ";");
  while (tok) {
    // strip leading "*"
    const char *pat = tok;
    if (pat[0] == '*') pat++;
    if (strstr(name, pat) != NULL) return true;
    tok = strtok(NULL, ";");
  }
  return false;
}

// ---------------------------------------------------------------------------
// Directory scanning
// ---------------------------------------------------------------------------

// Comparison for qsort: directories first, then alpha.
static int entry_cmp(const void *a, const void *b) {
  const dir_entry *ea = a, *eb = b;
  if (ea->is_dir != eb->is_dir) return ea->is_dir ? -1 : 1;
  return strcmp(ea->name, eb->name);
}

static void scan_dir(file_dialog *d) {
  d->n_entries = 0;
  DIR *dir = opendir(d->cwd);
  if (!dir) return;

  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL && d->n_entries < MAX_ENTRIES) {
    if (strcmp(ent->d_name, ".") == 0) continue;

    char full[PATH_CAP];
    snprintf(full, sizeof(full), "%s/%s", d->cwd, ent->d_name);

    struct stat st;
    if (stat(full, &st) != 0) continue;

    bool is_dir = S_ISDIR(st.st_mode);

    // Apply filter to files (directories are always shown)
    if (!is_dir && !matches_filter(d->filter, ent->d_name)) continue;

    dir_entry *e = &d->entries[d->n_entries++];
    snprintf(e->name, sizeof(e->name), "%s", ent->d_name);
    e->is_dir = is_dir;
  }
  closedir(dir);
  qsort(d->entries, (size_t)d->n_entries, sizeof(dir_entry), entry_cmp);
}

static void navigate_to(file_dialog *d, const char *path) {
  snprintf(d->cwd, sizeof(d->cwd), "%s", path);
  snprintf(d->path_buf, sizeof(d->path_buf), "%s", path);
  d->path_buf_len = (int)strlen(d->path_buf);
  d->selected[0]  = '\0';
  scan_dir(d);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

file_dialog *file_dialog_new(const char *title, const char *filter) {
  file_dialog *d = calloc(1, sizeof(*d));
  if (!d) return NULL;
  snprintf(d->title,  sizeof(d->title),  "%s", title  ? title  : "Open");
  snprintf(d->filter, sizeof(d->filter), "%s", filter ? filter : "");

  // Built-in bookmarks
  const char *home = getenv("HOME");
  if (home) file_dialog_add_bookmark(d, "Home", home);

  char data[PATH_CAP];
  snprintf(data, sizeof(data), "%s/data", home ? home : "/tmp");
  file_dialog_add_bookmark(d, "Data", data);

  return d;
}

void file_dialog_free(file_dialog *d) {
  free(d);
}

void file_dialog_show(file_dialog *d, const char *start_dir) {
  if (!d) return;
  const char *dir = start_dir;
  if (!dir || dir[0] == '\0') {
    char cwd[PATH_CAP];
    dir = getcwd(cwd, sizeof(cwd)) ? cwd : "/";
  }
  navigate_to(d, dir);
  d->committed = false;
  d->result[0] = '\0';
  d->visible   = true;
}

// ---------------------------------------------------------------------------
// Nuklear render
// ---------------------------------------------------------------------------

bool file_dialog_render(file_dialog *d, struct nk_context *ctx) {
  if (!d || !ctx || !d->visible) return false;
  d->committed = false;

  if (!nk_begin(ctx, d->title,
                nk_rect(80, 60, 600, 480),
                NK_WINDOW_BORDER | NK_WINDOW_MOVABLE |
                NK_WINDOW_SCALABLE | NK_WINDOW_TITLE | NK_WINDOW_CLOSABLE)) {
    d->visible = false;
    nk_end(ctx);
    return false;
  }

  // ---- Bookmarks column ----
  nk_layout_row_begin(ctx, NK_STATIC, 360, 2);
  nk_layout_row_push(ctx, 120);
  if (nk_group_begin(ctx, "Bookmarks", NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
    nk_layout_row_dynamic(ctx, 22, 1);
    for (int i = 0; i < d->n_bookmarks; i++) {
      if (nk_button_label(ctx, d->bookmarks[i].label))
        navigate_to(d, d->bookmarks[i].path);
    }
    // recent files section
    if (d->n_recent > 0) {
      nk_label(ctx, "Recent", NK_TEXT_LEFT);
      for (int i = 0; i < d->n_recent; i++) {
        // show just the filename as label
        const char *slash = strrchr(d->recent[i], '/');
        const char *base  = slash ? slash + 1 : d->recent[i];
        if (nk_button_label(ctx, base)) {
          snprintf(d->result, sizeof(d->result), "%s", d->recent[i]);
          d->committed = true;
          d->visible   = false;
        }
      }
    }
    nk_group_end(ctx);
  }

  // ---- Directory listing ----
  nk_layout_row_push(ctx, 460);
  if (nk_group_begin(ctx, "Files", NK_WINDOW_BORDER | NK_WINDOW_TITLE)) {
    nk_layout_row_dynamic(ctx, 22, 1);
    for (int i = 0; i < d->n_entries; i++) {
      dir_entry *e = &d->entries[i];
      char lbl[280];
      snprintf(lbl, sizeof(lbl), "%s%s", e->is_dir ? "[D] " : "    ", e->name);

      bool is_sel = (strcmp(e->name, d->selected) == 0);
      if (nk_select_label(ctx, lbl, NK_TEXT_LEFT, is_sel)) {
        snprintf(d->selected, sizeof(d->selected), "%s", e->name);
        // update path box
        snprintf(d->path_buf, sizeof(d->path_buf), "%s/%s", d->cwd, e->name);
        d->path_buf_len = (int)strlen(d->path_buf);

        if (e->is_dir) {
          // single-click navigates into directory
          char next[PATH_CAP];
          if (strcmp(e->name, "..") == 0) {
            // go up: strip last component
            snprintf(next, sizeof(next), "%s", d->cwd);
            char *sl = strrchr(next, '/');
            if (sl && sl != next) *sl = '\0';
            else                  snprintf(next, sizeof(next), "/");
          } else {
            snprintf(next, sizeof(next), "%s/%s", d->cwd, e->name);
          }
          navigate_to(d, next);
        }
      }
    }
    nk_group_end(ctx);
  }
  nk_layout_row_end(ctx);

  // ---- Path text entry ----
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_edit_string(ctx, NK_EDIT_FIELD, d->path_buf, &d->path_buf_len,
                 PATH_CAP - 1, nk_filter_default);

  // ---- OK / Cancel ----
  nk_layout_row_dynamic(ctx, 28, 2);
  if (nk_button_label(ctx, "Open")) {
    d->path_buf[d->path_buf_len] = '\0';
    snprintf(d->result, sizeof(d->result), "%s", d->path_buf);
    d->committed = true;
    d->visible   = false;
    // push to recent
    if (d->n_recent < RECENT_MAX) {
      snprintf(d->recent[d->n_recent++], PATH_CAP, "%s", d->result);
    }
  }
  if (nk_button_label(ctx, "Cancel")) {
    d->visible = false;
  }

  nk_end(ctx);
  return d->committed;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const char *file_dialog_get_path(const file_dialog *d) {
  return d ? d->result : NULL;
}

bool file_dialog_is_visible(const file_dialog *d) {
  return d && d->visible;
}

void file_dialog_add_bookmark(file_dialog *d, const char *label, const char *path) {
  if (!d || !label || !path || d->n_bookmarks >= MAX_BOOKMARKS) return;
  snprintf(d->bookmarks[d->n_bookmarks].label, 64, "%s", label);
  snprintf(d->bookmarks[d->n_bookmarks].path, PATH_CAP, "%s", path);
  d->n_bookmarks++;
}
