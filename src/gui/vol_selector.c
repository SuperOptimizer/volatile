// vol_selector.c — Nuklear volume combo-box picker
// Port of VC3D/elements/VolumeSelector (Qt) -> plain C + Nuklear.
//
// NK_IMPLEMENTATION is owned by app.c; include nuklear.h declaration-only.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/vol_selector.h"

#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// internals
// ---------------------------------------------------------------------------

#define VS_MAX_ENTRIES 256
#define VS_LABEL_MAX   256
#define VS_PATH_MAX    1024

typedef struct {
  char label[VS_LABEL_MAX];
  char path[VS_PATH_MAX];
} vol_entry_t;

struct vol_selector {
  vol_entry_t entries[VS_MAX_ENTRIES];
  int         count;
  int         selected;  // index into entries[], -1 if empty
};

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

vol_selector *vol_selector_new(void) {
  vol_selector *s = calloc(1, sizeof(vol_selector));
  if (s) s->selected = -1;
  return s;
}

void vol_selector_free(vol_selector *s) {
  free(s);
}

// ---------------------------------------------------------------------------
// mutation
// ---------------------------------------------------------------------------

void vol_selector_add(vol_selector *s, const char *name, const char *path) {
  if (!s || !name || !path) return;
  if (s->count >= VS_MAX_ENTRIES) return;

  vol_entry_t *e = &s->entries[s->count];
  strncpy(e->label, name, VS_LABEL_MAX - 1);
  e->label[VS_LABEL_MAX - 1] = '\0';
  strncpy(e->path, path, VS_PATH_MAX - 1);
  e->path[VS_PATH_MAX - 1] = '\0';

  if (s->selected < 0) s->selected = 0;
  s->count++;
}

void vol_selector_clear(vol_selector *s) {
  if (!s) return;
  s->count = 0;
  s->selected = -1;
}

// ---------------------------------------------------------------------------
// render
// ---------------------------------------------------------------------------

bool vol_selector_render(vol_selector *s, struct nk_context *ctx) {
  if (!s || !ctx) return false;
  if (s->count == 0) {
    nk_layout_row_dynamic(ctx, 22, 1);
    nk_label(ctx, "Volume: (none)", NK_TEXT_LEFT);
    return false;
  }

  // Build a pointer array for nk_combo.
  const char *labels[VS_MAX_ENTRIES];
  for (int i = 0; i < s->count; i++) labels[i] = s->entries[i].label;

  nk_layout_row_begin(ctx, NK_DYNAMIC, 22, 2);
    nk_layout_row_push(ctx, 0.25f);
    nk_label(ctx, "Volume:", NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.75f);
    int prev = s->selected < 0 ? 0 : s->selected;
    struct nk_vec2 combo_size = {300, 200};
    int next = nk_combo(ctx, labels, s->count, prev, 22, combo_size);
  nk_layout_row_end(ctx);

  if (next != prev) {
    s->selected = next;
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

int vol_selector_selected(const vol_selector *s) {
  return s ? s->selected : -1;
}

const char *vol_selector_selected_path(const vol_selector *s) {
  if (!s || s->selected < 0 || s->selected >= s->count) return NULL;
  return s->entries[s->selected].path;
}

int vol_selector_count(const vol_selector *s) {
  return s ? s->count : 0;
}
