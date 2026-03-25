#include "gui/surface_panel.h"
#include "core/log.h"

// NK_IMPLEMENTATION is defined once in app.c; only need declarations here.
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include <nuklear.h>

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
  surface_entry e;          // name and volume_id are owned by this struct
  int           insert_idx; // original insertion order for SORT_BY_DATE
} panel_entry;

struct surface_panel {
  panel_entry *entries;
  int          count;
  int          cap;
  int64_t      selected_id;   // -1 = nothing selected
  int          next_insert;   // monotonic counter for insertion order
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

surface_panel *surface_panel_new(void) {
  surface_panel *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->selected_id = -1;
  return p;
}

static void entry_clear(panel_entry *pe) {
  free(pe->e.name);
  free(pe->e.volume_id);
  pe->e.name      = NULL;
  pe->e.volume_id = NULL;
}

void surface_panel_free(surface_panel *p) {
  if (!p) return;
  for (int i = 0; i < p->count; i++) entry_clear(&p->entries[i]);
  free(p->entries);
  free(p);
}

// ---------------------------------------------------------------------------
// Mutation
// ---------------------------------------------------------------------------

void surface_panel_add(surface_panel *p, const surface_entry *entry) {
  if (!p || !entry) return;

  // Grow backing array if needed.
  if (p->count >= p->cap) {
    int new_cap = p->cap ? p->cap * 2 : 8;
    panel_entry *buf = realloc(p->entries, (size_t)new_cap * sizeof(panel_entry));
    if (!buf) { LOG_WARN("surface_panel_add: out of memory"); return; }
    p->entries = buf;
    p->cap = new_cap;
  }

  panel_entry *pe = &p->entries[p->count];
  pe->e             = *entry;
  pe->e.name        = entry->name      ? strdup(entry->name)      : strdup("");
  pe->e.volume_id   = entry->volume_id ? strdup(entry->volume_id) : NULL;
  pe->insert_idx    = p->next_insert++;

  if (!pe->e.name) {
    LOG_WARN("surface_panel_add: strdup failed");
    free(pe->e.volume_id);
    return;
  }

  p->count++;
}

void surface_panel_remove(surface_panel *p, int64_t id) {
  if (!p) return;
  for (int i = 0; i < p->count; i++) {
    if (p->entries[i].e.id == id) {
      entry_clear(&p->entries[i]);
      // Shift remaining entries down.
      int tail = p->count - i - 1;
      if (tail > 0)
        memmove(&p->entries[i], &p->entries[i + 1],
                (size_t)tail * sizeof(panel_entry));
      p->count--;
      if (p->selected_id == id) p->selected_id = -1;
      return;
    }
  }
}

void surface_panel_clear(surface_panel *p) {
  if (!p) return;
  for (int i = 0; i < p->count; i++) entry_clear(&p->entries[i]);
  p->count       = 0;
  p->selected_id = -1;
}

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

typedef struct { surface_panel *p; surface_sort_t by; bool ascending; } sort_ctx;
static sort_ctx g_sort_ctx;  // qsort has no user-data; use a static context

static int cmp_entries(const void *a, const void *b) {
  const panel_entry *ea = (const panel_entry *)a;
  const panel_entry *eb = (const panel_entry *)b;
  int cmp = 0;
  switch (g_sort_ctx.by) {
    case SORT_BY_ID:
      cmp = (ea->e.id > eb->e.id) - (ea->e.id < eb->e.id);
      break;
    case SORT_BY_NAME:
      cmp = strcmp(ea->e.name ? ea->e.name : "",
                   eb->e.name ? eb->e.name : "");
      break;
    case SORT_BY_AREA:
      cmp = (ea->e.area_vx2 > eb->e.area_vx2) -
            (ea->e.area_vx2 < eb->e.area_vx2);
      break;
    case SORT_BY_DATE:
      cmp = ea->insert_idx - eb->insert_idx;
      break;
  }
  return g_sort_ctx.ascending ? cmp : -cmp;
}

void surface_panel_sort(surface_panel *p, surface_sort_t by, bool ascending) {
  if (!p || p->count < 2) return;
  g_sort_ctx.p         = p;
  g_sort_ctx.by        = by;
  g_sort_ctx.ascending = ascending;
  qsort(p->entries, (size_t)p->count, sizeof(panel_entry), cmp_entries);
}

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

int64_t surface_panel_selected(const surface_panel *p) {
  return p ? p->selected_id : -1;
}

void surface_panel_select(surface_panel *p, int64_t id) {
  if (p) p->selected_id = id;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int surface_panel_count(const surface_panel *p) {
  return p ? p->count : 0;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

int64_t surface_panel_render(surface_panel *p, struct nk_context *ctx,
                             const char *title) {
  if (!p || !ctx) return -1;
  if (!title) title = "Surfaces";

  // Panel window — caller decides position/size via nk_begin flags.
  if (!nk_begin(ctx, title, nk_rect(0, 0, 320, 480),
                NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE |
                NK_WINDOW_TITLE)) {
    nk_end(ctx);
    return p->selected_id;
  }

  // Header row: count label + sort controls
  nk_layout_row_dynamic(ctx, 20, 1);
  {
    char hdr[64];
    snprintf(hdr, sizeof(hdr), "%d surface%s", p->count,
             p->count == 1 ? "" : "s");
    nk_label(ctx, hdr, NK_TEXT_LEFT);
  }

  nk_layout_row_dynamic(ctx, 20, 4);
  if (nk_button_label(ctx, "ID"))   surface_panel_sort(p, SORT_BY_ID,   true);
  if (nk_button_label(ctx, "Name")) surface_panel_sort(p, SORT_BY_NAME, true);
  if (nk_button_label(ctx, "Area")) surface_panel_sort(p, SORT_BY_AREA, false);
  if (nk_button_label(ctx, "Date")) surface_panel_sort(p, SORT_BY_DATE, true);

  nk_layout_row_dynamic(ctx, 1, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

  // One collapsible tree node per surface.
  for (int i = 0; i < p->count; i++) {
    surface_entry *e = &p->entries[i].e;

    // Build a unique tree-push id from the entry id.
    char node_label[128];
    snprintf(node_label, sizeof(node_label), "%s##%" PRId64,
             e->name && e->name[0] ? e->name : "(unnamed)", e->id);

    nk_layout_row_dynamic(ctx, 22, 1);

    // Highlight selected row.
    bool is_selected = (e->id == p->selected_id);
    if (is_selected) {
      struct nk_color sel_bg = nk_rgb(60, 100, 160);
      nk_style_push_color(ctx, &ctx->style.window.background, sel_bg);
    }

    bool open = nk_tree_push_id(ctx, NK_TREE_NODE, node_label,
                                NK_MINIMIZED, i);

    // Clicking the node header selects it.
    if (nk_widget_is_mouse_clicked(ctx, NK_BUTTON_LEFT))
      p->selected_id = e->id;

    if (is_selected) nk_style_pop_color(ctx);

    if (open) {
      // Dimensions
      nk_layout_row_dynamic(ctx, 18, 1);
      {
        char dim[64];
        snprintf(dim, sizeof(dim), "  Grid: %d x %d",
                 e->row_count, e->col_count);
        nk_label(ctx, dim, NK_TEXT_LEFT);
      }

      // Area in voxels^2
      {
        char area[64];
        snprintf(area, sizeof(area), "  Area: %.1f vx²", (double)e->area_vx2);
        nk_label(ctx, area, NK_TEXT_LEFT);
      }

      // Area in cm^2 if available
      if (e->area_cm2 > 0.0f) {
        char area_cm[64];
        snprintf(area_cm, sizeof(area_cm), "  Area: %.2f cm²",
                 (double)e->area_cm2);
        nk_label(ctx, area_cm, NK_TEXT_LEFT);
      }

      // Volume id
      if (e->volume_id && e->volume_id[0]) {
        char vid[128];
        snprintf(vid, sizeof(vid), "  Vol:  %s", e->volume_id);
        nk_label(ctx, vid, NK_TEXT_LEFT);
      }

      // Visible / approved checkboxes (modify in-place)
      nk_layout_row_dynamic(ctx, 20, 2);
      {
        int vis = (int)e->visible;
        nk_checkbox_label(ctx, "Visible", &vis);
        e->visible = (bool)vis;
      }
      {
        int app = (int)e->approved;
        nk_checkbox_label(ctx, "Approved", &app);
        e->approved = (bool)app;
      }

      // Select / remove buttons
      nk_layout_row_dynamic(ctx, 22, 2);
      if (nk_button_label(ctx, "Select"))
        p->selected_id = e->id;
      if (nk_button_label(ctx, "Remove")) {
        // Deferred removal: mark by setting id to sentinel, clear after loop.
        // We remove immediately here; nk_tree_pop is still required.
        nk_tree_pop(ctx);
        surface_panel_remove(p, e->id);
        // i now points at the next element (or past end); step back.
        i--;
        continue;
      }

      nk_tree_pop(ctx);
    }
  }

  nk_end(ctx);
  return p->selected_id;
}
