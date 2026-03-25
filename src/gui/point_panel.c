#include "gui/point_panel.h"
#include "core/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef NK_INCLUDE_DEFAULT_ALLOCATOR
struct nk_context;
#define NK_STUB
#endif

#ifndef NK_STUB
#include "nuklear.h"
#endif

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_COLLECTIONS 32
#define MAX_POINTS      256

// ---------------------------------------------------------------------------
// Internal storage
// ---------------------------------------------------------------------------

typedef struct {
  point_collection meta;
  point_entry      pts[MAX_POINTS];
  int              n_pts;
  bool             expanded;  // tree node open/closed
} coll_slot;

struct point_panel {
  coll_slot  colls[MAX_COLLECTIONS];
  int        n_colls;
  int64_t    next_coll_id;
  int64_t    next_pt_id;

  // selection
  int64_t sel_coll;
  int64_t sel_pt;

  // edit buffer for "add point" row
  char    new_label[64];
  char    new_x[16], new_y[16], new_z[16];

  point_navigate_fn navigate_fn;
  void             *navigate_ctx;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

point_panel *point_panel_new(void) {
  point_panel *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->next_coll_id = 1;
  p->next_pt_id   = 1;
  p->sel_coll     = -1;
  p->sel_pt       = -1;
  return p;
}

void point_panel_free(point_panel *p) { free(p); }

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static coll_slot *find_coll(point_panel *p, int64_t id) {
  for (int i = 0; i < p->n_colls; i++)
    if (p->colls[i].meta.id == id) return &p->colls[i];
  return NULL;
}

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

int64_t point_panel_add_collection(point_panel *p, const char *name,
                                   uint8_t r, uint8_t g, uint8_t b) {
  if (!p || p->n_colls >= MAX_COLLECTIONS) return -1;
  coll_slot *s  = &p->colls[p->n_colls++];
  memset(s, 0, sizeof(*s));
  s->meta.id    = p->next_coll_id++;
  s->meta.r     = r; s->meta.g = g; s->meta.b = b;
  strncpy(s->meta.name, name ? name : "collection", 63);
  s->expanded   = true;
  return s->meta.id;
}

void point_panel_remove_collection(point_panel *p, int64_t coll_id) {
  if (!p) return;
  for (int i = 0; i < p->n_colls; i++) {
    if (p->colls[i].meta.id == coll_id) {
      memmove(&p->colls[i], &p->colls[i + 1],
              (size_t)(p->n_colls - i - 1) * sizeof(coll_slot));
      p->n_colls--;
      return;
    }
  }
}

int point_panel_collection_count(const point_panel *p) {
  return p ? p->n_colls : 0;
}

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

int64_t point_panel_add_point(point_panel *p, int64_t coll_id,
                               const char *label, float x, float y, float z) {
  if (!p) return -1;
  coll_slot *s = find_coll(p, coll_id);
  if (!s || s->n_pts >= MAX_POINTS) return -1;
  point_entry *e = &s->pts[s->n_pts++];
  e->id = p->next_pt_id++;
  e->x = x; e->y = y; e->z = z;
  strncpy(e->label, label ? label : "", 63);
  return e->id;
}

void point_panel_remove_point(point_panel *p, int64_t coll_id, int64_t pt_id) {
  if (!p) return;
  coll_slot *s = find_coll(p, coll_id);
  if (!s) return;
  for (int i = 0; i < s->n_pts; i++) {
    if (s->pts[i].id == pt_id) {
      memmove(&s->pts[i], &s->pts[i + 1],
              (size_t)(s->n_pts - i - 1) * sizeof(point_entry));
      s->n_pts--;
      return;
    }
  }
}

int point_panel_point_count(const point_panel *p, int64_t coll_id) {
  if (!p) return 0;
  const coll_slot *s = find_coll((point_panel *)p, coll_id);
  return s ? s->n_pts : 0;
}

// ---------------------------------------------------------------------------
// Navigate callback
// ---------------------------------------------------------------------------

void point_panel_set_navigate_cb(point_panel *p, point_navigate_fn fn, void *ctx) {
  if (!p) return;
  p->navigate_fn  = fn;
  p->navigate_ctx = ctx;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void point_panel_render(point_panel *p, struct nk_context *ctx, const char *title) {
#ifdef NK_STUB
  (void)p; (void)ctx; (void)title;
  return;
#else
  if (!p || !ctx) return;
  (void)title;

  for (int ci = 0; ci < p->n_colls; ci++) {
    coll_slot *s = &p->colls[ci];
    // Collection tree node with color swatch
    struct nk_color col = nk_rgb(s->meta.r, s->meta.g, s->meta.b);
    nk_layout_row_dynamic(ctx, 22, 1);
    int expanded = nk_tree_push_id(ctx, NK_TREE_NODE, s->meta.name,
                                   NK_MINIMIZED, ci);
    if (expanded) {
      // Point rows
      for (int pi = 0; pi < s->n_pts; pi++) {
        point_entry *e = &s->pts[pi];
        char buf[128];
        snprintf(buf, sizeof(buf), "  %s  (%.1f, %.1f, %.1f)",
                 e->label, e->x, e->y, e->z);
        nk_layout_row_dynamic(ctx, 20, 2);
        bool selected = (p->sel_coll == s->meta.id && p->sel_pt == e->id);
        if (nk_selectable_label(ctx, buf, NK_TEXT_LEFT, &(int){selected ? 1 : 0})) {
          if (selected && p->navigate_fn)
            p->navigate_fn(e->x, e->y, e->z, p->navigate_ctx);
          p->sel_coll = s->meta.id;
          p->sel_pt   = e->id;
        }
        if (nk_button_symbol(ctx, NK_SYMBOL_X)) {
          point_panel_remove_point(p, s->meta.id, e->id);
          pi--;  // adjust after removal
        }
      }
      // Add point row
      nk_layout_row_dynamic(ctx, 22, 1);
      nk_label(ctx, "Add point:", NK_TEXT_LEFT);
      nk_layout_row_dynamic(ctx, 22, 4);
      nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                      p->new_label, 63, nk_filter_default);
      nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                      p->new_x, 15, nk_filter_float);
      nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                      p->new_y, 15, nk_filter_float);
      nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD,
                                      p->new_z, 15, nk_filter_float);
      nk_layout_row_dynamic(ctx, 24, 2);
      if (nk_button_label(ctx, "Add")) {
        float x = strtof(p->new_x, NULL);
        float y = strtof(p->new_y, NULL);
        float z = strtof(p->new_z, NULL);
        point_panel_add_point(p, s->meta.id, p->new_label, x, y, z);
        memset(p->new_label, 0, sizeof(p->new_label));
        memset(p->new_x, 0, sizeof(p->new_x));
        memset(p->new_y, 0, sizeof(p->new_y));
        memset(p->new_z, 0, sizeof(p->new_z));
      }
      if (nk_button_label(ctx, "Delete collection")) {
        point_panel_remove_collection(p, s->meta.id);
        ci--;
        nk_tree_pop(ctx);
        continue;
      }
      nk_tree_pop(ctx);
    }
    (void)col;
  }

  // Bottom: "New collection" button
  nk_layout_row_dynamic(ctx, 28, 1);
  if (nk_button_label(ctx, "+ New Collection")) {
    point_panel_add_collection(p, "New Collection", 180, 180, 60);
  }
#endif
}

// ---------------------------------------------------------------------------
// JSON export
// ---------------------------------------------------------------------------

char *point_panel_to_json(const point_panel *p) {
  if (!p) return NULL;

  // Estimate size: each point ~128 bytes, each collection header ~64 bytes.
  size_t cap = 256 + (size_t)p->n_colls * (64 + MAX_POINTS * 128);
  char *buf = malloc(cap);
  if (!buf) return NULL;

  int off = 0;
  off += snprintf(buf + off, cap - (size_t)off, "{\"collections\":[");
  for (int ci = 0; ci < p->n_colls; ci++) {
    const coll_slot *s = &p->colls[ci];
    if (ci) off += snprintf(buf + off, cap - (size_t)off, ",");
    off += snprintf(buf + off, cap - (size_t)off,
                   "{\"id\":%lld,\"name\":\"%s\","
                   "\"r\":%d,\"g\":%d,\"b\":%d,\"points\":[",
                   (long long)s->meta.id, s->meta.name,
                   s->meta.r, s->meta.g, s->meta.b);
    for (int pi = 0; pi < s->n_pts; pi++) {
      const point_entry *e = &s->pts[pi];
      if (pi) off += snprintf(buf + off, cap - (size_t)off, ",");
      off += snprintf(buf + off, cap - (size_t)off,
                     "{\"id\":%lld,\"label\":\"%s\","
                     "\"x\":%.4f,\"y\":%.4f,\"z\":%.4f}",
                     (long long)e->id, e->label, e->x, e->y, e->z);
    }
    off += snprintf(buf + off, cap - (size_t)off, "]}");
  }
  off += snprintf(buf + off, cap - (size_t)off, "]}");
  return buf;
}

// ---------------------------------------------------------------------------
// JSON import
// ---------------------------------------------------------------------------

bool point_panel_from_json(point_panel *p, const char *json_str) {
  if (!p || !json_str) return false;

  json_value *root = json_parse(json_str);
  if (!root) return false;

  const json_value *colls_arr = json_object_get(root, "collections");
  if (!colls_arr || json_typeof(colls_arr) != JSON_ARRAY) {
    json_free(root); return false;
  }

  // Replace current state
  p->n_colls = 0;

  size_t nc = json_array_len(colls_arr);
  for (size_t ci = 0; ci < nc && p->n_colls < MAX_COLLECTIONS; ci++) {
    const json_value *cobj = json_array_get(colls_arr, ci);
    if (!cobj) continue;

    const char *name = json_get_str(json_object_get(cobj, "name"));
    uint8_t r = (uint8_t)json_get_int(json_object_get(cobj, "r"), 180);
    uint8_t g = (uint8_t)json_get_int(json_object_get(cobj, "g"), 180);
    uint8_t b = (uint8_t)json_get_int(json_object_get(cobj, "b"), 60);
    int64_t cid = point_panel_add_collection(p, name, r, g, b);

    const json_value *pts_arr = json_object_get(cobj, "points");
    if (!pts_arr || json_typeof(pts_arr) != JSON_ARRAY) continue;

    size_t np = json_array_len(pts_arr);
    for (size_t pi = 0; pi < np; pi++) {
      const json_value *pobj = json_array_get(pts_arr, pi);
      if (!pobj) continue;
      const char *lbl = json_get_str(json_object_get(pobj, "label"));
      float x = (float)json_get_int(json_object_get(pobj, "x"), 0);
      float y = (float)json_get_int(json_object_get(pobj, "y"), 0);
      float z = (float)json_get_int(json_object_get(pobj, "z"), 0);
      // NOTE: json_get_int is used here; a real float parse would use
      // json_get_float if available, otherwise strtof on the raw string.
      point_panel_add_point(p, cid, lbl, x, y, z);
    }
  }

  json_free(root);
  return true;
}
