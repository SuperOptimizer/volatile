#include "gui/annotate.h"
#include "core/hash.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

struct annot_store {
  hash_map_int *map;   // id -> annotation*
  int64_t       next_id;
};

annot_store *annot_store_new(void) {
  annot_store *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->map     = hash_map_int_new();
  s->next_id = 1;
  if (!s->map) { free(s); return NULL; }
  return s;
}

static void free_annotation(annotation *a) {
  if (!a) return;
  free(a->points);
  free(a->label);
  free(a);
}

// Iter callback used by annot_store_free to release all entries.
typedef struct { int dummy; } _free_ctx;

static void _free_iter(const annotation *a, void *ctx) {
  (void)ctx;
  free_annotation((annotation *)a);
}

void annot_store_free(annot_store *s) {
  if (!s) return;
  annot_iter(s, _free_iter, NULL);
  hash_map_int_free(s->map);
  free(s);
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

int64_t annot_add(annot_store *s, annotation *a) {
  int64_t id = s->next_id++;
  a->id = id;
  hash_map_int_put(s->map, (uint64_t)id, a);
  return id;
}

annotation *annot_get(annot_store *s, int64_t id) {
  return hash_map_int_get(s->map, (uint64_t)id);
}

bool annot_remove(annot_store *s, int64_t id) {
  annotation *a = hash_map_int_get(s->map, (uint64_t)id);
  if (!a) return false;
  hash_map_int_del(s->map, (uint64_t)id);
  free_annotation(a);
  return true;
}

int annot_count(const annot_store *s) {
  return (int)hash_map_int_len(s->map);
}

// ---------------------------------------------------------------------------
// Iteration
// NOTE: hash_map_int has no iterator in the API; we use hash_map_int_get
//       indirectly via a stored key list approach. Since hash_map_int only
//       exposes get/put/del/len, we track a parallel sorted key array.
// ---------------------------------------------------------------------------
// We rebuild the key list on every iteration. This is fine for annotation
// counts (typically < 10 000). A future optimisation would add an iterator
// to hash_map_int, but we must not modify the dependency.

typedef struct {
  int64_t *keys;
  int      len;
  int      cap;
} key_list;

static void _collect_annots(annot_store *s, key_list *kl) {
  // Walk id range 1..next_id-1 and collect hits.
  // NOTE: This is O(next_id) not O(count). For moderate annotation counts
  //       (< 100k) this is acceptable; a proper iterator in hash_map_int
  //       would be better long-term.
  kl->len = 0;
  for (int64_t id = 1; id < s->next_id; id++) {
    if (!hash_map_int_get(s->map, (uint64_t)id)) continue;
    if (kl->len == kl->cap) {
      int new_cap = kl->cap ? kl->cap * 2 : 16;
      int64_t *tmp = realloc(kl->keys, (size_t)new_cap * sizeof(*kl->keys));
      if (!tmp) break;
      kl->keys = tmp;
      kl->cap  = new_cap;
    }
    kl->keys[kl->len++] = id;
  }
}

void annot_iter(const annot_store *s, annot_iter_fn fn, void *ctx) {
  key_list kl = {0};
  _collect_annots((annot_store *)s, &kl);
  for (int i = 0; i < kl.len; i++) {
    annotation *a = hash_map_int_get(s->map, (uint64_t)kl.keys[i]);
    if (a) fn(a, ctx);
  }
  free(kl.keys);
}

// ---------------------------------------------------------------------------
// Minimal JSON writer
// ---------------------------------------------------------------------------

typedef struct {
  FILE *f;
  bool  err;
} jw;

static void jw_raw(jw *w, const char *s)     { if (!w->err && fputs(s, w->f) < 0) w->err = true; }
static void jw_fmt(jw *w, const char *fmt, ...) {
  if (w->err) return;
  va_list ap; va_start(ap, fmt);
  if (vfprintf(w->f, fmt, ap) < 0) w->err = true;
  va_end(ap);
}

// Escape a C string for JSON output.
static void jw_str(jw *w, const char *s) {
  jw_raw(w, "\"");
  if (s) {
    for (const char *p = s; *p; p++) {
      if      (*p == '"')  jw_raw(w, "\\\"");
      else if (*p == '\\') jw_raw(w, "\\\\");
      else if (*p == '\n') jw_raw(w, "\\n");
      else if (*p == '\r') jw_raw(w, "\\r");
      else if (*p == '\t') jw_raw(w, "\\t");
      else                 fputc(*p, w->f);
    }
  }
  jw_raw(w, "\"");
}

static void write_annotation(jw *w, const annotation *a) {
  jw_raw(w, "{");
  jw_fmt(w, "\"id\":%"PRId64",", a->id);
  jw_fmt(w, "\"type\":%d,", (int)a->type);
  jw_fmt(w, "\"radius\":%.6g,", (double)a->radius);
  jw_fmt(w, "\"line_width\":%.6g,", (double)a->line_width);
  jw_fmt(w, "\"visible\":%s,", a->visible ? "true" : "false");
  jw_fmt(w, "\"color\":[%u,%u,%u,%u],", a->color[0], a->color[1], a->color[2], a->color[3]);
  jw_raw(w, "\"label\":"); jw_str(w, a->label); jw_raw(w, ",");
  jw_raw(w, "\"points\":[");
  for (int i = 0; i < a->num_points; i++) {
    if (i) jw_raw(w, ",");
    jw_fmt(w, "[%.6g,%.6g,%.6g]",
           (double)a->points[i].x, (double)a->points[i].y, (double)a->points[i].z);
  }
  jw_raw(w, "]}");
}

typedef struct { jw *w; int idx; } save_ctx;

static void _save_one(const annotation *a, void *ctx) {
  save_ctx *sc = ctx;
  if (sc->idx++) jw_raw(sc->w, ",");
  write_annotation(sc->w, a);
}

bool annot_save_json(const annot_store *s, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) { LOG_ERROR("annot_save_json: cannot open %s", path); return false; }
  jw w = {.f = f};
  jw_raw(&w, "{\"annotations\":[");
  save_ctx sc = {.w = &w, .idx = 0};
  annot_iter(s, _save_one, &sc);
  jw_raw(&w, "]}");
  fclose(f);
  return !w.err;
}

// ---------------------------------------------------------------------------
// JSON load helpers
// ---------------------------------------------------------------------------

#include "core/json.h"

static annotation *parse_annotation(const json_value *obj) {
  annotation *a = calloc(1, sizeof(*a));
  if (!a) return NULL;

  a->id         = json_get_int(json_object_get(obj, "id"),         0);
  a->type       = (annot_type)json_get_int(json_object_get(obj, "type"), 0);
  a->radius     = (float)json_get_number(json_object_get(obj, "radius"),     0.0);
  a->line_width = (float)json_get_number(json_object_get(obj, "line_width"), 1.0);
  a->visible    = json_get_bool(json_object_get(obj, "visible"), true);

  const json_value *color = json_object_get(obj, "color");
  if (color) {
    for (int i = 0; i < 4; i++)
      a->color[i] = (uint8_t)json_get_int(json_array_get(color, (size_t)i), 255);
  }

  const char *lbl = json_get_str(json_object_get(obj, "label"));
  if (lbl) a->label = strdup(lbl);

  const json_value *pts = json_object_get(obj, "points");
  a->num_points = pts ? (int)json_array_len(pts) : 0;
  if (a->num_points > 0) {
    a->points = malloc((size_t)a->num_points * sizeof(vec3f));
    if (!a->points) { free(a->label); free(a); return NULL; }
    for (int i = 0; i < a->num_points; i++) {
      const json_value *pt = json_array_get(pts, (size_t)i);
      a->points[i].x = (float)json_get_number(json_array_get(pt, 0), 0.0);
      a->points[i].y = (float)json_get_number(json_array_get(pt, 1), 0.0);
      a->points[i].z = (float)json_get_number(json_array_get(pt, 2), 0.0);
    }
  }
  return a;
}

annot_store *annot_load_json(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) { LOG_ERROR("annot_load_json: cannot open %s", path); return NULL; }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  fread(buf, 1, (size_t)sz, f);
  buf[sz] = '\0';
  fclose(f);

  json_value *root = json_parse(buf);
  free(buf);
  if (!root) { LOG_ERROR("annot_load_json: parse error in %s", path); return NULL; }

  annot_store *s = annot_store_new();
  if (!s) { json_free(root); return NULL; }

  const json_value *arr = json_object_get(root, "annotations");
  size_t n = arr ? json_array_len(arr) : 0;
  for (size_t i = 0; i < n; i++) {
    annotation *a = parse_annotation(json_array_get(arr, i));
    if (!a) continue;
    // Preserve original id by inserting directly and updating next_id.
    hash_map_int_put(s->map, (uint64_t)a->id, a);
    if (a->id >= s->next_id) s->next_id = a->id + 1;
  }

  json_free(root);
  return s;
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

// Minimum distance from point to a line segment (a, b).
static float seg_dist(vec3f p, vec3f a, vec3f b) {
  vec3f ab = vec3f_sub(b, a);
  vec3f ap = vec3f_sub(p, a);
  float len2 = vec3f_dot(ab, ab);
  if (len2 < 1e-12f) return vec3f_len(ap);
  float t = vec3f_dot(ap, ab) / len2;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  return vec3f_len(vec3f_sub(p, vec3f_add(a, vec3f_scale(ab, t))));
}

static float annot_dist(const annotation *a, vec3f p) {
  if (a->num_points == 0) return 1e30f;

  switch (a->type) {
    case ANNOT_POINT:
      return vec3f_len(vec3f_sub(p, a->points[0]));

    case ANNOT_CIRCLE: {
      // Distance to circle edge in the plane defined by its single center point.
      float d = vec3f_len(vec3f_sub(p, a->points[0]));
      return fabsf(d - a->radius);
    }

    case ANNOT_TEXT:
      return vec3f_len(vec3f_sub(p, a->points[0]));

    case ANNOT_LINE:
    case ANNOT_POLYLINE:
    case ANNOT_FREEHAND:
    case ANNOT_RECT:
    case ANNOT_POLYGON: {
      float min_d = 1e30f;
      int end = (a->type == ANNOT_POLYGON || a->type == ANNOT_RECT)
                ? a->num_points : a->num_points - 1;
      for (int i = 0; i < end && i < a->num_points - 1; i++) {
        float d = seg_dist(p, a->points[i], a->points[i+1]);
        if (d < min_d) min_d = d;
      }
      // Close polygon / rect
      if ((a->type == ANNOT_POLYGON || a->type == ANNOT_RECT) && a->num_points >= 2) {
        float d = seg_dist(p, a->points[a->num_points-1], a->points[0]);
        if (d < min_d) min_d = d;
      }
      return min_d;
    }
  }
  return 1e30f;
}

typedef struct {
  vec3f   point;
  float   tolerance;
  float   best_dist;
  int64_t best_id;
} hit_ctx;

static void _hit_one(const annotation *a, void *ctx) {
  if (!a->visible) return;
  hit_ctx *hc = ctx;
  float d = annot_dist(a, hc->point);
  if (d < hc->tolerance && d < hc->best_dist) {
    hc->best_dist = d;
    hc->best_id   = a->id;
  }
}

int64_t annot_hit_test(const annot_store *s, vec3f point, float tolerance) {
  hit_ctx hc = {.point = point, .tolerance = tolerance, .best_dist = 1e30f, .best_id = -1};
  annot_iter(s, _hit_one, &hc);
  return hc.best_id;
}
