#pragma once
#include "core/math.h"
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef enum {
  ANNOT_POINT,
  ANNOT_LINE,
  ANNOT_POLYLINE,
  ANNOT_POLYGON,
  ANNOT_CIRCLE,
  ANNOT_FREEHAND,
  ANNOT_TEXT,
  ANNOT_RECT,
} annot_type;

typedef struct {
  int64_t    id;
  annot_type type;
  vec3f     *points;      // array of 3D world-space points
  int        num_points;
  char      *label;       // text label (owned); NULL if unused
  float      radius;      // for ANNOT_CIRCLE
  uint8_t    color[4];    // RGBA
  float      line_width;
  bool       visible;
} annotation;

typedef struct annot_store annot_store;

// ---------------------------------------------------------------------------
// Store lifecycle
// ---------------------------------------------------------------------------

annot_store *annot_store_new(void);
void         annot_store_free(annot_store *s);

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

// Takes ownership of a (including a->points and a->label). Returns assigned id.
int64_t     annot_add(annot_store *s, annotation *a);
annotation *annot_get(annot_store *s, int64_t id);
bool        annot_remove(annot_store *s, int64_t id);
int         annot_count(const annot_store *s);

// ---------------------------------------------------------------------------
// Iteration
// ---------------------------------------------------------------------------

typedef void (*annot_iter_fn)(const annotation *a, void *ctx);
void annot_iter(const annot_store *s, annot_iter_fn fn, void *ctx);

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

bool         annot_save_json(const annot_store *s, const char *path);
annot_store *annot_load_json(const char *path);

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

// Returns id of nearest annotation whose geometry is within tolerance of
// point, or -1 if none found.
int64_t annot_hit_test(const annot_store *s, vec3f point, float tolerance);
