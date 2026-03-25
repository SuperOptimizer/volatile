#pragma once
#include <stdbool.h>
#include <stdint.h>

struct nk_context;

// ---------------------------------------------------------------------------
// point_panel — Hierarchical point collection panel.
//
// Replaces VC3D's CPointCollectionWidget. Supports:
//   - Multiple named collections, each with a color
//   - Per-point (x,y,z) coordinates and optional label
//   - Tree-view rendering via Nuklear
//   - Navigate callback on double-click
//   - JSON import/export
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Data types (caller-visible)
// ---------------------------------------------------------------------------

typedef struct {
  int64_t id;
  char    label[64];
  float   x, y, z;
} point_entry;

typedef struct {
  int64_t id;
  char    name[64];
  uint8_t r, g, b;     // collection color
} point_collection;

typedef struct point_panel point_panel;

// Called when the user double-clicks a point to navigate to it.
typedef void (*point_navigate_fn)(float x, float y, float z, void *ctx);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

point_panel *point_panel_new(void);
void         point_panel_free(point_panel *p);

// ---------------------------------------------------------------------------
// Collections
// ---------------------------------------------------------------------------

// Returns new collection id, or -1 on error.
int64_t point_panel_add_collection(point_panel *p, const char *name,
                                   uint8_t r, uint8_t g, uint8_t b);
void    point_panel_remove_collection(point_panel *p, int64_t coll_id);
int     point_panel_collection_count(const point_panel *p);

// ---------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------

// Returns new point id, or -1 on error.
int64_t point_panel_add_point(point_panel *p, int64_t coll_id,
                               const char *label, float x, float y, float z);
void    point_panel_remove_point(point_panel *p, int64_t coll_id, int64_t point_id);
int     point_panel_point_count(const point_panel *p, int64_t coll_id);

// ---------------------------------------------------------------------------
// Navigation callback
// ---------------------------------------------------------------------------

void point_panel_set_navigate_cb(point_panel *p, point_navigate_fn fn, void *ctx);

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void point_panel_render(point_panel *p, struct nk_context *ctx, const char *title);

// ---------------------------------------------------------------------------
// Import / export JSON
// ---------------------------------------------------------------------------

// Serialise all collections + points to a malloc'd JSON string (caller frees).
char *point_panel_to_json(const point_panel *p);

// Replace current state from JSON string.  Returns false on parse error.
bool  point_panel_from_json(point_panel *p, const char *json_str);
