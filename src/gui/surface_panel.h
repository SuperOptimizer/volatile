// WIDGET TYPE: WINDOW — renders its own nk_begin/nk_end window, call OUTSIDE any nk_begin block.
#pragma once
#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
struct nk_context;

// ---------------------------------------------------------------------------
// surface_entry — metadata for a single loaded surface
// ---------------------------------------------------------------------------

typedef struct {
  int64_t id;
  char   *name;        // display name (owned)
  char   *volume_id;   // parent volume id string (owned, may be NULL)
  float   area_vx2;    // surface area in voxels^2
  float   area_cm2;    // surface area in cm^2 (0 if voxel size unknown)
  bool    visible;
  bool    approved;
  int     row_count;
  int     col_count;
} surface_entry;

// ---------------------------------------------------------------------------
// Sort order
// ---------------------------------------------------------------------------

typedef enum {
  SORT_BY_ID,
  SORT_BY_NAME,
  SORT_BY_AREA,
  SORT_BY_DATE,   // insertion order (index)
} surface_sort_t;

// ---------------------------------------------------------------------------
// Opaque panel state
// ---------------------------------------------------------------------------

typedef struct surface_panel surface_panel;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

surface_panel *surface_panel_new(void);
void           surface_panel_free(surface_panel *p);

// ---------------------------------------------------------------------------
// Mutation — copies all fields from entry (strings are duplicated)
// ---------------------------------------------------------------------------

void surface_panel_add(surface_panel *p, const surface_entry *entry);
void surface_panel_remove(surface_panel *p, int64_t id);
void surface_panel_clear(surface_panel *p);

// ---------------------------------------------------------------------------
// Render — draws the panel using Nuklear.
// Returns the ID of the currently-selected surface, or -1 if none.
// ---------------------------------------------------------------------------

int64_t surface_panel_render(surface_panel *p, struct nk_context *ctx,
                             const char *title);

// ---------------------------------------------------------------------------
// Sort
// ---------------------------------------------------------------------------

void surface_panel_sort(surface_panel *p, surface_sort_t by, bool ascending);

// ---------------------------------------------------------------------------
// Selection
// ---------------------------------------------------------------------------

int64_t surface_panel_selected(const surface_panel *p);
void    surface_panel_select(surface_panel *p, int64_t id);

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int surface_panel_count(const surface_panel *p);
