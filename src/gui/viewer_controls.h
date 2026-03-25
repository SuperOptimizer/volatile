#pragma once
#include "render/composite.h"
#include <stdbool.h>

// Forward declarations — avoid pulling in nuklear.h or viewer.h transitively.
struct nk_context;
typedef struct slice_viewer slice_viewer;
typedef struct viewer_controls viewer_controls;

// ---------------------------------------------------------------------------
// viewer_controls — VC3D viewer controls dock panel (plain C + Nuklear)
//
// Sections (all collapsible):
//   Position · Zoom · Pyramid · Composite · Colormap · Window/Level
//   Overlay volume · Normal overlay · Intersection lines · Scale bar
// ---------------------------------------------------------------------------

viewer_controls *viewer_controls_new(void);
void             viewer_controls_free(viewer_controls *c);

// Render the panel into an already-open nk_begin()/nk_end() window.
// Reads current state from active_viewer (may be NULL — panel still renders).
// Returns true if any setting changed this frame.
bool viewer_controls_render(viewer_controls *c, struct nk_context *ctx,
                            slice_viewer *active_viewer);

// Accessors for the current display settings.
int              viewer_controls_get_cmap(const viewer_controls *c);
composite_params viewer_controls_get_composite(const viewer_controls *c);
