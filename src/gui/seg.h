#pragma once
#include "core/geom.h"

// ---------------------------------------------------------------------------
// Segmentation tool definitions
// ---------------------------------------------------------------------------

typedef enum {
  SEG_TOOL_BRUSH,
  SEG_TOOL_LINE,
  SEG_TOOL_PUSHPULL,
  SEG_TOOL_ERASER,
} seg_tool_id;

typedef struct {
  seg_tool_id tool;
  float radius;        // brush radius in grid units
  float sigma;         // gaussian falloff sigma
  float push_amount;   // for push-pull: displacement along normal
} seg_tool_params;

typedef struct seg_edit seg_edit;  // opaque edit record for undo

// Apply a brush stroke: deform quad_surface vertices near (u,v) by delta along normal.
seg_edit *seg_brush_apply(quad_surface *s, float u, float v, float delta,
                          const seg_tool_params *params);

// Apply a line stroke: deform vertices along a line from (u0,v0) to (u1,v1).
seg_edit *seg_line_apply(quad_surface *s, float u0, float v0, float u1, float v1,
                         float delta, const seg_tool_params *params);

// Push-pull: uniformly displace a region along the surface normal.
seg_edit *seg_pushpull_apply(quad_surface *s, float u, float v,
                             const seg_tool_params *params);

// Undo: restore vertices from the edit record.
void seg_edit_undo(quad_surface *s, const seg_edit *e);
void seg_edit_free(seg_edit *e);
