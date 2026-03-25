#pragma once
#include <stdint.h>
#include <stdbool.h>
#include "core/vol.h"
#include "render/cmap.h"

// ---------------------------------------------------------------------------
// overlay_volume — second volume composited on top of the primary slice
//
// Supports separate opacity, colormap, threshold, window/level, and blend mode.
// The composite step is called per-tile during rendering.
// ---------------------------------------------------------------------------

typedef enum {
  OVERLAY_BLEND_ALPHA     = 0,  // standard alpha over
  OVERLAY_BLEND_ADDITIVE,       // src * alpha + dst
  OVERLAY_BLEND_MULTIPLY,       // dst * src_norm
} overlay_blend_mode;

typedef struct overlay_volume overlay_volume;

// Lifecycle
overlay_volume *overlay_volume_new(void);
void            overlay_volume_free(overlay_volume *v);

// Data source — does NOT take ownership of vol.
void overlay_volume_set_volume(overlay_volume *v, volume *vol);

// Display controls
void overlay_volume_set_opacity(overlay_volume *v, float opacity);    // [0, 1]
void overlay_volume_set_cmap(overlay_volume *v, int cmap_id);          // cmap_id enum
void overlay_volume_set_threshold(overlay_volume *v, float threshold); // [0, 1] norm
void overlay_volume_set_visible(overlay_volume *v, bool visible);
void overlay_volume_set_blend(overlay_volume *v, overlay_blend_mode mode);
void overlay_volume_set_window(overlay_volume *v, float center, float width); // [0,255]

// Query
bool            overlay_volume_visible(const overlay_volume *v);
float           overlay_volume_opacity(const overlay_volume *v);
int             overlay_volume_cmap(const overlay_volume *v);
float           overlay_volume_threshold(const overlay_volume *v);
overlay_blend_mode overlay_volume_blend(const overlay_volume *v);

// Render a Nuklear controls panel. Returns true if any setting changed this frame.
// nk_context is forward-declared to avoid pulling in nuklear.h.
struct nk_context;
bool overlay_volume_render_controls(overlay_volume *v, struct nk_context *ctx);

// Composite overlay onto a tile in-place.
//
//   tile_rgba : RGBA8 buffer, w*h*4 bytes, modified in place
//   z         : slice position in voxels (axis-dependent meaning)
//   y0, x0    : top-left voxel of the tile at the current zoom
//   scale     : voxels per pixel (1.0 = 1:1)
//   axis      : 0=Z-plane, 1=Y-plane, 2=X-plane
void overlay_volume_composite_tile(const overlay_volume *v,
                                   uint8_t *tile_rgba, int w, int h,
                                   float z, float y0, float x0,
                                   float scale, int axis);
