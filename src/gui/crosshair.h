#pragma once
#include "core/math.h"

typedef struct crosshair_sync crosshair_sync;
typedef struct slice_viewer   slice_viewer;
typedef struct overlay_list   overlay_list;

crosshair_sync *crosshair_sync_new(void);
void            crosshair_sync_free(crosshair_sync *s);

void crosshair_sync_add_viewer(crosshair_sync *s, slice_viewer *v);
void crosshair_sync_remove_viewer(crosshair_sync *s, slice_viewer *v);

// set/get the current 3D focus point (world coordinates)
void  crosshair_sync_set_focus(crosshair_sync *s, vec3f world_pos);
vec3f crosshair_sync_get_focus(const crosshair_sync *s);

// generate overlay lines for a specific viewer based on current focus
// XY viewer (axis=0): vertical line at x=focus.x, horizontal line at y=focus.y
// XZ viewer (axis=1): vertical line at x=focus.x, horizontal line at z=focus.z
// YZ viewer (axis=2): vertical line at y=focus.y, horizontal line at z=focus.z
// Colors: XY=orange(255,165,0), XZ=red(255,50,50), YZ=yellow(255,220,0)
void crosshair_sync_render_overlays(const crosshair_sync *s, slice_viewer *v, overlay_list *out);
