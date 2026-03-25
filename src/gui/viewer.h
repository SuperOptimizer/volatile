#pragma once
#include <stdint.h>
#include "core/math.h"
#include "render/camera.h"
#include "render/composite.h"
#include "render/tile.h"

typedef struct volume volume;
typedef struct overlay_list overlay_list;

typedef struct {
  int view_axis;           // 0=XY, 1=XZ, 2=YZ
  viewer_camera camera;
  int cmap_id;             // colormap (cmap_id enum)
  float window, level;     // contrast
  composite_params composite;
} viewer_config;

typedef struct slice_viewer slice_viewer;

slice_viewer *viewer_new(viewer_config cfg, tile_renderer *renderer);
void          viewer_free(slice_viewer *v);

// update camera (from user input)
void viewer_pan(slice_viewer *v, float dx, float dy);
void viewer_zoom(slice_viewer *v, float factor, float cx, float cy);
void viewer_scroll_slice(slice_viewer *v, float delta);

// set the volume to display
void viewer_set_volume(slice_viewer *v, volume *vol);

// set overlays to draw on top
void viewer_set_overlays(slice_viewer *v, const overlay_list *overlays);

// render the current view to a pixel buffer (RGBA, width*height*4)
void viewer_render(slice_viewer *v, uint8_t *pixels, int width, int height);

// drive zoom-settle countdown and progressive refinement (~30 Hz).
// returns true while background work is pending.
bool viewer_tick(slice_viewer *v);

// get the 3D world position under a screen pixel
vec3f viewer_screen_to_world(const slice_viewer *v, float sx, float sy);

// get current slice info
float viewer_current_slice(const slice_viewer *v);
int   viewer_current_level(const slice_viewer *v);

// get view axis (0=XY, 1=XZ, 2=YZ)
int viewer_get_axis(const slice_viewer *v);
