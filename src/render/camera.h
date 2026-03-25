#pragma once
#include <stdint.h>
#include "core/math.h"

// ---------------------------------------------------------------------------
// viewer_camera — 2D surface viewer camera (pan/zoom/z-scroll)
// ---------------------------------------------------------------------------
// center is in normalised surface parameter space [0,1]x[0,1].
// scale is pixels-per-unit-surface-space at the current zoom level.
// z_offset drives scrolling through volume depth (in surface normal coords).

typedef struct {
  vec3f    center;         // (u, v, 0) — center of view in surface param space
  float    scale;          // zoom: surface units per screen pixel (clamped 1/32..4)
  float    z_offset;       // normal-axis offset into the volume
  int      pyramid_level;  // current LOD level (0 = full resolution)
  uint64_t epoch;          // monotonic counter — bump to signal re-render needed
} viewer_camera;

// ---------------------------------------------------------------------------
// viewport — screen/pixel context for coordinate transforms
// ---------------------------------------------------------------------------

typedef struct {
  int   screen_w, screen_h;
  float tile_size;   // pixels per tile (typically 256)
} viewport;

void camera_init(viewer_camera *cam);
void camera_pan(viewer_camera *cam, float dx, float dy);
// zoom centered on screen point (cx, cy) in absolute screen pixels; factor > 1 zooms in
void camera_zoom(viewer_camera *cam, const viewport *vp, float factor, float cx, float cy);
void camera_set_z_offset(viewer_camera *cam, float z);
void camera_step_z(viewer_camera *cam, float delta);
// select pyramid level given total number of LOD levels
int  camera_calc_pyramid_level(const viewer_camera *cam, int num_levels);
void camera_invalidate(viewer_camera *cam);

// surface param space <-> screen pixel space
void viewport_surface_to_screen(const viewer_camera *cam, const viewport *vp,
                                 float su, float sv, float *sx, float *sy);
void viewport_screen_to_surface(const viewer_camera *cam, const viewport *vp,
                                 float sx, float sy, float *su, float *sv);
