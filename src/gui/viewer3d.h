#pragma once
#include <stdint.h>
#include "core/math.h"
#include "core/vol.h"

typedef struct viewer3d viewer3d;

typedef enum {
  RENDER3D_MIP,           // Maximum Intensity Projection
  RENDER3D_ISO_SURFACE,   // Iso-surface (specify iso-value)
  RENDER3D_TRANSFER_FUNC, // Transfer function volume rendering
} render3d_mode;

typedef struct {
  render3d_mode mode;
  float iso_value;    // for ISO_SURFACE mode
  float step_size;    // ray step size (voxels, default 0.5)
  float fov_degrees;  // field of view (default 45)
  int   cmap_id;
  float window, level;
} viewer3d_config;

viewer3d *viewer3d_new(viewer3d_config cfg);
void      viewer3d_free(viewer3d *v);

void viewer3d_set_volume(viewer3d *v, volume *vol);
void viewer3d_set_camera(viewer3d *v, vec3f eye, vec3f target, vec3f up);
void viewer3d_orbit(viewer3d *v, float yaw_delta, float pitch_delta);
void viewer3d_dolly(viewer3d *v, float delta);

// render to pixel buffer (CPU reference, RGBA8, width*height*4 bytes)
void viewer3d_render_cpu(viewer3d *v, uint8_t *pixels, int width, int height);

// get the world-space ray for a screen pixel
void viewer3d_screen_ray(const viewer3d *v, float sx, float sy, int w, int h,
                         vec3f *origin, vec3f *dir);
