#include "render/camera.h"
#include "core/math.h"
#include <math.h>

#define SCALE_MIN (1.0f / 32.0f)
#define SCALE_MAX 4.0f

// ---------------------------------------------------------------------------
// camera
// ---------------------------------------------------------------------------

void camera_init(viewer_camera *cam) {
  cam->center        = (vec3f){0.5f, 0.5f, 0.0f};
  cam->scale         = 1.0f;
  cam->z_offset      = 0.0f;
  cam->pyramid_level = 0;
  cam->epoch         = 0;
}

// dx/dy are in screen pixels; convert to surface units and shift center
void camera_pan(viewer_camera *cam, float dx, float dy) {
  cam->center.x -= dx * cam->scale;
  cam->center.y -= dy * cam->scale;
  cam->epoch++;
}

// zoom toward screen point (cx, cy) in absolute screen pixels.
// factor > 1 zooms in (scale decreases — fewer surface units per pixel).
// The surface coordinate under (cx, cy) remains fixed after the zoom.
void camera_zoom(viewer_camera *cam, const viewport *vp, float factor, float cx, float cy) {
  // NOTE: keep pivot surface coord invariant.
  // su = center.x + (cx - screen_cx) * scale
  // After zoom: su = new_center.x + (cx - screen_cx) * new_scale
  // => new_center.x = center.x + (cx - screen_cx) * (scale - new_scale)
  float screen_cx = vp->screen_w * 0.5f;
  float screen_cy = vp->screen_h * 0.5f;
  float new_scale = clampf(cam->scale / factor, SCALE_MIN, SCALE_MAX);
  cam->center.x += (cx - screen_cx) * (cam->scale - new_scale);
  cam->center.y += (cy - screen_cy) * (cam->scale - new_scale);
  cam->scale = new_scale;
  cam->epoch++;
}

void camera_set_z_offset(viewer_camera *cam, float z) {
  cam->z_offset = z;
  cam->epoch++;
}

void camera_step_z(viewer_camera *cam, float delta) {
  cam->z_offset += delta;
  cam->epoch++;
}

// Select pyramid level so that one screen pixel ~ one texel at that level.
// Level 0 = full resolution, level k = 1/(2^k) resolution.
// We want 2^level ~ 1/scale (scale = surface_units/pixel; at level 0 one
// surface unit = one texel, so texels/pixel = 1/scale).
int camera_calc_pyramid_level(const viewer_camera *cam, int num_levels) {
  if (num_levels <= 1) return 0;
  // texels per pixel = 1 / scale; best level = log2(texels_per_pixel)
  float texels_per_pixel = 1.0f / cam->scale;
  if (texels_per_pixel <= 1.0f) return 0;
  int level = (int)log2f(texels_per_pixel);
  if (level < 0) level = 0;
  if (level >= num_levels) level = num_levels - 1;
  return level;
}

void camera_invalidate(viewer_camera *cam) {
  cam->epoch++;
}

// ---------------------------------------------------------------------------
// viewport transforms
//
// Convention:
//   screen origin (0,0) is top-left.
//   surface origin (0,0) is the top-left of the surface param space.
//   cam->center is the surface coord that maps to the screen centre.
//   cam->scale  is surface_units / screen_pixel.
//
//   su = cam->center.x + (sx - screen_cx) * cam->scale
//   sv = cam->center.y + (sy - screen_cy) * cam->scale
// ---------------------------------------------------------------------------

void viewport_surface_to_screen(const viewer_camera *cam, const viewport *vp,
                                  float su, float sv, float *sx, float *sy) {
  float screen_cx = vp->screen_w * 0.5f;
  float screen_cy = vp->screen_h * 0.5f;
  *sx = screen_cx + (su - cam->center.x) / cam->scale;
  *sy = screen_cy + (sv - cam->center.y) / cam->scale;
}

void viewport_screen_to_surface(const viewer_camera *cam, const viewport *vp,
                                  float sx, float sy, float *su, float *sv) {
  float screen_cx = vp->screen_w * 0.5f;
  float screen_cy = vp->screen_h * 0.5f;
  *su = cam->center.x + (sx - screen_cx) * cam->scale;
  *sv = cam->center.y + (sy - screen_cy) * cam->scale;
}
