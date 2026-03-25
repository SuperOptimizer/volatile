#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "gui/slice_controller.h"
#include "gui/viewer.h"
#include "core/math.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

#define DEBOUNCE_MS 200.0f

struct slice_controller {
  slice_viewer *viewer;

  // Current orientation as a 4x4 rotation matrix (we use mat4f for the
  // axis-angle API then extract the upper-left 3x3 for callers).
  mat4f rot;

  // Mouse drag state
  bool  dragging;
  float drag_x;     // last known drag position
  float drag_y;

  // Accumulated pending angle (degrees) not yet flushed to the viewer.
  float pending_deg;
  float quiet_ms;   // time since last drag event; flush after DEBOUNCE_MS
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build an axis-aligned rotation matrix from scratch for `axis`.
static mat4f make_axis_rot(int axis) {
  // For XY (0): identity — u=X, v=Y, n=Z
  // For XZ (1): rotate -90° around X so that u=X, v=Z, n=Y
  // For YZ (2): rotate +90° around Y so that u=Z, v=Y, n=X
  switch (axis) {
    case 1: {
      vec3f ax = {1.0f, 0.0f, 0.0f};
      return mat4f_rotate(ax, -(float)M_PI / 2.0f);
    }
    case 2: {
      vec3f ay = {0.0f, 1.0f, 0.0f};
      return mat4f_rotate(ay, (float)M_PI / 2.0f);
    }
    default:
      return mat4f_identity();
  }
}

// Extract the view-plane normal from the current rotation matrix.
// The normal is the third column (Z-axis after rotation).
static vec3f current_normal(const slice_controller *c) {
  return (vec3f){c->rot.m[8], c->rot.m[9], c->rot.m[10]};
}

// Apply all pending rotation to the stored matrix and reset accumulator.
static void flush_pending(slice_controller *c) {
  if (fabsf(c->pending_deg) < 1e-5f) return;
  vec3f n = current_normal(c);
  float rad = c->pending_deg * (float)(M_PI / 180.0);
  mat4f delta = mat4f_rotate(n, rad);
  c->rot = mat4f_mul(delta, c->rot);
  c->pending_deg = 0.0f;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

slice_controller *slice_controller_new(slice_viewer *viewer) {
  slice_controller *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->viewer = viewer;
  // default: XY plane (normal = Z)
  c->rot = mat4f_identity();
  // seed from viewer's current axis if available
  if (viewer) {
    int ax = viewer_get_axis(viewer);
    c->rot = make_axis_rot(ax);
  }
  return c;
}

void slice_controller_free(slice_controller *c) {
  free(c);
}

void slice_controller_set_axis(slice_controller *c, int axis) {
  if (!c) return;
  c->rot = make_axis_rot(axis);
  c->pending_deg = 0.0f;
  c->quiet_ms    = 0.0f;
}

void slice_controller_rotate(slice_controller *c, float angle_degrees) {
  if (!c) return;
  c->pending_deg += angle_degrees;
  c->quiet_ms    = 0.0f;  // reset debounce timer
}

// Mouse: horizontal drag → rotation angle proportional to pixel distance.
// 360° per ~500 px feels natural (0.72 deg/px).
#define DEG_PER_PX 0.72f

void slice_controller_on_mouse_down(slice_controller *c, float x, float y) {
  if (!c) return;
  c->dragging = true;
  c->drag_x   = x;
  c->drag_y   = y;
}

void slice_controller_on_mouse_drag(slice_controller *c, float x, float y) {
  if (!c || !c->dragging) return;
  float dx = x - c->drag_x;
  // horizontal drag → in-plane rotation around the slice normal
  slice_controller_rotate(c, dx * DEG_PER_PX);
  c->drag_x = x;
  c->drag_y = y;
}

void slice_controller_on_mouse_up(slice_controller *c) {
  if (!c) return;
  c->dragging = false;
}

void slice_controller_get_transform(const slice_controller *c, float *mat3x3_out) {
  if (!c || !mat3x3_out) return;
  // Include any unflushed pending rotation in the output but don't store it yet.
  mat4f r = c->rot;
  if (fabsf(c->pending_deg) > 1e-5f) {
    vec3f n = current_normal(c);
    float rad = c->pending_deg * (float)(M_PI / 180.0);
    mat4f delta = mat4f_rotate(n, rad);
    r = mat4f_mul(delta, r);
  }
  mat3f m3 = mat3f_from_mat4(r);
  memcpy(mat3x3_out, m3.m, 9 * sizeof(float));
}

void slice_controller_tick(slice_controller *c, float dt_ms) {
  if (!c) return;
  if (fabsf(c->pending_deg) < 1e-5f) return;

  c->quiet_ms += dt_ms;
  if (c->quiet_ms >= DEBOUNCE_MS) {
    flush_pending(c);
    c->quiet_ms = 0.0f;
    // notify viewer (axis hasn't changed, but a re-sample may be needed)
    // We use the existing set_axis pathway — pass the closest axis or -1.
    // For now just signal by keeping the matrix; the viewer polls via
    // slice_controller_get_transform() each frame anyway.
    (void)c->viewer;  // future: viewer_set_slice_transform(c->viewer, ...)
  }
}
