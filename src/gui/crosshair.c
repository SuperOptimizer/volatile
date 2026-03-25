#include "crosshair.h"
#include "viewer.h"
#include "render/overlay.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_VIEWERS 16

// Crosshair colors matching VC3D convention
#define COL_XY_R 255u
#define COL_XY_G 165u
#define COL_XY_B   0u   // orange

#define COL_XZ_R 255u
#define COL_XZ_G  50u
#define COL_XZ_B  50u   // red

#define COL_YZ_R 255u
#define COL_YZ_G 220u
#define COL_YZ_B   0u   // yellow

// Line extent in normalised surface units (large enough to span any view)
#define LINE_EXTENT 1e6f

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

struct crosshair_sync {
  slice_viewer *viewers[MAX_VIEWERS];
  int           num_viewers;
  vec3f         focus;          // current 3D world focus point
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

crosshair_sync *crosshair_sync_new(void) {
  crosshair_sync *s = calloc(1, sizeof(crosshair_sync));
  return s;
}

void crosshair_sync_free(crosshair_sync *s) {
  free(s);
}

// ---------------------------------------------------------------------------
// Viewer registry
// ---------------------------------------------------------------------------

void crosshair_sync_add_viewer(crosshair_sync *s, slice_viewer *v) {
  if (!s || !v || s->num_viewers >= MAX_VIEWERS) return;
  // avoid duplicates
  for (int i = 0; i < s->num_viewers; i++) {
    if (s->viewers[i] == v) return;
  }
  s->viewers[s->num_viewers++] = v;
}

void crosshair_sync_remove_viewer(crosshair_sync *s, slice_viewer *v) {
  if (!s || !v) return;
  for (int i = 0; i < s->num_viewers; i++) {
    if (s->viewers[i] == v) {
      // compact array
      s->viewers[i] = s->viewers[--s->num_viewers];
      s->viewers[s->num_viewers] = NULL;
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Focus point
// ---------------------------------------------------------------------------

void crosshair_sync_set_focus(crosshair_sync *s, vec3f world_pos) {
  if (!s) return;
  s->focus = world_pos;
}

vec3f crosshair_sync_get_focus(const crosshair_sync *s) {
  if (!s) return (vec3f){0, 0, 0};
  return s->focus;
}

// ---------------------------------------------------------------------------
// Overlay generation
//
// Each viewer shows two lines through the focus point:
//   one parallel to the horizontal axis (constant "row" coordinate)
//   one parallel to the vertical axis  (constant "column" coordinate)
//
// Surface coordinates for overlay_add_line are in screen-space pixels,
// but overlay.h works in pixel coords directly. We pass large absolute
// values spanning the full viewport so the lines always cross it.
//
// Axis mapping:
//   axis=0 (XY): horizontal axis = X, vertical axis = Y
//     vertical line  at x = focus.x   (spans full height)
//     horizontal line at y = focus.y  (spans full width)
//   axis=1 (XZ): horizontal axis = X, vertical axis = Z
//     vertical line  at x = focus.x
//     horizontal line at z = focus.z
//   axis=2 (YZ): horizontal axis = Y, vertical axis = Z
//     vertical line  at y = focus.y
//     horizontal line at z = focus.z
// ---------------------------------------------------------------------------

void crosshair_sync_render_overlays(const crosshair_sync *s, slice_viewer *v, overlay_list *out) {
  if (!s || !v || !out) return;

  int axis = viewer_get_axis(v);

  // Select color and which focus components to use
  uint8_t r, g, b;
  float horiz_pos, vert_pos;

  switch (axis) {
    case 1:  // XZ
      r = COL_XZ_R; g = COL_XZ_G; b = COL_XZ_B;
      vert_pos  = s->focus.x;
      horiz_pos = s->focus.z;
      break;
    case 2:  // YZ
      r = COL_YZ_R; g = COL_YZ_G; b = COL_YZ_B;
      vert_pos  = s->focus.y;
      horiz_pos = s->focus.z;
      break;
    default: // 0: XY
      r = COL_XY_R; g = COL_XY_G; b = COL_XY_B;
      vert_pos  = s->focus.x;
      horiz_pos = s->focus.y;
      break;
  }

  // Vertical line: x = vert_pos, spanning full height
  overlay_add_line(out,
    vert_pos, -LINE_EXTENT,
    vert_pos,  LINE_EXTENT,
    r, g, b, 1.0f);

  // Horizontal line: y = horiz_pos, spanning full width
  overlay_add_line(out,
    -LINE_EXTENT, horiz_pos,
     LINE_EXTENT, horiz_pos,
    r, g, b, 1.0f);
}
