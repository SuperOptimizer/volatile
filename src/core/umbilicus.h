#pragma once
#include "core/math.h"
#include <stddef.h>

// ---------------------------------------------------------------------------
// umbilicus — scroll center axis, ported from VC3D Umbilicus.cpp
//
// Built from a set of (x,y,z) control points sorted by z.  The center is
// linearly interpolated between control points so every z slice has one.
// ---------------------------------------------------------------------------

typedef struct {
  vec3f *points;    // one center per z slice (index 0 = z=0)
  int    count;     // number of z slices (= volume depth at the chosen level)
} umbilicus;

// Build from explicit control points (array of count vec3f, sorted by z).
// Returns NULL on allocation failure or empty input.
umbilicus *umbilicus_from_points(const vec3f *ctrl, int nctrl, int depth);

// Free.
void umbilicus_free(umbilicus *u);

// XY distance from point to the interpolated center at point.z.
float umbilicus_distance(const umbilicus *u, vec3f point);

// Winding angle (degrees, 0-360) around the scroll center, measured in
// the XY plane relative to the +X axis.  Returns 0 if u is NULL or empty.
float umbilicus_winding_angle(const umbilicus *u, vec3f point);
