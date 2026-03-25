#include "core/umbilicus.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compare vec3f by z for qsort.
static int cmp_z(const void *a, const void *b) {
  float za = ((const vec3f *)a)->z;
  float zb = ((const vec3f *)b)->z;
  return (za > zb) - (za < zb);
}

// Linear interpolation of the XY center at a given z, clamped to the
// range of control points.
static vec3f interp_center(const vec3f *ctrl, int nctrl, float z) {
  if (nctrl == 1)
    return (vec3f){ ctrl[0].x, ctrl[0].y, z };

  if (z <= ctrl[0].z)
    return (vec3f){ ctrl[0].x, ctrl[0].y, z };
  if (z >= ctrl[nctrl - 1].z)
    return (vec3f){ ctrl[nctrl-1].x, ctrl[nctrl-1].y, z };

  // Binary search for bracketing segment
  int lo = 0, hi = nctrl - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (ctrl[mid].z <= z) lo = mid; else hi = mid;
  }

  float z0 = ctrl[lo].z, z1 = ctrl[hi].z;
  float t  = (z1 - z0) > 1e-6f ? (z - z0) / (z1 - z0) : 0.0f;
  return (vec3f){
    ctrl[lo].x + t * (ctrl[hi].x - ctrl[lo].x),
    ctrl[lo].y + t * (ctrl[hi].y - ctrl[lo].y),
    z
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

umbilicus *umbilicus_from_points(const vec3f *ctrl, int nctrl, int depth) {
  if (!ctrl || nctrl <= 0 || depth <= 0) return NULL;

  // Sort a working copy by z
  vec3f *sorted = malloc((size_t)nctrl * sizeof(vec3f));
  if (!sorted) return NULL;
  memcpy(sorted, ctrl, (size_t)nctrl * sizeof(vec3f));
  qsort(sorted, (size_t)nctrl, sizeof(vec3f), cmp_z);

  umbilicus *u = malloc(sizeof(umbilicus));
  if (!u) { free(sorted); return NULL; }

  u->count  = depth;
  u->points = malloc((size_t)depth * sizeof(vec3f));
  if (!u->points) { free(sorted); free(u); return NULL; }

  for (int z = 0; z < depth; z++)
    u->points[z] = interp_center(sorted, nctrl, (float)z);

  free(sorted);
  return u;
}

void umbilicus_free(umbilicus *u) {
  if (!u) return;
  free(u->points);
  free(u);
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

// Clamp z to [0, count-1] and return the interpolated center.
static vec3f center_at_z(const umbilicus *u, float z) {
  int idx = (int)(z + 0.5f);
  if (idx < 0)           idx = 0;
  if (idx >= u->count)   idx = u->count - 1;
  return u->points[idx];
}

float umbilicus_distance(const umbilicus *u, vec3f point) {
  if (!u || u->count == 0) return 0.0f;
  vec3f c = center_at_z(u, point.z);
  float dx = point.x - c.x;
  float dy = point.y - c.y;
  return sqrtf(dx*dx + dy*dy);
}

float umbilicus_winding_angle(const umbilicus *u, vec3f point) {
  if (!u || u->count == 0) return 0.0f;
  vec3f c   = center_at_z(u, point.z);
  float dx  = point.x - c.x;
  float dy  = point.y - c.y;
  float ang = atan2f(dy, dx) * (180.0f / (float)M_PI);
  if (ang < 0.0f) ang += 360.0f;
  return ang;
}
