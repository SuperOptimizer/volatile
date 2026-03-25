#pragma once
#include "core/math.h"

// ---------------------------------------------------------------------------
// line_seg_list — dynamic growable polyline for path tracing.
// ---------------------------------------------------------------------------

typedef struct {
  vec3f *points;
  int    count;
  int    capacity;
} line_seg_list;

line_seg_list *lineseg_new(void);
void           lineseg_free(line_seg_list *l);

// Append a point; grows capacity as needed.
void           lineseg_add(line_seg_list *l, vec3f point);

// Total arc length of the polyline (sum of segment lengths).
float          lineseg_length(const line_seg_list *l);

// Sample at arc-length fraction t in [0,1].
// t=0 returns the first point, t=1 the last.
// Returns (0,0,0) if the list is empty.
vec3f          lineseg_sample(const line_seg_list *l, float t);
