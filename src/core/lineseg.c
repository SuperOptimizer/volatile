#include "core/lineseg.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define LINESEG_INIT_CAP 16

line_seg_list *lineseg_new(void) {
  line_seg_list *l = calloc(1, sizeof(*l));
  REQUIRE(l, "lineseg_new: calloc failed");
  l->points   = malloc(LINESEG_INIT_CAP * sizeof(vec3f));
  REQUIRE(l->points, "lineseg_new: malloc failed");
  l->capacity = LINESEG_INIT_CAP;
  return l;
}

void lineseg_free(line_seg_list *l) {
  if (!l) return;
  free(l->points);
  free(l);
}

void lineseg_add(line_seg_list *l, vec3f point) {
  REQUIRE(l, "lineseg_add: null list");
  if (l->count == l->capacity) {
    int   newcap = l->capacity * 2;
    vec3f *tmp   = realloc(l->points, (size_t)newcap * sizeof(vec3f));
    REQUIRE(tmp, "lineseg_add: realloc failed");
    l->points   = tmp;
    l->capacity = newcap;
  }
  l->points[l->count++] = point;
}

float lineseg_length(const line_seg_list *l) {
  if (!l || l->count < 2) return 0.0f;
  float total = 0.0f;
  for (int i = 1; i < l->count; i++)
    total += vec3f_len(vec3f_sub(l->points[i], l->points[i-1]));
  return total;
}

vec3f lineseg_sample(const line_seg_list *l, float t) {
  if (!l || l->count == 0) return (vec3f){0, 0, 0};
  if (l->count == 1)       return l->points[0];

  if (t <= 0.0f) return l->points[0];
  if (t >= 1.0f) return l->points[l->count - 1];

  float total  = lineseg_length(l);
  if (total <= 0.0f) return l->points[0];

  float target = t * total;
  float walked = 0.0f;

  for (int i = 1; i < l->count; i++) {
    float seg = vec3f_len(vec3f_sub(l->points[i], l->points[i-1]));
    if (walked + seg >= target) {
      float local_t = (seg > 0.0f) ? (target - walked) / seg : 0.0f;
      return (vec3f){
        l->points[i-1].x + local_t * (l->points[i].x - l->points[i-1].x),
        l->points[i-1].y + local_t * (l->points[i].y - l->points[i-1].y),
        l->points[i-1].z + local_t * (l->points[i].z - l->points[i-1].z),
      };
    }
    walked += seg;
  }
  return l->points[l->count - 1];
}
