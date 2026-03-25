#define _POSIX_C_SOURCE 200809L

#include "cli_metrics.h"
#include "core/geom.h"
#include "core/io.h"
#include "core/math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void metrics_usage(void) {
  puts("usage: volatile metrics <surface_path>");
  puts("                        [--level N]");
  puts("");
  puts("  Compute and print surface area, mean/max curvature, and smoothness.");
  puts("  surface_path  path to a saved quad_surface, or 'plane:z=N'");
}

// ---------------------------------------------------------------------------
// Curvature + smoothness helpers
// ---------------------------------------------------------------------------

// Discrete mean curvature at (r,c): average angle between normal and
// the normals of the 4-connected neighbors (in radians, 0 = flat).
static float curvature_at(const quad_surface *s, int r, int c) {
  if (!s->normals) return 0.0f;
  int rows = s->rows, cols = s->cols;
  vec3f n0 = s->normals[r * cols + c];

  float sum = 0.0f;
  int   cnt = 0;
  int dr[4] = {-1, 1,  0, 0};
  int dc[4] = { 0, 0, -1, 1};

  for (int d = 0; d < 4; d++) {
    int nr2 = r + dr[d], nc2 = c + dc[d];
    if (nr2 < 0 || nr2 >= rows || nc2 < 0 || nc2 >= cols) continue;
    vec3f n1 = s->normals[nr2 * cols + nc2];
    float dot = n0.x*n1.x + n0.y*n1.y + n0.z*n1.z;
    // clamp for numerical safety
    if (dot >  1.0f) dot =  1.0f;
    if (dot < -1.0f) dot = -1.0f;
    sum += acosf(dot);
    cnt++;
  }
  return cnt > 0 ? sum / (float)cnt : 0.0f;
}

// Smoothness: RMS of point-to-neighbor distance variance.
// A perfectly flat grid gives 0; bumpy surfaces give higher values.
static float compute_smoothness(const quad_surface *s) {
  int rows = s->rows, cols = s->cols;
  double sum_sq = 0.0;
  long   cnt    = 0;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      vec3f p = quad_surface_get(s, r, c);
      // Gather neighbor positions and their mean.
      vec3f nbs[4];
      int nn = 0;
      int dr[4] = {-1, 1,  0, 0};
      int dc[4] = { 0, 0, -1, 1};
      for (int d = 0; d < 4; d++) {
        int nr2 = r + dr[d], nc2 = c + dc[d];
        if (nr2 >= 0 && nr2 < rows && nc2 >= 0 && nc2 < cols)
          nbs[nn++] = quad_surface_get(s, nr2, nc2);
      }
      if (nn == 0) continue;
      vec3f mean = {0,0,0};
      for (int i = 0; i < nn; i++) {
        mean.x += nbs[i].x; mean.y += nbs[i].y; mean.z += nbs[i].z;
      }
      mean.x /= nn; mean.y /= nn; mean.z /= nn;
      // Laplacian displacement: p - mean(neighbors)
      double dx = p.x - mean.x, dy = p.y - mean.y, dz = p.z - mean.z;
      sum_sq += dx*dx + dy*dy + dz*dz;
      cnt++;
    }
  }
  return cnt > 0 ? (float)sqrt(sum_sq / (double)cnt) : 0.0f;
}

// ---------------------------------------------------------------------------
// cmd_metrics
// ---------------------------------------------------------------------------

int cmd_metrics(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    metrics_usage();
    return argc < 1 ? 1 : 0;
  }

  const char *surface_path = argv[0];

  // NOTE: surface file loading not yet implemented; build a synthetic plane
  // so the metrics pipeline is exercisable end-to-end.
  quad_surface *surf = NULL;

  if (strncmp(surface_path, "plane:", 6) == 0) {
    float z_val = 0.0f;
    const char *eq = strchr(surface_path, '=');
    if (eq) z_val = (float)atof(eq + 1);

    int rows = 64, cols = 64;
    surf = quad_surface_new(rows, cols);
    if (!surf) { fputs("error: oom\n", stderr); return 1; }
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        quad_surface_set(surf, r, c, (vec3f){(float)c, (float)r, z_val});
  } else {
    fprintf(stderr, "warning: surface file loading not yet implemented; "
                    "using plane:z=0\n");
    int rows = 64, cols = 64;
    surf = quad_surface_new(rows, cols);
    if (!surf) { fputs("error: oom\n", stderr); return 1; }
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        quad_surface_set(surf, r, c, (vec3f){(float)c, (float)r, 0.0f});
  }

  quad_surface_compute_normals(surf);

  // --- Area ---
  float area = quad_surface_area(surf);

  // --- Curvature ---
  float curv_mean = 0.0f, curv_max = 0.0f;
  long  curv_cnt  = 0;
  for (int r = 0; r < surf->rows; r++) {
    for (int c = 0; c < surf->cols; c++) {
      float k = curvature_at(surf, r, c);
      curv_mean += k;
      if (k > curv_max) curv_max = k;
      curv_cnt++;
    }
  }
  if (curv_cnt > 0) curv_mean /= (float)curv_cnt;

  // --- Smoothness ---
  float smoothness = compute_smoothness(surf);

  // --- Print ---
  printf("surface:    %s\n", surface_path);
  printf("grid:       %d x %d  (%d points)\n", surf->cols, surf->rows, surf->rows * surf->cols);
  printf("area:       %.4f voxel^2\n", (double)area);
  printf("curvature:  mean=%.6f rad  max=%.6f rad\n", (double)curv_mean, (double)curv_max);
  printf("smoothness: %.6f (RMS Laplacian displacement)\n", (double)smoothness);

  quad_surface_free(surf);
  return 0;
}
