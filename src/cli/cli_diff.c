#define _POSIX_C_SOURCE 200809L

#include "cli/cli_diff.h"
#include "core/geom.h"
#include "core/io.h"
#include "render/cmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Surface loader stub — identical to cli_render.c approach
// NOTE: Real surface file I/O will replace this when quad_surface serialisation
// is implemented. For now we synthesize a flat surface so the tool compiles
// and the stats pipeline is exercisable end-to-end.
// ---------------------------------------------------------------------------

static quad_surface *load_surface(const char *path, int rows, int cols) {
  (void)path;
  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;
  // Flat plane at z=0 as stand-in.
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      quad_surface_set(s, r, c,
        (vec3f){ (float)c, (float)r, 0.0f });
  quad_surface_compute_normals(s);
  return s;
}

// ---------------------------------------------------------------------------
// cmd_diff
// ---------------------------------------------------------------------------

int cmd_diff(int argc, char **argv) {
  if (argc < 2) {
    fputs("usage: volatile diff <surface1> <surface2> [--output <diff.tiff>]\n", stderr);
    return 1;
  }

  const char *path1  = argv[0];
  const char *path2  = argv[1];
  const char *out    = NULL;

  for (int i = 2; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) out = argv[i + 1];
  }

  // Load both surfaces at the same grid resolution.
  const int rows = 512, cols = 512;
  quad_surface *s1 = load_surface(path1, rows, cols);
  quad_surface *s2 = load_surface(path2, rows, cols);

  if (!s1 || !s2) {
    fputs("error: could not load surfaces\n", stderr);
    quad_surface_free(s1); quad_surface_free(s2); return 1;
  }

  if (s1->rows != s2->rows || s1->cols != s2->cols) {
    fputs("error: surfaces have different grid dimensions\n", stderr);
    quad_surface_free(s1); quad_surface_free(s2); return 1;
  }

  int n = s1->rows * s1->cols;
  float *dists = malloc((size_t)n * sizeof(float));
  if (!dists) {
    quad_surface_free(s1); quad_surface_free(s2); return 1;
  }

  // Per-vertex Euclidean distance.
  float dmin = FLT_MAX, dmax = 0.0f, dsum = 0.0f, dsum2 = 0.0f;
  for (int i = 0; i < n; i++) {
    vec3f a = s1->points[i];
    vec3f b = s2->points[i];
    float d = vec3f_len(vec3f_sub(a, b));
    dists[i] = d;
    if (d < dmin) dmin = d;
    if (d > dmax) dmax = d;
    dsum  += d;
    dsum2 += d * d;
  }

  float dmean = dsum / (float)n;
  float drms  = sqrtf(dsum2 / (float)n);

  printf("surface diff: %s  vs  %s\n", path1, path2);
  printf("  vertices : %d\n", n);
  printf("  min dist : %.6g\n", (double)dmin);
  printf("  max dist : %.6g\n", (double)dmax);
  printf("  mean     : %.6g\n", (double)dmean);
  printf("  RMS      : %.6g\n", (double)drms);

  // Optionally write colorized TIFF.
  if (out) {
    float range = (dmax > dmin) ? (dmax - dmin) : 1.0f;
    uint8_t *pixels = malloc((size_t)n * 3);
    if (pixels) {
      for (int i = 0; i < n; i++) {
        double norm = (double)(dists[i] - dmin) / (double)range;
        cmap_rgb rgb = cmap_apply(CMAP_VIRIDIS, norm);
        pixels[i*3+0] = rgb.r;
        pixels[i*3+1] = rgb.g;
        pixels[i*3+2] = rgb.b;
      }
      image img = {
        .width = s1->cols, .height = s1->rows, .depth = 1,
        .channels = 3, .dtype = DTYPE_U8,
        .data = pixels, .data_size = (size_t)n * 3,
      };
      if (tiff_write(out, &img))
        printf("  diff map : %s\n", out);
      else
        fprintf(stderr, "warning: failed to write diff map: %s\n", out);
      free(pixels);
    }
  }

  free(dists);
  quad_surface_free(s1);
  quad_surface_free(s2);
  return 0;
}
