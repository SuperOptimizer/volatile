#define _POSIX_C_SOURCE 200809L

#include "cli_mask.h"
#include "core/geom.h"
#include "core/vol.h"
#include "core/io.h"
#include "core/math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void mask_usage(void) {
  puts("usage: volatile mask <surface_path> --volume <vol_path> --output <mask.zarr>");
  puts("                     [--radius N] [--level N]");
  puts("");
  puts("  Generate a binary mask volume from a surface.");
  puts("  Voxels within --radius voxels of the surface are set to 1, others 0.");
  puts("  surface_path  path to a saved quad_surface, or 'plane:z=N'");
  puts("  --radius      distance threshold in voxels (default: 3)");
  puts("  --level       pyramid level to use for volume shape (default: 0)");
}

// ---------------------------------------------------------------------------
// cmd_mask
// ---------------------------------------------------------------------------

int cmd_mask(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    mask_usage();
    return argc < 1 ? 1 : 0;
  }

  const char *surface_path = argv[0];
  const char *vol_path     = NULL;
  const char *out_path     = NULL;
  float       radius       = 3.0f;
  int         level        = 0;

  for (int i = 1; i < argc; i++) {
    if      (strcmp(argv[i], "--volume") == 0 && i+1 < argc) vol_path = argv[++i];
    else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--radius") == 0 && i+1 < argc) radius   = (float)atof(argv[++i]);
    else if (strcmp(argv[i], "--level")  == 0 && i+1 < argc) level    = atoi(argv[++i]);
  }

  if (!vol_path) { fputs("error: --volume required\n", stderr); return 1; }
  if (!out_path) { fputs("error: --output required\n", stderr); return 1; }

  volume *v = vol_open(vol_path);
  if (!v) { fprintf(stderr, "error: cannot open volume: %s\n", vol_path); return 1; }

  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m) {
    fprintf(stderr, "error: level %d not available\n", level);
    vol_free(v);
    return 1;
  }

  // Determine volume shape (Z, Y, X from last 3 dims).
  int ndim = m->ndim;
  if (ndim < 3) {
    fputs("error: volume must have at least 3 dimensions\n", stderr);
    vol_free(v);
    return 1;
  }
  int depth  = (int)m->shape[ndim - 3];
  int height = (int)m->shape[ndim - 2];
  int width  = (int)m->shape[ndim - 1];
  size_t nvox = (size_t)depth * height * width;

  // Build or load surface.
  quad_surface *surf = NULL;
  if (strncmp(surface_path, "plane:", 6) == 0) {
    float z_val = 0.0f;
    const char *eq = strchr(surface_path, '=');
    if (eq) z_val = (float)atof(eq + 1);
    int rows = height, cols = width;
    surf = quad_surface_new(rows, cols);
    if (!surf) { vol_free(v); fputs("error: oom\n", stderr); return 1; }
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        quad_surface_set(surf, r, c, (vec3f){(float)c, (float)r, z_val});
  } else {
    fprintf(stderr, "warning: surface file loading not yet implemented; "
                    "using plane:z=0\n");
    surf = quad_surface_new(height, width);
    if (!surf) { vol_free(v); fputs("error: oom\n", stderr); return 1; }
    for (int r = 0; r < height; r++)
      for (int c = 0; c < width; c++)
        quad_surface_set(surf, r, c, (vec3f){(float)c, (float)r, 0.0f});
  }

  // Allocate mask buffer.
  uint8_t *mask = calloc(nvox, 1);
  if (!mask) {
    quad_surface_free(surf); vol_free(v);
    fputs("error: oom\n", stderr); return 1;
  }

  float r2 = radius * radius;

  // For each surface point, mark all voxels within radius.
  for (int sr = 0; sr < surf->rows; sr++) {
    for (int sc = 0; sc < surf->cols; sc++) {
      vec3f pt = quad_surface_get(surf, sr, sc);
      int iz0 = (int)(pt.z - radius) - 1;
      int iz1 = (int)(pt.z + radius) + 1;
      int iy0 = (int)(pt.y - radius) - 1;
      int iy1 = (int)(pt.y + radius) + 1;
      int ix0 = (int)(pt.x - radius) - 1;
      int ix1 = (int)(pt.x + radius) + 1;
      if (iz0 < 0) iz0 = 0;  if (iz1 >= depth)  iz1 = depth  - 1;
      if (iy0 < 0) iy0 = 0;  if (iy1 >= height) iy1 = height - 1;
      if (ix0 < 0) ix0 = 0;  if (ix1 >= width)  ix1 = width  - 1;

      for (int iz = iz0; iz <= iz1; iz++) {
        float dz = (float)iz - pt.z;
        for (int iy = iy0; iy <= iy1; iy++) {
          float dy = (float)iy - pt.y;
          for (int ix = ix0; ix <= ix1; ix++) {
            float dx = (float)ix - pt.x;
            if (dz*dz + dy*dy + dx*dx <= r2)
              mask[(size_t)iz * height * width + (size_t)iy * width + ix] = 1;
          }
        }
      }
    }
  }

  // Count marked voxels.
  size_t marked = 0;
  for (size_t i = 0; i < nvox; i++) marked += mask[i];

  // Write mask as a raw uint8 image (single slice output for now — write to
  // a PPM/binary for portability since we don't have zarr write yet).
  // NOTE: Real zarr write would use vol_write; for now emit a flat binary file
  // with a simple header so the output is usable.
  FILE *f = fopen(out_path, "wb");
  if (!f) {
    fprintf(stderr, "error: cannot write: %s\n", out_path);
    free(mask); quad_surface_free(surf); vol_free(v);
    return 1;
  }
  // Header: magic + dimensions (little-endian).
  uint8_t hdr[16];
  memcpy(hdr, "VMSK", 4);
  hdr[4]  = (uint8_t)(depth  & 0xFF); hdr[5]  = (uint8_t)((depth  >> 8) & 0xFF);
  hdr[6]  = (uint8_t)(height & 0xFF); hdr[7]  = (uint8_t)((height >> 8) & 0xFF);
  hdr[8]  = (uint8_t)(width  & 0xFF); hdr[9]  = (uint8_t)((width  >> 8) & 0xFF);
  hdr[10] = hdr[11] = hdr[12] = hdr[13] = hdr[14] = hdr[15] = 0;
  fwrite(hdr, 1, 16, f);
  fwrite(mask, 1, nvox, f);
  fclose(f);

  printf("mask: %d x %d x %d  → %s\n", width, height, depth, out_path);
  printf("marked voxels: %zu / %zu  (%.2f%%)\n",
         marked, nvox, 100.0 * (double)marked / (double)nvox);

  free(mask);
  quad_surface_free(surf);
  vol_free(v);
  return 0;
}
