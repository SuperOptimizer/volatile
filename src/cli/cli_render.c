#define _POSIX_C_SOURCE 200809L

#include "cli_render.h"
#include "core/vol.h"
#include "core/geom.h"
#include "core/io.h"
#include "render/cmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void render_usage(void) {
  puts("usage: volatile render <surface_path> --volume <vol_path> --output <out.tiff>");
  puts("                       [--composite max] [--cmap viridis]");
  puts("                       [--layers-front N] [--layers-behind N]");
  puts("                       [--level N] [--width W] [--height H]");
  puts("");
  puts("  surface_path  path to a saved quad_surface, or 'plane:z=N' for a flat Z-slice");
  puts("  --composite   compositing mode: max (default), mean");
  puts("  --cmap        colormap: grayscale, viridis, magma, inferno, plasma (default: grayscale)");
  puts("  --layers-front  voxel layers to sample in front of surface (default 3)");
  puts("  --layers-behind voxel layers to sample behind surface (default 3)");
  puts("  --level       pyramid level to sample from (default 0)");
  puts("  --width/--height  output image dimensions (default: surface grid or 512x512)");
}

// ---------------------------------------------------------------------------
// Compositing helpers
// ---------------------------------------------------------------------------

typedef enum { COMP_MAX, COMP_MEAN } comp_mode;

// Sample `n_front + n_behind + 1` voxels along `normal` through `pt` and
// composite them into a single float value.
static float composite_along_normal(const volume *v, int level,
                                    float px, float py, float pz,
                                    float nx, float ny, float nz,
                                    int n_front, int n_behind,
                                    comp_mode mode) {
  float result = 0.0f;
  int   count  = 0;

  for (int d = -n_behind; d <= n_front; d++) {
    float sz = pz + (float)d * nz;
    float sy = py + (float)d * ny;
    float sx = px + (float)d * nx;
    float val = vol_sample(v, level, sz, sy, sx);

    if (mode == COMP_MAX) {
      if (count == 0 || val > result) result = val;
    } else {
      result += val;
    }
    count++;
  }

  if (mode == COMP_MEAN && count > 0) result /= (float)count;
  return result;
}

// ---------------------------------------------------------------------------
// Parse "plane:z=N" pseudo-path and build a flat quad_surface at z=N
// ---------------------------------------------------------------------------

static quad_surface *make_plane_surface(const char *spec, int rows, int cols,
                                        const volume *v, int level) {
  // spec = "plane:z=N"
  float z_val = 0.0f;
  const char *eq = strchr(spec, '=');
  if (eq) z_val = (float)atof(eq + 1);

  // Use volume shape to fill Y,X extents if available.
  float y_max = (float)(rows - 1), x_max = (float)(cols - 1);
  if (v) {
    const zarr_level_meta *m = vol_level_meta(v, level);
    if (m && m->ndim >= 3) {
      y_max = (float)(m->shape[m->ndim - 2] - 1);
      x_max = (float)(m->shape[m->ndim - 1] - 1);
    }
  }

  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      vec3f pt = {
        .x = x_max * (float)c / (float)(cols - 1),
        .y = y_max * (float)r / (float)(rows - 1),
        .z = z_val,
      };
      quad_surface_set(s, r, c, pt);
    }
  }
  quad_surface_compute_normals(s);
  return s;
}

// ---------------------------------------------------------------------------
// cmd_render
// ---------------------------------------------------------------------------

int cmd_render(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    render_usage();
    return argc < 1 ? 1 : 0;
  }

  const char *surface_path = argv[0];
  const char *vol_path     = NULL;
  const char *out_path     = NULL;
  const char *cmap_str     = "grayscale";
  const char *comp_str     = "max";
  int  layers_front        = 3;
  int  layers_behind       = 3;
  int  level               = 0;
  int  out_w               = 0;  // 0 = use surface grid
  int  out_h               = 0;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--volume") == 0 && i+1 < argc)        vol_path     = argv[++i];
    else if (strcmp(argv[i], "--output") == 0 && i+1 < argc)   out_path     = argv[++i];
    else if (strcmp(argv[i], "--cmap") == 0 && i+1 < argc)     cmap_str     = argv[++i];
    else if (strcmp(argv[i], "--composite") == 0 && i+1 < argc) comp_str    = argv[++i];
    else if (strcmp(argv[i], "--layers-front") == 0 && i+1 < argc) layers_front = atoi(argv[++i]);
    else if (strcmp(argv[i], "--layers-behind") == 0 && i+1 < argc) layers_behind = atoi(argv[++i]);
    else if (strcmp(argv[i], "--level") == 0 && i+1 < argc)    level        = atoi(argv[++i]);
    else if (strcmp(argv[i], "--width") == 0 && i+1 < argc)    out_w        = atoi(argv[++i]);
    else if (strcmp(argv[i], "--height") == 0 && i+1 < argc)   out_h        = atoi(argv[++i]);
  }

  if (!vol_path) { fputs("error: --volume required\n", stderr); return 1; }
  if (!out_path) { fputs("error: --output required\n", stderr); return 1; }

  // Parse colormap.
  cmap_id cmap = CMAP_GRAYSCALE;
  for (int i = 0; i < cmap_count(); i++) {
    if (strcmp(cmap_name((cmap_id)i), cmap_str) == 0) { cmap = (cmap_id)i; break; }
  }

  comp_mode mode = (strcmp(comp_str, "mean") == 0) ? COMP_MEAN : COMP_MAX;

  // Open volume.
  volume *v = vol_open(vol_path);
  if (!v) { fprintf(stderr, "error: cannot open volume: %s\n", vol_path); return 1; }

  // Load or generate surface.
  quad_surface *surf = NULL;
  int render_rows = out_h > 0 ? out_h : 512;
  int render_cols = out_w > 0 ? out_w : 512;

  if (strncmp(surface_path, "plane:", 6) == 0) {
    surf = make_plane_surface(surface_path, render_rows, render_cols, v, level);
  } else {
    // NOTE: Future: load quad_surface from a binary file.  For now, fall back
    // to plane:z=0 with a warning so the plumbing is exercisable end-to-end.
    fprintf(stderr, "warning: surface file loading not yet implemented; "
                    "using plane:z=0\n");
    surf = make_plane_surface("plane:z=0", render_rows, render_cols, v, level);
  }

  if (!surf) {
    fputs("error: could not create surface\n", stderr);
    vol_free(v);
    return 1;
  }

  // Compute normals if not already present.
  if (!surf->normals) quad_surface_compute_normals(surf);

  int rows = surf->rows;
  int cols = surf->cols;

  // Sample volume along the surface into a float buffer.
  float *samples = malloc((size_t)rows * (size_t)cols * sizeof(float));
  if (!samples) { quad_surface_free(surf); vol_free(v); fputs("error: oom\n", stderr); return 1; }

  float vmin = 1e38f, vmax = -1e38f;

  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      vec3f pt = quad_surface_get(surf, r, c);
      vec3f n  = surf->normals ? surf->normals[r * cols + c] : (vec3f){0,0,1};

      float val = composite_along_normal(v, level,
                                         pt.x, pt.y, pt.z,
                                         n.x,  n.y,  n.z,
                                         layers_front, layers_behind, mode);
      samples[r * cols + c] = val;
      if (val < vmin) vmin = val;
      if (val > vmax) vmax = val;
    }
  }

  // Normalise and apply colormap -> RGB uint8 image.
  float range = (vmax > vmin) ? (vmax - vmin) : 1.0f;

  uint8_t *pixels = malloc((size_t)rows * (size_t)cols * 3);
  if (!pixels) {
    free(samples); quad_surface_free(surf); vol_free(v);
    fputs("error: oom\n", stderr); return 1;
  }

  for (int i = 0; i < rows * cols; i++) {
    double norm = (double)(samples[i] - vmin) / (double)range;
    cmap_rgb rgb = cmap_apply(cmap, norm);
    pixels[i*3+0] = rgb.r;
    pixels[i*3+1] = rgb.g;
    pixels[i*3+2] = rgb.b;
  }

  // Write output TIFF.
  image out_img = {
    .width     = cols,
    .height    = rows,
    .depth     = 1,
    .channels  = 3,
    .dtype     = DTYPE_U8,
    .data      = pixels,
    .data_size = (size_t)rows * (size_t)cols * 3,
  };

  bool ok = tiff_write(out_path, &out_img);
  if (!ok) fprintf(stderr, "error: failed to write TIFF: %s\n", out_path);
  else     printf("rendered %dx%d → %s\n", cols, rows, out_path);

  free(pixels);
  free(samples);
  quad_surface_free(surf);
  vol_free(v);
  return ok ? 0 : 1;
}
