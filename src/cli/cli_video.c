#define _POSIX_C_SOURCE 200809L

#include "cli/cli_video.h"
#include "core/geom.h"
#include "core/io.h"
#include "core/vol.h"
#include "render/cmap.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Surface loader stub (same pattern as cli_render.c)
// ---------------------------------------------------------------------------

static quad_surface *load_surface(const char *path, int rows, int cols,
                                  const volume *v, int level) {
  // plane:z=N shortcut
  if (strncmp(path, "plane:", 6) == 0) {
    float z_val = 0.0f;
    const char *eq = strchr(path, '=');
    if (eq) z_val = (float)atof(eq + 1);

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
    for (int r = 0; r < rows; r++)
      for (int c = 0; c < cols; c++)
        quad_surface_set(s, r, c, (vec3f){
          x_max * (float)c / (float)(cols - 1),
          y_max * (float)r / (float)(rows - 1),
          z_val,
        });
    quad_surface_compute_normals(s);
    return s;
  }

  // Generic path: stub flat plane at z=0 (replace once surface I/O exists).
  (void)v; (void)level;
  fprintf(stderr, "warning: surface file loading not yet implemented; using plane:z=0\n");
  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      quad_surface_set(s, r, c, (vec3f){ (float)c, (float)r, 0.0f });
  quad_surface_compute_normals(s);
  return s;
}

// ---------------------------------------------------------------------------
// Write one PPM frame to `fp` (raw RGB, P6 binary format)
// ---------------------------------------------------------------------------

static void write_ppm_frame(FILE *fp, const uint8_t *pixels, int w, int h) {
  fprintf(fp, "P6\n%d %d\n255\n", w, h);
  fwrite(pixels, 3, (size_t)(w * h), fp);
}

// ---------------------------------------------------------------------------
// cmd_video
// ---------------------------------------------------------------------------

int cmd_video(int argc, char **argv) {
  if (argc < 1) {
    fputs("usage: volatile video <surface> --volume <vol> --output <out.mp4>\n"
          "                      [--slices N] [--level L] [--cmap NAME]\n"
          "                      [--width W] [--height H]\n", stderr);
    return 1;
  }

  const char *surface_path = argv[0];
  const char *vol_path     = NULL;
  const char *out_path     = NULL;
  const char *cmap_str     = "grayscale";
  int slices  = 100;
  int level   = 0;
  int out_w   = 512;
  int out_h   = 512;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--volume") == 0 && i+1 < argc)  vol_path  = argv[++i];
    else if (strcmp(argv[i], "--output") == 0 && i+1 < argc)  out_path = argv[++i];
    else if (strcmp(argv[i], "--slices") == 0 && i+1 < argc)  slices   = atoi(argv[++i]);
    else if (strcmp(argv[i], "--level")  == 0 && i+1 < argc)  level    = atoi(argv[++i]);
    else if (strcmp(argv[i], "--cmap")   == 0 && i+1 < argc)  cmap_str = argv[++i];
    else if (strcmp(argv[i], "--width")  == 0 && i+1 < argc)  out_w    = atoi(argv[++i]);
    else if (strcmp(argv[i], "--height") == 0 && i+1 < argc)  out_h    = atoi(argv[++i]);
  }

  if (!vol_path) { fputs("error: --volume required\n", stderr); return 1; }
  if (!out_path) { fputs("error: --output required\n", stderr); return 1; }
  if (slices < 1) slices = 1;

  // Resolve colormap.
  cmap_id cmap = CMAP_GRAYSCALE;
  for (int i = 0; i < cmap_count(); i++) {
    if (strcmp(cmap_name((cmap_id)i), cmap_str) == 0) { cmap = (cmap_id)i; break; }
  }

  volume *v = vol_open(vol_path);
  if (!v) { fprintf(stderr, "error: cannot open volume: %s\n", vol_path); return 1; }

  quad_surface *base = load_surface(surface_path, out_h, out_w, v, level);
  if (!base) {
    fputs("error: could not create surface\n", stderr);
    vol_free(v); return 1;
  }
  if (!base->normals) quad_surface_compute_normals(base);

  int n = out_h * out_w;
  uint8_t *pixels = malloc((size_t)n * 3);
  if (!pixels) {
    quad_surface_free(base); vol_free(v);
    fputs("error: oom\n", stderr); return 1;
  }

  // Try to pipe to ffmpeg; fall back to writing PPM frames to stdout or files.
  bool use_ffmpeg = false;
  FILE *pipe_fp   = NULL;

  char ffmpeg_cmd[1024];
  snprintf(ffmpeg_cmd, sizeof(ffmpeg_cmd),
    "ffmpeg -y -f rawvideo -pixel_format rgb24 -video_size %dx%d "
    "-framerate 25 -i pipe:0 -c:v libx264 -pix_fmt yuv420p \"%s\" 2>/dev/null",
    out_w, out_h, out_path);

  pipe_fp = popen(ffmpeg_cmd, "w");
  if (pipe_fp) {
    use_ffmpeg = true;
  } else {
    // NOTE: If ffmpeg is unavailable, write raw PPM frames next to out_path.
    fprintf(stderr, "warning: ffmpeg not available; writing PPM frames instead\n");
  }

  printf("rendering %d slices (%dx%d) -> %s\n", slices, out_w, out_h, out_path);
  fflush(stdout);

  for (int s = 0; s < slices; s++) {
    // Offset along the mean surface normal: step = s - slices/2 voxels.
    float step = (float)(s - slices / 2);

    float vmin = 1e38f, vmax = -1e38f;
    float *vals = malloc((size_t)n * sizeof(float));
    if (!vals) break;

    for (int i = 0; i < n; i++) {
      vec3f pt = base->points[i];
      vec3f nm = base->normals[i];
      float val = vol_sample(v, level,
                             pt.z + step * nm.z,
                             pt.y + step * nm.y,
                             pt.x + step * nm.x);
      vals[i] = val;
      if (val < vmin) vmin = val;
      if (val > vmax) vmax = val;
    }

    float range = (vmax > vmin) ? (vmax - vmin) : 1.0f;
    for (int i = 0; i < n; i++) {
      double norm = (double)(vals[i] - vmin) / (double)range;
      cmap_rgb rgb = cmap_apply(cmap, norm);
      pixels[i*3+0] = rgb.r;
      pixels[i*3+1] = rgb.g;
      pixels[i*3+2] = rgb.b;
    }
    free(vals);

    if (use_ffmpeg) {
      // Raw RGB frames fed directly to ffmpeg stdin.
      fwrite(pixels, 3, (size_t)n, pipe_fp);
    } else {
      char frame_path[4096];
      snprintf(frame_path, sizeof(frame_path), "%s_frame%04d.ppm", out_path, s);
      FILE *fp = fopen(frame_path, "wb");
      if (fp) { write_ppm_frame(fp, pixels, out_w, out_h); fclose(fp); }
    }
  }

  if (use_ffmpeg) pclose(pipe_fp);
  else            printf("  wrote %d PPM frames alongside %s\n", slices, out_path);

  free(pixels);
  quad_surface_free(base);
  vol_free(v);
  return 0;
}
