#define _POSIX_C_SOURCE 200809L
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "cli_winding.h"
#include "cli_progress.h"
#include "core/vol.h"
#include "core/io.h"
#include "core/geom.h"
#include "core/math.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void winding_usage(void) {
  puts("usage: volatile winding <surface.obj> --output <winding.zarr>");
  puts("                        --shape Z,Y,X [--chunk-z Z]");
  puts("");
  puts("  Compute the winding number field from a triangle mesh surface.");
  puts("  For each voxel, evaluates the signed solid angle subtended by the");
  puts("  surface (inside ≈ 1, outside ≈ 0). Output: float32 scalar zarr.");
  puts("");
  puts("  --shape Z,Y,X   output volume shape (required)");
  puts("  --chunk-z Z     z slices per processing chunk (default 16)");
}

// ---------------------------------------------------------------------------
// Signed solid angle subtended by a triangle at a query point p.
// Uses the formula by Oosterom & Strackee (1983):
//   Ω = 2 * atan2(|a·(b×c)|, |a||b||c| + (a·b)|c| + (b·c)|a| + (c·a)|b|)
// where a, b, c are vectors from p to each triangle vertex, normalised.
// ---------------------------------------------------------------------------

static float triangle_solid_angle(vec3f p,
                                   vec3f v0, vec3f v1, vec3f v2) {
  vec3f a = vec3f_sub(v0, p);
  vec3f b = vec3f_sub(v1, p);
  vec3f c = vec3f_sub(v2, p);

  float la = vec3f_len(a);
  float lb = vec3f_len(b);
  float lc = vec3f_len(c);

  if (la < 1e-10f || lb < 1e-10f || lc < 1e-10f) return 0.0f;

  float numerator   = fabsf(vec3f_dot(a, vec3f_cross(b, c)));
  float denominator = la*lb*lc
                    + vec3f_dot(a, b) * lc
                    + vec3f_dot(b, c) * la
                    + vec3f_dot(c, a) * lb;

  // Preserve sign: flip if denominator < 0 (point on back side of all edges)
  float angle = 2.0f * atan2f(numerator, denominator);
  // Sign from orientation: use dot of centroid normal with query offset
  vec3f cent = vec3f_scale(vec3f_add(vec3f_add(v0, v1), v2), 1.0f/3.0f);
  vec3f n = vec3f_cross(vec3f_sub(v1, v0), vec3f_sub(v2, v0));
  if (vec3f_dot(n, vec3f_sub(p, cent)) > 0.0f) angle = -angle;
  return angle;
}

// ---------------------------------------------------------------------------
// cmd_winding
// ---------------------------------------------------------------------------

int cmd_winding(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    winding_usage(); return argc < 1 ? 1 : 0;
  }

  const char *surf_path = argv[0];
  const char *out_path  = NULL;
  int D = 0, H = 0, W = 0;
  int chunk_z = 16;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--output") == 0 && i+1 < argc) {
      out_path = argv[++i];
    } else if (strcmp(argv[i], "--shape") == 0 && i+1 < argc) {
      sscanf(argv[++i], "%d,%d,%d", &D, &H, &W);
    } else if (strcmp(argv[i], "--chunk-z") == 0 && i+1 < argc) {
      chunk_z = atoi(argv[++i]);
    }
  }

  if (!out_path)            { fputs("error: --output required\n",        stderr); return 1; }
  if (D <= 0 || H <= 0 || W <= 0) { fputs("error: --shape Z,Y,X required\n", stderr); return 1; }

  // Load surface mesh
  obj_mesh *mesh = obj_read(surf_path);
  if (!mesh) { fprintf(stderr, "error: cannot load surface: %s\n", surf_path); return 1; }
  if (mesh->index_count == 0) {
    fputs("error: mesh has no triangles\n", stderr); obj_free(mesh); return 1;
  }

  int n_tris = mesh->index_count / 3;
  fprintf(stderr, "loaded %d triangles from %s\n", n_tris, surf_path);

  // Create output zarr: scalar float32
  vol_create_params p = {
    .zarr_version = 2, .ndim = 3,
    .shape        = {D, H, W},
    .chunk_shape  = {chunk_z > D ? D : chunk_z, 64, 64},
    .dtype        = DTYPE_F32,
    .compressor   = "blosc", .clevel = 5,
  };
  volume *out = vol_create(out_path, p);
  if (!out) {
    fprintf(stderr, "error: cannot create output: %s\n", out_path);
    obj_free(mesh); return 1;
  }

  int n_steps = (D + chunk_z - 1) / chunk_z;
  size_t slab_n = (size_t)chunk_z * (size_t)H * (size_t)W;
  float *winding = malloc(slab_n * sizeof(float));
  if (!winding) {
    fputs("error: out of memory\n", stderr);
    obj_free(mesh); vol_free(out); return 1;
  }

  const float inv4pi = 1.0f / (4.0f * (float)M_PI);

  for (int step = 0; step < n_steps; step++) {
    int z0 = step * chunk_z;
    int z1 = z0 + chunk_z;
    if (z1 > D) z1 = D;
    int dz = z1 - z0;

    // For each voxel in the slab, accumulate solid angles over all triangles
    for (int z = z0; z < z1; z++) {
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          vec3f query = {(float)x, (float)y, (float)z};
          float total = 0.0f;

          for (int t = 0; t < n_tris; t++) {
            int i0 = mesh->indices[t*3+0];
            int i1 = mesh->indices[t*3+1];
            int i2 = mesh->indices[t*3+2];
            vec3f v0 = {mesh->vertices[i0*3], mesh->vertices[i0*3+1], mesh->vertices[i0*3+2]};
            vec3f v1 = {mesh->vertices[i1*3], mesh->vertices[i1*3+1], mesh->vertices[i1*3+2]};
            vec3f v2 = {mesh->vertices[i2*3], mesh->vertices[i2*3+1], mesh->vertices[i2*3+2]};
            total += triangle_solid_angle(query, v0, v1, v2);
          }

          size_t idx = (size_t)(z - z0)*(size_t)H*W + (size_t)y*W + x;
          // Normalise: winding number = total / (4π); inside ≈ 1, outside ≈ 0
          winding[idx] = total * inv4pi;
        }
      }
    }

    int64_t cz = (int64_t)(z0 / chunk_z);
    int64_t chunk_coords[3] = {cz, 0, 0};
    vol_write_chunk(out, 0, chunk_coords,
                    winding, (size_t)dz * H * W * sizeof(float));

    cli_progress(step + 1, n_steps, "winding");
  }

  free(winding);
  obj_free(mesh);
  vol_free(out);
  printf("wrote winding field → %s\n", out_path);
  return 0;
}
