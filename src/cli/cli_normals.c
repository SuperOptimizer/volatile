#define _POSIX_C_SOURCE 200809L

#include "cli_normals.h"
#include "cli_progress.h"
#include "core/vol.h"
#include "core/imgproc.h"
#include "core/io.h"

#include "core/math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void normals_usage(void) {
  puts("usage: volatile normals <volume> --output <normals.zarr>");
  puts("                        [--level N] [--sigma S] [--chunk-z Z]");
  puts("");
  puts("  Generate per-voxel surface normal vectors via structure tensor");
  puts("  eigendecomposition. The smallest eigenvector is the normal direction.");
  puts("  Output: 3-channel float32 zarr (nx, ny, nz per voxel).");
  puts("");
  puts("  --level N     pyramid level to process (default 0)");
  puts("  --sigma S     derivative sigma for structure tensor (default 1.0)");
  puts("  --chunk-z Z   number of Z slices to process at once (default 32)");
}

// ---------------------------------------------------------------------------
// Symmetric 3x3 eigenvector for the SMALLEST eigenvalue
// Using the structure tensor upper triangle: [Jzz, Jzy, Jzx, Jyy, Jyx, Jxx]
//
// We use power-iteration on (A - lambda_min * I)^{-1} — but for small 3x3
// it's faster to use the analytical Cardano method for eigenvalues, then
// solve (A - lambda*I)v = 0 via cross products of two rows.
// ---------------------------------------------------------------------------

// Compute all three eigenvalues of a symmetric 3x3 matrix via Cardano's formula.
// M = [m00, m01, m02; m01, m11, m12; m02, m12, m22]
static void sym3_eigenvalues(float m00, float m01, float m02,
                              float m11, float m12, float m22,
                              float *l0, float *l1, float *l2) {
  double a = m00, b = m01, c = m02, d = m11, e = m12, f = m22;
  double p1 = b*b + c*c + e*e;

  if (p1 < 1e-30) {
    // already diagonal
    double v[3] = {a, d, f};
    // sort ascending
    if (v[0] > v[1]) { double t = v[0]; v[0] = v[1]; v[1] = t; }
    if (v[1] > v[2]) { double t = v[1]; v[1] = v[2]; v[2] = t; }
    if (v[0] > v[1]) { double t = v[0]; v[0] = v[1]; v[1] = t; }
    *l0 = (float)v[0]; *l1 = (float)v[1]; *l2 = (float)v[2];
    return;
  }

  double q  = (a + d + f) / 3.0;
  double p2 = (a-q)*(a-q) + (d-q)*(d-q) + (f-q)*(f-q) + 2.0*p1;
  double p  = sqrt(p2 / 6.0);
  double B00 = (a-q)/p, B01 = b/p, B02 = c/p;
  double B11 = (d-q)/p, B12 = e/p, B22 = (f-q)/p;
  double r = (B00*(B11*B22 - B12*B12) - B01*(B01*B22 - B12*B02)
              + B02*(B01*B12 - B11*B02)) / 2.0;
  double phi;
  if      (r <= -1.0) phi = M_PI / 3.0;
  else if (r >=  1.0) phi = 0.0;
  else                phi = acos(r) / 3.0;

  *l2 = (float)(q + 2.0*p*cos(phi));
  *l0 = (float)(q + 2.0*p*cos(phi + 2.0*M_PI/3.0));
  *l1 = (float)(q*3.0 - (double)*l0 - (double)*l2);
}

// Compute the eigenvector of sym-3x3 for eigenvalue lambda (smallest).
// Returns unit vector via cross product of two non-parallel rows of (M - lI).
static void sym3_eigenvec_min(float m00, float m01, float m02,
                               float m11, float m12, float m22,
                               float lambda, float *vx, float *vy, float *vz) {
  float r0x = m00 - lambda, r0y = m01,          r0z = m02;
  float r1x = m01,          r1y = m11 - lambda, r1z = m12;
  float r2x = m02,          r2y = m12,          r2z = m22 - lambda;

  // cross products of pairs of rows to find null vector
  float cx = r0y*r1z - r0z*r1y;
  float cy = r0z*r1x - r0x*r1z;
  float cz = r0x*r1y - r0y*r1x;
  float len2 = cx*cx + cy*cy + cz*cz;

  if (len2 < 1e-20f) {
    cx = r0y*r2z - r0z*r2y;
    cy = r0z*r2x - r0x*r2z;
    cz = r0x*r2y - r0y*r2x;
    len2 = cx*cx + cy*cy + cz*cz;
  }
  if (len2 < 1e-20f) {
    cx = r1y*r2z - r1z*r2y;
    cy = r1z*r2x - r1x*r2z;
    cz = r1x*r2y - r1y*r2x;
    len2 = cx*cx + cy*cy + cz*cz;
  }

  float inv = (len2 > 1e-20f) ? 1.0f / sqrtf(len2) : 0.0f;
  *vx = cx * inv;
  *vy = cy * inv;
  *vz = cz * inv;
}

// ---------------------------------------------------------------------------
// cmd_normals
// ---------------------------------------------------------------------------

int cmd_normals(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    normals_usage(); return argc < 1 ? 1 : 0;
  }

  const char *vol_path = argv[0];
  const char *out_path = NULL;
  int   level          = 0;
  float sigma          = 1.0f;
  int   chunk_z        = 32;

  for (int i = 1; i < argc; i++) {
    if      (strcmp(argv[i], "--output")  == 0 && i+1 < argc) out_path = argv[++i];
    else if (strcmp(argv[i], "--level")   == 0 && i+1 < argc) level    = atoi(argv[++i]);
    else if (strcmp(argv[i], "--sigma")   == 0 && i+1 < argc) sigma    = (float)atof(argv[++i]);
    else if (strcmp(argv[i], "--chunk-z") == 0 && i+1 < argc) chunk_z  = atoi(argv[++i]);
  }

  if (!out_path) { fputs("error: --output required\n", stderr); return 1; }

  volume *v = vol_open(vol_path);
  if (!v) { fprintf(stderr, "error: cannot open volume: %s\n", vol_path); return 1; }

  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m || m->ndim < 3) {
    fputs("error: volume must be 3D\n", stderr); vol_free(v); return 1;
  }

  int D = (int)m->shape[m->ndim - 3];
  int H = (int)m->shape[m->ndim - 2];
  int W = (int)m->shape[m->ndim - 1];

  // Create output zarr: same spatial shape, 3 channels (nx,ny,nz), float32
  vol_create_params p = {
    .zarr_version = 2, .ndim = 4,
    .shape       = {3, D, H, W},
    .chunk_shape = {3, (int64_t)(chunk_z > D ? D : chunk_z), 64, 64},
    .dtype       = DTYPE_F32,
    .compressor  = "blosc", .clevel = 5,
  };
  volume *out = vol_create(out_path, p);
  if (!out) { fprintf(stderr, "error: cannot create output: %s\n", out_path); vol_free(v); return 1; }

  int z_step = chunk_z;
  int n_steps = (D + z_step - 1) / z_step;
  size_t slab_floats = (size_t)z_step * H * W;

  float *slab      = malloc(slab_floats * sizeof(float));
  float *tensor    = malloc(slab_floats * 6 * sizeof(float));  // 6-component ST
  float *normals_buf = malloc(slab_floats * 3 * sizeof(float));

  if (!slab || !tensor || !normals_buf) {
    fputs("error: out of memory\n", stderr);
    free(slab); free(tensor); free(normals_buf);
    vol_free(v); vol_free(out); return 1;
  }

  for (int step = 0; step < n_steps; step++) {
    int z0 = step * z_step;
    int z1 = z0 + z_step;
    if (z1 > D) z1 = D;
    int dz = z1 - z0;

    // Read slab from volume (float samples)
    for (int z = z0; z < z1; z++) {
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          slab[(z - z0)*(size_t)H*W + (size_t)y*W + x] =
            vol_sample(v, level, (float)z, (float)y, (float)x);
        }
      }
    }

    // Compute structure tensor for this slab
    structure_tensor_3d(slab, tensor, dz, H, W, sigma, sigma * 2.0f);

    // Decompose each voxel's tensor → smallest eigenvector = normal
    for (int z = 0; z < dz; z++) {
      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          size_t vi = ((size_t)z*H + y)*(size_t)W + x;
          float *t = tensor + vi * 6;
          // tensor layout: {Jzz, Jzy, Jzx, Jyy, Jyx, Jxx}
          float l0, l1, l2;
          sym3_eigenvalues(t[0], t[1], t[2], t[3], t[4], t[5], &l0, &l1, &l2);
          float nx, ny, nz;
          sym3_eigenvec_min(t[0], t[1], t[2], t[3], t[4], t[5], l0, &nx, &ny, &nz);
          normals_buf[vi*3+0] = nx;
          normals_buf[vi*3+1] = ny;
          normals_buf[vi*3+2] = nz;
        }
      }
    }

    // Write normals slab to output (3 channel zarr)
    // NOTE: vol_write_chunk expects chunk coords, not voxel coords
    int64_t cz = (int64_t)(z0 / chunk_z);
    int64_t chunk_coords[4] = {0, cz, 0, 0};
    vol_write_chunk(out, 0, chunk_coords,
                    normals_buf, (size_t)dz * H * W * 3 * sizeof(float));

    cli_progress(step + 1, n_steps, "normals");
  }

  free(slab); free(tensor); free(normals_buf);
  vol_free(v); vol_free(out);
  printf("wrote normals → %s\n", out_path);
  return 0;
}
