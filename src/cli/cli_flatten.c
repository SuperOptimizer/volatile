#define _POSIX_C_SOURCE 200809L

#include "cli_flatten.h"
#include "core/geom.h"
#include "core/log.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// quad_surface binary loader
// Format: "QSRF" magic (4 bytes), rows (int32), cols (int32),
//         then rows*cols * 3 floats (x,y,z) in row-major order.
// ---------------------------------------------------------------------------

static quad_surface *load_surface(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "flatten: cannot open %s\n", path); return NULL; }

  char magic[4];
  int32_t rows, cols;
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "QSRF", 4) != 0) {
    fprintf(stderr, "flatten: not a QSRF surface file: %s\n", path);
    fclose(f); return NULL;
  }
  if (fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1 ||
      rows <= 1 || cols <= 1) {
    fprintf(stderr, "flatten: bad surface dimensions\n");
    fclose(f); return NULL;
  }

  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) { fclose(f); return NULL; }

  size_t n = (size_t)(rows * cols);
  if (fread(s->points, sizeof(vec3f), n, f) != n) {
    fprintf(stderr, "flatten: truncated surface data\n");
    quad_surface_free(s); fclose(f); return NULL;
  }
  fclose(f);
  return s;
}

// ---------------------------------------------------------------------------
// Cotangent Laplacian LSCM UV flattening
//
// We pin two boundary corners to fix translation/rotation:
//   vertex (0,0)            -> uv (0, 0)
//   vertex (0, cols-1)      -> uv (1, 0)
//
// For each interior vertex i, the cotangent equation is:
//   sum_j  w_ij * (u_i - u_j) = 0
// where w_ij = (cot(alpha_ij) + cot(beta_ij)) / 2 over the two triangles
// sharing edge (i,j).  We solve the resulting sparse system with conjugate
// gradient separately for u and v.
// ---------------------------------------------------------------------------

// Sparse matrix in COO form — we'll sort and compress per-row in CG.
typedef struct { int row, col; float val; } coo_entry;

typedef struct {
  coo_entry *entries;
  int        n_entries, cap;
  int        dim;
} sp_mat;

static sp_mat *sp_new(int dim) {
  sp_mat *m = calloc(1, sizeof(sp_mat));
  if (!m) return NULL;
  m->dim     = dim;
  m->cap     = dim * 6;
  m->entries = malloc((size_t)m->cap * sizeof(coo_entry));
  if (!m->entries) { free(m); return NULL; }
  return m;
}

static void sp_free(sp_mat *m) { if (m) { free(m->entries); free(m); } }

static void sp_add(sp_mat *m, int r, int c, float v) {
  if (m->n_entries == m->cap) {
    m->cap *= 2;
    m->entries = realloc(m->entries, (size_t)m->cap * sizeof(coo_entry));
  }
  m->entries[m->n_entries++] = (coo_entry){r, c, v};
}

// Compute A*x -> y  (COO, no symmetry assumed)
static void sp_matvec(const sp_mat *A, const float *x, float *y) {
  memset(y, 0, (size_t)A->dim * sizeof(float));
  for (int k = 0; k < A->n_entries; k++)
    y[A->entries[k].row] += A->entries[k].val * x[A->entries[k].col];
}

// Conjugate gradient: solve A*x = b for symmetric positive-definite A.
// x is initialised on entry.  max_iter ~= 2*dim is usually enough.
static void cg_solve(const sp_mat *A, const float *b, float *x, int max_iter) {
  int n = A->dim;
  float *r  = malloc((size_t)n * sizeof(float));
  float *p  = malloc((size_t)n * sizeof(float));
  float *Ap = malloc((size_t)n * sizeof(float));
  if (!r || !p || !Ap) { free(r); free(p); free(Ap); return; }

  sp_matvec(A, x, Ap);
  float r_dot = 0.0f;
  for (int i = 0; i < n; i++) { r[i] = b[i] - Ap[i]; p[i] = r[i]; r_dot += r[i]*r[i]; }

  for (int iter = 0; iter < max_iter && r_dot > 1e-12f; iter++) {
    sp_matvec(A, p, Ap);
    float pAp = 0.0f;
    for (int i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (fabsf(pAp) < 1e-30f) break;
    float alpha = r_dot / pAp;
    float r_dot_new = 0.0f;
    for (int i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; r_dot_new += r[i]*r[i]; }
    float beta = r_dot_new / r_dot;
    for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    r_dot = r_dot_new;
  }
  free(r); free(p); free(Ap);
}

// Cotangent of angle at vertex b in triangle (a,b,c).
static float cot_angle(vec3f a, vec3f b, vec3f c) {
  vec3f ba = vec3f_sub(a, b);
  vec3f bc = vec3f_sub(c, b);
  float dot   = vec3f_dot(ba, bc);
  float cross = vec3f_len(vec3f_cross(ba, bc));
  return (cross > 1e-8f) ? dot / cross : 0.0f;
}

// Flatten a quad_surface -> UV coords (n=rows*cols, allocated by caller).
static bool flatten_surface(const quad_surface *s, vec2f *uv) {
  int R = s->rows, C = s->cols, N = R * C;

  // Pin vertices: idx 0 = (0,0), idx C-1 = (0,C-1)
  int pin0 = 0, pin1 = C - 1;

  // interior dof count
  int n_dof = N - 2;
  // mapping: vertex -> dof index (-1 = pinned)
  int *dof = malloc((size_t)N * sizeof(int));
  if (!dof) return false;
  int d = 0;
  for (int i = 0; i < N; i++) {
    if (i == pin0 || i == pin1) { dof[i] = -1; continue; }
    dof[i] = d++;
  }

  sp_mat *A  = sp_new(n_dof);
  float  *bu = calloc((size_t)n_dof, sizeof(float));
  float  *bv = calloc((size_t)n_dof, sizeof(float));
  float  *xu = calloc((size_t)n_dof, sizeof(float));
  float  *xv = calloc((size_t)n_dof, sizeof(float));
  if (!A || !bu || !bv || !xu || !xv) {
    sp_free(A); free(bu); free(bv); free(xu); free(xv); free(dof); return false;
  }

  // Set pin UVs (normalised by grid width for a unit domain).
  uv[pin0] = (vec2f){0.0f, 0.0f};
  uv[pin1] = (vec2f){1.0f, 0.0f};

  // Accumulate cotangent weights over all quads (2 triangles each).
  for (int r = 0; r < R - 1; r++) {
    for (int c = 0; c < C - 1; c++) {
      int v[4] = { r*C+c, r*C+c+1, (r+1)*C+c, (r+1)*C+c+1 };
      // Two triangles: (v0,v1,v2) and (v1,v3,v2)
      int tris[2][3] = { {v[0],v[1],v[2]}, {v[1],v[3],v[2]} };
      for (int t = 0; t < 2; t++) {
        int *tri = tris[t];
        for (int k = 0; k < 3; k++) {
          int a = tri[(k+1)%3], b_v = tri[k], c_v = tri[(k+2)%3];
          float w = cot_angle(s->points[a], s->points[b_v], s->points[c_v]) * 0.5f;
          if (w == 0.0f) continue;
          // Edge (a, c_v) — add cotangent weight w to both
          for (int side = 0; side < 2; side++) {
            int i = (side == 0) ? a : c_v;
            int j = (side == 0) ? c_v : a;
            int di = dof[i], dj = dof[j];
            if (di < 0) continue;  // pinned row, skip
            sp_add(A, di, di,  w);
            if (dj >= 0) sp_add(A, di, dj, -w);
            else {
              // j is pinned — move contribution to RHS
              float pj_u = (j == pin0) ? 0.0f : 1.0f;
              float pj_v = 0.0f;
              bu[di] += w * pj_u;
              bv[di] += w * pj_v;
            }
          }
        }
      }
    }
  }

  cg_solve(A, bu, xu, n_dof * 2);
  cg_solve(A, bv, xv, n_dof * 2);

  for (int i = 0; i < N; i++) {
    if (dof[i] >= 0) uv[i] = (vec2f){xu[dof[i]], xv[dof[i]]};
  }

  sp_free(A); free(bu); free(bv); free(xu); free(xv); free(dof);
  return true;
}

// ---------------------------------------------------------------------------
// OBJ export
// ---------------------------------------------------------------------------

static bool write_obj(const quad_surface *s, const vec2f *uv, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) { fprintf(stderr, "flatten: cannot write %s\n", path); return false; }

  int R = s->rows, C = s->cols, N = R * C;

  fputs("# volatile UV-flattened surface\n", f);
  for (int i = 0; i < N; i++)
    fprintf(f, "v %f %f %f\n", (double)s->points[i].x,
            (double)s->points[i].y, (double)s->points[i].z);
  for (int i = 0; i < N; i++)
    fprintf(f, "vt %f %f\n", (double)uv[i].x, (double)uv[i].y);

  // Quads as two triangles (1-based OBJ indices).
  for (int r = 0; r < R - 1; r++) {
    for (int c = 0; c < C - 1; c++) {
      int a = r*C+c+1, b = r*C+c+2, d = (r+1)*C+c+1, e = (r+1)*C+c+2;
      fprintf(f, "f %d/%d %d/%d %d/%d\n", a,a, b,b, d,d);
      fprintf(f, "f %d/%d %d/%d %d/%d\n", b,b, e,e, d,d);
    }
  }
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

int cmd_flatten(int argc, char **argv) {
  if (argc < 1) goto usage;

  const char *surface_path = NULL;
  const char *output_path  = "out.obj";

  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) goto usage;
    if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { output_path = argv[++i]; continue; }
    if (!surface_path) { surface_path = argv[i]; continue; }
  }
  if (!surface_path) goto usage;

  quad_surface *s = load_surface(surface_path);
  if (!s) return 1;

  int N = s->rows * s->cols;
  vec2f *uv = calloc((size_t)N, sizeof(vec2f));
  if (!uv) { quad_surface_free(s); return 1; }

  if (!flatten_surface(s, uv)) {
    fprintf(stderr, "flatten: solver failed\n");
    free(uv); quad_surface_free(s); return 1;
  }

  bool ok = write_obj(s, uv, output_path);
  free(uv);
  quad_surface_free(s);

  if (ok) printf("flatten: wrote %s\n", output_path);
  return ok ? 0 : 1;

usage:
  puts("Usage: volatile flatten <surface_path> --output <out.obj>");
  puts("");
  puts("UV-flatten a quad surface using a cotangent Laplacian conformal map.");
  puts("Surface file format: QSRF binary (magic + int32 rows + int32 cols + float32 xyz...).");
  puts("Output is an OBJ file with per-vertex UV coordinates.");
  return argc < 1 ? 1 : 0;
}
