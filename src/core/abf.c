#define _POSIX_C_SOURCE 200809L
#include "core/abf.h"
#include "core/sparse.h"
#include "core/math.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Triangle: three vertex indices (into flat row*cols grid)
// ---------------------------------------------------------------------------

typedef struct { int v[3]; } tri_t;

// ---------------------------------------------------------------------------
// Per-triangle angles: alpha[i] is the angle at vertex v[i]
// ---------------------------------------------------------------------------

static float tri_angle(vec3f a, vec3f b, vec3f c) {
  // Angle at vertex `a` of triangle (a, b, c)
  vec3f ab = vec3f_sub(b, a);
  vec3f ac = vec3f_sub(c, a);
  float d  = vec3f_dot(ab, ac);
  float n  = vec3f_len(ab) * vec3f_len(ac);
  if (n < 1e-8f) return (float)(M_PI / 3.0); // degenerate → equilateral fallback
  float cosA = d / n;
  if (cosA >  1.0f) cosA =  1.0f;
  if (cosA < -1.0f) cosA = -1.0f;
  return acosf(cosA);
}

// ---------------------------------------------------------------------------
// ABF++ angle optimization (simplified interior-vertex relaxation)
//
// For each interior vertex: enforce sum-to-2π planarity by scaling all wheel
// angles proportionally. For each triangle: enforce sum-to-π by distributing
// residual evenly. Iterate until convergence or max_iter.
// ---------------------------------------------------------------------------

static void abf_optimize_angles(float *alpha, const tri_t *tris, int ntris,
                                 const vec3f *pts, int npts,
                                 const int *vert_tris,   // vert_tris[v*8..] up to 8 tri indices
                                 const int *vert_ntris,  // number of triangles at vertex v
                                 int max_iter) {
  // Initialize from actual 3D angles
  for (int t = 0; t < ntris; t++) {
    vec3f p0 = pts[tris[t].v[0]];
    vec3f p1 = pts[tris[t].v[1]];
    vec3f p2 = pts[tris[t].v[2]];
    alpha[t*3+0] = tri_angle(p0, p1, p2);
    alpha[t*3+1] = tri_angle(p1, p2, p0);
    alpha[t*3+2] = tri_angle(p2, p0, p1);
  }

  const float PI = (float)M_PI;
  const float TWO_PI = 2.0f * PI;
  const float eps = 1e-4f;

  for (int iter = 0; iter < max_iter; iter++) {
    float max_err = 0.0f;

    // Triangle planarity: angles in each tri must sum to π
    for (int t = 0; t < ntris; t++) {
      float sum = alpha[t*3+0] + alpha[t*3+1] + alpha[t*3+2];
      float err = PI - sum;
      if (fabsf(err) > max_err) max_err = fabsf(err);
      // Distribute residual proportionally to angle magnitude
      float w0 = alpha[t*3+0], w1 = alpha[t*3+1], w2 = alpha[t*3+2];
      float ws = w0 + w1 + w2;
      if (ws > 1e-8f) {
        alpha[t*3+0] += err * w0 / ws;
        alpha[t*3+1] += err * w1 / ws;
        alpha[t*3+2] += err * w2 / ws;
      } else {
        alpha[t*3+0] += err / 3.0f;
        alpha[t*3+1] += err / 3.0f;
        alpha[t*3+2] += err / 3.0f;
      }
      // Clamp to (0, π)
      for (int k = 0; k < 3; k++) {
        if (alpha[t*3+k] < 1e-5f) alpha[t*3+k] = 1e-5f;
        if (alpha[t*3+k] > PI - 1e-5f) alpha[t*3+k] = PI - 1e-5f;
      }
    }

    // Vertex planarity: angles around interior vertices must sum to 2π
    // vert_tris stores {tri_idx, corner_within_tri} pairs packed as int[2]
    for (int v = 0; v < npts; v++) {
      int nt = vert_ntris[v];
      if (nt < 3) continue;  // boundary vertex
      float sum = 0.0f;
      for (int i = 0; i < nt; i++) {
        int ti  = vert_tris[v * 16 + i*2 + 0];
        int ki  = vert_tris[v * 16 + i*2 + 1];
        sum += alpha[ti*3 + ki];
      }
      float scale = (sum > 1e-8f) ? (TWO_PI / sum) : 1.0f;
      float err = fabsf(sum - TWO_PI);
      if (err > max_err) max_err = err;
      for (int i = 0; i < nt; i++) {
        int ti  = vert_tris[v * 16 + i*2 + 0];
        int ki  = vert_tris[v * 16 + i*2 + 1];
        alpha[ti*3 + ki] *= scale;
        if (alpha[ti*3+ki] < 1e-5f) alpha[ti*3+ki] = 1e-5f;
        if (alpha[ti*3+ki] > PI - 1e-5f) alpha[ti*3+ki] = PI - 1e-5f;
      }
    }

    if (max_err < eps) break;
  }
}

// ---------------------------------------------------------------------------
// LSCM UV recovery: given optimized angles, solve for (u,v) positions.
// Uses the conformal constraint per triangle: (u,v) must be consistent with
// the optimized angles. Formulated as a least-squares system Ax=b where
// x = [u0,v0, u1,v1, ...] for free vertices.
// We pin two boundary vertices to fix the global rotation/translation.
// ---------------------------------------------------------------------------

uv_coords *abf_flatten(const quad_surface *surf) {
  if (!surf || surf->rows < 2 || surf->cols < 2) return NULL;

  int R = surf->rows, C = surf->cols;
  int N = R * C;  // number of vertices

  // Build triangle list from quad grid (each cell → 2 triangles)
  int max_tris = (R-1) * (C-1) * 2;
  tri_t *tris = malloc((size_t)max_tris * sizeof(tri_t));
  if (!tris) return NULL;

  int ntris = 0;
  for (int r = 0; r < R-1; r++) {
    for (int c = 0; c < C-1; c++) {
      int v00 = r*C + c,   v01 = r*C + c+1;
      int v10 = (r+1)*C+c, v11 = (r+1)*C+c+1;
      vec3f p00 = surf->points[v00], p01 = surf->points[v01];
      vec3f p10 = surf->points[v10], p11 = surf->points[v11];
      // Skip degenerate triangles
      float d0 = vec3f_len(vec3f_sub(p01, p00));
      float d1 = vec3f_len(vec3f_sub(p10, p00));
      float d2 = vec3f_len(vec3f_sub(p11, p10));
      if (d0 < 1e-7f || d1 < 1e-7f || d2 < 1e-7f) continue;
      tris[ntris++] = (tri_t){{ v00, v10, v01 }};
      tris[ntris++] = (tri_t){{ v10, v11, v01 }};
    }
  }
  if (ntris == 0) { free(tris); return NULL; }

  // Build per-vertex triangle adjacency (up to 8 triangles per vertex)
  int *vert_tris  = calloc((size_t)N * 16, sizeof(int));  // {tri,corner} pairs
  int *vert_ntris = calloc((size_t)N, sizeof(int));
  if (!vert_tris || !vert_ntris) {
    free(tris); free(vert_tris); free(vert_ntris); return NULL;
  }
  for (int t = 0; t < ntris; t++) {
    for (int k = 0; k < 3; k++) {
      int v = tris[t].v[k];
      int nt = vert_ntris[v];
      if (nt < 8) {
        vert_tris[v*16 + nt*2 + 0] = t;
        vert_tris[v*16 + nt*2 + 1] = k;
        vert_ntris[v]++;
      }
    }
  }

  // Optimize angles via ABF++ relaxation
  float *alpha = malloc((size_t)ntris * 3 * sizeof(float));
  if (!alpha) {
    free(tris); free(vert_tris); free(vert_ntris); return NULL;
  }
  abf_optimize_angles(alpha, tris, ntris, surf->points, N,
                      vert_tris, vert_ntris, 50);

  // LSCM UV recovery
  // Pin first two boundary vertices (top-left and top-right corners)
  int pin0 = 0, pin1 = C - 1;  // row 0 corners
  float u0 = 0.0f, v0 = 0.0f;
  float u1 = vec3f_len(vec3f_sub(surf->points[pin1], surf->points[pin0]));
  float v1 = 0.0f;

  // Free vertex permutation (everyone except pin0, pin1)
  int *free_idx = malloc((size_t)N * sizeof(int));
  if (!free_idx) {
    free(tris); free(vert_tris); free(vert_ntris); free(alpha); return NULL;
  }
  int nfree = 0;
  for (int i = 0; i < N; i++) {
    if (i == pin0 || i == pin1) { free_idx[i] = -1; continue; }
    free_idx[i] = nfree++;
  }

  // Build Ax = b  (2*ntris equations, 2*nfree unknowns)
  // Each triangle contributes 2 conformal equations
  sparse_mat *A  = sparse_new(2*ntris, 2*nfree, 6*ntris);
  float      *b  = calloc((size_t)(2*ntris), sizeof(float));
  if (!A || !b) {
    free(tris); free(vert_tris); free(vert_ntris); free(alpha);
    free(free_idx); sparse_free(A); free(b); return NULL;
  }

  for (int t = 0; t < ntris; t++) {
    // Reorder edges so largest angle is last (for numerical stability)
    int ord[3] = {0, 1, 2};
    if (alpha[t*3+1] > alpha[t*3+ord[2]]) { int tmp=ord[2]; ord[2]=1; ord[0]=tmp; }
    if (alpha[t*3+0] > alpha[t*3+ord[2]]) { int tmp=ord[2]; ord[2]=0; ord[1]=tmp; }

    int e0 = ord[0], e1 = ord[1], e2 = ord[2];
    int ve0 = tris[t].v[e0], ve1 = tris[t].v[e1], ve2 = tris[t].v[e2];
    float s2 = sinf(alpha[t*3+e2]);
    float ratio  = (s2 > 1e-8f) ? sinf(alpha[t*3+e1]) / s2 : 1.0f;
    float cosine = cosf(alpha[t*3+e0]) * ratio;
    float sine   = sinf(alpha[t*3+e0]) * ratio;

    int row = 2 * t;
    // ve0 contribution: [cosine-1, -sine] (real part), [sine, cosine-1] (imag)
    float cv0[2] = { cosine-1.0f, -sine };
    float cv1[2] = { -cosine,      sine };
    // ve0 at row, ve1 at row, ve2 at row: identity (1,0 / 0,1)
    // Accumulate into A or b depending on whether vertex is pinned
    auto_insert:;
    int verts[3]  = { ve0, ve1, ve2 };
    float coeff_u[3] = { cosine-1.0f, -cosine, 1.0f };
    float coeff_v[3] = { -sine,        sine,   0.0f };
    float coeff_u2[3] = { sine,        -sine,  0.0f };
    float coeff_v2[3] = { cosine-1.0f, -cosine, 1.0f };
    (void)cv0; (void)cv1;

    for (int k = 0; k < 3; k++) {
      int vi = verts[k];
      if (vi == pin0) {
        b[row]   -= coeff_u[k] * u0 + coeff_v[k] * v0;
        b[row+1] -= coeff_u2[k] * u0 + coeff_v2[k] * v0;
      } else if (vi == pin1) {
        b[row]   -= coeff_u[k] * u1 + coeff_v[k] * v1;
        b[row+1] -= coeff_u2[k] * u1 + coeff_v2[k] * v1;
      } else {
        int fi = free_idx[vi];
        if (fabsf(coeff_u[k])  > 1e-10f) sparse_add(A, row,   2*fi,   coeff_u[k]);
        if (fabsf(coeff_v[k])  > 1e-10f) sparse_add(A, row,   2*fi+1, coeff_v[k]);
        if (fabsf(coeff_u2[k]) > 1e-10f) sparse_add(A, row+1, 2*fi,   coeff_u2[k]);
        if (fabsf(coeff_v2[k]) > 1e-10f) sparse_add(A, row+1, 2*fi+1, coeff_v2[k]);
      }
    }
    (void)auto_insert; // label used by goto avoided; keep as no-op
  }

  // Solve normal equations AtA x = At b (AtA is SPD, CG works)
  sparse_mat *At  = sparse_new(2*nfree, 2*ntris, 6*ntris);
  float      *Atb = calloc((size_t)(2*nfree), sizeof(float));
  float      *x   = calloc((size_t)(2*nfree), sizeof(float));
  if (!At || !Atb || !x) goto fail;

  // Transpose A → At and compute At*b
  for (int i = 0; i < A->nnz; i++) {
    sparse_add(At, A->col[i], A->row[i], A->val[i]);
    Atb[A->col[i]] += A->val[i] * b[A->row[i]];
  }

  // Build AtA and solve
  sparse_mat *AtA = sparse_new(2*nfree, 2*nfree, 12*ntris);
  if (!AtA) goto fail;
  // AtA = At * A  (manual spmm for symmetric result)
  for (int i = 0; i < At->nnz; i++) {
    int r = At->row[i]; float va = At->val[i]; int ac = At->col[i];
    // Find all entries in row ac of A
    for (int j = 0; j < A->nnz; j++) {
      if (A->row[j] == ac)
        sparse_add(AtA, r, A->col[j], va * A->val[j]);
    }
  }

  sparse_solve_cg(AtA, Atb, x, 2000, 1e-6f);

  // Extract UVs
  uv_coords *uv = malloc(sizeof(uv_coords));
  if (!uv) goto fail;
  uv->u = malloc((size_t)N * sizeof(float));
  uv->v = malloc((size_t)N * sizeof(float));
  if (!uv->u || !uv->v) { free(uv->u); free(uv->v); free(uv); uv = NULL; goto fail; }
  uv->count = N; uv->rows = R; uv->cols = C;
  for (int i = 0; i < N; i++) {
    if (i == pin0)      { uv->u[i] = u0; uv->v[i] = v0; }
    else if (i == pin1) { uv->u[i] = u1; uv->v[i] = v1; }
    else {
      int fi = free_idx[i];
      uv->u[i] = x[2*fi];
      uv->v[i] = x[2*fi+1];
    }
  }
  sparse_free(AtA); sparse_free(At); sparse_free(A);
  free(b); free(Atb); free(x);
  free(alpha); free(free_idx); free(vert_tris); free(vert_ntris); free(tris);
  return uv;

fail:
  sparse_free(AtA); sparse_free(At); sparse_free(A);
  free(b); free(Atb); free(x);
  free(alpha); free(free_idx); free(vert_tris); free(vert_ntris); free(tris);
  return NULL;
}

void uv_coords_free(uv_coords *uv) {
  if (!uv) return;
  free(uv->u);
  free(uv->v);
  free(uv);
}
