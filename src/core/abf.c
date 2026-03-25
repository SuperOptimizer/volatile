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
// Triangle: three vertex indices into the flat N=rows*cols grid.
// alpha[t*3+k] is the optimized angle at vertex tris[t].v[k].
// ---------------------------------------------------------------------------

typedef struct { int v[3]; } tri_t;

static float tri_angle_at(vec3f a, vec3f b, vec3f c) {
  // Interior angle at vertex `a`.
  vec3f ab = vec3f_sub(b, a);
  vec3f ac = vec3f_sub(c, a);
  float n  = vec3f_len(ab) * vec3f_len(ac);
  if (n < 1e-8f) return (float)(M_PI / 3.0);
  float cosA = vec3f_dot(ab, ac) / n;
  if (cosA >  1.0f) cosA =  1.0f;
  if (cosA < -1.0f) cosA = -1.0f;
  return acosf(cosA);
}

// ---------------------------------------------------------------------------
// ABF++ angle optimization.
// Per-iteration: enforce triangle sum-to-π, then interior vertex sum-to-2π.
// ---------------------------------------------------------------------------

static void abf_optimize(float *alpha, const tri_t *tris, int ntris,
                          const vec3f *pts, int N,
                          const int *vtris,   // vtris[v*16]: {tri,corner} pairs
                          const int *vntris,  // vntris[v]: count
                          int max_iter) {
  for (int t = 0; t < ntris; t++) {
    alpha[t*3+0] = tri_angle_at(pts[tris[t].v[0]], pts[tris[t].v[1]], pts[tris[t].v[2]]);
    alpha[t*3+1] = tri_angle_at(pts[tris[t].v[1]], pts[tris[t].v[2]], pts[tris[t].v[0]]);
    alpha[t*3+2] = tri_angle_at(pts[tris[t].v[2]], pts[tris[t].v[0]], pts[tris[t].v[1]]);
  }
  const float PI = (float)M_PI;
  for (int iter = 0; iter < max_iter; iter++) {
    float max_err = 0.0f;
    // Triangle constraint: angles sum to π
    for (int t = 0; t < ntris; t++) {
      float s = alpha[t*3+0] + alpha[t*3+1] + alpha[t*3+2];
      float e = PI - s;
      if (fabsf(e) > max_err) max_err = fabsf(e);
      float ws = s > 1e-8f ? s : 1.0f;
      for (int k = 0; k < 3; k++) {
        alpha[t*3+k] += e * alpha[t*3+k] / ws;
        if (alpha[t*3+k] < 1e-5f)  alpha[t*3+k] = 1e-5f;
        if (alpha[t*3+k] > PI-1e-5f) alpha[t*3+k] = PI-1e-5f;
      }
    }
    // Planarity constraint: angles around interior vertex sum to 2π
    for (int v = 0; v < N; v++) {
      if (vntris[v] < 3) continue;
      float s = 0.0f;
      for (int i = 0; i < vntris[v]; i++)
        s += alpha[vtris[v*16+i*2] * 3 + vtris[v*16+i*2+1]];
      float scale = s > 1e-8f ? 2.0f * PI / s : 1.0f;
      if (fabsf(s - 2.0f*PI) > max_err) max_err = fabsf(s - 2.0f*PI);
      for (int i = 0; i < vntris[v]; i++) {
        float *a = &alpha[vtris[v*16+i*2] * 3 + vtris[v*16+i*2+1]];
        *a *= scale;
        if (*a < 1e-5f)  *a = 1e-5f;
        if (*a > PI-1e-5f) *a = PI-1e-5f;
      }
    }
    if (max_err < 1e-4f) break;
  }
}

// ---------------------------------------------------------------------------
// LSCM: given optimized angles, solve for UV positions.
// Pin two boundary vertices; solve the normal equations AtA*x = At*b via CG.
// ---------------------------------------------------------------------------

uv_coords *abf_flatten(const quad_surface *surf) {
  if (!surf || surf->rows < 2 || surf->cols < 2) return NULL;
  int R = surf->rows, C = surf->cols, N = R * C;

  // Triangulate quad grid
  tri_t *tris = malloc((size_t)(R-1)*(C-1)*2 * sizeof(tri_t));
  if (!tris) return NULL;
  int ntris = 0;
  for (int r = 0; r < R-1; r++) {
    for (int c = 0; c < C-1; c++) {
      int v00=r*C+c, v01=r*C+c+1, v10=(r+1)*C+c, v11=(r+1)*C+c+1;
      // Skip degenerate cells
      if (vec3f_len(vec3f_sub(surf->points[v01],surf->points[v00])) < 1e-7f) continue;
      if (vec3f_len(vec3f_sub(surf->points[v10],surf->points[v00])) < 1e-7f) continue;
      tris[ntris++] = (tri_t){{v00, v10, v01}};
      tris[ntris++] = (tri_t){{v10, v11, v01}};
    }
  }
  if (ntris == 0) { free(tris); return NULL; }

  // Per-vertex triangle adjacency (up to 8 triangles, 2 ints each → 16)
  int *vtris  = calloc((size_t)N * 16, sizeof(int));
  int *vntris = calloc((size_t)N,      sizeof(int));
  if (!vtris || !vntris) goto fail_early;
  for (int t = 0; t < ntris; t++)
    for (int k = 0; k < 3; k++) {
      int v = tris[t].v[k], nt = vntris[v];
      if (nt < 8) { vtris[v*16+nt*2]=t; vtris[v*16+nt*2+1]=k; vntris[v]++; }
    }

  float *alpha = malloc((size_t)ntris * 3 * sizeof(float));
  if (!alpha) goto fail_early;
  abf_optimize(alpha, tris, ntris, surf->points, N, vtris, vntris, 50);

  // Pin corners: pin0 = (0,0), pin1 = (0, C-1)
  int pin0 = 0, pin1 = C-1;
  float pu0 = 0.0f, pv0 = 0.0f;
  float pu1 = vec3f_len(vec3f_sub(surf->points[pin1], surf->points[pin0]));
  float pv1 = 0.0f;

  int *fidx = malloc((size_t)N * sizeof(int));
  if (!fidx) goto fail_early;
  int nfree = 0;
  for (int i = 0; i < N; i++)
    fidx[i] = (i==pin0||i==pin1) ? -1 : nfree++;
  if (nfree == 0) goto fail_early;

  // Build A (2*ntris × 2*nfree) and rhs b (2*ntris) for LSCM
  sparse_mat *A = sparse_new(2*ntris, 2*nfree, 6*ntris);
  float      *b = calloc((size_t)(2*ntris), sizeof(float));
  if (!A || !b) { sparse_free(A); free(b); goto fail_early; }

  for (int t = 0; t < ntris; t++) {
    // Rotate triangle so largest angle is last (numerical stability)
    int e[3] = {0,1,2};
    if (alpha[t*3+e[0]] > alpha[t*3+e[2]]) { int tmp=e[2]; e[2]=e[0]; e[0]=tmp; }
    if (alpha[t*3+e[1]] > alpha[t*3+e[2]]) { int tmp=e[2]; e[2]=e[1]; e[1]=tmp; }

    float s2 = sinf(alpha[t*3+e[2]]);
    float r  = s2 > 1e-8f ? sinf(alpha[t*3+e[1]]) / s2 : 1.0f;
    float cs = cosf(alpha[t*3+e[0]]) * r;
    float sn = sinf(alpha[t*3+e[0]]) * r;

    // ve0: coeff [cs-1, -sn], ve1: [-cs, sn], ve2: [1, 0]
    int verts[3] = { tris[t].v[e[0]], tris[t].v[e[1]], tris[t].v[e[2]] };
    float cu[3]  = { cs-1.0f, -cs,  1.0f };
    float cv_[3] = { -sn,      sn,  0.0f };
    float cu2[3] = { sn,      -sn,  0.0f };
    float cv2[3] = { cs-1.0f, -cs,  1.0f };
    int row = 2*t;

    for (int k = 0; k < 3; k++) {
      int vi = verts[k];
      if (vi == pin0) {
        b[row]   -= cu[k]*pu0  + cv_[k]*pv0;
        b[row+1] -= cu2[k]*pu0 + cv2[k]*pv0;
      } else if (vi == pin1) {
        b[row]   -= cu[k]*pu1  + cv_[k]*pv1;
        b[row+1] -= cu2[k]*pu1 + cv2[k]*pv1;
      } else {
        int fi = fidx[vi];
        if (fabsf(cu[k])  > 1e-10f) sparse_add(A, row,   2*fi,   cu[k]);
        if (fabsf(cv_[k]) > 1e-10f) sparse_add(A, row,   2*fi+1, cv_[k]);
        if (fabsf(cu2[k]) > 1e-10f) sparse_add(A, row+1, 2*fi,   cu2[k]);
        if (fabsf(cv2[k]) > 1e-10f) sparse_add(A, row+1, 2*fi+1, cv2[k]);
      }
    }
  }

  // Normal equations: AtA*x = At*b
  sparse_mat *AtA = sparse_new(2*nfree, 2*nfree, 12*ntris);
  float      *Atb = calloc((size_t)(2*nfree), sizeof(float));
  float      *x   = calloc((size_t)(2*nfree), sizeof(float));
  if (!AtA || !Atb || !x) {
    sparse_free(A); free(b); sparse_free(AtA); free(Atb); free(x);
    goto fail_early;
  }

  // Compute AtA and Atb in one pass over A's nonzeros
  for (int i = 0; i < A->nnz; i++) {
    int ri = A->row[i]; int ci = A->col[i]; float vi = A->val[i];
    Atb[ci] += vi * b[ri];
    for (int j = i; j < A->nnz; j++) {
      if (A->row[j] != ri) continue;
      sparse_add(AtA, ci, A->col[j], vi * A->val[j]);
      if (A->col[j] != ci)
        sparse_add(AtA, A->col[j], ci, vi * A->val[j]);
    }
  }

  sparse_solve_cg(AtA, Atb, x, 2000, 1e-6f);

  uv_coords *uv = malloc(sizeof(uv_coords));
  float     *uu = uv ? malloc((size_t)N*sizeof(float)) : NULL;
  float     *vv = uu ? malloc((size_t)N*sizeof(float)) : NULL;
  if (!vv) {
    if (uv) { free(uu); free(uv); }
    sparse_free(A); free(b); sparse_free(AtA); free(Atb); free(x);
    goto fail_early;
  }
  uv->u=uu; uv->v=vv; uv->count=N; uv->rows=R; uv->cols=C;
  for (int i = 0; i < N; i++) {
    if (i==pin0)       { uu[i]=pu0; vv[i]=pv0; }
    else if (i==pin1)  { uu[i]=pu1; vv[i]=pv1; }
    else { int fi=fidx[i]; uu[i]=x[2*fi]; vv[i]=x[2*fi+1]; }
  }

  sparse_free(A); free(b); sparse_free(AtA); free(Atb); free(x);
  free(alpha); free(fidx); free(vtris); free(vntris); free(tris);
  return uv;

fail_early:
  free(alpha); free(fidx); free(vtris); free(vntris); free(tris);
  return NULL;
}

void uv_coords_free(uv_coords *uv) {
  if (!uv) return;
  free(uv->u); free(uv->v); free(uv);
}
