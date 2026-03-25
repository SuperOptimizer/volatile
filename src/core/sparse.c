#include "core/sparse.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

sparse_mat *sparse_new(int rows, int cols, int nnz_hint) {
  sparse_mat *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  int cap = nnz_hint > 0 ? nnz_hint : 16;
  m->row = malloc((size_t)cap * sizeof(int));
  m->col = malloc((size_t)cap * sizeof(int));
  m->val = malloc((size_t)cap * sizeof(float));
  if (!m->row || !m->col || !m->val) { sparse_free(m); return NULL; }
  m->nnz_cap = cap;
  m->rows = rows;
  m->cols = cols;
  return m;
}

void sparse_free(sparse_mat *m) {
  if (!m) return;
  free(m->row); free(m->col); free(m->val);
  free(m);
}

void sparse_add(sparse_mat *m, int row, int col, float val) {
  if (m->nnz == m->nnz_cap) {
    int new_cap = m->nnz_cap * 2;
    int   *nr = realloc(m->row, (size_t)new_cap * sizeof(int));
    int   *nc = realloc(m->col, (size_t)new_cap * sizeof(int));
    float *nv = realloc(m->val, (size_t)new_cap * sizeof(float));
    if (!nr || !nc || !nv) return;
    m->row = nr; m->col = nc; m->val = nv;
    m->nnz_cap = new_cap;
  }
  m->row[m->nnz] = row;
  m->col[m->nnz] = col;
  m->val[m->nnz] = val;
  m->nnz++;
}

// ---------------------------------------------------------------------------
// Internal: sparse matrix-vector product y = A*x
// Duplicate (row,col) entries are summed automatically.
// ---------------------------------------------------------------------------

static void spmv(const sparse_mat *A, const float *x, float *y) {
  memset(y, 0, (size_t)A->rows * sizeof(float));
  for (int k = 0; k < A->nnz; k++)
    y[A->row[k]] += A->val[k] * x[A->col[k]];
}

// ---------------------------------------------------------------------------
// Conjugate gradient solver — standard textbook implementation.
// Assumes A is symmetric positive definite.
// ---------------------------------------------------------------------------

int sparse_solve_cg(const sparse_mat *A, const float *b, float *x,
                    int max_iter, float tol) {
  int n = A->rows;
  float *r  = malloc((size_t)n * sizeof(float));
  float *p  = malloc((size_t)n * sizeof(float));
  float *Ap = malloc((size_t)n * sizeof(float));
  if (!r || !p || !Ap) { free(r); free(p); free(Ap); return -1; }

  memset(x, 0, (size_t)n * sizeof(float));

  // r = b - A*x = b (since x=0)
  memcpy(r, b, (size_t)n * sizeof(float));
  memcpy(p, r, (size_t)n * sizeof(float));

  float rsq = 0.0f;
  for (int i = 0; i < n; i++) rsq += r[i] * r[i];

  float tol2 = tol * tol;
  int iter = 0;

  for (iter = 0; iter < max_iter; iter++) {
    if (rsq < tol2) break;

    spmv(A, p, Ap);

    float pAp = 0.0f;
    for (int i = 0; i < n; i++) pAp += p[i] * Ap[i];
    if (pAp == 0.0f) break;

    float alpha = rsq / pAp;
    for (int i = 0; i < n; i++) x[i] += alpha * p[i];
    for (int i = 0; i < n; i++) r[i] -= alpha * Ap[i];

    float rsq_new = 0.0f;
    for (int i = 0; i < n; i++) rsq_new += r[i] * r[i];

    float beta = rsq_new / rsq;
    for (int i = 0; i < n; i++) p[i] = r[i] + beta * p[i];
    rsq = rsq_new;
  }

  free(r); free(p); free(Ap);
  return (rsq < tol2) ? iter : -1;
}
