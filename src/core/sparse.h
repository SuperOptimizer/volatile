#pragma once
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Sparse matrix (COO format, with deduplication on solve)
// ---------------------------------------------------------------------------

typedef struct {
  int   *row;
  int   *col;
  float *val;
  int    nnz;      // number of stored entries
  int    nnz_cap;  // allocated capacity
  int    rows;
  int    cols;
} sparse_mat;

sparse_mat *sparse_new(int rows, int cols, int nnz_hint);
void        sparse_free(sparse_mat *m);

// Accumulate a value at (row, col). Duplicate entries are summed at solve time.
void sparse_add(sparse_mat *m, int row, int col, float val);

// Solve A*x = b using conjugate gradient (A must be SPD).
// x is initialised to zero on entry. Returns number of iterations used,
// or -1 if the solver did not converge within max_iter iterations.
int sparse_solve_cg(const sparse_mat *A, const float *b, float *x,
                    int max_iter, float tol);
