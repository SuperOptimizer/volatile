#include "greatest.h"
#include "core/sparse.h"
#include "core/imgproc.h"

#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define EPS 1e-4f

// ---------------------------------------------------------------------------
// sparse_mat: add / structural checks
// ---------------------------------------------------------------------------

TEST test_sparse_new_free(void) {
  sparse_mat *m = sparse_new(4, 4, 8);
  ASSERT(m != NULL);
  ASSERT_EQ(4, m->rows);
  ASSERT_EQ(4, m->cols);
  ASSERT_EQ(0, m->nnz);
  sparse_free(m);
  PASS();
}

TEST test_sparse_add(void) {
  sparse_mat *m = sparse_new(3, 3, 4);
  sparse_add(m, 0, 0, 1.0f);
  sparse_add(m, 1, 1, 2.0f);
  sparse_add(m, 2, 2, 3.0f);
  ASSERT_EQ(3, m->nnz);
  ASSERT(fabsf(m->val[1] - 2.0f) < EPS);
  sparse_free(m);
  PASS();
}

TEST test_sparse_grow(void) {
  // Start with tiny hint, force several grows.
  sparse_mat *m = sparse_new(100, 100, 2);
  for (int i = 0; i < 50; i++) sparse_add(m, i, i, (float)i);
  ASSERT_EQ(50, m->nnz);
  ASSERT(fabsf(m->val[49] - 49.0f) < EPS);
  sparse_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// CG solver: 2x2 diagonal system
//   [2 0] [x0]   [4]      x0=2, x1=1
//   [0 3] [x1] = [3]
// ---------------------------------------------------------------------------

TEST test_cg_diagonal_2x2(void) {
  sparse_mat *A = sparse_new(2, 2, 2);
  sparse_add(A, 0, 0, 2.0f);
  sparse_add(A, 1, 1, 3.0f);

  float b[2] = {4.0f, 3.0f};
  float x[2] = {0};
  int iters = sparse_solve_cg(A, b, x, 100, 1e-6f);

  ASSERT(iters >= 0);
  ASSERT(fabsf(x[0] - 2.0f) < EPS);
  ASSERT(fabsf(x[1] - 1.0f) < EPS);
  sparse_free(A);
  PASS();
}

// ---------------------------------------------------------------------------
// CG solver: 3x3 SPD system
//   A = [4 1 0]    b = [6]    exact x = [1, 2, 3]
//       [1 3 1]        [10]   (verified by hand)
//       [0 1 2]        [8]
// ---------------------------------------------------------------------------

TEST test_cg_3x3(void) {
  sparse_mat *A = sparse_new(3, 3, 7);
  sparse_add(A, 0, 0, 4.0f); sparse_add(A, 0, 1, 1.0f);
  sparse_add(A, 1, 0, 1.0f); sparse_add(A, 1, 1, 3.0f); sparse_add(A, 1, 2, 1.0f);
  sparse_add(A, 2, 1, 1.0f); sparse_add(A, 2, 2, 2.0f);

  float b[3] = {6.0f, 10.0f, 8.0f};
  float x[3] = {0};
  int iters = sparse_solve_cg(A, b, x, 200, 1e-6f);

  ASSERT(iters >= 0);
  ASSERT(fabsf(x[0] - 1.0f) < EPS);
  ASSERT(fabsf(x[1] - 2.0f) < EPS);
  ASSERT(fabsf(x[2] - 3.0f) < EPS);
  sparse_free(A);
  PASS();
}

// CG with duplicate entries: same position added twice should sum.
TEST test_cg_duplicate_entries(void) {
  sparse_mat *A = sparse_new(2, 2, 4);
  // Diagonal = 2+2 = 4, off-diag 0
  sparse_add(A, 0, 0, 2.0f); sparse_add(A, 0, 0, 2.0f);
  sparse_add(A, 1, 1, 2.0f); sparse_add(A, 1, 1, 2.0f);

  float b[2] = {8.0f, 4.0f};
  float x[2] = {0};
  int iters = sparse_solve_cg(A, b, x, 100, 1e-6f);

  ASSERT(iters >= 0);
  ASSERT(fabsf(x[0] - 2.0f) < EPS);
  ASSERT(fabsf(x[1] - 1.0f) < EPS);
  sparse_free(A);
  PASS();
}

// ---------------------------------------------------------------------------
// connected_components_3d
// ---------------------------------------------------------------------------

// Two disconnected 1-voxel blobs in a 3x3x3 volume.
TEST test_cc3d_two_blobs(void) {
  uint8_t mask[27] = {0};
  int     labels[27];
  mask[0]  = 1;  // voxel (0,0,0)
  mask[26] = 1;  // voxel (2,2,2) — not 6-connected to (0,0,0)

  int n = connected_components_3d(mask, labels, 3, 3, 3);
  ASSERT_EQ(2, n);
  ASSERT(labels[0]  != 0);
  ASSERT(labels[26] != 0);
  ASSERT(labels[0]  != labels[26]);
  // All background voxels must be 0.
  for (int i = 1; i < 26; i++) ASSERT_EQ(0, labels[i]);
  PASS();
}

// Single connected blob: 3x1x1 bar.
TEST test_cc3d_single_bar(void) {
  uint8_t mask[9] = {0};
  int     labels[9];
  mask[0] = mask[1] = mask[2] = 1;  // x=0,1,2 in 3x1x3... let's use 1x1x3

  int n = connected_components_3d(mask, labels, 1, 1, 3);
  ASSERT_EQ(1, n);
  for (int i = 0; i < 3; i++) ASSERT_EQ(1, labels[i]);
  PASS();
}

// Empty mask → 0 components.
TEST test_cc3d_empty(void) {
  uint8_t mask[8]   = {0};
  int     labels[8] = {0};
  int n = connected_components_3d(mask, labels, 2, 2, 2);
  ASSERT_EQ(0, n);
  PASS();
}

// Full cube → 1 component.
TEST test_cc3d_full_cube(void) {
  uint8_t mask[8];
  int     labels[8];
  memset(mask, 1, 8);
  int n = connected_components_3d(mask, labels, 2, 2, 2);
  ASSERT_EQ(1, n);
  for (int i = 0; i < 8; i++) ASSERT_EQ(1, labels[i]);
  PASS();
}

// ---------------------------------------------------------------------------
// dijkstra_3d
// ---------------------------------------------------------------------------

// 1D corridor of 5 voxels, uniform cost 1.  Start at idx 0.
// Expected dist: 0, 1, 2, 3, 4
TEST test_dijkstra_1d_corridor(void) {
  float cost[5]  = {1, 1, 1, 1, 1};
  float dist[5];
  dijkstra_3d(cost, 0, dist, 1, 1, 5);

  for (int i = 0; i < 5; i++)
    ASSERT(fabsf(dist[i] - (float)i) < EPS);
  PASS();
}

// 3D grid 3x3x3, all cost 1, start at centre (idx 13).
// All neighbours at 6-distance 1 should have dist 1.
TEST test_dijkstra_3d_centre(void) {
  float cost[27], dist[27];
  for (int i = 0; i < 27; i++) cost[i] = 1.0f;
  dijkstra_3d(cost, 13, dist, 3, 3, 3);  // centre = z=1,y=1,x=1 → 1*9+1*3+1=13

  ASSERT(fabsf(dist[13]) < EPS);  // source
  // Face neighbours at Manhattan distance 1 in 3D grid
  ASSERT(fabsf(dist[13 - 1]  - 1.0f) < EPS);  // x-1
  ASSERT(fabsf(dist[13 + 1]  - 1.0f) < EPS);  // x+1
  ASSERT(fabsf(dist[13 - 3]  - 1.0f) < EPS);  // y-1
  ASSERT(fabsf(dist[13 + 3]  - 1.0f) < EPS);  // y+1
  ASSERT(fabsf(dist[13 - 9]  - 1.0f) < EPS);  // z-1
  ASSERT(fabsf(dist[13 + 9]  - 1.0f) < EPS);  // z+1
  PASS();
}

// High-cost wall: path must route around it.
// 1x1x5: cost[2] = 100, rest = 1. Shortest path from 0 to 4 must go through 2.
TEST test_dijkstra_wall(void) {
  float cost[5] = {1, 1, 100, 1, 1};
  float dist[5];
  dijkstra_3d(cost, 0, dist, 1, 1, 5);

  ASSERT(fabsf(dist[0] - 0.0f)   < EPS);
  ASSERT(fabsf(dist[1] - 1.0f)   < EPS);
  ASSERT(fabsf(dist[2] - 101.0f) < EPS);  // 1 + 100
  ASSERT(fabsf(dist[3] - 102.0f) < EPS);
  ASSERT(fabsf(dist[4] - 103.0f) < EPS);
  PASS();
}

// Isolated voxel: zero-cost single voxel, rest are unreachable because grid is 1x1x1.
TEST test_dijkstra_single_voxel(void) {
  float cost[1] = {5.0f};
  float dist[1];
  dijkstra_3d(cost, 0, dist, 1, 1, 1);
  ASSERT(fabsf(dist[0]) < EPS);  // source always 0
  PASS();
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(sparse_suite) {
  RUN_TEST(test_sparse_new_free);
  RUN_TEST(test_sparse_add);
  RUN_TEST(test_sparse_grow);
  RUN_TEST(test_cg_diagonal_2x2);
  RUN_TEST(test_cg_3x3);
  RUN_TEST(test_cg_duplicate_entries);
}

SUITE(cc3d_suite) {
  RUN_TEST(test_cc3d_two_blobs);
  RUN_TEST(test_cc3d_single_bar);
  RUN_TEST(test_cc3d_empty);
  RUN_TEST(test_cc3d_full_cube);
}

SUITE(dijkstra_suite) {
  RUN_TEST(test_dijkstra_1d_corridor);
  RUN_TEST(test_dijkstra_3d_centre);
  RUN_TEST(test_dijkstra_wall);
  RUN_TEST(test_dijkstra_single_voxel);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(sparse_suite);
  RUN_SUITE(cc3d_suite);
  RUN_SUITE(dijkstra_suite);
  GREATEST_MAIN_END();
}
