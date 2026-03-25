#include "greatest.h"
#include "core/graph_solver.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Tests: lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  graph *g = graph_new(4);
  ASSERT_NEQ(NULL, g);
  ASSERT_EQ(4, graph_num_nodes(g));
  ASSERT_EQ(0, graph_num_edges(g));
  graph_free(g);
  PASS();
}

TEST test_add_edges(void) {
  graph *g = graph_new(5);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 0.9f);
  graph_add_edge(g, 1, 2, 0.8f);
  graph_add_edge(g, 2, 3, 0.7f);
  ASSERT_EQ(3, graph_num_edges(g));
  graph_free(g);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: connected components
// ---------------------------------------------------------------------------

TEST test_connected_components_chain(void) {
  // 0-1-2-3 chain: one component
  graph *g = graph_new(4);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 1.0f);
  graph_add_edge(g, 1, 2, 1.0f);
  graph_add_edge(g, 2, 3, 1.0f);

  int comp[4];
  int n = graph_connected_components(g, comp);
  ASSERT_EQ(1, n);
  // All nodes in same component
  ASSERT_EQ(comp[0], comp[1]);
  ASSERT_EQ(comp[1], comp[2]);
  ASSERT_EQ(comp[2], comp[3]);

  graph_free(g);
  PASS();
}

TEST test_connected_components_split(void) {
  // 0-1  2-3 — two components
  graph *g = graph_new(4);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 1.0f);
  graph_add_edge(g, 2, 3, 1.0f);

  int comp[4];
  int n = graph_connected_components(g, comp);
  ASSERT_EQ(2, n);
  ASSERT_EQ(comp[0], comp[1]);
  ASSERT_EQ(comp[2], comp[3]);
  ASSERT(comp[0] != comp[2]);

  graph_free(g);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: MST
// ---------------------------------------------------------------------------

TEST test_mst_triangle(void) {
  // Triangle 0-1 (w=0.5), 1-2 (w=0.9), 0-2 (w=0.3)
  // MST should include the two heaviest: 1-2 and 0-1 (or 1-2 and 0-2 depending
  // on tie-breaking), but NOT both 0-2 AND 0-1 AND 1-2 all three.
  graph *g = graph_new(3);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 0.5f);
  graph_add_edge(g, 1, 2, 0.9f);
  graph_add_edge(g, 0, 2, 0.3f);

  graph *mst = graph_mst(g);
  ASSERT_NEQ(NULL, mst);
  ASSERT_EQ(3, graph_num_nodes(mst));
  // MST of 3 nodes has exactly 2 edges
  ASSERT_EQ(2, graph_num_edges(mst));

  graph_free(mst);
  graph_free(g);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: BP on 4-node chain
// ---------------------------------------------------------------------------

TEST test_bp_chain(void) {
  // 0-1-2-3 chain with uniform high weights.
  // With 2 labels, all nodes should converge to the same label.
  graph *g = graph_new(4);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 0.95f);
  graph_add_edge(g, 1, 2, 0.95f);
  graph_add_edge(g, 2, 3, 0.95f);

  int labels[4] = {0};
  int iters = graph_solve_bp(g, labels, 2, 20);
  ASSERT(iters >= 0);

  // All nodes should agree on one label
  ASSERT_EQ(labels[0], labels[1]);
  ASSERT_EQ(labels[1], labels[2]);
  ASSERT_EQ(labels[2], labels[3]);

  graph_free(g);
  PASS();
}

TEST test_bp_single_label(void) {
  graph *g = graph_new(3);
  ASSERT_NEQ(NULL, g);
  graph_add_edge(g, 0, 1, 1.0f);
  graph_add_edge(g, 1, 2, 1.0f);

  int labels[3] = {0};
  int iters = graph_solve_bp(g, labels, 1, 10);
  ASSERT(iters >= 0);
  ASSERT_EQ(0, labels[0]);
  ASSERT_EQ(0, labels[1]);
  ASSERT_EQ(0, labels[2]);

  graph_free(g);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: winding angle assignment
// ---------------------------------------------------------------------------

TEST test_assign_winding_chain(void) {
  // 4-node chain where each edge has k=360 (one full winding apart).
  // Starting from init_f[0]=0, expect out_f = {0, 360, 720, 1080}.
  graph *g = graph_new(4);
  ASSERT_NEQ(NULL, g);
  graph_add_edge_k(g, 0, 1, 1.0f, 360.0f);
  graph_add_edge_k(g, 1, 2, 1.0f, 360.0f);
  graph_add_edge_k(g, 2, 3, 1.0f, 360.0f);

  float init_f[4] = {0.0f, 360.0f, 720.0f, 1080.0f};
  float out_f[4]  = {0.0f};
  int n = graph_assign_winding(g, init_f, out_f);
  ASSERT_EQ(4, n);

  // Each node should be snapped to its correct winding
  ASSERT(fabsf(out_f[0] -    0.0f) < 1.0f);
  ASSERT(fabsf(out_f[1] -  360.0f) < 1.0f);
  ASSERT(fabsf(out_f[2] -  720.0f) < 1.0f);
  ASSERT(fabsf(out_f[3] - 1080.0f) < 1.0f);

  graph_free(g);
  PASS();
}

TEST test_assign_winding_consistent(void) {
  // Triangle where all k offsets are self-consistent (sum around cycle = 0).
  graph *g = graph_new(3);
  ASSERT_NEQ(NULL, g);
  graph_add_edge_k(g, 0, 1, 1.0f,  360.0f);
  graph_add_edge_k(g, 1, 2, 1.0f,  360.0f);
  graph_add_edge_k(g, 0, 2, 0.5f,  720.0f);  // lower weight, may not be in MST

  float init_f[3] = {0.0f, 360.0f, 720.0f};
  float out_f[3]  = {0.0f};
  int n = graph_assign_winding(g, init_f, out_f);
  ASSERT_EQ(3, n);

  // MST will pick the two highest-weight edges (0-1 and 1-2)
  ASSERT(fabsf(out_f[1] - out_f[0] -  360.0f) < 1.0f);
  ASSERT(fabsf(out_f[2] - out_f[1] -  360.0f) < 1.0f);

  graph_free(g);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(graph_solver_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_add_edges);
  RUN_TEST(test_connected_components_chain);
  RUN_TEST(test_connected_components_split);
  RUN_TEST(test_mst_triangle);
  RUN_TEST(test_bp_chain);
  RUN_TEST(test_bp_single_label);
  RUN_TEST(test_assign_winding_chain);
  RUN_TEST(test_assign_winding_consistent);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(graph_solver_suite);
  GREATEST_MAIN_END();
}
