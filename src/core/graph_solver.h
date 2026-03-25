#pragma once
#include <stdbool.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// graph — adjacency list for an undirected weighted graph.
//
// Edge weights represent "certainty" (confidence that the edge is correct).
// Each edge also carries a k-offset (winding angle difference in degrees)
// used during label assignment.
//
// For winding-number assignment the solver:
//   1. Runs loopy belief propagation to find consistent f_tilde values.
//   2. Builds a max-weight spanning tree (Prim) and propagates labels via k.
// ---------------------------------------------------------------------------

typedef struct {
  int   src, dst;
  float weight;   // certainty [0..1]
  float k;        // winding offset (degrees); used during label assignment
} graph_edge;

typedef struct graph graph;

graph *graph_new(int num_nodes);
void   graph_add_edge(graph *g, int src, int dst, float weight);
void   graph_add_edge_k(graph *g, int src, int dst, float weight, float k);
void   graph_free(graph *g);

int graph_num_nodes(const graph *g);
int graph_num_edges(const graph *g);

// ---------------------------------------------------------------------------
// Algorithms
// ---------------------------------------------------------------------------

// Loopy belief propagation on a pairwise MRF.
// Each node gets an integer label in [0, max_labels).
// Returns number of iterations run; -1 on allocation failure.
int graph_solve_bp(const graph *g, int *labels, int max_labels, int max_iter);

// Maximum spanning tree (Prim, by edge weight).
// Returns a new graph (caller must free) containing only MST edges.
graph *graph_mst(const graph *g);

// Label nodes via MST + winding-angle propagation (f_star assignment).
// init_f[i] = initial winding estimate for node i (degrees).
// out_f[i]  = solved winding angle for node i.
// Returns number of labelled nodes.
int graph_assign_winding(const graph *g, const float *init_f, float *out_f);

// Connected components (iterative DFS).
// component_ids[i] = component index for node i (0-based).
// Returns number of components.
int graph_connected_components(const graph *g, int *component_ids);
