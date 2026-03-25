#include "core/graph_solver.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define EDGE_CAP_INIT 8
#define BP_DAMPING    0.5f   // message damping for loopy BP convergence

// ---------------------------------------------------------------------------
// Adjacency list
// ---------------------------------------------------------------------------

typedef struct adj_entry {
  int   dst;
  float weight;
  float k;
} adj_entry;

typedef struct {
  adj_entry *adj;
  int        n_adj;
  int        cap_adj;
} node_t;

struct graph {
  node_t    *nodes;
  int        num_nodes;
  graph_edge *edges;    // flat edge list (mirrors adjacency)
  int        num_edges;
  int        cap_edges;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

graph *graph_new(int num_nodes) {
  if (num_nodes <= 0) return NULL;
  graph *g = calloc(1, sizeof(*g));
  if (!g) return NULL;
  g->nodes = calloc((size_t)num_nodes, sizeof(node_t));
  if (!g->nodes) { free(g); return NULL; }
  g->num_nodes  = num_nodes;
  g->cap_edges  = EDGE_CAP_INIT;
  g->edges      = malloc((size_t)g->cap_edges * sizeof(graph_edge));
  if (!g->edges) { free(g->nodes); free(g); return NULL; }
  return g;
}

static bool adj_push(node_t *n, int dst, float w, float k) {
  if (n->n_adj == n->cap_adj) {
    int nc = n->cap_adj ? n->cap_adj * 2 : 4;
    adj_entry *na = realloc(n->adj, (size_t)nc * sizeof(adj_entry));
    if (!na) return false;
    n->adj = na; n->cap_adj = nc;
  }
  n->adj[n->n_adj++] = (adj_entry){dst, w, k};
  return true;
}

void graph_add_edge_k(graph *g, int src, int dst, float weight, float k) {
  if (!g || src < 0 || dst < 0 || src >= g->num_nodes || dst >= g->num_nodes)
    return;
  // Grow flat edge list
  if (g->num_edges == g->cap_edges) {
    int nc = g->cap_edges * 2;
    graph_edge *ne = realloc(g->edges, (size_t)nc * sizeof(graph_edge));
    if (!ne) return;
    g->edges = ne; g->cap_edges = nc;
  }
  g->edges[g->num_edges++] = (graph_edge){src, dst, weight, k};
  // Undirected adjacency (both directions)
  adj_push(&g->nodes[src], dst, weight,  k);
  adj_push(&g->nodes[dst], src, weight, -k);
}

void graph_add_edge(graph *g, int src, int dst, float weight) {
  graph_add_edge_k(g, src, dst, weight, 0.0f);
}

void graph_free(graph *g) {
  if (!g) return;
  for (int i = 0; i < g->num_nodes; i++) free(g->nodes[i].adj);
  free(g->nodes);
  free(g->edges);
  free(g);
}

int graph_num_nodes(const graph *g) { return g ? g->num_nodes : 0; }
int graph_num_edges(const graph *g) { return g ? g->num_edges : 0; }

// ---------------------------------------------------------------------------
// Loopy Belief Propagation (sum-product on pairwise MRF)
//
// Unary: uniform (no prior).
// Pairwise: Potts-like — if labels match, cost 0; otherwise cost = 1/weight.
// Messages indexed as msg[edge_idx * max_labels + label].
// We use the undirected flat edge list; each undirected edge has two
// directed messages: fwd (src→dst) and bwd (dst→src).
// ---------------------------------------------------------------------------

int graph_solve_bp(const graph *g, int *labels, int max_labels, int max_iter) {
  if (!g || !labels || max_labels < 1 || max_iter < 1) return -1;
  int N = g->num_nodes;
  int E = g->num_edges;
  int L = max_labels;

  // messages: 2*E*L floats (fwd and bwd per undirected edge)
  float *msg = calloc((size_t)(2 * E * L), sizeof(float));
  float *tmp = calloc((size_t)(2 * E * L), sizeof(float));
  if (!msg || !tmp) { free(msg); free(tmp); return -1; }

  // Initialise messages to uniform (1/L)
  float uni = 1.0f / (float)L;
  for (int i = 0; i < 2 * E * L; i++) msg[i] = uni;

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    memcpy(tmp, msg, (size_t)(2 * E * L) * sizeof(float));

    // Update each directed message
    for (int ei = 0; ei < E; ei++) {
      int src = g->edges[ei].src;
      int dst = g->edges[ei].dst;
      float w = g->edges[ei].weight;
      if (w < 1e-6f) w = 1e-6f;

      // fwd msg (src→dst) stored at [2*ei*L .. (2*ei+1)*L)
      // bwd msg (dst→src) stored at [(2*ei+1)*L .. (2*ei+2)*L)
      float *fwd = &tmp[2 * ei * L];
      float *bwd = &tmp[(2 * ei + 1) * L];

      for (int l_dst = 0; l_dst < L; l_dst++) {
        // Compute new fwd message: max over l_src of (psi(l_src,l_dst) * product of all incoming to src except this edge)
        float sum = 0.0f;
        for (int l_src = 0; l_src < L; l_src++) {
          float psi = (l_src == l_dst) ? 1.0f : w;  // Potts: agree=1, disagree=weight
          // Incoming to src from all other edges
          float belief = 1.0f;
          for (int ej = 0; ej < E; ej++) {
            if (ej == ei) continue;
            float *m_in = NULL;
            if (g->edges[ej].dst == src)
              m_in = &msg[2 * ej * L];       // fwd arrives at src
            else if (g->edges[ej].src == src)
              m_in = &msg[(2 * ej + 1) * L]; // bwd arrives at src
            if (m_in) belief *= m_in[l_src];
          }
          sum += psi * belief;
        }
        fwd[l_dst] = (1.0f - BP_DAMPING) * sum + BP_DAMPING * msg[2 * ei * L + l_dst];
      }

      // Symmetric bwd message (dst→src)
      for (int l_src = 0; l_src < L; l_src++) {
        float sum = 0.0f;
        for (int l_dst = 0; l_dst < L; l_dst++) {
          float psi = (l_src == l_dst) ? 1.0f : w;
          float belief = 1.0f;
          for (int ej = 0; ej < E; ej++) {
            if (ej == ei) continue;
            float *m_in = NULL;
            if (g->edges[ej].dst == dst)
              m_in = &msg[2 * ej * L];
            else if (g->edges[ej].src == dst)
              m_in = &msg[(2 * ej + 1) * L];
            if (m_in) belief *= m_in[l_dst];
          }
          sum += psi * belief;
        }
        bwd[l_src] = (1.0f - BP_DAMPING) * sum + BP_DAMPING * msg[(2 * ei + 1) * L + l_src];
      }

      // Normalise
      float s = 0.0f;
      for (int l = 0; l < L; l++) s += fwd[l];
      if (s > 0) for (int l = 0; l < L; l++) fwd[l] /= s;
      s = 0.0f;
      for (int l = 0; l < L; l++) s += bwd[l];
      if (s > 0) for (int l = 0; l < L; l++) bwd[l] /= s;
    }

    memcpy(msg, tmp, (size_t)(2 * E * L) * sizeof(float));
  }

  // Decode: for each node, compute belief = product of incoming messages
  float *belief = malloc((size_t)L * sizeof(float));
  if (!belief) { free(msg); free(tmp); return -1; }

  for (int n = 0; n < N; n++) {
    for (int l = 0; l < L; l++) belief[l] = 1.0f;
    for (int ei = 0; ei < E; ei++) {
      float *m_in = NULL;
      if (g->edges[ei].dst == n)
        m_in = &msg[2 * ei * L];
      else if (g->edges[ei].src == n)
        m_in = &msg[(2 * ei + 1) * L];
      if (!m_in) continue;
      for (int l = 0; l < L; l++) belief[l] *= m_in[l];
    }
    int best = 0;
    for (int l = 1; l < L; l++)
      if (belief[l] > belief[best]) best = l;
    labels[n] = best;
  }

  free(belief);
  free(msg);
  free(tmp);
  return iter;
}

// ---------------------------------------------------------------------------
// Maximum spanning tree (Prim, by weight)
// ---------------------------------------------------------------------------

graph *graph_mst(const graph *g) {
  if (!g || g->num_nodes == 0) return NULL;
  int N = g->num_nodes;

  graph *mst = graph_new(N);
  if (!mst) return NULL;

  bool  *in_mst = calloc((size_t)N, sizeof(bool));
  float *max_w  = malloc((size_t)N * sizeof(float));
  int   *parent = malloc((size_t)N * sizeof(int));
  float *parent_k = malloc((size_t)N * sizeof(float));
  if (!in_mst || !max_w || !parent || !parent_k) goto oom;

  for (int i = 0; i < N; i++) { max_w[i] = -FLT_MAX; parent[i] = -1; }
  max_w[0] = 0.0f;

  for (int step = 0; step < N; step++) {
    // Pick non-MST node with max weight
    int u = -1;
    for (int i = 0; i < N; i++)
      if (!in_mst[i] && (u == -1 || max_w[i] > max_w[u])) u = i;
    if (u == -1) break;
    in_mst[u] = true;

    if (parent[u] >= 0)
      graph_add_edge_k(mst, parent[u], u, max_w[u], parent_k[u]);

    for (int j = 0; j < g->nodes[u].n_adj; j++) {
      int v = g->nodes[u].adj[j].dst;
      float w = g->nodes[u].adj[j].weight;
      float k = g->nodes[u].adj[j].k;
      if (!in_mst[v] && w > max_w[v]) {
        max_w[v] = w; parent[v] = u; parent_k[v] = k;
      }
    }
  }

  free(in_mst); free(max_w); free(parent); free(parent_k);
  return mst;

oom:
  free(in_mst); free(max_w); free(parent); free(parent_k);
  graph_free(mst);
  return NULL;
}

// ---------------------------------------------------------------------------
// Winding-angle assignment via MST propagation
// Mirrors prim_mst_assign_f_star from main.cpp.
// ---------------------------------------------------------------------------

// Round f to the nearest multiple of 360 from f_init.
static float snap_winding(float f_init, float f_target) {
  int x = (int)roundf((f_target - f_init) / 360.0f);
  return f_init + (float)x * 360.0f;
}

int graph_assign_winding(const graph *g, const float *init_f, float *out_f) {
  if (!g || !init_f || !out_f || g->num_nodes == 0) return 0;
  int N = g->num_nodes;

  graph *mst = graph_mst(g);
  if (!mst) return 0;

  bool  *visited = calloc((size_t)N, sizeof(bool));
  int   *stack   = malloc((size_t)N * sizeof(int));
  if (!visited || !stack) { free(visited); free(stack); graph_free(mst); return 0; }

  for (int i = 0; i < N; i++) out_f[i] = init_f[i];

  // DFS from node 0
  int sp = 0;
  stack[sp++] = 0;
  visited[0]  = true;

  int count = 0;
  while (sp > 0) {
    int cur = stack[--sp];
    count++;
    for (int j = 0; j < mst->nodes[cur].n_adj; j++) {
      int   nb = mst->nodes[cur].adj[j].dst;
      float k  = mst->nodes[cur].adj[j].k;
      if (!visited[nb]) {
        visited[nb] = true;
        float candidate = out_f[cur] + k;
        out_f[nb] = snap_winding(init_f[nb], candidate);
        stack[sp++] = nb;
      }
    }
  }

  free(visited); free(stack);
  graph_free(mst);
  return count;
}

// ---------------------------------------------------------------------------
// Connected components (iterative DFS)
// ---------------------------------------------------------------------------

int graph_connected_components(const graph *g, int *component_ids) {
  if (!g || !component_ids) return 0;
  int N = g->num_nodes;

  for (int i = 0; i < N; i++) component_ids[i] = -1;

  int  *stack = malloc((size_t)N * sizeof(int));
  if (!stack) return 0;

  int comp = 0;
  for (int start = 0; start < N; start++) {
    if (component_ids[start] != -1) continue;
    int sp = 0;
    stack[sp++] = start;
    component_ids[start] = comp;
    while (sp > 0) {
      int cur = stack[--sp];
      for (int j = 0; j < g->nodes[cur].n_adj; j++) {
        int nb = g->nodes[cur].adj[j].dst;
        if (component_ids[nb] == -1) {
          component_ids[nb] = comp;
          stack[sp++] = nb;
        }
      }
    }
    comp++;
  }

  free(stack);
  return comp;
}
