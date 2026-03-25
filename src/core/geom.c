#include "geom.h"
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// quad_surface
// ---------------------------------------------------------------------------

quad_surface *quad_surface_new(int rows, int cols) {
  assert(rows > 0 && cols > 0);
  quad_surface *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->rows = rows;
  s->cols = cols;
  s->points = calloc((size_t)(rows * cols), sizeof(vec3f));
  if (!s->points) { free(s); return NULL; }
  return s;
}

void quad_surface_free(quad_surface *s) {
  if (!s) return;
  free(s->points);
  free(s->normals);
  free(s->mask);
  free(s->id);
  free(s);
}

vec3f quad_surface_get(const quad_surface *s, int row, int col) {
  assert(s && row >= 0 && row < s->rows && col >= 0 && col < s->cols);
  return s->points[row * s->cols + col];
}

void quad_surface_set(quad_surface *s, int row, int col, vec3f point) {
  assert(s && row >= 0 && row < s->rows && col >= 0 && col < s->cols);
  s->points[row * s->cols + col] = point;
  // invalidate cached normals
  if (s->normals) { free(s->normals); s->normals = NULL; }
}

vec3f quad_surface_sample(const quad_surface *s, float u, float v) {
  assert(s);
  // clamp u,v to [0,1]
  if (u < 0.0f) u = 0.0f;
  if (u > 1.0f) u = 1.0f;
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;

  float fu = u * (float)(s->cols - 1);
  float fv = v * (float)(s->rows - 1);
  int c0 = (int)fu;
  int r0 = (int)fv;
  int c1 = c0 + 1 < s->cols ? c0 + 1 : c0;
  int r1 = r0 + 1 < s->rows ? r0 + 1 : r0;
  float tc = fu - (float)c0;
  float tr = fv - (float)r0;

  vec3f p00 = s->points[r0 * s->cols + c0];
  vec3f p01 = s->points[r0 * s->cols + c1];
  vec3f p10 = s->points[r1 * s->cols + c0];
  vec3f p11 = s->points[r1 * s->cols + c1];

  vec3f top = vec3f_lerp(p00, p01, tc);
  vec3f bot = vec3f_lerp(p10, p11, tc);
  return vec3f_lerp(top, bot, tr);
}

void quad_surface_compute_normals(quad_surface *s) {
  assert(s);
  if (!s->normals) {
    s->normals = calloc((size_t)(s->rows * s->cols), sizeof(vec3f));
    if (!s->normals) return;
  }

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      vec3f p = s->points[r * s->cols + c];

      // finite-difference neighbors; clamp at edges
      int rp = r + 1 < s->rows ? r + 1 : r;
      int rm = r - 1 >= 0      ? r - 1 : r;
      int cp = c + 1 < s->cols ? c + 1 : c;
      int cm = c - 1 >= 0      ? c - 1 : c;

      vec3f drow = vec3f_sub(s->points[rp * s->cols + c], s->points[rm * s->cols + c]);
      vec3f dcol = vec3f_sub(s->points[r  * s->cols + cp], s->points[r  * s->cols + cm]);
      (void)p;

      s->normals[r * s->cols + c] = vec3f_normalize(vec3f_cross(dcol, drow));
    }
  }
}

float quad_surface_area(const quad_surface *s) {
  assert(s);
  float area = 0.0f;
  for (int r = 0; r < s->rows - 1; r++) {
    for (int c = 0; c < s->cols - 1; c++) {
      vec3f p00 = s->points[r       * s->cols + c    ];
      vec3f p01 = s->points[r       * s->cols + c + 1];
      vec3f p10 = s->points[(r + 1) * s->cols + c    ];
      vec3f p11 = s->points[(r + 1) * s->cols + c + 1];

      // two triangles per quad
      vec3f d0 = vec3f_cross(vec3f_sub(p01, p00), vec3f_sub(p10, p00));
      vec3f d1 = vec3f_cross(vec3f_sub(p10, p11), vec3f_sub(p01, p11));
      area += 0.5f * vec3f_len(d0) + 0.5f * vec3f_len(d1);
    }
  }
  return area;
}

quad_surface *quad_surface_clone(const quad_surface *s) {
  assert(s);
  quad_surface *c = quad_surface_new(s->rows, s->cols);
  if (!c) return NULL;
  memcpy(c->points, s->points, (size_t)(s->rows * s->cols) * sizeof(vec3f));
  if (s->normals) {
    c->normals = malloc((size_t)(s->rows * s->cols) * sizeof(vec3f));
    if (c->normals) memcpy(c->normals, s->normals, (size_t)(s->rows * s->cols) * sizeof(vec3f));
  }
  if (s->mask) {
    c->mask = malloc((size_t)(s->rows * s->cols));
    if (c->mask) memcpy(c->mask, s->mask, (size_t)(s->rows * s->cols));
  }
  if (s->id) {
    c->id = strdup(s->id);
  }
  return c;
}

// ---------------------------------------------------------------------------
// plane_surface
// ---------------------------------------------------------------------------

plane_surface plane_surface_from_normal(vec3f origin, vec3f normal) {
  normal = vec3f_normalize(normal);

  // pick an arbitrary vector not parallel to normal to build basis
  vec3f ref = (fabsf(normal.x) < 0.9f) ? (vec3f){1,0,0} : (vec3f){0,1,0};
  vec3f u = vec3f_normalize(vec3f_cross(ref, normal));
  vec3f v = vec3f_cross(normal, u);

  return (plane_surface){ .origin = origin, .normal = normal, .u_axis = u, .v_axis = v };
}

vec3f plane_surface_project(const plane_surface *p, vec3f world_point) {
  assert(p);
  float d = vec3f_dot(vec3f_sub(world_point, p->origin), p->normal);
  return vec3f_sub(world_point, vec3f_scale(p->normal, d));
}

vec3f plane_surface_sample(const plane_surface *p, float u, float v) {
  assert(p);
  return vec3f_add(p->origin, vec3f_add(vec3f_scale(p->u_axis, u), vec3f_scale(p->v_axis, v)));
}

float plane_surface_dist(const plane_surface *p, vec3f point) {
  assert(p);
  return vec3f_dot(vec3f_sub(point, p->origin), p->normal);
}

// ---------------------------------------------------------------------------
// HAMT (Hash Array Mapped Trie)
// 32-way branching (5 bits per level), up to 13 levels for 64-bit keys
// ---------------------------------------------------------------------------

#define HAMT_BITS    5
#define HAMT_WIDTH   32   // 1 << HAMT_BITS
#define HAMT_MASK    0x1f

// Node types
#define HAMT_LEAF    0
#define HAMT_BRANCH  1

struct hamt_node {
  atomic_int refcount;
  int        type;      // HAMT_LEAF or HAMT_BRANCH
  union {
    struct {
      uint64_t    key;
      void       *val;
    } leaf;
    struct {
      uint32_t    bitmap;
      size_t      count;   // hamt_len cache (-1 = not computed)
      hamt_node **children; // packed array of popcount(bitmap) children
    } branch;
  };
};

static hamt_node *node_alloc_leaf(uint64_t key, void *val) {
  hamt_node *n = calloc(1, sizeof(*n));
  if (!n) return NULL;
  atomic_init(&n->refcount, 1);
  n->type = HAMT_LEAF;
  n->leaf.key = key;
  n->leaf.val = val;
  return n;
}

static hamt_node *node_alloc_branch(uint32_t bitmap, int nchildren) {
  hamt_node *n = calloc(1, sizeof(*n));
  if (!n) return NULL;
  atomic_init(&n->refcount, 1);
  n->type = HAMT_BRANCH;
  n->branch.bitmap = bitmap;
  n->branch.count = (size_t)-1;
  if (nchildren > 0) {
    n->branch.children = calloc((size_t)nchildren, sizeof(hamt_node *));
    if (!n->branch.children) { free(n); return NULL; }
  }
  return n;
}

static inline int popcount32(uint32_t x) {
  return __builtin_popcount(x);
}

static inline int child_index(uint32_t bitmap, int bit) {
  // number of set bits below this bit position
  return popcount32(bitmap & ((1u << bit) - 1u));
}

hamt_node *hamt_retain(hamt_node *n) {
  if (n) atomic_fetch_add(&n->refcount, 1);
  return n;
}

void hamt_release(hamt_node *n) {
  if (!n) return;
  if (atomic_fetch_sub(&n->refcount, 1) > 1) return;
  // refcount hit 0 — free
  if (n->type == HAMT_BRANCH) {
    int nc = popcount32(n->branch.bitmap);
    for (int i = 0; i < nc; i++) hamt_release(n->branch.children[i]);
    free(n->branch.children);
  }
  free(n);
}

hamt_node *hamt_empty(void) {
  return node_alloc_branch(0, 0);
}

void *hamt_get(const hamt_node *root, uint64_t key) {
  const hamt_node *cur = root;
  uint64_t k = key;
  for (int level = 0; level < 13; level++) {
    if (!cur) return NULL;
    if (cur->type == HAMT_LEAF) {
      return cur->leaf.key == key ? cur->leaf.val : NULL;
    }
    int bit = (int)(k & HAMT_MASK);
    uint32_t mask = 1u << bit;
    if (!(cur->branch.bitmap & mask)) return NULL;
    int idx = child_index(cur->branch.bitmap, bit);
    cur = cur->branch.children[idx];
    k >>= HAMT_BITS;
  }
  return NULL;
}

// Returns a new branch that is a copy of src with child at position idx replaced by new_child.
// Retains all kept children. Does NOT release old node.
static hamt_node *branch_copy_replace(const hamt_node *src, int idx, hamt_node *new_child) {
  int nc = popcount32(src->branch.bitmap);
  hamt_node *n = node_alloc_branch(src->branch.bitmap, nc);
  if (!n) return NULL;
  for (int i = 0; i < nc; i++) {
    if (i == idx) {
      n->branch.children[i] = new_child; // already retained by caller
    } else {
      n->branch.children[i] = hamt_retain(src->branch.children[i]);
    }
  }
  return n;
}

// Returns a new branch that is a copy of src with a new child inserted at position idx.
static hamt_node *branch_copy_insert(const hamt_node *src, int bit, hamt_node *new_child) {
  int nc = popcount32(src->branch.bitmap);
  hamt_node *n = node_alloc_branch(src->branch.bitmap | (1u << bit), nc + 1);
  if (!n) return NULL;
  int new_idx = child_index(src->branch.bitmap | (1u << bit), bit);
  for (int i = 0, j = 0; j <= nc; j++) {
    if (j == new_idx) {
      n->branch.children[j] = new_child;
    } else {
      n->branch.children[j] = hamt_retain(src->branch.children[i++]);
    }
  }
  return n;
}

// Returns a new branch that is a copy of src with child at idx removed.
static hamt_node *branch_copy_remove(const hamt_node *src, int bit) {
  int nc = popcount32(src->branch.bitmap);
  int rm_idx = child_index(src->branch.bitmap, bit);
  hamt_node *n = node_alloc_branch(src->branch.bitmap & ~(1u << bit), nc - 1);
  if (!n) return NULL;
  for (int i = 0, j = 0; i < nc; i++) {
    if (i != rm_idx) n->branch.children[j++] = hamt_retain(src->branch.children[i]);
  }
  return n;
}

// Recursive set; returns new root or NULL on alloc failure.
// shift is how many bits have already been consumed (0 at top, +5 per level).
static hamt_node *hamt_set_rec(const hamt_node *node, uint64_t key, void *val, int shift) {
  if (!node) {
    // create a leaf
    return node_alloc_leaf(key, val);
  }

  if (node->type == HAMT_LEAF) {
    if (node->leaf.key == key) {
      // replace value
      return node_alloc_leaf(key, val);
    }
    // collision: create a branch holding both leaves
    int bit_existing = (int)((node->leaf.key >> shift) & HAMT_MASK);
    int bit_new      = (int)((key >> shift) & HAMT_MASK);

    if (bit_existing == bit_new) {
      // need to go deeper
      hamt_node *sub = hamt_set_rec(node, key, val, shift + HAMT_BITS);
      if (!sub) return NULL;
      hamt_node *branch = node_alloc_branch(1u << bit_new, 1);
      if (!branch) { hamt_release(sub); return NULL; }
      branch->branch.children[0] = sub;
      return branch;
    } else {
      // two distinct bits: new branch with both leaves
      hamt_node *leaf_new = node_alloc_leaf(key, val);
      if (!leaf_new) return NULL;
      hamt_node *leaf_ex = hamt_retain((hamt_node *)node);
      uint32_t bmap = (1u << bit_existing) | (1u << bit_new);
      hamt_node *branch = node_alloc_branch(bmap, 2);
      if (!branch) { hamt_release(leaf_new); hamt_release(leaf_ex); return NULL; }
      int idx_ex  = child_index(bmap, bit_existing);
      int idx_new = child_index(bmap, bit_new);
      branch->branch.children[idx_ex]  = leaf_ex;
      branch->branch.children[idx_new] = leaf_new;
      return branch;
    }
  }

  // node is a branch
  int bit = (int)((key >> shift) & HAMT_MASK);
  uint32_t mask = 1u << bit;

  if (node->branch.bitmap & mask) {
    int idx = child_index(node->branch.bitmap, bit);
    hamt_node *new_child = hamt_set_rec(node->branch.children[idx], key, val, shift + HAMT_BITS);
    if (!new_child) return NULL;
    return branch_copy_replace(node, idx, new_child);
  } else {
    hamt_node *leaf = node_alloc_leaf(key, val);
    if (!leaf) return NULL;
    return branch_copy_insert(node, bit, leaf);
  }
}

hamt_node *hamt_set(hamt_node *root, uint64_t key, void *val) {
  hamt_node *new_root = hamt_set_rec(root, key, val, 0);
  return new_root ? new_root : root;
}

static hamt_node *hamt_del_rec(const hamt_node *node, uint64_t key, int shift, bool *deleted) {
  if (!node) return NULL;

  if (node->type == HAMT_LEAF) {
    if (node->leaf.key == key) { *deleted = true; return NULL; }
    return hamt_retain((hamt_node *)node);
  }

  int bit = (int)((key >> shift) & HAMT_MASK);
  uint32_t mask = 1u << bit;
  if (!(node->branch.bitmap & mask)) {
    // key not present
    return hamt_retain((hamt_node *)node);
  }

  int idx = child_index(node->branch.bitmap, bit);
  hamt_node *new_child = hamt_del_rec(node->branch.children[idx], key, shift + HAMT_BITS, deleted);

  if (!*deleted) {
    // Nothing changed, return a retained copy
    hamt_release(new_child);
    return hamt_retain((hamt_node *)node);
  }

  if (new_child == NULL) {
    // child was removed entirely
    int nc = popcount32(node->branch.bitmap);
    if (nc == 1) {
      // branch becomes empty; return NULL so parent can prune
      return NULL;
    }
    return branch_copy_remove(node, bit);
  }

  return branch_copy_replace(node, idx, new_child);
}

hamt_node *hamt_del(hamt_node *root, uint64_t key) {
  bool deleted = false;
  hamt_node *new_root = hamt_del_rec(root, key, 0, &deleted);
  if (!new_root) {
    // deleted last element — return empty branch
    return hamt_empty();
  }
  return new_root;
}

static size_t hamt_len_rec(const hamt_node *n) {
  if (!n) return 0;
  if (n->type == HAMT_LEAF) return 1;
  if (n->branch.count != (size_t)-1) return n->branch.count;
  size_t total = 0;
  int nc = popcount32(n->branch.bitmap);
  for (int i = 0; i < nc; i++) total += hamt_len_rec(n->branch.children[i]);
  // cache result (safe: count is logically immutable after construction)
  ((hamt_node *)n)->branch.count = total;
  return total;
}

size_t hamt_len(const hamt_node *root) {
  return hamt_len_rec(root);
}
