#pragma once
#include "math.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// quad_surface: regular grid of 3D points (u,v parameterization)
// ---------------------------------------------------------------------------

typedef struct {
  int rows, cols;       // grid dimensions
  vec3f *points;        // rows*cols points in row-major order
  vec3f *normals;       // cached normals (NULL if not computed)
  uint8_t *mask;        // validity mask (1=valid, 0=invalid), NULL if all valid
  char *id;             // surface identifier
} quad_surface;

quad_surface *quad_surface_new(int rows, int cols);
void          quad_surface_free(quad_surface *s);
vec3f         quad_surface_get(const quad_surface *s, int row, int col);
void          quad_surface_set(quad_surface *s, int row, int col, vec3f point);
vec3f         quad_surface_sample(const quad_surface *s, float u, float v);  // bilinear interpolation
void          quad_surface_compute_normals(quad_surface *s);                 // cross product of grid neighbors
float         quad_surface_area(const quad_surface *s);                      // total surface area in voxel^2
quad_surface *quad_surface_clone(const quad_surface *s);

// ---------------------------------------------------------------------------
// plane_surface: infinite plane defined by origin + normal + basis vectors
// ---------------------------------------------------------------------------

typedef struct {
  vec3f origin;
  vec3f normal;
  vec3f u_axis, v_axis;  // basis vectors in the plane
} plane_surface;

plane_surface plane_surface_from_normal(vec3f origin, vec3f normal);         // auto-compute basis
vec3f         plane_surface_project(const plane_surface *p, vec3f world_point);  // project onto plane
vec3f         plane_surface_sample(const plane_surface *p, float u, float v);    // (u,v) -> world point
float         plane_surface_dist(const plane_surface *p, vec3f point);           // signed distance

// ---------------------------------------------------------------------------
// HAMT (Hash Array Mapped Trie) for persistent undo
// Immutable map: every "mutation" returns a new root sharing structure with the old one
// ---------------------------------------------------------------------------

typedef struct hamt_node hamt_node;

hamt_node *hamt_empty(void);
hamt_node *hamt_set(hamt_node *root, uint64_t key, void *val);  // returns NEW root
void      *hamt_get(const hamt_node *root, uint64_t key);
hamt_node *hamt_del(hamt_node *root, uint64_t key);             // returns NEW root
size_t     hamt_len(const hamt_node *root);

// reference counting for structural sharing
hamt_node *hamt_retain(hamt_node *n);
void       hamt_release(hamt_node *n);

// ---------------------------------------------------------------------------
// tri_mesh: indexed triangle mesh
// ---------------------------------------------------------------------------

typedef struct {
  vec3f   *verts;      // vertex positions
  int      num_verts;
  int     *indices;    // triangle indices, 3 per face
  int      num_faces;
} tri_mesh;

tri_mesh *tri_mesh_new(int num_verts, int num_faces);
void      tri_mesh_free(tri_mesh *m);

// Voxelize a closed triangle mesh to a binary volume using ray casting.
// Returns a calloc'd mask of size d*h*w (z-major); caller must free.
uint8_t  *mesh_voxelize(const tri_mesh *m, int d, int h, int w);

// Per-face and aggregate quality metrics.
typedef struct {
  float   min_angle_deg;
  float   max_angle_deg;
  float   avg_angle_deg;
  float   max_aspect_ratio;   // longest edge / shortest edge, worst face
  int     self_intersections; // approximate count (O(n^2), capped at 10000 pairs)
} mesh_quality_t;

mesh_quality_t mesh_quality(const tri_mesh *m);

// Quadric error metric edge collapse simplification.
// Returns a new mesh with at most target_faces faces; caller owns result.
tri_mesh *mesh_simplify(const tri_mesh *m, int target_faces);

// Laplacian smoothing in-place.
void mesh_smooth(tri_mesh *m, int iterations, float lambda);
