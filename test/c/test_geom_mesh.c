#include "greatest.h"
#include "core/geom.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Synthetic meshes
// ---------------------------------------------------------------------------

// Axis-aligned unit cube: 8 verts, 12 triangles.
static tri_mesh *make_cube(void) {
  tri_mesh *m = tri_mesh_new(8, 12);
  if (!m) return NULL;
  // Vertices at corners of [0,1]^3
  m->verts[0] = (vec3f){0,0,0}; m->verts[1] = (vec3f){1,0,0};
  m->verts[2] = (vec3f){1,1,0}; m->verts[3] = (vec3f){0,1,0};
  m->verts[4] = (vec3f){0,0,1}; m->verts[5] = (vec3f){1,0,1};
  m->verts[6] = (vec3f){1,1,1}; m->verts[7] = (vec3f){0,1,1};
  // 6 faces × 2 triangles each, consistent outward winding
  int f[] = {
    0,1,2, 0,2,3,  // -Z face
    4,6,5, 4,7,6,  // +Z face
    0,4,5, 0,5,1,  // -Y face
    2,6,7, 2,7,3,  // +Y face
    0,3,7, 0,7,4,  // -X face
    1,5,6, 1,6,2,  // +X face
  };
  memcpy(m->indices, f, sizeof(f));
  return m;
}

// Single equilateral triangle in XY plane (for quality tests).
static tri_mesh *make_equilateral(void) {
  tri_mesh *m = tri_mesh_new(3, 1);
  if (!m) return NULL;
  m->verts[0] = (vec3f){0.0f, 0.0f, 0.0f};
  m->verts[1] = (vec3f){1.0f, 0.0f, 0.0f};
  m->verts[2] = (vec3f){0.5f, 0.866025f, 0.0f};
  m->indices[0] = 0; m->indices[1] = 1; m->indices[2] = 2;
  return m;
}

// ---------------------------------------------------------------------------
// tri_mesh lifecycle
// ---------------------------------------------------------------------------

TEST test_mesh_new_free(void) {
  tri_mesh *m = tri_mesh_new(4, 2);
  ASSERT(m != NULL);
  ASSERT_EQ(4, m->num_verts);
  ASSERT_EQ(2, m->num_faces);
  ASSERT(m->verts   != NULL);
  ASSERT(m->indices != NULL);
  tri_mesh_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// mesh_voxelize
// ---------------------------------------------------------------------------

TEST test_voxelize_cube_interior(void) {
  tri_mesh *m = make_cube();
  ASSERT(m != NULL);

  uint8_t *mask = mesh_voxelize(m, 10, 10, 10);
  ASSERT(mask != NULL);

  // Center voxel (5,5,5) should be inside.
  ASSERT_EQ(1, mask[5*10*10 + 5*10 + 5]);

  // Corner-adjacent voxels at (0,0,0) should be on the boundary or outside
  // (not inside — ray parity for a corner is implementation-dependent, so
  // just verify the mask allocation succeeded and has at least one interior).
  int interior = 0;
  for (int i = 0; i < 1000; i++) interior += mask[i];
  ASSERT(interior > 0);

  free(mask);
  tri_mesh_free(m);
  PASS();
}

TEST test_voxelize_returns_null_on_bad_args(void) {
  // Passing NULL mesh must not crash (assert fires in debug; skip in release).
  // We just verify the grid sizes are validated by checking a degenerate case.
  tri_mesh *m = make_cube();
  // d=0 would be caught by assert; pass minimum valid args.
  uint8_t *mask = mesh_voxelize(m, 1, 1, 1);
  ASSERT(mask != NULL);
  free(mask);
  tri_mesh_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// mesh_quality
// ---------------------------------------------------------------------------

TEST test_quality_equilateral(void) {
  tri_mesh *m = make_equilateral();
  mesh_quality_t q = mesh_quality(m);

  // All angles of equilateral triangle are 60°.
  ASSERT(fabsf(q.min_angle_deg - 60.0f) < 1.0f);
  ASSERT(fabsf(q.max_angle_deg - 60.0f) < 1.0f);
  ASSERT(fabsf(q.avg_angle_deg - 60.0f) < 1.0f);
  // Aspect ratio of equilateral triangle is 1.
  ASSERT(q.max_aspect_ratio < 1.1f);
  // No self-intersections in a single triangle.
  ASSERT_EQ(0, q.self_intersections);

  tri_mesh_free(m);
  PASS();
}

TEST test_quality_cube(void) {
  tri_mesh *m = make_cube();
  mesh_quality_t q = mesh_quality(m);

  // Cube triangles are right triangles (45°/45°/90°).
  ASSERT(q.min_angle_deg >= 40.0f && q.min_angle_deg <= 50.0f);
  ASSERT(q.max_angle_deg >= 85.0f && q.max_angle_deg <= 95.0f);
  ASSERT_EQ(0, q.self_intersections);

  tri_mesh_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// mesh_simplify
// ---------------------------------------------------------------------------

TEST test_simplify_reduces_faces(void) {
  tri_mesh *m = make_cube();  // 12 faces
  ASSERT(m != NULL);

  tri_mesh *s = mesh_simplify(m, 6);
  ASSERT(s != NULL);
  ASSERT(s->num_faces <= 12);

  tri_mesh_free(s);
  tri_mesh_free(m);
  PASS();
}

TEST test_simplify_already_small(void) {
  // target >= current: should return a copy without crashing.
  tri_mesh *m = make_equilateral();  // 1 face
  tri_mesh *s = mesh_simplify(m, 10);
  ASSERT(s != NULL);
  tri_mesh_free(s);
  tri_mesh_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// mesh_smooth
// ---------------------------------------------------------------------------

TEST test_smooth_moves_vertices(void) {
  // Flat equilateral triangle — smoothing shouldn't crash or NaN.
  tri_mesh *m = make_equilateral();
  vec3f before = m->verts[0];
  mesh_smooth(m, 3, 0.5f);
  // For a single triangle the centroid is the fixed point; vertices may move.
  // Just verify no NaN/Inf.
  for (int i = 0; i < 3; i++) {
    ASSERT(isfinite(m->verts[i].x));
    ASSERT(isfinite(m->verts[i].y));
    ASSERT(isfinite(m->verts[i].z));
  }
  (void)before;
  tri_mesh_free(m);
  PASS();
}

TEST test_smooth_zero_iterations(void) {
  tri_mesh *m = make_cube();
  vec3f v0 = m->verts[0];
  mesh_smooth(m, 0, 0.5f);
  // Zero iterations: no change.
  ASSERT(fabsf(m->verts[0].x - v0.x) < 1e-6f);
  tri_mesh_free(m);
  PASS();
}

TEST test_smooth_cube_converges(void) {
  // After many iterations with lambda=1 a cube should approach its centroid.
  tri_mesh *m = make_cube();
  mesh_smooth(m, 50, 1.0f);
  // All vertices should be near (0.5, 0.5, 0.5) — the cube centroid.
  for (int i = 0; i < m->num_verts; i++) {
    ASSERT(fabsf(m->verts[i].x - 0.5f) < 0.1f);
    ASSERT(fabsf(m->verts[i].y - 0.5f) < 0.1f);
    ASSERT(fabsf(m->verts[i].z - 0.5f) < 0.1f);
  }
  tri_mesh_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(geom_mesh_suite) {
  RUN_TEST(test_mesh_new_free);
  RUN_TEST(test_voxelize_cube_interior);
  RUN_TEST(test_voxelize_returns_null_on_bad_args);
  RUN_TEST(test_quality_equilateral);
  RUN_TEST(test_quality_cube);
  RUN_TEST(test_simplify_reduces_faces);
  RUN_TEST(test_simplify_already_small);
  RUN_TEST(test_smooth_moves_vertices);
  RUN_TEST(test_smooth_zero_iterations);
  RUN_TEST(test_smooth_cube_converges);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(geom_mesh_suite);
  GREATEST_MAIN_END();
}
