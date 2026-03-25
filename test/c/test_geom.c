#include "greatest.h"
#include "core/geom.h"

#include <math.h>
#include <stdint.h>

#define EPS 1e-5f

// ---------------------------------------------------------------------------
// quad_surface tests
// ---------------------------------------------------------------------------

TEST test_quad_surface_create(void) {
  quad_surface *s = quad_surface_new(4, 5);
  ASSERT(s != NULL);
  ASSERT_EQ(4, s->rows);
  ASSERT_EQ(5, s->cols);
  ASSERT(s->points != NULL);
  ASSERT_EQ(NULL, s->normals);
  ASSERT_EQ(NULL, s->mask);
  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_set_get(void) {
  quad_surface *s = quad_surface_new(3, 3);
  ASSERT(s != NULL);

  vec3f p = {1.0f, 2.0f, 3.0f};
  quad_surface_set(s, 1, 2, p);
  vec3f got = quad_surface_get(s, 1, 2);

  ASSERT(vec3f_eq(p, got, EPS));
  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_sample_corners(void) {
  // 2x2 grid at known positions; sample at corners must return those positions
  quad_surface *s = quad_surface_new(2, 2);
  quad_surface_set(s, 0, 0, (vec3f){0,0,0});
  quad_surface_set(s, 0, 1, (vec3f){1,0,0});
  quad_surface_set(s, 1, 0, (vec3f){0,1,0});
  quad_surface_set(s, 1, 1, (vec3f){1,1,0});

  vec3f p00 = quad_surface_sample(s, 0.0f, 0.0f);
  vec3f p10 = quad_surface_sample(s, 1.0f, 0.0f);
  vec3f p01 = quad_surface_sample(s, 0.0f, 1.0f);
  vec3f p11 = quad_surface_sample(s, 1.0f, 1.0f);

  ASSERT(vec3f_eq(p00, (vec3f){0,0,0}, EPS));
  ASSERT(vec3f_eq(p10, (vec3f){1,0,0}, EPS));
  ASSERT(vec3f_eq(p01, (vec3f){0,1,0}, EPS));
  ASSERT(vec3f_eq(p11, (vec3f){1,1,0}, EPS));

  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_sample_center(void) {
  // bilinear center of a flat grid should be the average
  quad_surface *s = quad_surface_new(2, 2);
  quad_surface_set(s, 0, 0, (vec3f){0,0,0});
  quad_surface_set(s, 0, 1, (vec3f){2,0,0});
  quad_surface_set(s, 1, 0, (vec3f){0,2,0});
  quad_surface_set(s, 1, 1, (vec3f){2,2,0});

  vec3f mid = quad_surface_sample(s, 0.5f, 0.5f);
  ASSERT(vec3f_eq(mid, (vec3f){1,1,0}, EPS));

  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_normals_flat(void) {
  // flat XY plane: all normals should point in +Z
  quad_surface *s = quad_surface_new(3, 3);
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      quad_surface_set(s, r, c, (vec3f){(float)c, (float)r, 0.0f});

  quad_surface_compute_normals(s);
  ASSERT(s->normals != NULL);

  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 3; c++) {
      vec3f n = s->normals[r * 3 + c];
      // should point in +Z or -Z with unit length
      ASSERT(fabsf(vec3f_len(n) - 1.0f) < EPS);
      ASSERT(fabsf(fabsf(n.z) - 1.0f) < EPS);
    }
  }

  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_area_flat(void) {
  // 3x3 grid of unit spacing on XY plane => 2*2 = 4 quads => area = 4
  quad_surface *s = quad_surface_new(3, 3);
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++)
      quad_surface_set(s, r, c, (vec3f){(float)c, (float)r, 0.0f});

  float area = quad_surface_area(s);
  ASSERT(fabsf(area - 4.0f) < EPS);

  quad_surface_free(s);
  PASS();
}

TEST test_quad_surface_clone(void) {
  quad_surface *s = quad_surface_new(2, 2);
  quad_surface_set(s, 0, 0, (vec3f){1,2,3});
  quad_surface_set(s, 1, 1, (vec3f){4,5,6});
  s->id = strdup("original");

  quad_surface *c = quad_surface_clone(s);
  ASSERT(c != NULL);
  ASSERT(vec3f_eq(quad_surface_get(c, 0, 0), (vec3f){1,2,3}, EPS));
  ASSERT(vec3f_eq(quad_surface_get(c, 1, 1), (vec3f){4,5,6}, EPS));
  ASSERT(c->id != NULL && strcmp(c->id, "original") == 0);

  // mutating clone must not affect original
  quad_surface_set(c, 0, 0, (vec3f){0,0,0});
  ASSERT(vec3f_eq(quad_surface_get(s, 0, 0), (vec3f){1,2,3}, EPS));

  quad_surface_free(s);
  quad_surface_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// plane_surface tests
// ---------------------------------------------------------------------------

TEST test_plane_surface_basis_orthogonal(void) {
  vec3f origin = {0,0,0};
  vec3f normal = {0,0,1};
  plane_surface p = plane_surface_from_normal(origin, normal);

  // basis vectors should be orthogonal to normal and to each other
  ASSERT(fabsf(vec3f_dot(p.normal, p.u_axis)) < EPS);
  ASSERT(fabsf(vec3f_dot(p.normal, p.v_axis)) < EPS);
  ASSERT(fabsf(vec3f_dot(p.u_axis, p.v_axis)) < EPS);

  // unit length
  ASSERT(fabsf(vec3f_len(p.u_axis) - 1.0f) < EPS);
  ASSERT(fabsf(vec3f_len(p.v_axis) - 1.0f) < EPS);
  PASS();
}

TEST test_plane_surface_project(void) {
  vec3f origin = {0,0,5};
  vec3f normal = {0,0,1};
  plane_surface p = plane_surface_from_normal(origin, normal);

  vec3f pt = {3, 4, 10};
  vec3f proj = plane_surface_project(&p, pt);

  // projected point should be on the plane (z=5) and retain x,y
  ASSERT(fabsf(proj.x - 3.0f) < EPS);
  ASSERT(fabsf(proj.y - 4.0f) < EPS);
  ASSERT(fabsf(proj.z - 5.0f) < EPS);
  PASS();
}

TEST test_plane_surface_sample_roundtrip(void) {
  // sample a point on the plane then project it back: should get the same point
  vec3f origin = {1,2,3};
  vec3f normal = vec3f_normalize((vec3f){1,1,1});
  plane_surface p = plane_surface_from_normal(origin, normal);

  float u = 3.5f, v = -2.1f;
  vec3f world = plane_surface_sample(&p, u, v);
  vec3f proj  = plane_surface_project(&p, world);

  ASSERT(vec3f_eq(world, proj, EPS));
  PASS();
}

TEST test_plane_surface_dist(void) {
  vec3f origin = {0,0,0};
  vec3f normal = {0,0,1};
  plane_surface p = plane_surface_from_normal(origin, normal);

  float d_above = plane_surface_dist(&p, (vec3f){0,0, 3.0f});
  float d_below = plane_surface_dist(&p, (vec3f){0,0,-2.0f});
  float d_on    = plane_surface_dist(&p, (vec3f){5,7, 0.0f});

  ASSERT(fabsf(d_above -  3.0f) < EPS);
  ASSERT(fabsf(d_below - -2.0f) < EPS);
  ASSERT(fabsf(d_on) < EPS);
  PASS();
}

// ---------------------------------------------------------------------------
// HAMT tests
// ---------------------------------------------------------------------------

TEST test_hamt_empty(void) {
  hamt_node *h = hamt_empty();
  ASSERT(h != NULL);
  ASSERT_EQ(0u, hamt_len(h));
  ASSERT_EQ(NULL, hamt_get(h, 42));
  hamt_release(h);
  PASS();
}

TEST test_hamt_insert_get(void) {
  hamt_node *h = hamt_empty();
  int val1 = 100, val2 = 200;

  hamt_node *h1 = hamt_set(h, 1, &val1);
  hamt_node *h2 = hamt_set(h1, 2, &val2);

  ASSERT_EQ(&val1, hamt_get(h2, 1));
  ASSERT_EQ(&val2, hamt_get(h2, 2));
  ASSERT_EQ(NULL,  hamt_get(h2, 3));
  ASSERT_EQ(2u, hamt_len(h2));

  hamt_release(h);
  hamt_release(h1);
  hamt_release(h2);
  PASS();
}

TEST test_hamt_update(void) {
  hamt_node *h = hamt_empty();
  int v1 = 10, v2 = 20;

  hamt_node *h1 = hamt_set(h,  42, &v1);
  hamt_node *h2 = hamt_set(h1, 42, &v2);  // update same key

  ASSERT_EQ(&v2, hamt_get(h2, 42));
  ASSERT_EQ(1u, hamt_len(h2));

  hamt_release(h);
  hamt_release(h1);
  hamt_release(h2);
  PASS();
}

TEST test_hamt_delete(void) {
  hamt_node *h = hamt_empty();
  int v1 = 1, v2 = 2, v3 = 3;

  hamt_node *h1 = hamt_set(h,  10, &v1);
  hamt_node *h2 = hamt_set(h1, 20, &v2);
  hamt_node *h3 = hamt_set(h2, 30, &v3);

  hamt_node *h4 = hamt_del(h3, 20);

  ASSERT_EQ(&v1,  hamt_get(h4, 10));
  ASSERT_EQ(NULL, hamt_get(h4, 20));
  ASSERT_EQ(&v3,  hamt_get(h4, 30));
  ASSERT_EQ(2u, hamt_len(h4));

  hamt_release(h);
  hamt_release(h1);
  hamt_release(h2);
  hamt_release(h3);
  hamt_release(h4);
  PASS();
}

TEST test_hamt_delete_missing(void) {
  // deleting a key that doesn't exist returns a valid (unchanged) tree
  hamt_node *h = hamt_empty();
  int v = 99;
  hamt_node *h1 = hamt_set(h, 5, &v);
  hamt_node *h2 = hamt_del(h1, 99);  // key not present

  ASSERT_EQ(&v, hamt_get(h2, 5));
  ASSERT_EQ(1u, hamt_len(h2));

  hamt_release(h);
  hamt_release(h1);
  hamt_release(h2);
  PASS();
}

TEST test_hamt_persistence(void) {
  // old root must still be valid (with old data) after mutation
  hamt_node *h0 = hamt_empty();
  int v1 = 111, v2 = 222;

  hamt_node *h1 = hamt_set(h0, 7, &v1);
  hamt_node *h2 = hamt_set(h1, 7, &v2);  // overwrite in h2

  // h1 must still see v1
  ASSERT_EQ(&v1, hamt_get(h1, 7));
  // h2 sees v2
  ASSERT_EQ(&v2, hamt_get(h2, 7));

  hamt_release(h0);
  hamt_release(h1);
  hamt_release(h2);
  PASS();
}

TEST test_hamt_many_keys(void) {
  // insert 200 keys, check all present, delete half, check remaining
  hamt_node *h = hamt_empty();
  static int vals[200];

  for (int i = 0; i < 200; i++) {
    vals[i] = i * 10;
    hamt_node *next = hamt_set(h, (uint64_t)i, &vals[i]);
    hamt_release(h);
    h = next;
  }
  ASSERT_EQ(200u, hamt_len(h));

  for (int i = 0; i < 200; i++) {
    int *got = hamt_get(h, (uint64_t)i);
    ASSERT(got != NULL);
    ASSERT_EQ(i * 10, *got);
  }

  // delete even keys
  for (int i = 0; i < 200; i += 2) {
    hamt_node *next = hamt_del(h, (uint64_t)i);
    hamt_release(h);
    h = next;
  }
  ASSERT_EQ(100u, hamt_len(h));

  for (int i = 0; i < 200; i++) {
    int *got = hamt_get(h, (uint64_t)i);
    if (i % 2 == 0) {
      ASSERT_EQ(NULL, got);
    } else {
      ASSERT(got != NULL);
      ASSERT_EQ(i * 10, *got);
    }
  }

  hamt_release(h);
  PASS();
}

TEST test_hamt_structural_sharing(void) {
  // After inserting a key, the old root and new root share structure.
  // We verify this indirectly: release order must not double-free.
  hamt_node *h0 = hamt_empty();
  int v = 42;
  hamt_node *h1 = hamt_set(h0, 1000, &v);

  // Both roots reference-count correctly. If sharing is broken this would crash/assert.
  hamt_node *h2 = hamt_retain(h1);
  hamt_release(h1);  // h2 still holds it
  ASSERT_EQ(&v, hamt_get(h2, 1000));
  hamt_release(h2);
  hamt_release(h0);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(quad_surface_suite) {
  RUN_TEST(test_quad_surface_create);
  RUN_TEST(test_quad_surface_set_get);
  RUN_TEST(test_quad_surface_sample_corners);
  RUN_TEST(test_quad_surface_sample_center);
  RUN_TEST(test_quad_surface_normals_flat);
  RUN_TEST(test_quad_surface_area_flat);
  RUN_TEST(test_quad_surface_clone);
}

SUITE(plane_surface_suite) {
  RUN_TEST(test_plane_surface_basis_orthogonal);
  RUN_TEST(test_plane_surface_project);
  RUN_TEST(test_plane_surface_sample_roundtrip);
  RUN_TEST(test_plane_surface_dist);
}

SUITE(hamt_suite) {
  RUN_TEST(test_hamt_empty);
  RUN_TEST(test_hamt_insert_get);
  RUN_TEST(test_hamt_update);
  RUN_TEST(test_hamt_delete);
  RUN_TEST(test_hamt_delete_missing);
  RUN_TEST(test_hamt_persistence);
  RUN_TEST(test_hamt_many_keys);
  RUN_TEST(test_hamt_structural_sharing);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(quad_surface_suite);
  RUN_SUITE(plane_surface_suite);
  RUN_SUITE(hamt_suite);
  GREATEST_MAIN_END();
}
