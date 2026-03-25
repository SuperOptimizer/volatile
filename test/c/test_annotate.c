#include "greatest.h"
#include "gui/annotate.h"
#include "core/math.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static annotation *make_point(float x, float y, float z) {
  annotation *a = calloc(1, sizeof(*a));
  a->type       = ANNOT_POINT;
  a->num_points = 1;
  a->points     = malloc(sizeof(vec3f));
  a->points[0]  = (vec3f){x, y, z};
  a->visible    = true;
  a->color[0] = a->color[1] = a->color[2] = a->color[3] = 255;
  a->line_width = 1.0f;
  return a;
}

static annotation *make_line(vec3f a, vec3f b) {
  annotation *ann = calloc(1, sizeof(*ann));
  ann->type       = ANNOT_LINE;
  ann->num_points = 2;
  ann->points     = malloc(2 * sizeof(vec3f));
  ann->points[0]  = a;
  ann->points[1]  = b;
  ann->visible    = true;
  ann->line_width = 1.0f;
  return ann;
}

static annotation *make_circle(vec3f center, float radius) {
  annotation *a = calloc(1, sizeof(*a));
  a->type       = ANNOT_CIRCLE;
  a->num_points = 1;
  a->points     = malloc(sizeof(vec3f));
  a->points[0]  = center;
  a->radius     = radius;
  a->visible    = true;
  a->line_width = 1.0f;
  return a;
}

// ---------------------------------------------------------------------------
// add / get / remove
// ---------------------------------------------------------------------------

TEST test_add_get(void) {
  annot_store *s = annot_store_new();
  ASSERT(s != NULL);
  ASSERT_EQ(0, annot_count(s));

  annotation *a = make_point(1, 2, 3);
  int64_t id = annot_add(s, a);
  ASSERT(id > 0);
  ASSERT_EQ(1, annot_count(s));

  annotation *got = annot_get(s, id);
  ASSERT(got != NULL);
  ASSERT_EQ(id, got->id);
  ASSERT_EQ(ANNOT_POINT, got->type);

  annot_store_free(s);
  PASS();
}

TEST test_remove(void) {
  annot_store *s = annot_store_new();

  int64_t id1 = annot_add(s, make_point(0, 0, 0));
  int64_t id2 = annot_add(s, make_point(1, 1, 1));
  ASSERT_EQ(2, annot_count(s));

  ASSERT(annot_remove(s, id1));
  ASSERT_EQ(1, annot_count(s));
  ASSERT(annot_get(s, id1) == NULL);
  ASSERT(annot_get(s, id2) != NULL);

  // Double-remove returns false.
  ASSERT(!annot_remove(s, id1));

  annot_store_free(s);
  PASS();
}

TEST test_unique_ids(void) {
  annot_store *s = annot_store_new();
  int64_t ids[8];
  for (int i = 0; i < 8; i++) ids[i] = annot_add(s, make_point((float)i, 0, 0));
  for (int i = 0; i < 8; i++)
    for (int j = i+1; j < 8; j++)
      ASSERT(ids[i] != ids[j]);
  annot_store_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Iteration
// ---------------------------------------------------------------------------

typedef struct { int count; int64_t sum; } iter_result;

static void count_fn(const annotation *a, void *ctx) {
  iter_result *r = ctx;
  r->count++;
  r->sum += a->id;
}

TEST test_iter(void) {
  annot_store *s = annot_store_new();
  int64_t id1 = annot_add(s, make_point(0, 0, 0));
  int64_t id2 = annot_add(s, make_point(1, 1, 1));
  int64_t id3 = annot_add(s, make_point(2, 2, 2));

  iter_result r = {0};
  annot_iter(s, count_fn, &r);
  ASSERT_EQ(3, r.count);
  ASSERT_EQ(id1 + id2 + id3, r.sum);

  annot_store_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Hit testing
// ---------------------------------------------------------------------------

TEST test_hit_point(void) {
  annot_store *s = annot_store_new();
  int64_t id = annot_add(s, make_point(5, 5, 5));

  // Within tolerance.
  int64_t hit = annot_hit_test(s, (vec3f){5.05f, 5.0f, 5.0f}, 0.1f);
  ASSERT_EQ(id, hit);

  // Outside tolerance.
  int64_t miss = annot_hit_test(s, (vec3f){6.0f, 5.0f, 5.0f}, 0.1f);
  ASSERT_EQ(-1, miss);

  annot_store_free(s);
  PASS();
}

TEST test_hit_line(void) {
  annot_store *s = annot_store_new();
  int64_t id = annot_add(s, make_line((vec3f){0,0,0}, (vec3f){10,0,0}));

  // Point on the line.
  ASSERT_EQ(id, annot_hit_test(s, (vec3f){5, 0.05f, 0}, 0.1f));
  // Past endpoint — clamped to endpoint, still within tolerance.
  ASSERT_EQ(id, annot_hit_test(s, (vec3f){10.05f, 0, 0}, 0.1f));
  // Far away.
  ASSERT_EQ(-1, annot_hit_test(s, (vec3f){5, 5, 0}, 0.1f));

  annot_store_free(s);
  PASS();
}

TEST test_hit_circle(void) {
  annot_store *s = annot_store_new();
  int64_t id = annot_add(s, make_circle((vec3f){0,0,0}, 5.0f));

  // On the rim.
  ASSERT_EQ(id, annot_hit_test(s, (vec3f){5.05f, 0, 0}, 0.1f));
  // At center — distance to rim = radius, outside tolerance 0.1.
  ASSERT_EQ(-1, annot_hit_test(s, (vec3f){0, 0, 0}, 0.1f));

  annot_store_free(s);
  PASS();
}

TEST test_hit_invisible(void) {
  annot_store *s = annot_store_new();
  annotation *a = make_point(1, 1, 1);
  a->visible = false;
  annot_add(s, a);

  // Invisible annotations are ignored.
  ASSERT_EQ(-1, annot_hit_test(s, (vec3f){1, 1, 1}, 0.5f));

  annot_store_free(s);
  PASS();
}

TEST test_hit_nearest(void) {
  annot_store *s = annot_store_new();
  int64_t id1 = annot_add(s, make_point(0, 0, 0));
  int64_t id2 = annot_add(s, make_point(1, 0, 0));

  // Query at 0.3 — nearer to id1.
  int64_t hit = annot_hit_test(s, (vec3f){0.3f, 0, 0}, 1.0f);
  ASSERT_EQ(id1, hit);

  // Query at 0.8 — nearer to id2.
  hit = annot_hit_test(s, (vec3f){0.8f, 0, 0}, 1.0f);
  ASSERT_EQ(id2, hit);

  annot_store_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Save / load roundtrip
// ---------------------------------------------------------------------------

TEST test_save_load_roundtrip(void) {
  annot_store *s = annot_store_new();

  annotation *pt = make_point(1, 2, 3);
  pt->label = strdup("origin");
  pt->color[0] = 255; pt->color[1] = 0; pt->color[2] = 128; pt->color[3] = 200;
  int64_t id_pt = annot_add(s, pt);

  annotation *ln = make_line((vec3f){0,0,0}, (vec3f){10,5,2});
  ln->line_width = 2.5f;
  int64_t id_ln = annot_add(s, ln);

  const char *path = "/tmp/test_annot_roundtrip.json";
  ASSERT(annot_save_json(s, path));
  annot_store_free(s);

  annot_store *loaded = annot_load_json(path);
  ASSERT(loaded != NULL);
  ASSERT_EQ(2, annot_count(loaded));

  annotation *got_pt = annot_get(loaded, id_pt);
  ASSERT(got_pt != NULL);
  ASSERT_EQ(ANNOT_POINT, got_pt->type);
  ASSERT(got_pt->label && strcmp(got_pt->label, "origin") == 0);
  ASSERT_EQ(255, got_pt->color[0]);
  ASSERT_EQ(0,   got_pt->color[1]);
  ASSERT_EQ(128, got_pt->color[2]);
  ASSERT_EQ(200, got_pt->color[3]);
  ASSERT(fabsf(got_pt->points[0].x - 1.0f) < 1e-4f);
  ASSERT(fabsf(got_pt->points[0].y - 2.0f) < 1e-4f);
  ASSERT(fabsf(got_pt->points[0].z - 3.0f) < 1e-4f);

  annotation *got_ln = annot_get(loaded, id_ln);
  ASSERT(got_ln != NULL);
  ASSERT_EQ(ANNOT_LINE, got_ln->type);
  ASSERT_EQ(2, got_ln->num_points);
  ASSERT(fabsf(got_ln->line_width - 2.5f) < 1e-4f);
  ASSERT(fabsf(got_ln->points[1].x - 10.0f) < 1e-4f);

  annot_store_free(loaded);
  remove(path);
  PASS();
}

TEST test_load_missing_file(void) {
  annot_store *s = annot_load_json("/tmp/this_file_does_not_exist_volatile.json");
  ASSERT(s == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(annot_suite) {
  RUN_TEST(test_add_get);
  RUN_TEST(test_remove);
  RUN_TEST(test_unique_ids);
  RUN_TEST(test_iter);
  RUN_TEST(test_hit_point);
  RUN_TEST(test_hit_line);
  RUN_TEST(test_hit_circle);
  RUN_TEST(test_hit_invisible);
  RUN_TEST(test_hit_nearest);
  RUN_TEST(test_save_load_roundtrip);
  RUN_TEST(test_load_missing_file);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(annot_suite);
  GREATEST_MAIN_END();
}
