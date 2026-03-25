#include "greatest.h"
#include "gui/seg.h"
#include "core/geom.h"

#include <math.h>

#define EPS 1e-5f

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a flat rows×cols grid in the XY plane with unit spacing.
static quad_surface *flat_grid(int rows, int cols) {
  quad_surface *s = quad_surface_new(rows, cols);
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      quad_surface_set(s, r, c, (vec3f){(float)c, (float)r, 0.0f});
  return s;
}

// ---------------------------------------------------------------------------
// Brush tests
// ---------------------------------------------------------------------------

TEST test_brush_moves_center_vertex(void) {
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 2.0f, .sigma = 1.0f };
  float delta = 1.0f;

  // Apply brush at center (u=0.5, v=0.5) → grid point (5,5).
  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, delta, &p);
  ASSERT(e != NULL);

  vec3f center = quad_surface_get(s, 5, 5);
  // Flat grid normals point in +Z; center vertex should have moved in +Z.
  ASSERT(center.z > 0.0f);

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

TEST test_brush_falloff(void) {
  // Vertex closer to brush center should move more than one farther away.
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 3.0f, .sigma = 1.5f };

  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, 1.0f, &p);

  vec3f near_pt = quad_surface_get(s, 5, 5);  // dist=0
  vec3f far_pt  = quad_surface_get(s, 5, 7);  // dist=2
  ASSERT(near_pt.z > far_pt.z);

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

TEST test_brush_undo_restores(void) {
  quad_surface *s = flat_grid(11, 11);
  quad_surface *orig = quad_surface_clone(s);

  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 2.0f, .sigma = 1.0f };
  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, 5.0f, &p);

  seg_edit_undo(s, e);

  // All points should match original.
  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      vec3f got = quad_surface_get(s, r, c);
      vec3f exp = quad_surface_get(orig, r, c);
      ASSERT(vec3f_eq(got, exp, EPS));
    }
  }

  seg_edit_free(e);
  quad_surface_free(s);
  quad_surface_free(orig);
  PASS();
}

TEST test_brush_no_effect_outside_radius(void) {
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 1.0f, .sigma = 0.5f };

  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, 10.0f, &p);

  // Vertex at (0,0) is far from center (5,5) — should be untouched.
  vec3f corner = quad_surface_get(s, 0, 0);
  ASSERT(fabsf(corner.z) < EPS);

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

TEST test_brush_negative_delta(void) {
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 2.0f, .sigma = 1.0f };

  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, -1.0f, &p);

  vec3f center = quad_surface_get(s, 5, 5);
  ASSERT(center.z < 0.0f);

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Line tests
// ---------------------------------------------------------------------------

TEST test_line_moves_vertices_along_path(void) {
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_LINE, .radius = 1.5f, .sigma = 0.75f };

  // Horizontal line at v=0.5: u from 0.1 to 0.9.
  seg_edit *e = seg_line_apply(s, 0.1f, 0.5f, 0.9f, 0.5f, 1.0f, &p);
  ASSERT(e != NULL);

  // Vertices along row 5 between col 1 and col 9 should have moved in +Z.
  for (int c = 1; c <= 9; c++) {
    vec3f pt = quad_surface_get(s, 5, c);
    ASSERT(pt.z > 0.0f);
  }

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

TEST test_line_undo_restores(void) {
  quad_surface *s = flat_grid(11, 11);
  quad_surface *orig = quad_surface_clone(s);
  seg_tool_params p = { .tool = SEG_TOOL_LINE, .radius = 1.5f, .sigma = 0.75f };

  seg_edit *e = seg_line_apply(s, 0.1f, 0.5f, 0.9f, 0.5f, 3.0f, &p);
  seg_edit_undo(s, e);

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      ASSERT(vec3f_eq(quad_surface_get(s, r, c), quad_surface_get(orig, r, c), EPS));
    }
  }

  seg_edit_free(e);
  quad_surface_free(s);
  quad_surface_free(orig);
  PASS();
}

// ---------------------------------------------------------------------------
// Push-pull tests
// ---------------------------------------------------------------------------

TEST test_pushpull_uniform_displacement(void) {
  quad_surface *s = flat_grid(11, 11);
  seg_tool_params p = { .tool = SEG_TOOL_PUSHPULL, .radius = 2.0f, .push_amount = 2.0f };

  seg_edit *e = seg_pushpull_apply(s, 0.5f, 0.5f, &p);
  ASSERT(e != NULL);

  // All vertices within radius should have moved the same amount (+Z on flat grid).
  vec3f c55 = quad_surface_get(s, 5, 5);
  vec3f c56 = quad_surface_get(s, 5, 6);  // dist=1, within radius=2
  ASSERT(fabsf(c55.z - 2.0f) < EPS);
  ASSERT(fabsf(c56.z - 2.0f) < EPS);

  seg_edit_free(e);
  quad_surface_free(s);
  PASS();
}

TEST test_pushpull_undo_restores(void) {
  quad_surface *s = flat_grid(11, 11);
  quad_surface *orig = quad_surface_clone(s);
  seg_tool_params p = { .tool = SEG_TOOL_PUSHPULL, .radius = 3.0f, .push_amount = 5.0f };

  seg_edit *e = seg_pushpull_apply(s, 0.5f, 0.5f, &p);
  seg_edit_undo(s, e);

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      ASSERT(vec3f_eq(quad_surface_get(s, r, c), quad_surface_get(orig, r, c), EPS));
    }
  }

  seg_edit_free(e);
  quad_surface_free(s);
  quad_surface_free(orig);
  PASS();
}

// ---------------------------------------------------------------------------
// Double-undo safety: applying undo twice must not corrupt (it should be idempotent
// when the surface hasn't changed between the two undo calls).
// ---------------------------------------------------------------------------

TEST test_undo_idempotent(void) {
  quad_surface *s = flat_grid(9, 9);
  quad_surface *orig = quad_surface_clone(s);
  seg_tool_params p = { .tool = SEG_TOOL_BRUSH, .radius = 2.0f, .sigma = 1.0f };

  seg_edit *e = seg_brush_apply(s, 0.5f, 0.5f, 1.0f, &p);
  seg_edit_undo(s, e);
  seg_edit_undo(s, e);  // second undo: re-applies old values over already-restored surface

  for (int r = 0; r < s->rows; r++) {
    for (int c = 0; c < s->cols; c++) {
      ASSERT(vec3f_eq(quad_surface_get(s, r, c), quad_surface_get(orig, r, c), EPS));
    }
  }

  seg_edit_free(e);
  quad_surface_free(s);
  quad_surface_free(orig);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(seg_suite) {
  RUN_TEST(test_brush_moves_center_vertex);
  RUN_TEST(test_brush_falloff);
  RUN_TEST(test_brush_undo_restores);
  RUN_TEST(test_brush_no_effect_outside_radius);
  RUN_TEST(test_brush_negative_delta);
  RUN_TEST(test_line_moves_vertices_along_path);
  RUN_TEST(test_line_undo_restores);
  RUN_TEST(test_pushpull_uniform_displacement);
  RUN_TEST(test_pushpull_undo_restores);
  RUN_TEST(test_undo_idempotent);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(seg_suite);
  GREATEST_MAIN_END();
}
