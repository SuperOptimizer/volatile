#include "greatest.h"
#include "gui/slice_controller.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns true if two 3x3 matrices are element-wise within `eps`.
static bool mat3_near(const float *a, const float *b, float eps) {
  for (int i = 0; i < 9; i++) {
    if (fabsf(a[i] - b[i]) > eps) return false;
  }
  return true;
}

// Identity 3x3 (column-major, but symmetric so layout doesn't matter here)
static const float g_identity3[9] = {
  1,0,0,
  0,1,0,
  0,0,1,
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_create_free(void) {
  // NULL viewer is valid (viewer is optional)
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);
  slice_controller_free(c);
  PASS();
}

TEST test_free_null_no_crash(void) {
  slice_controller_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Default orientation
// ---------------------------------------------------------------------------

TEST test_default_transform_is_identity(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float m[9];
  slice_controller_get_transform(c, m);
  ASSERT(mat3_near(m, g_identity3, 1e-5f));

  slice_controller_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// set_axis
// ---------------------------------------------------------------------------

TEST test_set_axis_xy_resets_to_identity(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  // rotate to mess up the matrix, then snap back to XY
  slice_controller_rotate(c, 45.0f);
  slice_controller_set_axis(c, 0);  // XY

  float m[9];
  slice_controller_get_transform(c, m);
  ASSERT(mat3_near(m, g_identity3, 1e-5f));

  slice_controller_free(c);
  PASS();
}

TEST test_set_axis_changes_normal(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float m0[9], m1[9], m2[9];

  slice_controller_set_axis(c, 0);
  slice_controller_get_transform(c, m0);

  slice_controller_set_axis(c, 1);
  slice_controller_get_transform(c, m1);

  slice_controller_set_axis(c, 2);
  slice_controller_get_transform(c, m2);

  // All three orientations must differ from each other
  ASSERT(!mat3_near(m0, m1, 1e-3f));
  ASSERT(!mat3_near(m0, m2, 1e-3f));
  ASSERT(!mat3_near(m1, m2, 1e-3f));

  slice_controller_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// rotate — transform must change
// ---------------------------------------------------------------------------

TEST test_rotate_changes_transform(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9], after[9];
  slice_controller_get_transform(c, before);

  slice_controller_rotate(c, 45.0f);
  slice_controller_get_transform(c, after);

  // Matrices must differ
  ASSERT(!mat3_near(before, after, 1e-3f));

  slice_controller_free(c);
  PASS();
}

TEST test_rotate_360_returns_to_identity(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  // Apply 360° in one shot — should round-trip to identity
  slice_controller_rotate(c, 360.0f);

  float m[9];
  slice_controller_get_transform(c, m);
  ASSERT(mat3_near(m, g_identity3, 1e-4f));

  slice_controller_free(c);
  PASS();
}

TEST test_rotate_accumulates(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float m45[9], m90[9];

  slice_controller_rotate(c, 45.0f);
  slice_controller_get_transform(c, m45);

  slice_controller_rotate(c, 45.0f);  // now 90° total
  slice_controller_get_transform(c, m90);

  ASSERT(!mat3_near(m45, m90, 1e-3f));

  slice_controller_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Mouse drag sequence
// ---------------------------------------------------------------------------

TEST test_mouse_drag_rotates(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9];
  slice_controller_get_transform(c, before);

  // Drag 100 px to the right
  slice_controller_on_mouse_down(c, 100.0f, 200.0f);
  slice_controller_on_mouse_drag(c, 200.0f, 200.0f);  // dx = +100
  slice_controller_on_mouse_up(c);

  float after[9];
  slice_controller_get_transform(c, after);

  ASSERT(!mat3_near(before, after, 1e-3f));

  slice_controller_free(c);
  PASS();
}

TEST test_mouse_drag_no_movement_no_change(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9];
  slice_controller_get_transform(c, before);

  // Down and up at the same position
  slice_controller_on_mouse_down(c, 100.0f, 100.0f);
  slice_controller_on_mouse_drag(c, 100.0f, 100.0f);  // dx = 0
  slice_controller_on_mouse_up(c);

  float after[9];
  slice_controller_get_transform(c, after);

  ASSERT(mat3_near(before, after, 1e-6f));

  slice_controller_free(c);
  PASS();
}

TEST test_drag_ignored_without_mouse_down(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9];
  slice_controller_get_transform(c, before);

  // drag without a preceding mouse_down — must be ignored
  slice_controller_on_mouse_drag(c, 500.0f, 200.0f);

  float after[9];
  slice_controller_get_transform(c, after);

  ASSERT(mat3_near(before, after, 1e-6f));

  slice_controller_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Debounce / tick
// ---------------------------------------------------------------------------

TEST test_tick_flushes_after_debounce(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9];
  slice_controller_get_transform(c, before);

  slice_controller_rotate(c, 30.0f);

  // Tick for less than debounce period — not yet flushed (matrix unchanged
  // in internal state, but get_transform still shows the pending rotation).
  slice_controller_tick(c, 100.0f);

  // Tick past 200 ms debounce — should flush internally
  slice_controller_tick(c, 150.0f);

  float after[9];
  slice_controller_get_transform(c, after);

  // After flush, rotation should be baked in and visible
  ASSERT(!mat3_near(before, after, 1e-3f));

  slice_controller_free(c);
  PASS();
}

TEST test_tick_no_pending_no_change(void) {
  slice_controller *c = slice_controller_new(NULL);
  ASSERT(c != NULL);

  float before[9];
  slice_controller_get_transform(c, before);

  // Tick with nothing pending
  slice_controller_tick(c, 300.0f);

  float after[9];
  slice_controller_get_transform(c, after);

  ASSERT(mat3_near(before, after, 1e-6f));

  slice_controller_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// NULL safety
// ---------------------------------------------------------------------------

TEST test_null_safety(void) {
  // None of these should crash
  slice_controller_set_axis(NULL, 0);
  slice_controller_rotate(NULL, 45.0f);
  slice_controller_on_mouse_down(NULL, 0, 0);
  slice_controller_on_mouse_drag(NULL, 0, 0);
  slice_controller_on_mouse_up(NULL);
  slice_controller_get_transform(NULL, NULL);
  slice_controller_tick(NULL, 16.0f);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(slice_controller_suite) {
  RUN_TEST(test_create_free);
  RUN_TEST(test_free_null_no_crash);
  RUN_TEST(test_default_transform_is_identity);
  RUN_TEST(test_set_axis_xy_resets_to_identity);
  RUN_TEST(test_set_axis_changes_normal);
  RUN_TEST(test_rotate_changes_transform);
  RUN_TEST(test_rotate_360_returns_to_identity);
  RUN_TEST(test_rotate_accumulates);
  RUN_TEST(test_mouse_drag_rotates);
  RUN_TEST(test_mouse_drag_no_movement_no_change);
  RUN_TEST(test_drag_ignored_without_mouse_down);
  RUN_TEST(test_tick_flushes_after_debounce);
  RUN_TEST(test_tick_no_pending_no_change);
  RUN_TEST(test_null_safety);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(slice_controller_suite);
  GREATEST_MAIN_END();
}
