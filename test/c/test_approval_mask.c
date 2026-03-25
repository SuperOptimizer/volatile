#include "greatest.h"
#include "gui/approval_mask.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Basic creation / paint / query
// ---------------------------------------------------------------------------

TEST test_new_all_unpainted(void) {
  approval_mask *m = approval_mask_new(10, 10);
  ASSERT(m != NULL);
  for (int r = 0; r < 10; r++)
    for (int c = 0; c < 10; c++)
      ASSERT_FALSE(approval_mask_is_approved(m, r, c));
  ASSERT(approval_mask_coverage(m) < 1e-6f);
  approval_mask_free(m);
  PASS();
}

TEST test_paint_approved(void) {
  approval_mask *m = approval_mask_new(20, 20);
  // Paint center at grid coords (10,10) with radius 2
  approval_mask_paint(m, 10.0f, 10.0f, 2.0f, true);
  // Center cell must be approved
  ASSERT(approval_mask_is_approved(m, 10, 10));
  // Corner cell must be untouched
  ASSERT_FALSE(approval_mask_is_approved(m, 0, 0));
  approval_mask_free(m);
  PASS();
}

TEST test_paint_rejected(void) {
  approval_mask *m = approval_mask_new(20, 20);
  approval_mask_paint(m, 10.0f, 10.0f, 2.0f, false);
  // Center must NOT be approved (it's rejected)
  ASSERT_FALSE(approval_mask_is_approved(m, 10, 10));
  approval_mask_free(m);
  PASS();
}

TEST test_erase_clears_paint(void) {
  approval_mask *m = approval_mask_new(20, 20);
  approval_mask_paint(m, 10.0f, 10.0f, 3.0f, true);
  ASSERT(approval_mask_is_approved(m, 10, 10));
  approval_mask_erase(m, 10.0f, 10.0f, 3.0f);
  ASSERT_FALSE(approval_mask_is_approved(m, 10, 10));
  approval_mask_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Coverage
// ---------------------------------------------------------------------------

TEST test_coverage(void) {
  approval_mask *m = approval_mask_new(4, 4);  // 16 cells
  // Approve the center 2x2 block (4 cells)
  approval_mask_paint(m, 1.0f, 1.0f, 0.4f, true);  // radius <1, hits center cell (1,1)
  float cov = approval_mask_coverage(m);
  ASSERT(cov > 0.0f && cov <= 1.0f);
  approval_mask_free(m);
  PASS();
}

TEST test_coverage_full(void) {
  approval_mask *m = approval_mask_new(5, 5);
  // Approve entire grid with a large radius
  approval_mask_paint(m, 2.0f, 2.0f, 10.0f, true);
  float cov = approval_mask_coverage(m);
  ASSERT(fabsf(cov - 1.0f) < 1e-6f);
  approval_mask_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Undo / redo
// ---------------------------------------------------------------------------

TEST test_undo_restores_to_unpainted(void) {
  approval_mask *m = approval_mask_new(10, 10);
  approval_mask_paint(m, 5.0f, 5.0f, 2.0f, true);
  ASSERT(approval_mask_is_approved(m, 5, 5));
  ASSERT(approval_mask_can_undo(m));

  approval_mask_undo(m);
  ASSERT_FALSE(approval_mask_is_approved(m, 5, 5));
  ASSERT_FALSE(approval_mask_can_undo(m));

  approval_mask_free(m);
  PASS();
}

TEST test_redo_reapplies(void) {
  approval_mask *m = approval_mask_new(10, 10);
  approval_mask_paint(m, 5.0f, 5.0f, 2.0f, true);
  approval_mask_undo(m);
  ASSERT(approval_mask_can_redo(m));

  approval_mask_redo(m);
  ASSERT(approval_mask_is_approved(m, 5, 5));
  ASSERT_FALSE(approval_mask_can_redo(m));

  approval_mask_free(m);
  PASS();
}

TEST test_new_paint_clears_redo(void) {
  approval_mask *m = approval_mask_new(10, 10);
  approval_mask_paint(m, 5.0f, 5.0f, 1.0f, true);
  approval_mask_undo(m);
  ASSERT(approval_mask_can_redo(m));

  // New paint should clear redo history
  approval_mask_paint(m, 3.0f, 3.0f, 1.0f, false);
  ASSERT_FALSE(approval_mask_can_redo(m));

  approval_mask_free(m);
  PASS();
}

TEST test_multiple_undo_redo(void) {
  approval_mask *m = approval_mask_new(20, 20);
  approval_mask_paint(m, 5.0f, 5.0f, 1.5f, true);   // op 1
  approval_mask_paint(m, 10.0f, 10.0f, 1.5f, false); // op 2

  approval_mask_undo(m);  // undo op2
  ASSERT(approval_mask_is_approved(m, 5, 5));
  ASSERT_FALSE(approval_mask_is_approved(m, 10, 10));

  approval_mask_undo(m);  // undo op1
  ASSERT_FALSE(approval_mask_is_approved(m, 5, 5));

  approval_mask_redo(m);  // redo op1
  ASSERT(approval_mask_is_approved(m, 5, 5));

  approval_mask_redo(m);  // redo op2 — but op2 painted rejected, not approved
  ASSERT_FALSE(approval_mask_is_approved(m, 10, 10));  // rejected, not approved

  approval_mask_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Save / load roundtrip
// ---------------------------------------------------------------------------

TEST test_save_load_roundtrip(void) {
  approval_mask *m = approval_mask_new(8, 12);
  approval_mask_paint(m, 6.0f, 4.0f, 2.5f, true);
  approval_mask_paint(m, 2.0f, 2.0f, 1.5f, false);

  const char *path = "/tmp/test_approval_mask.bin";
  ASSERT(approval_mask_save(m, path));

  approval_mask *loaded = approval_mask_load(path);
  ASSERT(loaded != NULL);

  for (int r = 0; r < 8; r++) {
    for (int c = 0; c < 12; c++) {
      bool orig_approved   = approval_mask_is_approved(m, r, c);
      bool loaded_approved = approval_mask_is_approved(loaded, r, c);
      ASSERT_EQ(orig_approved, loaded_approved);
    }
  }

  float orig_cov   = approval_mask_coverage(m);
  float loaded_cov = approval_mask_coverage(loaded);
  ASSERT(fabsf(orig_cov - loaded_cov) < 1e-6f);

  approval_mask_free(m);
  approval_mask_free(loaded);
  remove(path);
  PASS();
}

TEST test_load_bad_path_returns_null(void) {
  approval_mask *m = approval_mask_load("/tmp/nonexistent_approval_mask_xyz.bin");
  ASSERT(m == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Overlay generation
// ---------------------------------------------------------------------------

TEST test_overlay_colors(void) {
  approval_mask *m = approval_mask_new(2, 2);
  // (u=0,v=0) → approved;  (u=1,v=1) → rejected;  rest unpainted
  approval_mask_paint(m, 0.0f, 0.0f, 0.4f, true);   // top-left cell
  approval_mask_paint(m, 1.0f, 1.0f, 0.4f, false);  // bottom-right cell

  uint8_t rgba[2 * 2 * 4] = {0};
  approval_mask_to_overlay(m, rgba, 2, 2);

  // Top-left pixel: approved = green (g channel dominant)
  ASSERT(rgba[1] > rgba[0] && rgba[1] > rgba[2]);  // g > r, g > b
  ASSERT(rgba[3] > 0);                              // not transparent

  // Bottom-right pixel: rejected = red (r channel dominant)
  uint8_t *br = rgba + (1 * 2 + 1) * 4;
  ASSERT(br[0] > br[1] && br[0] > br[2]);
  ASSERT(br[3] > 0);

  approval_mask_free(m);
  PASS();
}

TEST test_overlay_unpainted_transparent(void) {
  approval_mask *m = approval_mask_new(4, 4);
  uint8_t rgba[4 * 4 * 4] = {0};
  approval_mask_to_overlay(m, rgba, 4, 4);
  // All cells unpainted → all alpha == 0
  for (int i = 0; i < 4 * 4; i++)
    ASSERT_EQ(rgba[i * 4 + 3], 0);
  approval_mask_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(approval_mask_suite) {
  RUN_TEST(test_new_all_unpainted);
  RUN_TEST(test_paint_approved);
  RUN_TEST(test_paint_rejected);
  RUN_TEST(test_erase_clears_paint);
  RUN_TEST(test_coverage);
  RUN_TEST(test_coverage_full);
  RUN_TEST(test_undo_restores_to_unpainted);
  RUN_TEST(test_redo_reapplies);
  RUN_TEST(test_new_paint_clears_redo);
  RUN_TEST(test_multiple_undo_redo);
  RUN_TEST(test_save_load_roundtrip);
  RUN_TEST(test_load_bad_path_returns_null);
  RUN_TEST(test_overlay_colors);
  RUN_TEST(test_overlay_unpainted_transparent);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(approval_mask_suite);
  GREATEST_MAIN_END();
}
