#include "greatest.h"
#include "gui/drawing.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static draw_params red_brush(float radius) {
  return (draw_params){
    .tool         = DRAW_FREEHAND,
    .brush_radius = radius,
    .color        = {255, 0, 0, 255},
    .line_width   = 1.0f,
  };
}

// Count non-zero alpha pixels in a 64x64 canvas.
static int count_painted(const drawing_canvas *c) {
  const uint8_t *px = drawing_get_pixels(c);
  int n = 0;
  for (int i = 0; i < 64 * 64; i++)
    if (px[i * 4 + 3]) n++;
  return n;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_canvas_new_free(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  ASSERT(c != NULL);
  ASSERT(drawing_get_pixels(c) != NULL);
  // New canvas is blank.
  const uint8_t *px = drawing_get_pixels(c);
  for (int i = 0; i < 64 * 64 * 4; i++) ASSERT_EQ(0, px[i]);
  drawing_canvas_free(c);
  PASS();
}

TEST test_freehand_stroke_paints_pixels(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(4.0f);

  drawing_begin_stroke(c, 10, 10, &p);
  drawing_continue_stroke(c, 20, 10);
  drawing_continue_stroke(c, 30, 10);
  drawing_end_stroke(c);

  ASSERT(count_painted(c) > 0);
  drawing_canvas_free(c);
  PASS();
}

TEST test_eraser_clears_pixels(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(8.0f);

  // Paint a wide stroke.
  drawing_begin_stroke(c, 5, 32, &p);
  drawing_continue_stroke(c, 58, 32);
  drawing_end_stroke(c);
  int after_draw = count_painted(c);
  ASSERT(after_draw > 0);

  // Erase the center.
  draw_params ep = {
    .tool = DRAW_ERASER, .brush_radius = 10.0f,
    .color = {0,0,0,0}, .line_width = 1.0f
  };
  drawing_begin_stroke(c, 28, 32, &ep);
  drawing_continue_stroke(c, 36, 32);
  drawing_end_stroke(c);

  int after_erase = count_painted(c);
  ASSERT(after_erase < after_draw);

  drawing_canvas_free(c);
  PASS();
}

TEST test_undo_restores_blank(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(4.0f);

  drawing_begin_stroke(c, 10, 10, &p);
  drawing_continue_stroke(c, 30, 10);
  drawing_end_stroke(c);
  ASSERT(count_painted(c) > 0);

  drawing_undo(c);
  ASSERT_EQ(0, count_painted(c));

  drawing_canvas_free(c);
  PASS();
}

TEST test_redo_reapplies_stroke(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(4.0f);

  drawing_begin_stroke(c, 10, 10, &p);
  drawing_continue_stroke(c, 30, 10);
  drawing_end_stroke(c);
  int painted = count_painted(c);
  ASSERT(painted > 0);

  drawing_undo(c);
  ASSERT_EQ(0, count_painted(c));

  drawing_redo(c);
  ASSERT_EQ(painted, count_painted(c));

  drawing_canvas_free(c);
  PASS();
}

TEST test_clear(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(5.0f);
  drawing_begin_stroke(c, 20, 20, &p);
  drawing_end_stroke(c);
  ASSERT(count_painted(c) > 0);

  drawing_clear(c);
  ASSERT_EQ(0, count_painted(c));

  // Undo clear should restore.
  drawing_undo(c);
  ASSERT(count_painted(c) > 0);

  drawing_canvas_free(c);
  PASS();
}

TEST test_export_mask(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = red_brush(5.0f);
  drawing_begin_stroke(c, 32, 32, &p);
  drawing_end_stroke(c);

  uint8_t *mask = calloc(64 * 64, 1);
  drawing_export_mask(c, mask);

  // The painted region should have mask value 255.
  int mask_set = 0;
  for (int i = 0; i < 64 * 64; i++) if (mask[i]) mask_set++;
  ASSERT(mask_set > 0);

  free(mask);
  drawing_canvas_free(c);
  PASS();
}

TEST test_rect_shape(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = {
    .tool = DRAW_RECT, .brush_radius = 2.0f,
    .color = {0, 255, 0, 255}, .line_width = 2.0f
  };

  drawing_begin_shape(c, 5, 5, DRAW_RECT, &p);
  drawing_update_shape(c, 50, 50);
  drawing_finish_shape(c);

  ASSERT(count_painted(c) > 0);
  drawing_canvas_free(c);
  PASS();
}

TEST test_circle_shape(void) {
  drawing_canvas *c = drawing_canvas_new(64, 64);
  draw_params p = {
    .tool = DRAW_CIRCLE, .brush_radius = 2.0f,
    .color = {0, 0, 255, 255}, .line_width = 2.0f
  };

  drawing_begin_shape(c, 32, 32, DRAW_CIRCLE, &p);
  drawing_update_shape(c, 48, 32);  // radius = 16
  drawing_finish_shape(c);

  ASSERT(count_painted(c) > 0);
  drawing_canvas_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(drawing_suite) {
  RUN_TEST(test_canvas_new_free);
  RUN_TEST(test_freehand_stroke_paints_pixels);
  RUN_TEST(test_eraser_clears_pixels);
  RUN_TEST(test_undo_restores_blank);
  RUN_TEST(test_redo_reapplies_stroke);
  RUN_TEST(test_clear);
  RUN_TEST(test_export_mask);
  RUN_TEST(test_rect_shape);
  RUN_TEST(test_circle_shape);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(drawing_suite);
  GREATEST_MAIN_END();
}
