#include "greatest.h"
#include "gui/draw_panel.h"
#include "gui/drawing.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_create_free(void) {
  draw_panel *p = draw_panel_new(64, 64);
  ASSERT(p != NULL);
  draw_panel_free(p);
  PASS();
}

TEST test_free_null_no_crash(void) {
  draw_panel_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Layer management
// ---------------------------------------------------------------------------

TEST test_starts_with_one_layer(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);
  ASSERT_EQ(1, draw_panel_layer_count(p));
  ASSERT_EQ(0, draw_panel_active_layer(p));
  draw_panel_free(p);
  PASS();
}

TEST test_add_layer(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);
  ASSERT(draw_panel_add_layer(p));
  ASSERT_EQ(2, draw_panel_layer_count(p));
  ASSERT_EQ(1, draw_panel_active_layer(p));  // new layer becomes active
  draw_panel_free(p);
  PASS();
}

TEST test_set_layer(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);
  draw_panel_add_layer(p);
  draw_panel_set_layer(p, 0);
  ASSERT_EQ(0, draw_panel_active_layer(p));
  draw_panel_free(p);
  PASS();
}

TEST test_remove_layer_shifts(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);
  draw_panel_add_layer(p);
  draw_panel_add_layer(p);
  ASSERT_EQ(3, draw_panel_layer_count(p));

  ASSERT(draw_panel_remove_layer(p, 1));
  ASSERT_EQ(2, draw_panel_layer_count(p));

  draw_panel_free(p);
  PASS();
}

TEST test_cannot_remove_last_layer(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);
  ASSERT_FALSE(draw_panel_remove_layer(p, 0));
  ASSERT_EQ(1, draw_panel_layer_count(p));
  draw_panel_free(p);
  PASS();
}

TEST test_cannot_exceed_max_layers(void) {
  draw_panel *p = draw_panel_new(8, 8);
  ASSERT(p != NULL);
  for (int i = 1; i < DRAW_PANEL_MAX_LAYERS; i++)
    draw_panel_add_layer(p);
  ASSERT_EQ(DRAW_PANEL_MAX_LAYERS, draw_panel_layer_count(p));
  // One more should fail
  ASSERT_FALSE(draw_panel_add_layer(p));
  ASSERT_EQ(DRAW_PANEL_MAX_LAYERS, draw_panel_layer_count(p));
  draw_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Pixels / canvas access
// ---------------------------------------------------------------------------

TEST test_get_pixels_not_null(void) {
  draw_panel *p = draw_panel_new(16, 16);
  ASSERT(p != NULL);
  const uint8_t *px = draw_panel_get_pixels(p);
  ASSERT(px != NULL);
  draw_panel_free(p);
  PASS();
}

TEST test_pixels_start_transparent(void) {
  draw_panel *p = draw_panel_new(8, 8);
  ASSERT(p != NULL);
  const uint8_t *px = draw_panel_get_pixels(p);
  ASSERT(px != NULL);
  // Canvas is cleared on creation — all pixels should be zero (transparent)
  for (int i = 0; i < 8 * 8 * 4; i++) ASSERT_EQ(0, px[i]);
  draw_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Mouse interaction — freehand stroke leaves marks
// ---------------------------------------------------------------------------

TEST test_stroke_changes_pixels(void) {
  draw_panel *p = draw_panel_new(64, 64);
  ASSERT(p != NULL);

  // Sample a pixel before stroke
  const uint8_t *px = draw_panel_get_pixels(p);
  uint8_t before = px[4 * (32 * 64 + 32) + 3];  // alpha at centre

  draw_panel_mouse_down(p, 32.0f, 32.0f);
  draw_panel_mouse_drag(p, 33.0f, 32.0f);
  draw_panel_mouse_up(p);

  uint8_t after = px[4 * (32 * 64 + 32) + 3];
  ASSERT(after != before || before == 255);  // pixel was touched

  draw_panel_free(p);
  PASS();
}

TEST test_drag_without_down_is_ignored(void) {
  draw_panel *p = draw_panel_new(32, 32);
  ASSERT(p != NULL);

  const uint8_t *px = draw_panel_get_pixels(p);
  // snapshot first 64 bytes
  uint8_t snap[64];
  memcpy(snap, px, 64);

  draw_panel_mouse_drag(p, 10.0f, 10.0f);  // no preceding mouse_down

  ASSERT_EQ(0, memcmp(snap, px, 64));

  draw_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Export mask
// ---------------------------------------------------------------------------

TEST test_export_mask_empty_canvas(void) {
  draw_panel *p = draw_panel_new(8, 8);
  ASSERT(p != NULL);

  uint8_t mask[64] = {0xFF};  // pre-fill with garbage
  draw_panel_export_mask(p, mask);

  // Empty canvas → all zeros
  for (int i = 0; i < 64; i++) ASSERT_EQ(0, mask[i]);

  draw_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// NULL safety
// ---------------------------------------------------------------------------

TEST test_null_safety(void) {
  draw_panel_render(NULL, NULL);
  draw_panel_mouse_down(NULL, 0, 0);
  draw_panel_mouse_drag(NULL, 0, 0);
  draw_panel_mouse_up(NULL);
  ASSERT(draw_panel_get_pixels(NULL) == NULL);
  draw_panel_export_mask(NULL, NULL);
  ASSERT_EQ(0, draw_panel_active_layer(NULL));
  draw_panel_set_layer(NULL, 0);
  ASSERT_EQ(0, draw_panel_layer_count(NULL));
  ASSERT_FALSE(draw_panel_add_layer(NULL));
  ASSERT_FALSE(draw_panel_remove_layer(NULL, 0));
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(draw_panel_suite) {
  RUN_TEST(test_create_free);
  RUN_TEST(test_free_null_no_crash);
  RUN_TEST(test_starts_with_one_layer);
  RUN_TEST(test_add_layer);
  RUN_TEST(test_set_layer);
  RUN_TEST(test_remove_layer_shifts);
  RUN_TEST(test_cannot_remove_last_layer);
  RUN_TEST(test_cannot_exceed_max_layers);
  RUN_TEST(test_get_pixels_not_null);
  RUN_TEST(test_pixels_start_transparent);
  RUN_TEST(test_stroke_changes_pixels);
  RUN_TEST(test_drag_without_down_is_ignored);
  RUN_TEST(test_export_mask_empty_canvas);
  RUN_TEST(test_null_safety);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(draw_panel_suite);
  GREATEST_MAIN_END();
}
