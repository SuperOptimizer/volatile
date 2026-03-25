#include "greatest.h"
#include "render/overlay.h"

#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Check that pixel at (x,y) in a width*height RGBA buffer is non-zero.
static int pixel_nonzero(const uint8_t *pixels, int width, int x, int y) {
  const uint8_t *p = pixels + (y * width + x) * 4;
  return (p[0] | p[1] | p[2] | p[3]) != 0;
}

// ---------------------------------------------------------------------------
// overlay_list lifecycle
// ---------------------------------------------------------------------------

TEST test_overlay_list_create(void) {
  overlay_list *l = overlay_list_new();
  ASSERT(l != NULL);
  ASSERT_EQ(0, overlay_count(l));
  overlay_list_free(l);
  PASS();
}

TEST test_overlay_list_clear(void) {
  overlay_list *l = overlay_list_new();
  overlay_add_point(l, 10, 10, 255, 0, 0, 3.0f);
  overlay_add_point(l, 20, 20, 0, 255, 0, 3.0f);
  ASSERT_EQ(2, overlay_count(l));
  overlay_list_clear(l);
  ASSERT_EQ(0, overlay_count(l));
  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Render: point
// ---------------------------------------------------------------------------

TEST test_render_point(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_point(l, 32, 32, 255, 0, 0, 2.0f);
  overlay_render(l, buf, W, H);

  // center pixel must be set
  ASSERT(pixel_nonzero(buf, W, 32, 32));
  // red channel must dominate
  ASSERT(buf[(32 * W + 32) * 4 + 0] > 200);

  overlay_list_free(l);
  PASS();
}

TEST test_render_point_offscreen(void) {
  // points outside the buffer must not crash and must leave buffer clean
  int W = 32, H = 32;
  uint8_t buf[32 * 32 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_point(l, -10, -10, 255, 0, 0, 3.0f);
  overlay_add_point(l, 200, 200, 255, 0, 0, 3.0f);
  overlay_render(l, buf, W, H);  // must not crash

  // buffer should remain all zero (off-screen)
  int any = 0;
  for (int i = 0; i < W * H * 4; i++) any |= buf[i];
  ASSERT_EQ(0, any);

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Render: line
// ---------------------------------------------------------------------------

TEST test_render_line_horizontal(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_line(l, 10, 32, 50, 32, 0, 255, 0, 1.0f);
  overlay_render(l, buf, W, H);

  // pixels along y=32 between x=10..50 must be set
  ASSERT(pixel_nonzero(buf, W, 10, 32));
  ASSERT(pixel_nonzero(buf, W, 30, 32));
  ASSERT(pixel_nonzero(buf, W, 50, 32));

  overlay_list_free(l);
  PASS();
}

TEST test_render_line_diagonal(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_line(l, 5, 5, 55, 55, 0, 0, 255, 1.0f);
  overlay_render(l, buf, W, H);

  ASSERT(pixel_nonzero(buf, W,  5,  5));
  ASSERT(pixel_nonzero(buf, W, 30, 30));
  ASSERT(pixel_nonzero(buf, W, 55, 55));

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Render: rect
// ---------------------------------------------------------------------------

TEST test_render_rect(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_rect(l, 10, 10, 20, 20, 255, 255, 0);
  overlay_render(l, buf, W, H);

  // corners must be drawn
  ASSERT(pixel_nonzero(buf, W, 10, 10));
  ASSERT(pixel_nonzero(buf, W, 30, 10));
  ASSERT(pixel_nonzero(buf, W, 10, 30));
  ASSERT(pixel_nonzero(buf, W, 30, 30));
  // interior should not be filled
  ASSERT(!pixel_nonzero(buf, W, 20, 20));

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Render: circle
// ---------------------------------------------------------------------------

TEST test_render_circle(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_circle(l, 32, 32, 10, 255, 128, 0);
  overlay_render(l, buf, W, H);

  // cardinal points on circle outline must be set
  ASSERT(pixel_nonzero(buf, W, 42, 32));  // right
  ASSERT(pixel_nonzero(buf, W, 22, 32));  // left
  ASSERT(pixel_nonzero(buf, W, 32, 42));  // bottom
  ASSERT(pixel_nonzero(buf, W, 32, 22));  // top
  // center must NOT be filled (outline only)
  ASSERT(!pixel_nonzero(buf, W, 32, 32));

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Render: text marker
// ---------------------------------------------------------------------------

TEST test_render_text(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_text(l, 32, 32, "hello", 200, 200, 200);
  overlay_render(l, buf, W, H);

  // text marker draws a cross; anchor must be set
  ASSERT(pixel_nonzero(buf, W, 32, 32));

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Multiple overlays, count
// ---------------------------------------------------------------------------

TEST test_overlay_count_multiple(void) {
  overlay_list *l = overlay_list_new();
  overlay_add_point(l, 5,  5,  255, 0,   0,   3.0f);
  overlay_add_line(l,  0,  0,  10, 10, 0, 255,   0,   1.0f);
  overlay_add_rect(l,  1,  1,  8,  8,  0,   0, 255);
  overlay_add_circle(l, 5, 5,  4,  255, 255, 0);
  overlay_add_text(l,  5,  5,  "x", 255, 255, 255);
  ASSERT_EQ(5, overlay_count(l));
  overlay_list_free(l);
  PASS();
}

TEST test_render_multiple_colors(void) {
  int W = 64, H = 64;
  uint8_t buf[64 * 64 * 4];
  memset(buf, 0, sizeof(buf));

  overlay_list *l = overlay_list_new();
  overlay_add_point(l, 10, 10, 255, 0,   0,   3.0f);  // red
  overlay_add_point(l, 50, 50, 0,   255, 0,   3.0f);  // green
  overlay_render(l, buf, W, H);

  // red point: red channel dominant
  uint8_t *pr = buf + (10 * W + 10) * 4;
  ASSERT(pr[0] > 200 && pr[1] < 50);

  // green point: green channel dominant
  uint8_t *pg = buf + (50 * W + 50) * 4;
  ASSERT(pg[1] > 200 && pg[0] < 50);

  overlay_list_free(l);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(overlay_suite) {
  RUN_TEST(test_overlay_list_create);
  RUN_TEST(test_overlay_list_clear);
  RUN_TEST(test_render_point);
  RUN_TEST(test_render_point_offscreen);
  RUN_TEST(test_render_line_horizontal);
  RUN_TEST(test_render_line_diagonal);
  RUN_TEST(test_render_rect);
  RUN_TEST(test_render_circle);
  RUN_TEST(test_render_text);
  RUN_TEST(test_overlay_count_multiple);
  RUN_TEST(test_render_multiple_colors);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(overlay_suite);
  GREATEST_MAIN_END();
}
