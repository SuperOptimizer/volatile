/* test_seg_panels.c — lifecycle and render tests for seg_panels.c */
#include "greatest.h"
#include "gui/seg_panels.h"
#include "gui/seg.h"
#include "gui/seg_growth.h"
#include "core/geom.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Nuklear stub — GUI tests that don't exercise actual rendering need a
// minimal nk_context so render calls don't crash.  We provide NULL and
// guard on it inside seg_panels_render.
// ---------------------------------------------------------------------------

/* -------------------------------------------------------------------------
 * 1. lifecycle: new returns non-NULL, free doesn't crash
 * --------------------------------------------------------------------- */

TEST test_lifecycle(void) {
  seg_panels *p = seg_panels_new();
  ASSERT(p != NULL);
  seg_panels_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 2. free NULL: must not crash
 * --------------------------------------------------------------------- */

TEST test_free_null(void) {
  seg_panels_free(NULL);
  PASS();
}

/* -------------------------------------------------------------------------
 * 3. get_tool_params on fresh panel: sensible defaults
 * --------------------------------------------------------------------- */

TEST test_default_tool_params(void) {
  seg_panels *p = seg_panels_new();
  ASSERT(p != NULL);

  seg_tool_params tp = seg_panels_get_tool_params(p);
  ASSERT_EQ(tp.tool, SEG_TOOL_BRUSH);
  ASSERT(tp.radius > 0.0f);
  ASSERT(tp.sigma  > 0.0f);

  seg_panels_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 4. get_tool_params on NULL: returns zeroed struct without crash
 * --------------------------------------------------------------------- */

TEST test_tool_params_null(void) {
  seg_tool_params tp = seg_panels_get_tool_params(NULL);
  ASSERT_EQ((int)tp.tool, 0);
  ASSERT_EQ(tp.radius, 0.0f);
  PASS();
}

/* -------------------------------------------------------------------------
 * 5. get_growth_params on fresh panel: sensible defaults
 * --------------------------------------------------------------------- */

TEST test_default_growth_params(void) {
  seg_panels *p = seg_panels_new();
  ASSERT(p != NULL);

  growth_params gp = seg_panels_get_growth_params(p);
  ASSERT_EQ(gp.method, GROWTH_TRACER);
  ASSERT_EQ(gp.direction, GROWTH_DIR_ALL);
  ASSERT(gp.generations >= 1);
  ASSERT(gp.step_size   >  0.0f);

  seg_panels_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 6. get_growth_params on NULL: returns zeroed struct
 * --------------------------------------------------------------------- */

TEST test_growth_params_null(void) {
  growth_params gp = seg_panels_get_growth_params(NULL);
  ASSERT_EQ((int)gp.method, 0);
  ASSERT_EQ(gp.generations, 0);
  PASS();
}

/* -------------------------------------------------------------------------
 * 7. render with NULL ctx: must not crash (guard is inside render)
 * --------------------------------------------------------------------- */

TEST test_render_null_ctx(void) {
  seg_panels *p = seg_panels_new();
  ASSERT(p != NULL);
  seg_panels_render(p, NULL, NULL);  // must not crash
  seg_panels_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * 8. render with NULL panels: must not crash
 * --------------------------------------------------------------------- */

TEST test_render_null_panels(void) {
  seg_panels_render(NULL, NULL, NULL);  // must not crash
  PASS();
}

/* -------------------------------------------------------------------------
 * 9. multiple new/free cycles: no leak / double-free
 * --------------------------------------------------------------------- */

TEST test_multiple_lifecycle(void) {
  for (int i = 0; i < 16; i++) {
    seg_panels *p = seg_panels_new();
    ASSERT(p != NULL);
    // touch fields to ensure allocation is real
    seg_tool_params tp = seg_panels_get_tool_params(p);
    (void)tp;
    growth_params gp = seg_panels_get_growth_params(p);
    (void)gp;
    seg_panels_free(p);
  }
  PASS();
}

/* -------------------------------------------------------------------------
 * 10. returned structs are independent copies (no shared state)
 * --------------------------------------------------------------------- */

TEST test_params_are_copies(void) {
  seg_panels *p = seg_panels_new();
  ASSERT(p != NULL);

  seg_tool_params tp1 = seg_panels_get_tool_params(p);
  seg_tool_params tp2 = seg_panels_get_tool_params(p);

  /* modifying a copy must not alter the next call */
  tp1.radius = 999.0f;
  seg_tool_params tp3 = seg_panels_get_tool_params(p);
  ASSERT(tp3.radius != 999.0f);
  (void)tp2;

  seg_panels_free(p);
  PASS();
}

/* -------------------------------------------------------------------------
 * Suite
 * --------------------------------------------------------------------- */

SUITE(seg_panels_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_free_null);
  RUN_TEST(test_default_tool_params);
  RUN_TEST(test_tool_params_null);
  RUN_TEST(test_default_growth_params);
  RUN_TEST(test_growth_params_null);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_null_panels);
  RUN_TEST(test_multiple_lifecycle);
  RUN_TEST(test_params_are_copies);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(seg_panels_suite);
  GREATEST_MAIN_END();
}
