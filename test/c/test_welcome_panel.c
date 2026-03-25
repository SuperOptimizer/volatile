// test_welcome_panel.c — lifecycle, recent files, null-safety.
// Does not call welcome_panel_render (requires a live Nuklear context).

#include "greatest.h"

struct nk_context;  // stub

#include "gui/welcome_panel.h"
#include <string.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  welcome_panel_free(w);
  PASS();
}

TEST test_free_null(void) {
  welcome_panel_free(NULL);
  PASS();
}

TEST test_add_recent_basic(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  welcome_panel_add_recent(w, "/data/vol1.zarr", "vol1");
  welcome_panel_add_recent(w, "/data/vol2.zarr", "vol2");
  welcome_panel_add_recent(w, "s3://bucket/vol3.zarr/", "vol3");
  welcome_panel_free(w);
  PASS();
}

TEST test_add_recent_dedup(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  welcome_panel_add_recent(w, "/a.zarr", "a");
  welcome_panel_add_recent(w, "/b.zarr", "b");
  welcome_panel_add_recent(w, "/a.zarr", "a-again");  // should move to front
  // Just verify no crash and function ran; internal order is opaque
  welcome_panel_free(w);
  PASS();
}

TEST test_add_recent_overflow(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  // RECENT_MAX is 8; add 12 entries
  for (int i = 0; i < 12; i++) {
    char path[64], name[32];
    snprintf(path, sizeof(path), "/vol%d.zarr", i);
    snprintf(name, sizeof(name), "vol%d", i);
    welcome_panel_add_recent(w, path, name);
  }
  welcome_panel_free(w);
  PASS();
}

TEST test_add_recent_null(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  welcome_panel_add_recent(w, NULL,  "name");   // null path: no-op
  welcome_panel_add_recent(w, "",    "name");   // empty path: no-op
  welcome_panel_add_recent(w, "/x.zarr", NULL); // null name: uses path
  welcome_panel_add_recent(NULL, "/x.zarr", "x");  // null panel: no-op
  welcome_panel_free(w);
  PASS();
}

TEST test_render_null_ctx(void) {
  welcome_panel *w = welcome_panel_new();
  ASSERT(w != NULL);
  welcome_result r = welcome_panel_render(w, NULL, 800, 600);
  ASSERT_EQ(WELCOME_NONE, r.action);
  welcome_panel_free(w);
  PASS();
}

TEST test_render_null_panel(void) {
  welcome_result r = welcome_panel_render(NULL, NULL, 800, 600);
  ASSERT_EQ(WELCOME_NONE, r.action);
  PASS();
}

TEST test_action_enum_values(void) {
  // Verify enum is distinct and NONE is 0
  ASSERT_EQ(0, (int)WELCOME_NONE);
  ASSERT(WELCOME_OPEN_ZARR    != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_VOLPKG  != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_S3      != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_URL     != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_PROJECT != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_RECENT  != WELCOME_NONE);
  ASSERT(WELCOME_OPEN_ZARR    != WELCOME_OPEN_VOLPKG);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(welcome_panel_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_add_recent_basic);
  RUN_TEST(test_add_recent_dedup);
  RUN_TEST(test_add_recent_overflow);
  RUN_TEST(test_add_recent_null);
  RUN_TEST(test_render_null_ctx);
  RUN_TEST(test_render_null_panel);
  RUN_TEST(test_action_enum_values);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(welcome_panel_suite);
  GREATEST_MAIN_END();
}
