// test_menubar.c — unit tests for menubar lifecycle and callback registration.
// Does NOT call menubar_render (requires a live Nuklear context).

#include "greatest.h"

// Stub out nuklear before including menubar.h
struct nk_context;

#include "gui/menubar.h"

#include <stdbool.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_fired = 0;
static void fire_cb(void *ctx) { (void)ctx; g_fired++; }
static void fire_ctx_cb(void *ctx) { if (ctx) *(int *)ctx += 1; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_menubar_new_free(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);
  menubar_free(m);
  PASS();
}

TEST test_menubar_free_null(void) {
  menubar_free(NULL);  // must not crash
  PASS();
}

TEST test_menubar_register_and_fire(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);

  g_fired = 0;
  menubar_on_open_volpkg(m, fire_cb, NULL);
  menubar_on_open_zarr(m, fire_cb, NULL);
  menubar_on_open_remote(m, fire_cb, NULL);
  menubar_on_attach_remote_zarr(m, fire_cb, NULL);
  menubar_on_generate_report(m, fire_cb, NULL);
  menubar_on_settings(m, fire_cb, NULL);
  menubar_on_import_obj(m, fire_cb, NULL);
  menubar_on_exit(m, fire_cb, NULL);

  menubar_on_toggle_volumes(m, fire_cb, NULL);
  menubar_on_toggle_segmentation(m, fire_cb, NULL);
  menubar_on_toggle_distance_transform(m, fire_cb, NULL);
  menubar_on_toggle_drawing(m, fire_cb, NULL);
  menubar_on_toggle_viewer_controls(m, fire_cb, NULL);
  menubar_on_toggle_point_collection(m, fire_cb, NULL);
  menubar_on_sync_cursor(m, fire_cb, NULL);
  menubar_on_reset_seg_views(m, fire_cb, NULL);
  menubar_on_show_console(m, fire_cb, NULL);

  menubar_on_draw_bbox(m, fire_cb, NULL);
  menubar_on_surface_from_selection(m, fire_cb, NULL);
  menubar_on_clear_selection(m, fire_cb, NULL);
  menubar_on_inpaint_rebuild(m, fire_cb, NULL);

  menubar_on_keybinds(m, fire_cb, NULL);
  menubar_on_about(m, fire_cb, NULL);

  // All registrations succeed — g_fired is still 0 (no callbacks fired yet)
  ASSERT_EQ(0, g_fired);

  menubar_free(m);
  PASS();
}

TEST test_menubar_ctx_callback(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);

  int counter = 0;
  menubar_on_exit(m, fire_ctx_cb, &counter);
  // Callbacks only fire during render; just verify registration doesn't corrupt state
  ASSERT_EQ(0, counter);

  menubar_free(m);
  PASS();
}

TEST test_menubar_add_recent_basic(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);

  menubar_add_recent(m, "/data/vol1.volpkg");
  menubar_add_recent(m, "/data/vol2.volpkg");
  menubar_add_recent(m, "/data/vol3.volpkg");

  // No crash, no assertion — recent list is internal state accessed via render
  menubar_free(m);
  PASS();
}

TEST test_menubar_add_recent_dedup(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);

  menubar_add_recent(m, "/a.volpkg");
  menubar_add_recent(m, "/b.volpkg");
  menubar_add_recent(m, "/a.volpkg");  // duplicate — should move to front, not add

  // Verify add_recent with NULL / empty doesn't crash
  menubar_add_recent(m, NULL);
  menubar_add_recent(m, "");

  menubar_free(m);
  PASS();
}

TEST test_menubar_add_recent_overflow(void) {
  menubar *m = menubar_new();
  ASSERT(m != NULL);

  // Add 20 entries — RECENT_MAX is 16, oldest should be evicted silently
  for (int i = 0; i < 20; i++) {
    char path[64];
    snprintf(path, sizeof(path), "/vol%d.volpkg", i);
    menubar_add_recent(m, path);
  }

  menubar_free(m);
  PASS();
}

TEST test_menubar_null_register(void) {
  // Registering on NULL menubar must not crash
  menubar_on_open_volpkg(NULL, fire_cb, NULL);
  menubar_on_exit(NULL, fire_cb, NULL);
  menubar_on_about(NULL, fire_cb, NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(menubar_suite) {
  RUN_TEST(test_menubar_new_free);
  RUN_TEST(test_menubar_free_null);
  RUN_TEST(test_menubar_register_and_fire);
  RUN_TEST(test_menubar_ctx_callback);
  RUN_TEST(test_menubar_add_recent_basic);
  RUN_TEST(test_menubar_add_recent_dedup);
  RUN_TEST(test_menubar_add_recent_overflow);
  RUN_TEST(test_menubar_null_register);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(menubar_suite);
  GREATEST_MAIN_END();
}
