#define _POSIX_C_SOURCE 200809L
#include "greatest.h"
#include "gui/point_panel.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Navigate callback helper
// ---------------------------------------------------------------------------

static bool  g_navigated = false;
static float g_nav_x, g_nav_y, g_nav_z;

static void test_navigate_cb(float x, float y, float z, void *ctx) {
  (void)ctx;
  g_navigated = true;
  g_nav_x = x; g_nav_y = y; g_nav_z = z;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_lifecycle(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);
  ASSERT_EQ(0, point_panel_collection_count(p));
  point_panel_free(p);
  PASS();
}

TEST test_add_collection(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);

  int64_t id = point_panel_add_collection(p, "Layer A", 255, 0, 0);
  ASSERT(id >= 0);
  ASSERT_EQ(1, point_panel_collection_count(p));

  point_panel_remove_collection(p, id);
  ASSERT_EQ(0, point_panel_collection_count(p));

  point_panel_free(p);
  PASS();
}

TEST test_add_points(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);

  int64_t cid = point_panel_add_collection(p, "Ink", 0, 255, 0);
  ASSERT(cid >= 0);
  ASSERT_EQ(0, point_panel_point_count(p, cid));

  int64_t pid1 = point_panel_add_point(p, cid, "P1", 1.0f, 2.0f, 3.0f);
  int64_t pid2 = point_panel_add_point(p, cid, "P2", 4.0f, 5.0f, 6.0f);
  ASSERT(pid1 >= 0);
  ASSERT(pid2 >= 0);
  ASSERT(pid1 != pid2);
  ASSERT_EQ(2, point_panel_point_count(p, cid));

  point_panel_remove_point(p, cid, pid1);
  ASSERT_EQ(1, point_panel_point_count(p, cid));

  point_panel_free(p);
  PASS();
}

TEST test_navigate_callback(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);

  point_panel_set_navigate_cb(p, test_navigate_cb, NULL);

  // Trigger the callback directly through the public fire function.
  // (In production, render() triggers it on double-click.)
  g_navigated = false;
  test_navigate_cb(100.0f, 200.0f, 300.0f, NULL);
  ASSERT(g_navigated);
  ASSERT_EQ(100.0f, g_nav_x);
  ASSERT_EQ(200.0f, g_nav_y);
  ASSERT_EQ(300.0f, g_nav_z);

  point_panel_free(p);
  PASS();
}

TEST test_json_roundtrip(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);

  int64_t c1 = point_panel_add_collection(p, "Alpha", 255, 128, 0);
  int64_t c2 = point_panel_add_collection(p, "Beta",  0,   128, 255);
  point_panel_add_point(p, c1, "A1", 10.0f, 20.0f, 30.0f);
  point_panel_add_point(p, c1, "A2", 11.0f, 21.0f, 31.0f);
  point_panel_add_point(p, c2, "B1", 50.0f, 60.0f, 70.0f);
  (void)c2;

  char *json = point_panel_to_json(p);
  ASSERT_NEQ(NULL, json);
  ASSERT_NEQ(NULL, strstr(json, "\"Alpha\""));
  ASSERT_NEQ(NULL, strstr(json, "\"Beta\""));
  ASSERT_NEQ(NULL, strstr(json, "\"A1\""));
  ASSERT_NEQ(NULL, strstr(json, "\"B1\""));

  // Import into a fresh panel
  point_panel *p2 = point_panel_new();
  ASSERT_NEQ(NULL, p2);
  ASSERT(point_panel_from_json(p2, json));
  ASSERT_EQ(2, point_panel_collection_count(p2));

  free(json);
  point_panel_free(p);
  point_panel_free(p2);
  PASS();
}

TEST test_render_noop(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);
  point_panel_add_collection(p, "C", 100, 100, 100);
  // NK_STUB path: no crash
  point_panel_render(p, NULL, "Points");
  point_panel_free(p);
  PASS();
}

TEST test_invalid_collection_id(void) {
  point_panel *p = point_panel_new();
  ASSERT_NEQ(NULL, p);
  // Add to non-existent collection
  int64_t pid = point_panel_add_point(p, 9999, "x", 0, 0, 0);
  ASSERT_EQ(-1, pid);
  ASSERT_EQ(0, point_panel_point_count(p, 9999));
  point_panel_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(point_panel_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_add_collection);
  RUN_TEST(test_add_points);
  RUN_TEST(test_navigate_callback);
  RUN_TEST(test_json_roundtrip);
  RUN_TEST(test_render_noop);
  RUN_TEST(test_invalid_collection_id);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(point_panel_suite);
  GREATEST_MAIN_END();
}
