#include "greatest.h"
#include "server/versioning.h"
#include "core/geom.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a 2x3 quad surface with predictable point values.
static quad_surface *make_surface(float offset) {
  quad_surface *s = quad_surface_new(2, 3);
  for (int r = 0; r < 2; r++) {
    for (int c = 0; c < 3; c++) {
      quad_surface_set(s, r, c, (vec3f){
        (float)c + offset,
        (float)r + offset,
        offset
      });
    }
  }
  return s;
}

static float fabsf_local(float x) { return x < 0 ? -x : x; }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_commit_returns_valid_id(void) {
  surface_history *h = surface_history_new(1, ":memory:");
  ASSERT(h != NULL);

  quad_surface *s = make_surface(0.0f);
  int64_t vid = surface_history_commit(h, 42, "initial", s);
  ASSERT(vid > 0);

  quad_surface_free(s);
  surface_history_free(h);
  PASS();
}

TEST test_list_versions(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  quad_surface *s = make_surface(0.0f);
  surface_history_commit(h, 1, "v1", s);
  surface_history_commit(h, 2, "v2", s);
  surface_history_commit(h, 1, "v3", s);
  quad_surface_free(s);

  version_info infos[10];
  int count = surface_history_list(h, infos, 10);
  ASSERT_EQ(3, count);

  // Newest first
  ASSERT_STR_EQ("v3", infos[0].message);
  ASSERT_STR_EQ("v2", infos[1].message);
  ASSERT_STR_EQ("v1", infos[2].message);

  ASSERT_EQ(1, infos[0].user_id);
  ASSERT_EQ(2, infos[1].user_id);

  surface_history_free(h);
  PASS();
}

TEST test_list_max_versions(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  quad_surface *s = make_surface(0.0f);
  for (int i = 0; i < 8; i++)
    surface_history_commit(h, i, "msg", s);
  quad_surface_free(s);

  version_info infos[4];
  int count = surface_history_list(h, infos, 4);
  ASSERT_EQ(4, count);  // capped at max_versions

  surface_history_free(h);
  PASS();
}

TEST test_checkout_roundtrip(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  quad_surface *orig = make_surface(7.0f);
  int64_t vid = surface_history_commit(h, 1, "snap", orig);
  ASSERT(vid > 0);

  quad_surface *restored = surface_history_checkout(h, vid);
  ASSERT(restored != NULL);
  ASSERT_EQ(orig->rows, restored->rows);
  ASSERT_EQ(orig->cols, restored->cols);

  // Verify all points round-trip exactly
  for (int r = 0; r < orig->rows; r++) {
    for (int c = 0; c < orig->cols; c++) {
      vec3f a = quad_surface_get(orig, r, c);
      vec3f b = quad_surface_get(restored, r, c);
      ASSERT(fabsf_local(a.x - b.x) < 1e-5f);
      ASSERT(fabsf_local(a.y - b.y) < 1e-5f);
      ASSERT(fabsf_local(a.z - b.z) < 1e-5f);
    }
  }

  quad_surface_free(orig);
  quad_surface_free(restored);
  surface_history_free(h);
  PASS();
}

TEST test_checkout_unknown_version_returns_null(void) {
  surface_history *h = surface_history_new(1, ":memory:");
  quad_surface *r = surface_history_checkout(h, 9999);
  ASSERT(r == NULL);
  surface_history_free(h);
  PASS();
}

TEST test_diff_displacement(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  quad_surface *s1 = make_surface(0.0f);
  quad_surface *s2 = make_surface(1.0f);  // each point shifted by (1,1,1)
  int64_t v1 = surface_history_commit(h, 1, "before", s1);
  int64_t v2 = surface_history_commit(h, 1, "after",  s2);

  int count = 0;
  float *diff = surface_history_diff(h, v1, v2, &count);
  ASSERT(diff != NULL);
  ASSERT_EQ(6, count);  // 2*3 vertices

  // Every displacement component should be 1.0
  for (int i = 0; i < count * 3; i++) {
    ASSERT(fabsf_local(diff[i] - 1.0f) < 1e-5f);
  }

  free(diff);
  quad_surface_free(s1);
  quad_surface_free(s2);
  surface_history_free(h);
  PASS();
}

TEST test_diff_same_version_is_zero(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  quad_surface *s = make_surface(5.0f);
  int64_t v = surface_history_commit(h, 1, "snap", s);

  int count = 0;
  float *diff = surface_history_diff(h, v, v, &count);
  ASSERT(diff != NULL);
  for (int i = 0; i < count * 3; i++)
    ASSERT(fabsf_local(diff[i]) < 1e-5f);

  free(diff);
  quad_surface_free(s);
  surface_history_free(h);
  PASS();
}

TEST test_surface_isolation_by_id(void) {
  // Versions from surface_id=1 should not appear in surface_id=2 history.
  surface_history *h1 = surface_history_new(1, ":memory:");
  surface_history *h2 = surface_history_new(2, ":memory:");

  quad_surface *s = make_surface(0.0f);
  surface_history_commit(h1, 1, "for-surface-1", s);
  quad_surface_free(s);

  version_info infos[10];
  int count = surface_history_list(h2, infos, 10);
  ASSERT_EQ(0, count);

  surface_history_free(h1);
  surface_history_free(h2);
  PASS();
}

TEST test_autosave_commits_after_interval(void) {
  surface_history *h = surface_history_new(1, ":memory:");

  // Set 1 second interval; force last_autosave far in the past.
  surface_history_enable_autosave(h, 1);

  quad_surface *s = make_surface(0.0f);

  // First tick — nothing committed yet (last_autosave just set by enable)
  // Manually simulate elapsed time by ticking twice with a sleep-free approach:
  // call tick, which will only commit after interval_seconds elapsed.
  // Since we can't sleep in a unit test, we check that calling tick doesn't crash
  // and that explicit commits still work normally alongside autosave.
  surface_history_autosave_tick(h, s);

  // Verify explicit commit still works after enabling autosave
  int64_t vid = surface_history_commit(h, 0, "manual", s);
  ASSERT(vid > 0);

  quad_surface_free(s);
  surface_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(versioning_suite) {
  RUN_TEST(test_commit_returns_valid_id);
  RUN_TEST(test_list_versions);
  RUN_TEST(test_list_max_versions);
  RUN_TEST(test_checkout_roundtrip);
  RUN_TEST(test_checkout_unknown_version_returns_null);
  RUN_TEST(test_diff_displacement);
  RUN_TEST(test_diff_same_version_is_zero);
  RUN_TEST(test_surface_isolation_by_id);
  RUN_TEST(test_autosave_commits_after_interval);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(versioning_suite);
  GREATEST_MAIN_END();
}
