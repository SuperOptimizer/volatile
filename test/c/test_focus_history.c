#include "greatest.h"
#include "gui/focus_history.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static vec3f v(float x, float y, float z) { return (vec3f){x, y, z}; }
static bool  veq(vec3f a, vec3f b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

// ---------------------------------------------------------------------------
// Basic lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  focus_history *h = focus_history_new(10);
  ASSERT(h != NULL);
  ASSERT_EQ(0, focus_history_count(h));
  ASSERT_FALSE(focus_history_can_back(h));
  ASSERT_FALSE(focus_history_can_forward(h));
  focus_history_free(h);
  PASS();
}

TEST test_free_null(void) {
  focus_history_free(NULL);  // must not crash
  PASS();
}

TEST test_default_capacity(void) {
  focus_history *h = focus_history_new(0);
  ASSERT(h != NULL);
  focus_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Push and count
// ---------------------------------------------------------------------------

TEST test_push_increments_count(void) {
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,0,0), 0);
  ASSERT_EQ(1, focus_history_count(h));
  focus_history_push(h, v(2,0,0), 0);
  ASSERT_EQ(2, focus_history_count(h));
  focus_history_free(h);
  PASS();
}

TEST test_single_entry_no_back_no_forward(void) {
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,2,3), 1);
  ASSERT_FALSE(focus_history_can_back(h));
  ASSERT_FALSE(focus_history_can_forward(h));
  focus_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Back / forward navigation
// ---------------------------------------------------------------------------

TEST test_back_returns_previous(void) {
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,0,0), 0);
  focus_history_push(h, v(2,0,0), 1);

  vec3f pos; int lvl;
  ASSERT(focus_history_back(h, &pos, &lvl));
  ASSERT(veq(pos, v(1,0,0)));
  ASSERT_EQ(0, lvl);
  ASSERT_FALSE(focus_history_can_back(h));
  focus_history_free(h);
  PASS();
}

TEST test_back_then_forward(void) {
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,0,0), 0);
  focus_history_push(h, v(2,0,0), 1);
  focus_history_push(h, v(3,0,0), 2);

  vec3f pos; int lvl;
  ASSERT(focus_history_back(h, &pos, &lvl));  // -> pos2
  ASSERT(veq(pos, v(2,0,0)));
  ASSERT(focus_history_back(h, &pos, &lvl));  // -> pos1
  ASSERT(veq(pos, v(1,0,0)));
  ASSERT_FALSE(focus_history_can_back(h));
  ASSERT(focus_history_can_forward(h));

  ASSERT(focus_history_forward(h, &pos, &lvl));  // -> pos2
  ASSERT(veq(pos, v(2,0,0)));
  ASSERT(focus_history_forward(h, &pos, &lvl));  // -> pos3
  ASSERT(veq(pos, v(3,0,0)));
  ASSERT_FALSE(focus_history_can_forward(h));
  focus_history_free(h);
  PASS();
}

TEST test_back_at_start_returns_false(void) {
  focus_history *h = focus_history_new(10);
  vec3f pos; int lvl;
  ASSERT_FALSE(focus_history_back(h, &pos, &lvl));  // empty
  focus_history_push(h, v(1,0,0), 0);
  ASSERT_FALSE(focus_history_back(h, &pos, &lvl));  // already at start
  focus_history_free(h);
  PASS();
}

TEST test_forward_at_end_returns_false(void) {
  focus_history *h = focus_history_new(10);
  vec3f pos; int lvl;
  focus_history_push(h, v(1,0,0), 0);
  ASSERT_FALSE(focus_history_forward(h, &pos, &lvl));
  focus_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Push truncates forward history
// ---------------------------------------------------------------------------

TEST test_push_truncates_forward(void) {
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,0,0), 0);
  focus_history_push(h, v(2,0,0), 0);
  focus_history_push(h, v(3,0,0), 0);

  vec3f pos; int lvl;
  focus_history_back(h, &pos, &lvl);  // cur -> pos2
  focus_history_back(h, &pos, &lvl);  // cur -> pos1

  // Push new entry: forward history (pos2, pos3) should be discarded.
  focus_history_push(h, v(10,0,0), 5);
  ASSERT_FALSE(focus_history_can_forward(h));
  ASSERT_EQ(2, focus_history_count(h));  // pos1 + new

  // Back should give pos1.
  ASSERT(focus_history_back(h, &pos, &lvl));
  ASSERT(veq(pos, v(1,0,0)));
  focus_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Circular buffer wrapping at max_entries
// ---------------------------------------------------------------------------

TEST test_wrapping_at_capacity(void) {
  const int cap = 4;
  focus_history *h = focus_history_new(cap);

  // Push cap+2 entries: oldest two should be evicted.
  for (int i = 1; i <= cap + 2; i++) {
    focus_history_push(h, v((float)i, 0, 0), i);
  }

  // Count should stay capped.
  ASSERT_EQ(cap, focus_history_count(h));
  ASSERT_FALSE(focus_history_can_forward(h));

  // Walk back through all entries: newest first, then older.
  vec3f pos; int lvl;
  int expected = cap + 2;
  // We're at the last entry (cap+2). Walk back cap-1 times.
  for (int i = 0; i < cap - 1; i++) {
    ASSERT(focus_history_back(h, &pos, &lvl));
    expected--;
    ASSERT(veq(pos, v((float)expected, 0, 0)));
  }
  ASSERT_FALSE(focus_history_can_back(h));
  focus_history_free(h);
  PASS();
}

TEST test_wrapping_null_out_params(void) {
  // back/forward must not crash when output pointers are NULL.
  focus_history *h = focus_history_new(10);
  focus_history_push(h, v(1,0,0), 0);
  focus_history_push(h, v(2,0,0), 0);
  ASSERT(focus_history_back(h, NULL, NULL));
  ASSERT(focus_history_forward(h, NULL, NULL));
  focus_history_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(focus_history_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_default_capacity);
  RUN_TEST(test_push_increments_count);
  RUN_TEST(test_single_entry_no_back_no_forward);
  RUN_TEST(test_back_returns_previous);
  RUN_TEST(test_back_then_forward);
  RUN_TEST(test_back_at_start_returns_false);
  RUN_TEST(test_forward_at_end_returns_false);
  RUN_TEST(test_push_truncates_forward);
  RUN_TEST(test_wrapping_at_capacity);
  RUN_TEST(test_wrapping_null_out_params);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(focus_history_suite);
  GREATEST_MAIN_END();
}
