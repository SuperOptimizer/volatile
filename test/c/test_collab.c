#include "greatest.h"
#include "server/collab.h"

#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Create a session without a real server (NULL is accepted).
static collab_session *open_session(void) {
  return collab_new(NULL);
}

// ---------------------------------------------------------------------------
// Tests: user management
// ---------------------------------------------------------------------------

TEST test_add_remove_users(void) {
  collab_session *s = open_session();
  ASSERT(s);

  ASSERT_EQ(0, collab_user_count(s));

  int id1 = collab_add_user(s, "alice");
  int id2 = collab_add_user(s, "bob");
  ASSERT(id1 > 0);
  ASSERT(id2 > 0);
  ASSERT(id1 != id2);
  ASSERT_EQ(2, collab_user_count(s));

  const collab_user *u1 = collab_get_user(s, id1);
  ASSERT(u1);
  ASSERT_STR_EQ("alice", u1->username);
  ASSERT(u1->active);

  collab_remove_user(s, id1);
  ASSERT_EQ(1, collab_user_count(s));
  ASSERT_EQ(NULL, collab_get_user(s, id1));

  collab_free(s);
  PASS();
}

TEST test_unique_colors(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int ids[8];
  for (int i = 0; i < 8; i++)
    ids[i] = collab_add_user(s, "x");

  // Each of the first 8 users should get a different palette color.
  for (int i = 0; i < 8; i++) {
    const collab_user *u = collab_get_user(s, ids[i]);
    ASSERT(u);
    for (int j = i + 1; j < 8; j++) {
      const collab_user *v = collab_get_user(s, ids[j]);
      ASSERT(v);
      // Colors should differ (palette has 8 distinct entries).
      bool same = (u->color[0] == v->color[0] &&
                   u->color[1] == v->color[1] &&
                   u->color[2] == v->color[2]);
      ASSERT(!same);
    }
  }

  collab_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: cursor updates
// ---------------------------------------------------------------------------

TEST test_cursor_update(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int id = collab_add_user(s, "charlie");
  ASSERT(id > 0);

  vec3f pos = { 10.f, 20.f, 30.f };
  // With srv=NULL, broadcast is a no-op; we just check state is updated.
  collab_update_cursor(s, id, pos, 3);

  const collab_user *u = collab_get_user(s, id);
  ASSERT(u);
  ASSERT_IN_RANGE(10.f, u->cursor_pos.x, 0.001f);
  ASSERT_IN_RANGE(20.f, u->cursor_pos.y, 0.001f);
  ASSERT_IN_RANGE(30.f, u->cursor_pos.z, 0.001f);
  ASSERT_EQ(3, u->active_tool);

  collab_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: region locking
// ---------------------------------------------------------------------------

TEST test_lock_basic(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int alice = collab_add_user(s, "alice");
  int bob   = collab_add_user(s, "bob");

  // Alice locks a region on surface 1.
  bool ok = collab_lock_region(s, alice, 1, 0.5f, 0.5f, 0.08f);
  ASSERT(ok);

  // Bob tries to lock the same region — should fail.
  ok = collab_lock_region(s, bob, 1, 0.5f, 0.5f, 0.08f);
  ASSERT(!ok);

  // Bob can lock a non-overlapping region.
  ok = collab_lock_region(s, bob, 1, 0.9f, 0.9f, 0.02f);
  ASSERT(ok);

  collab_free(s);
  PASS();
}

TEST test_lock_unlock(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int alice = collab_add_user(s, "alice");
  int bob   = collab_add_user(s, "bob");

  ASSERT(collab_lock_region(s, alice, 1, 0.5f, 0.5f, 0.08f));
  ASSERT(!collab_lock_region(s, bob, 1, 0.5f, 0.5f, 0.08f));

  collab_unlock_region(s, alice, 1);

  // After unlock, Bob should succeed.
  ASSERT(collab_lock_region(s, bob, 1, 0.5f, 0.5f, 0.08f));

  collab_free(s);
  PASS();
}

TEST test_lock_different_surfaces(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int alice = collab_add_user(s, "alice");
  int bob   = collab_add_user(s, "bob");

  // Alice locks region on surface 1.
  ASSERT(collab_lock_region(s, alice, 1, 0.5f, 0.5f, 0.08f));

  // Bob can lock the same UV region on a different surface.
  ASSERT(collab_lock_region(s, bob, 2, 0.5f, 0.5f, 0.08f));

  collab_free(s);
  PASS();
}

TEST test_lock_released_on_remove(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int alice = collab_add_user(s, "alice");
  int bob   = collab_add_user(s, "bob");

  ASSERT(collab_lock_region(s, alice, 1, 0.5f, 0.5f, 0.08f));
  // Remove alice — all her locks should be released.
  collab_remove_user(s, alice);

  ASSERT(collab_lock_region(s, bob, 1, 0.5f, 0.5f, 0.08f));

  collab_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: event ring buffer
// ---------------------------------------------------------------------------

TEST test_event_ring(void) {
  collab_session *s = open_session();
  ASSERT(s);

  ASSERT_EQ(0, collab_event_count(s));

  // Push a few events.
  uint8_t data[4] = { 0xDE, 0xAD, 0xBE, 0xEF };
  for (int i = 0; i < 5; i++) {
    collab_event ev = {
      .user_id      = 1,
      .timestamp_ms = (int64_t)(i + 1) * 100,
      .edit_type    = EDIT_SEG_PAINT,
      .payload      = data,
      .payload_len  = 4,
    };
    collab_push_event(s, &ev);
  }

  ASSERT_EQ(5, collab_event_count(s));

  const collab_event *e0 = collab_get_event(s, 0);
  ASSERT(e0);
  ASSERT_EQ(100, e0->timestamp_ms);
  ASSERT_EQ(EDIT_SEG_PAINT, e0->edit_type);
  ASSERT(memcmp(e0->payload, data, 4) == 0);

  const collab_event *e4 = collab_get_event(s, 4);
  ASSERT(e4);
  ASSERT_EQ(500, e4->timestamp_ms);

  // Out-of-bounds returns NULL.
  ASSERT_EQ(NULL, collab_get_event(s, 5));

  collab_free(s);
  PASS();
}

TEST test_event_ring_overflow(void) {
  collab_session *s = open_session();
  ASSERT(s);

  // Push 520 events (> ring capacity of 512) to test wrap-around.
  for (int i = 0; i < 520; i++) {
    collab_event ev = {
      .user_id      = 1,
      .timestamp_ms = (int64_t)(i + 1),
      .edit_type    = EDIT_ANNOT_ADD,
      .payload      = NULL,
      .payload_len  = 0,
    };
    collab_push_event(s, &ev);
  }

  // Ring is capped at 512; oldest events are overwritten.
  ASSERT_EQ(512, collab_event_count(s));

  // The oldest remaining event should have timestamp 520-512+1 = 9.
  const collab_event *first = collab_get_event(s, 0);
  ASSERT(first);
  ASSERT_EQ(9, first->timestamp_ms);

  // The newest should be 520.
  const collab_event *last = collab_get_event(s, 511);
  ASSERT(last);
  ASSERT_EQ(520, last->timestamp_ms);

  collab_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: cursor overlay
// ---------------------------------------------------------------------------

TEST test_render_cursors(void) {
  collab_session *s = open_session();
  ASSERT(s);

  int alice = collab_add_user(s, "alice");
  int bob   = collab_add_user(s, "bob");

  collab_update_cursor(s, alice, (vec3f){ 10.f, 20.f, 0.f }, 0);
  collab_update_cursor(s, bob,   (vec3f){ 50.f, 60.f, 0.f }, 1);

  overlay_list *ov = overlay_list_new();
  ASSERT(ov);

  // Exclude alice: only bob's cursor should appear.
  collab_render_cursors(s, alice, ov);
  // Each user adds a point + a text label = 2 overlay items.
  ASSERT_EQ(2, overlay_count(ov));

  overlay_list_clear(ov);
  // No exclusion: both cursors (4 items).
  collab_render_cursors(s, -1, ov);
  ASSERT_EQ(4, overlay_count(ov));

  overlay_list_free(ov);
  collab_free(s);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(suite_collab) {
  RUN_TEST(test_add_remove_users);
  RUN_TEST(test_unique_colors);
  RUN_TEST(test_cursor_update);
  RUN_TEST(test_lock_basic);
  RUN_TEST(test_lock_unlock);
  RUN_TEST(test_lock_different_surfaces);
  RUN_TEST(test_lock_released_on_remove);
  RUN_TEST(test_event_ring);
  RUN_TEST(test_event_ring_overflow);
  RUN_TEST(test_render_cursors);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_collab);
  GREATEST_MAIN_END();
}
