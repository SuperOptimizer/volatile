#define _POSIX_C_SOURCE 200809L

#include "server/collab.h"
#include "core/hash.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_USERS        64
#define EVENT_RING_CAP  512

// Region lock granularity: uv space is divided into cells of this size.
// Two users collide if their lock circles cover the same cell.
#define LOCK_CELL_SIZE  0.05f

// Preset color palette — 8 distinct hues cycling for new users.
static const uint8_t PALETTE[][3] = {
  { 230,  80,  60 },   // red
  {  60, 180, 110 },   // green
  {  70, 130, 220 },   // blue
  { 220, 170,  50 },   // amber
  { 180,  80, 220 },   // purple
  {  60, 200, 210 },   // cyan
  { 220, 120,  60 },   // orange
  { 160, 200,  70 },   // lime
};
#define PALETTE_LEN  (int)(sizeof PALETTE / sizeof PALETTE[0])

// ---------------------------------------------------------------------------
// Lock entry: one row in the lock table per (surface_id, grid_cell) pair.
// Key is a composite uint64: upper 32 bits = cell_u, lower 32 bits = cell_v
// combined with surface_id in a string key "surfid:cu:cv".
// We use the string hash_map keyed by "surface_id:cell_u:cell_v".
// ---------------------------------------------------------------------------

typedef struct {
  int     owner_id;
  int64_t surface_id;
  int     cell_u;
  int     cell_v;
} lock_entry;

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

struct collab_session {
  vol_server *srv;

  collab_user  users[MAX_USERS];
  int          next_user_id;
  int          user_count;

  // Region locks: string key -> lock_entry*
  hash_map    *locks;

  // Event ring buffer
  collab_event events[EVENT_RING_CAP];
  int          ev_head;    // oldest slot index
  int          ev_count;   // live entries (<= EVENT_RING_CAP)
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int64_t now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000 + (int64_t)(ts.tv_nsec / 1000000);
}

static collab_user *find_user(collab_session *s, int user_id) {
  for (int i = 0; i < MAX_USERS; i++) {
    if (s->users[i].active && s->users[i].user_id == user_id)
      return &s->users[i];
  }
  return NULL;
}

static collab_user *find_slot(collab_session *s) {
  for (int i = 0; i < MAX_USERS; i++) {
    if (!s->users[i].active)
      return &s->users[i];
  }
  return NULL;
}

// Build the composite lock key string into buf.
static void lock_key(char *buf, size_t sz, int64_t surface_id, int cu, int cv) {
  snprintf(buf, sz, "%lld:%d:%d", (long long)surface_id, cu, cv);
}

// Release all locks held by user_id on a specific surface (-1 means all surfaces).
static void release_locks_for(collab_session *s, int user_id, int64_t surface_filter) {
  hash_map_iter *it = hash_map_iter_new(s->locks);
  if (!it) return;

  // Collect keys to remove (can't delete during iteration).
  char *to_del[4096];
  int nd = 0;

  hash_map_entry ent;
  while (hash_map_iter_next(it, &ent)) {
    lock_entry *le = ent.val;
    if (le->owner_id == user_id &&
        (surface_filter < 0 || le->surface_id == surface_filter)) {
      to_del[nd++] = (char *)ent.key;
      if (nd >= (int)(sizeof to_del / sizeof to_del[0])) break;
    }
  }
  hash_map_iter_free(it);

  for (int i = 0; i < nd; i++) {
    lock_entry *le = hash_map_get(s->locks, to_del[i]);
    if (le) free(le);
    hash_map_del(s->locks, to_del[i]);
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

collab_session *collab_new(vol_server *srv) {
  collab_session *s = calloc(1, sizeof *s);
  if (!s) return NULL;
  s->srv = srv;
  s->locks = hash_map_new();
  if (!s->locks) { free(s); return NULL; }
  s->next_user_id = 1;
  return s;
}

void collab_free(collab_session *s) {
  if (!s) return;

  // Free all lock entries.
  hash_map_iter *it = hash_map_iter_new(s->locks);
  if (it) {
    hash_map_entry ent;
    while (hash_map_iter_next(it, &ent))
      free(ent.val);
    hash_map_iter_free(it);
  }
  hash_map_free(s->locks);

  // Free event payloads in ring buffer.
  for (int i = 0; i < s->ev_count; i++) {
    int idx = (s->ev_head + i) % EVENT_RING_CAP;
    free(s->events[idx].payload);
    s->events[idx].payload = NULL;
  }

  free(s);
}

// ---------------------------------------------------------------------------
// User management
// ---------------------------------------------------------------------------

int collab_add_user(collab_session *s, const char *username) {
  collab_user *u = find_slot(s);
  if (!u) return -1;

  memset(u, 0, sizeof *u);
  u->active  = true;
  u->user_id = s->next_user_id++;
  strncpy(u->username, username ? username : "user", sizeof u->username - 1);

  // Assign palette color based on slot position.
  int palette_idx = (u->user_id - 1) % PALETTE_LEN;
  u->color[0] = PALETTE[palette_idx][0];
  u->color[1] = PALETTE[palette_idx][1];
  u->color[2] = PALETTE[palette_idx][2];

  s->user_count++;
  return u->user_id;
}

void collab_remove_user(collab_session *s, int user_id) {
  collab_user *u = find_user(s, user_id);
  if (!u) return;
  u->active = false;
  s->user_count--;
  // Release all locks held by this user.
  release_locks_for(s, user_id, -1);
}

int collab_user_count(const collab_session *s) {
  return s->user_count;
}

const collab_user *collab_get_user(const collab_session *s, int user_id) {
  for (int i = 0; i < MAX_USERS; i++) {
    if (s->users[i].active && s->users[i].user_id == user_id)
      return &s->users[i];
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// Cursor updates
// ---------------------------------------------------------------------------

// Wire format for cursor broadcast (sent as MSG_SEG_UPDATE payload):
//   1  byte  : subtype = 0xC0 (cursor)
//   4  bytes : user_id (big-endian int32)
//   12 bytes : cursor_pos (3x float32 LE)
//   4  bytes : active_tool (int32 LE)
//   3  bytes : color RGB
#define CURSOR_PAYLOAD_SZ  24

void collab_update_cursor(collab_session *s, int user_id, vec3f pos, int tool) {
  collab_user *u = find_user(s, user_id);
  if (!u) return;

  u->cursor_pos  = pos;
  u->active_tool = tool;

  if (!s->srv) return;

  uint8_t buf[CURSOR_PAYLOAD_SZ];
  buf[0] = 0xC0;
  buf[1] = (uint8_t)((user_id >> 24) & 0xFF);
  buf[2] = (uint8_t)((user_id >> 16) & 0xFF);
  buf[3] = (uint8_t)((user_id >>  8) & 0xFF);
  buf[4] = (uint8_t)( user_id        & 0xFF);
  memcpy(buf + 5,  &pos.x, 4);
  memcpy(buf + 9,  &pos.y, 4);
  memcpy(buf + 13, &pos.z, 4);
  buf[17] = (uint8_t)((tool >> 24) & 0xFF);
  buf[18] = (uint8_t)((tool >> 16) & 0xFF);
  buf[19] = (uint8_t)((tool >>  8) & 0xFF);
  buf[20] = (uint8_t)( tool        & 0xFF);
  buf[21] = u->color[0];
  buf[22] = u->color[1];
  buf[23] = u->color[2];

  server_broadcast(s->srv, MSG_SEG_UPDATE, buf, CURSOR_PAYLOAD_SZ);
}

// ---------------------------------------------------------------------------
// Edit broadcast
// ---------------------------------------------------------------------------

void collab_broadcast_edit(collab_session *s, int user_id, const collab_event *evt) {
  (void)user_id;
  if (!s->srv || !evt) return;
  server_broadcast(s->srv, MSG_SEG_UPDATE, evt->payload, evt->payload_len);
}

// ---------------------------------------------------------------------------
// Region locking
// ---------------------------------------------------------------------------

bool collab_lock_region(collab_session *s, int user_id, int64_t surface_id,
                        float u, float v, float radius) {
  if (!find_user(s, user_id)) return false;

  // Enumerate all cells within the bounding box of the circle.
  int cu_min = (int)((u - radius) / LOCK_CELL_SIZE);
  int cu_max = (int)((u + radius) / LOCK_CELL_SIZE);
  int cv_min = (int)((v - radius) / LOCK_CELL_SIZE);
  int cv_max = (int)((v + radius) / LOCK_CELL_SIZE);

  // First pass: check for conflicts.
  char key[128];
  for (int cu = cu_min; cu <= cu_max; cu++) {
    for (int cv = cv_min; cv <= cv_max; cv++) {
      lock_key(key, sizeof key, surface_id, cu, cv);
      lock_entry *le = hash_map_get(s->locks, key);
      if (le && le->owner_id != user_id)
        return false;  // conflict
    }
  }

  // Second pass: claim all cells.
  for (int cu = cu_min; cu <= cu_max; cu++) {
    for (int cv = cv_min; cv <= cv_max; cv++) {
      lock_key(key, sizeof key, surface_id, cu, cv);
      lock_entry *le = hash_map_get(s->locks, key);
      if (!le) {
        le = malloc(sizeof *le);
        if (!le) return false;
        le->surface_id = surface_id;
        le->cell_u     = cu;
        le->cell_v     = cv;
        hash_map_put(s->locks, key, le);
      }
      le->owner_id = user_id;
    }
  }
  return true;
}

void collab_unlock_region(collab_session *s, int user_id, int64_t surface_id) {
  release_locks_for(s, user_id, surface_id);
}

// ---------------------------------------------------------------------------
// Event ring buffer
// ---------------------------------------------------------------------------

void collab_push_event(collab_session *s, const collab_event *evt) {
  if (!evt) return;

  // Copy the event; duplicate payload.
  collab_event copy = *evt;
  if (evt->payload && evt->payload_len > 0) {
    copy.payload = malloc(evt->payload_len);
    if (!copy.payload) return;
    memcpy(copy.payload, evt->payload, evt->payload_len);
  } else {
    copy.payload = NULL;
  }
  if (!copy.timestamp_ms)
    copy.timestamp_ms = now_ms();

  if (s->ev_count < EVENT_RING_CAP) {
    int slot = (s->ev_head + s->ev_count) % EVENT_RING_CAP;
    s->events[slot] = copy;
    s->ev_count++;
  } else {
    // Overwrite oldest slot.
    free(s->events[s->ev_head].payload);
    s->events[s->ev_head] = copy;
    s->ev_head = (s->ev_head + 1) % EVENT_RING_CAP;
  }
}

int collab_event_count(const collab_session *s) {
  return s->ev_count;
}

const collab_event *collab_get_event(const collab_session *s, int index) {
  if (index < 0 || index >= s->ev_count) return NULL;
  int slot = (s->ev_head + index) % EVENT_RING_CAP;
  return &s->events[slot];
}

// ---------------------------------------------------------------------------
// Cursor overlay rendering
// ---------------------------------------------------------------------------

void collab_render_cursors(const collab_session *s, int exclude_user,
                           overlay_list *out) {
  if (!out) return;
  for (int i = 0; i < MAX_USERS; i++) {
    const collab_user *u = &s->users[i];
    if (!u->active) continue;
    if (u->user_id == exclude_user) continue;

    // Project 3D cursor onto XY plane (slice viewer convention).
    overlay_add_point(out,
                      u->cursor_pos.x, u->cursor_pos.y,
                      u->color[0], u->color[1], u->color[2],
                      6.0f);
    // Label with username.
    overlay_add_text(out,
                     u->cursor_pos.x + 8.0f, u->cursor_pos.y,
                     u->username,
                     u->color[0], u->color[1], u->color[2]);
  }
}
