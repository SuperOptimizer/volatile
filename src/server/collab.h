#pragma once
#include "server/srv.h"
#include "core/geom.h"
#include "render/overlay.h"
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Edit types
// ---------------------------------------------------------------------------

typedef enum {
  EDIT_SEG_PAINT    = 0,   // painted/modified segmentation
  EDIT_ANNOT_ADD    = 1,   // added annotation
  EDIT_ANNOT_DEL    = 2,   // deleted annotation
  EDIT_CURSOR_MOVE  = 3,   // cursor position update (not undoable)
  EDIT_SEG_ERASE    = 4,   // erased segmentation
} collab_edit_type;

// ---------------------------------------------------------------------------
// User and event types
// ---------------------------------------------------------------------------

typedef struct {
  int     user_id;
  char    username[64];
  uint8_t color[3];         // unique per-user RGB
  vec3f   cursor_pos;       // current 3D cursor position
  int     active_tool;      // tool enum (client-defined)
  bool    active;
} collab_user;

typedef struct {
  int      user_id;
  int64_t  timestamp_ms;
  int      edit_type;       // collab_edit_type
  uint8_t *payload;
  uint32_t payload_len;
} collab_event;

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

typedef struct collab_session collab_session;

collab_session *collab_new(vol_server *srv);
void            collab_free(collab_session *s);

// User management
int               collab_add_user(collab_session *s, const char *username);
void              collab_remove_user(collab_session *s, int user_id);
int               collab_user_count(const collab_session *s);
const collab_user *collab_get_user(const collab_session *s, int user_id);

// Cursor broadcast (~10 Hz per user)
void collab_update_cursor(collab_session *s, int user_id, vec3f pos, int tool);

// Broadcast an edit event to all other users
void collab_broadcast_edit(collab_session *s, int user_id, const collab_event *evt);

// Region locking: lock a (surface_id, grid_cell) -> user_id
// Returns false if already locked by another user.
bool collab_lock_region(collab_session *s, int user_id, int64_t surface_id,
                        float u, float v, float radius);
void collab_unlock_region(collab_session *s, int user_id, int64_t surface_id);

// Event log (ring buffer)
void              collab_push_event(collab_session *s, const collab_event *evt);
int               collab_event_count(const collab_session *s);
const collab_event *collab_get_event(const collab_session *s, int index);

// Render other users' cursors as overlay points (exclude_user skipped)
void collab_render_cursors(const collab_session *s, int exclude_user,
                           overlay_list *out);
