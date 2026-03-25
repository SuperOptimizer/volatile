#pragma once
#include "server/gitstore.h"
#include "server/srv.h"
#include "core/geom.h"
#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// sync_manager — bridges vol_server edit events to git_store operations.
// Each user edit stages the change and optionally auto-commits on a timer.
// ---------------------------------------------------------------------------

typedef struct sync_manager sync_manager;

// Lifecycle.  Both store and srv are borrowed — caller owns them.
sync_manager *sync_new(git_store *store, vol_server *srv);
void          sync_free(sync_manager *s);

// Called by server handlers when a client edits a surface or annotation.
// Serialises the change to the repo working tree and calls git_store_add.
void sync_on_segment_edit(sync_manager *s, int user_id, int64_t segment_id,
                          const quad_surface *surface);
void sync_on_annotation_edit(sync_manager *s, int user_id, int64_t annot_id,
                              const char *json);

// Enable periodic auto-commit every `interval_seconds` of pending changes.
// Spawns an internal background thread; safe to call multiple times
// (re-sets the interval).  Pass 0 to disable.
void sync_enable_autocommit(sync_manager *s, int interval_seconds);

// Pull latest from remote, load/reload all segments.
// Returns false if a conflict was detected (call sync_resolve_conflicts).
bool sync_pull_latest(sync_manager *s);

// Push local commits to remote.
bool sync_push(sync_manager *s);

// Conflict resolution.
typedef enum {
  CONFLICT_THEIRS = 0,
  CONFLICT_OURS   = 1,
  CONFLICT_MERGE  = 2,
} conflict_strategy;

bool sync_resolve_conflicts(sync_manager *s, conflict_strategy strategy);

// Trigger an immediate commit of any staged changes (also called by autocommit).
bool sync_commit_pending(sync_manager *s, int user_id, const char *message);
