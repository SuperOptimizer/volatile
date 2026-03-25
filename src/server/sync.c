#define _POSIX_C_SOURCE 200809L
#include "server/sync.h"
#include "server/srv.h"
#include "core/log.h"

#include <inttypes.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// sync_manager
// ---------------------------------------------------------------------------

struct sync_manager {
  git_store  *store;
  vol_server *srv;
  int         autocommit_interval; // seconds; 0 = disabled
  pthread_t   ac_thread;
  int         ac_running;
  pthread_mutex_t mu;
};

sync_manager *sync_new(git_store *store, vol_server *srv) {
  if (!store) return NULL;
  sync_manager *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->store = store;
  s->srv   = srv;
  pthread_mutex_init(&s->mu, NULL);
  return s;
}

void sync_free(sync_manager *s) {
  if (!s) return;
  if (s->ac_running) {
    s->ac_running = 0;
    pthread_join(s->ac_thread, NULL);
  }
  pthread_mutex_destroy(&s->mu);
  free(s);
}

// ---------------------------------------------------------------------------
// Edit handlers — use gitstore's high-level write helpers
// ---------------------------------------------------------------------------

void sync_on_segment_edit(sync_manager *s, int user_id, int64_t segment_id,
                          const quad_surface *surface) {
  if (!s || !surface) return;

  char name[32];
  snprintf(name, sizeof(name), "%" PRId64, segment_id);

  pthread_mutex_lock(&s->mu);
  bool ok = git_store_write_surface(s->store, name, surface);
  pthread_mutex_unlock(&s->mu);

  if (!ok) {
    LOG_ERROR("sync_on_segment_edit: write failed (seg=%" PRId64 ")", segment_id);
    return;
  }
  LOG_INFO("sync: user %d staged segment %" PRId64, user_id, segment_id);

  if (s->srv) {
    uint8_t msg[12];
    memcpy(msg,     &user_id,    4);
    memcpy(msg + 4, &segment_id, 8);
    server_broadcast(s->srv, MSG_SEG_UPDATE, msg, sizeof(msg));
  }
}

void sync_on_annotation_edit(sync_manager *s, int user_id, int64_t annot_id,
                              const char *json) {
  if (!s || !json) return;

  char name[32];
  snprintf(name, sizeof(name), "%" PRId64, annot_id);

  pthread_mutex_lock(&s->mu);
  bool ok = git_store_write_annotation(s->store, name, json);
  pthread_mutex_unlock(&s->mu);

  if (!ok) {
    LOG_ERROR("sync_on_annotation_edit: write failed (annot=%" PRId64 ")", annot_id);
    return;
  }
  LOG_INFO("sync: user %d staged annotation %" PRId64, user_id, annot_id);
}

// ---------------------------------------------------------------------------
// Commit pending
// ---------------------------------------------------------------------------

bool sync_commit_pending(sync_manager *s, int user_id, const char *message) {
  if (!s) return false;
  pthread_mutex_lock(&s->mu);
  bool clean = git_store_is_clean(s->store);
  pthread_mutex_unlock(&s->mu);
  if (clean) return true;

  char author[64];
  snprintf(author, sizeof(author), "user:%d", user_id);
  char msg[256];
  snprintf(msg, sizeof(msg), "%s",
           (message && message[0]) ? message : "auto-commit");

  pthread_mutex_lock(&s->mu);
  bool ok = git_store_commit(s->store, author, msg);
  pthread_mutex_unlock(&s->mu);

  if (!ok) LOG_ERROR("sync_commit_pending: commit failed");
  else     LOG_INFO("sync: committed (%s)", msg);
  return ok;
}

// ---------------------------------------------------------------------------
// Auto-commit background thread
// ---------------------------------------------------------------------------

static void *ac_thread_fn(void *arg) {
  sync_manager *s = arg;
  while (s->ac_running) {
    sleep((unsigned)s->autocommit_interval);
    if (!s->ac_running) break;
    sync_commit_pending(s, 0, "autocommit");
  }
  return NULL;
}

void sync_enable_autocommit(sync_manager *s, int interval_seconds) {
  if (!s) return;
  if (s->ac_running) {
    s->ac_running = 0;
    pthread_join(s->ac_thread, NULL);
    s->ac_thread = 0;
  }
  s->autocommit_interval = interval_seconds;
  if (interval_seconds <= 0) return;
  s->ac_running = 1;
  if (pthread_create(&s->ac_thread, NULL, ac_thread_fn, s) != 0) {
    s->ac_running = 0;
    LOG_ERROR("sync_enable_autocommit: pthread_create failed");
  }
}

// ---------------------------------------------------------------------------
// Pull / push / conflict resolution
// ---------------------------------------------------------------------------

bool sync_pull_latest(sync_manager *s) {
  if (!s) return false;
  pthread_mutex_lock(&s->mu);
  bool ok = git_store_pull(s->store, "origin");
  pthread_mutex_unlock(&s->mu);
  if (!ok) LOG_WARN("sync_pull_latest: pull failed or conflicts detected");
  return ok;
}

bool sync_push(sync_manager *s) {
  if (!s) return false;
  pthread_mutex_lock(&s->mu);
  bool ok = git_store_push(s->store, "origin");
  pthread_mutex_unlock(&s->mu);
  if (!ok) LOG_ERROR("sync_push: push failed");
  return ok;
}

bool sync_resolve_conflicts(sync_manager *s, conflict_strategy strategy) {
  if (!s) return false;
  // Resolve by checking out the appropriate version of each conflicting file,
  // then committing. The gitstore merge helper handles 3-way text merge.
  bool ok = false;
  pthread_mutex_lock(&s->mu);
  switch (strategy) {
    case CONFLICT_THEIRS:
      ok = git_store_merge(s->store, "MERGE_HEAD");
      break;
    case CONFLICT_OURS:
      // Re-commit our working-tree state over the merge conflict markers.
      ok = git_store_commit(s->store, "resolver", "resolve: keep ours");
      break;
    case CONFLICT_MERGE:
      // 3-way merge already staged by git_store_pull; just commit it.
      ok = git_store_commit(s->store, "resolver", "resolve: 3-way merge");
      break;
  }
  pthread_mutex_unlock(&s->mu);
  if (!ok) LOG_ERROR("sync_resolve_conflicts: strategy %d failed", (int)strategy);
  return ok;
}
