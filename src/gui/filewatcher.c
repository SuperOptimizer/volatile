// clock_gettime, inotify, NAME_MAX require POSIX and Linux extensions
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE

#include "gui/filewatcher.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <limits.h>

#include <sys/inotify.h>
#include <poll.h>

#include "gui/inotify_util.h"

#define MAX_WATCHES 64
#define INOTIFY_MASK (IN_MODIFY | IN_CLOSE_WRITE | IN_CREATE)

// ---------------------------------------------------------------------------
// watch entry
// ---------------------------------------------------------------------------

typedef struct {
  int             wd;              // inotify watch descriptor; -1 = unused
  char           *path;
  file_changed_fn callback;
  void           *ctx;
  int             min_interval_ms; // 0 = no debounce
  struct timespec last_fired;      // time of last callback
} watch_entry;

struct file_watcher {
  int          ifd;                // inotify fd
  watch_entry  entries[MAX_WATCHES];
  int          count;
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static long timespec_ms(struct timespec ts) {
  return ts.tv_sec * 1000L + ts.tv_nsec / 1000000L;
}

static struct timespec now_monotonic(void) {
  struct timespec ts = {0};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts;
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

file_watcher *file_watcher_new(void) {
  int ifd = inotify_init1(IN_NONBLOCK);
  if (ifd < 0) {
    LOG_ERROR("file_watcher_new: inotify_init1: %s", strerror(errno));
    return NULL;
  }
  file_watcher *w = calloc(1, sizeof(file_watcher));
  if (!w) { close(ifd); return NULL; }
  w->ifd = ifd;
  for (int i = 0; i < MAX_WATCHES; i++) w->entries[i].wd = -1;
  return w;
}

void file_watcher_free(file_watcher *w) {
  if (!w) return;
  for (int i = 0; i < MAX_WATCHES; i++) {
    if (w->entries[i].wd >= 0) {
      inotify_rm_watch(w->ifd, w->entries[i].wd);
      free(w->entries[i].path);
    }
  }
  close(w->ifd);
  free(w);
}

// ---------------------------------------------------------------------------
// add / remove
// ---------------------------------------------------------------------------

static bool do_add(file_watcher *w, const char *path,
                   file_changed_fn callback, void *ctx, int min_interval_ms) {
  // find free slot
  int slot = -1;
  for (int i = 0; i < MAX_WATCHES; i++) {
    if (w->entries[i].wd < 0) { slot = i; break; }
  }
  if (slot < 0) { LOG_WARN("file_watcher_add: MAX_WATCHES reached"); return false; }

  int wd = inotify_add_watch(w->ifd, path, INOTIFY_MASK);
  if (wd < 0) {
    LOG_ERROR("file_watcher_add: inotify_add_watch(%s): %s", path, strerror(errno));
    return false;
  }

  watch_entry *e = &w->entries[slot];
  e->wd              = wd;
  e->path            = strdup(path);
  e->callback        = callback;
  e->ctx             = ctx;
  e->min_interval_ms = min_interval_ms;
  e->last_fired      = (struct timespec){0};
  if (slot >= w->count) w->count = slot + 1;
  return true;
}

bool file_watcher_add(file_watcher *w, const char *path,
                      file_changed_fn callback, void *ctx) {
  return do_add(w, path, callback, ctx, 0);
}

bool file_watcher_add_debounced(file_watcher *w, const char *path,
                                 file_changed_fn callback, void *ctx,
                                 int min_interval_ms) {
  return do_add(w, path, callback, ctx, min_interval_ms);
}

bool file_watcher_remove(file_watcher *w, const char *path) {
  for (int i = 0; i < w->count; i++) {
    watch_entry *e = &w->entries[i];
    if (e->wd >= 0 && strcmp(e->path, path) == 0) {
      inotify_rm_watch(w->ifd, e->wd);
      free(e->path);
      e->wd   = -1;
      e->path = NULL;
      return true;
    }
  }
  return false;
}

// ---------------------------------------------------------------------------
// poll
// ---------------------------------------------------------------------------

int file_watcher_poll(file_watcher *w) {
  if (!w || w->ifd < 0) return 0;

  // check if data is available without blocking
  struct pollfd pfd = { .fd = w->ifd, .events = POLLIN };
  if (poll(&pfd, 1, 0) <= 0) return 0;

  // read all available events
  char buf[16 * INOTIFY_BUF] __attribute__((aligned(__alignof__(struct inotify_event))));
  int fired = 0;

  ssize_t n;
  while ((n = read(w->ifd, buf, sizeof(buf))) > 0) {
    for (char *p = buf; p < buf + n; ) {
      struct inotify_event *ev = (struct inotify_event *)p;
      p += sizeof(struct inotify_event) + ev->len;

      // find the matching watch entry
      for (int i = 0; i < w->count; i++) {
        watch_entry *e = &w->entries[i];
        if (e->wd != ev->wd) continue;

        // debounce check
        if (e->min_interval_ms > 0) {
          struct timespec now = now_monotonic();
          long elapsed = timespec_ms(now) - timespec_ms(e->last_fired);
          if (elapsed < e->min_interval_ms) break;
          e->last_fired = now;
        }

        // build effective path: if directory watch and event has a name, append it
        char full_path[4096];
        if (ev->len > 0 && ev->name[0] != '\0') {
          snprintf(full_path, sizeof(full_path), "%s/%s", e->path, ev->name);
          e->callback(full_path, e->ctx);
        } else {
          e->callback(e->path, e->ctx);
        }
        fired++;
        break;
      }
    }
  }
  return fired;
}
