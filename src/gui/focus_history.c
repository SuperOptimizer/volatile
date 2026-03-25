#include "gui/focus_history.h"
#include <stdlib.h>

#define DEFAULT_MAX_ENTRIES 100

typedef struct {
  vec3f position;
  int   pyramid_level;
} focus_entry;

// ---------------------------------------------------------------------------
// Circular buffer layout
//
// `buf` holds up to `cap` entries at indices [0, cap).
// `head` is the index of the oldest valid entry.
// `count` is how many entries are logically present (may be < cap when not
//   yet wrapped, and counts forward entries too).
// `cur` is the index (0-based from head) of the current entry the user is
//   viewing. Entries at [cur+1 .. count-1] are "forward" history.
//
// Physical index of logical offset i: (head + i) % cap
// ---------------------------------------------------------------------------

struct focus_history {
  focus_entry *buf;
  int          cap;   // allocated capacity
  int          head;  // physical index of logical[0]
  int          count; // total entries (cur + forward + 1 at most)
  int          cur;   // 0-based logical index of current position
};

static int phys(const focus_history *h, int logical) {
  return (h->head + logical) % h->cap;
}

focus_history *focus_history_new(int max_entries) {
  if (max_entries <= 0) max_entries = DEFAULT_MAX_ENTRIES;
  focus_history *h = malloc(sizeof(*h));
  if (!h) return NULL;
  h->buf = malloc((size_t)max_entries * sizeof(focus_entry));
  if (!h->buf) { free(h); return NULL; }
  h->cap   = max_entries;
  h->head  = 0;
  h->count = 0;
  h->cur   = -1;
  return h;
}

void focus_history_free(focus_history *h) {
  if (!h) return;
  free(h->buf);
  free(h);
}

void focus_history_push(focus_history *h, vec3f position, int pyramid_level) {
  if (!h) return;

  // Truncate forward history — everything after cur is discarded.
  if (h->cur >= 0) h->count = h->cur + 1;
  else             h->count = 0;

  if (h->count < h->cap) {
    // There is room; just append at logical index `count`.
    h->buf[phys(h, h->count)] = (focus_entry){ position, pyramid_level };
    h->count++;
    h->cur = h->count - 1;
  } else {
    // Buffer is full — overwrite the oldest entry by advancing head.
    // NOTE: cur stays at cap-1 (the last logical slot) because we slide
    // the window forward rather than growing it.
    h->head = (h->head + 1) % h->cap;
    h->buf[phys(h, h->cap - 1)] = (focus_entry){ position, pyramid_level };
    // count stays == cap; cur stays at cap-1.
    h->cur = h->cap - 1;
  }
}

bool focus_history_back(focus_history *h, vec3f *pos_out, int *level_out) {
  if (!h || h->cur <= 0) return false;
  h->cur--;
  focus_entry *e = &h->buf[phys(h, h->cur)];
  if (pos_out)   *pos_out   = e->position;
  if (level_out) *level_out = e->pyramid_level;
  return true;
}

bool focus_history_forward(focus_history *h, vec3f *pos_out, int *level_out) {
  if (!h || h->cur < 0 || h->cur >= h->count - 1) return false;
  h->cur++;
  focus_entry *e = &h->buf[phys(h, h->cur)];
  if (pos_out)   *pos_out   = e->position;
  if (level_out) *level_out = e->pyramid_level;
  return true;
}

bool focus_history_can_back(const focus_history *h) {
  return h && h->cur > 0;
}

bool focus_history_can_forward(const focus_history *h) {
  return h && h->cur >= 0 && h->cur < h->count - 1;
}

int focus_history_count(const focus_history *h) {
  return h ? h->count : 0;
}
