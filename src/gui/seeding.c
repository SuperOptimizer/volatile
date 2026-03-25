#include "gui/seeding.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

typedef struct {
  int   id;
  bool  active;
  vec3f point;
} seed_entry;

struct seed_manager {
  seed_entry *entries;
  int         cap;
  int         count;
  int         next_id;
};

// Marker colors cycle through a small palette (RGBA).
static const uint8_t SEED_COLORS[][4] = {
  {255, 100, 100, 220},
  {100, 255, 100, 220},
  {100, 150, 255, 220},
  {255, 220,  80, 220},
  {220,  80, 255, 220},
};
#define NCOLORS (int)(sizeof(SEED_COLORS) / sizeof(SEED_COLORS[0]))

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

seed_manager *seed_mgr_new(void) {
  seed_manager *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  m->next_id = 1;
  return m;
}

void seed_mgr_free(seed_manager *m) {
  if (!m) return;
  free(m->entries);
  free(m);
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

int seed_mgr_add(seed_manager *m, vec3f point) {
  if (!m) return -1;
  // Find a vacant slot first.
  for (int i = 0; i < m->cap; i++) {
    if (!m->entries[i].active) {
      m->entries[i].active = true;
      m->entries[i].id     = m->next_id++;
      m->entries[i].point  = point;
      m->count++;
      return m->entries[i].id;
    }
  }
  // Grow.
  int new_cap = m->cap ? m->cap * 2 : 8;
  seed_entry *tmp = realloc(m->entries, (size_t)new_cap * sizeof(*tmp));
  if (!tmp) return -1;
  // Zero-initialise new slots.
  memset(tmp + m->cap, 0, (size_t)(new_cap - m->cap) * sizeof(*tmp));
  m->entries = tmp;
  int slot = m->cap;
  m->cap = new_cap;
  m->entries[slot].active = true;
  m->entries[slot].id     = m->next_id++;
  m->entries[slot].point  = point;
  m->count++;
  return m->entries[slot].id;
}

bool seed_mgr_remove(seed_manager *m, int id) {
  if (!m || id <= 0) return false;
  for (int i = 0; i < m->cap; i++) {
    if (m->entries[i].active && m->entries[i].id == id) {
      m->entries[i].active = false;
      m->count--;
      return true;
    }
  }
  return false;
}

vec3f seed_mgr_get(const seed_manager *m, int id) {
  if (!m || id <= 0) return (vec3f){0};
  for (int i = 0; i < m->cap; i++) {
    if (m->entries[i].active && m->entries[i].id == id)
      return m->entries[i].point;
  }
  return (vec3f){0};
}

int seed_mgr_count(const seed_manager *m) {
  return m ? m->count : 0;
}

// ---------------------------------------------------------------------------
// Overlay
// ---------------------------------------------------------------------------

void seed_mgr_to_overlay(const seed_manager *m, overlay_list *out,
                          float marker_radius) {
  if (!m || !out) return;
  int color_idx = 0;
  for (int i = 0; i < m->cap; i++) {
    if (!m->entries[i].active) continue;
    const uint8_t *col = SEED_COLORS[color_idx % NCOLORS];
    color_idx++;
    float x = m->entries[i].point.x;
    float y = m->entries[i].point.y;
    overlay_add_circle(out, x, y, marker_radius, col[0], col[1], col[2]);
    // Small crosshair for precise position feedback.
    float d = marker_radius * 0.5f;
    overlay_add_line(out, x - d, y, x + d, y, col[0], col[1], col[2], 1.0f);
    overlay_add_line(out, x, y - d, x, y + d, col[0], col[1], col[2], 1.0f);
  }
}
