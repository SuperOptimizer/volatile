#include "core/hash.h"
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define INITIAL_CAP   16     // must be power of 2
#define LOAD_NUM      3      // grow when len > cap * LOAD_NUM / LOAD_DEN
#define LOAD_DEN      4      // i.e. 75% load factor

// ---------------------------------------------------------------------------
// FNV-1a hash for strings
// ---------------------------------------------------------------------------

static uint64_t fnv1a(const char *s) {
  uint64_t h = UINT64_C(14695981039346656037);
  while (*s) {
    h ^= (uint8_t)*s++;
    h *= UINT64_C(1099511628211);
  }
  return h;
}

// NOTE: Fibonacci hashing — multiplying by 2^64/phi reduces clustering for
// sequential integer keys compared to plain modulo.
static uint64_t fib_hash(uint64_t k) {
  return k * UINT64_C(11400714819323198485);
}

// ---------------------------------------------------------------------------
// String-keyed map internals
// ---------------------------------------------------------------------------

typedef struct {
  const char *key;   // NULL = empty slot
  void       *val;
  uint64_t    hash;
  uint32_t    psl;   // probe sequence length (distance from ideal slot)
} str_slot;

struct hash_map {
  str_slot *slots;
  size_t    len;
  size_t    cap;
};

struct hash_map_iter {
  hash_map *map;
  size_t    idx;
};

// ---------------------------------------------------------------------------
// String-keyed map implementation
// ---------------------------------------------------------------------------

hash_map *hash_map_new(void) {
  hash_map *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  m->cap   = INITIAL_CAP;
  m->slots = calloc(m->cap, sizeof(str_slot));
  if (!m->slots) { free(m); return NULL; }
  return m;
}

void hash_map_free(hash_map *m) {
  if (!m) return;
  for (size_t i = 0; i < m->cap; i++) {
    if (m->slots[i].key) free((void *)m->slots[i].key);
  }
  free(m->slots);
  free(m);
}

static bool str_map_grow(hash_map *m);

// NOTE: Robin Hood insertion — if the current slot's occupant has a shorter
// PSL than the incoming entry, we evict it and continue inserting the evicted
// entry. This bounds the maximum PSL and keeps lookup variance low.
static bool str_map_insert_slot(str_slot *slots, size_t cap,
                                const char *key, void *val, uint64_t h,
                                bool *is_new) {
  uint32_t psl  = 0;
  size_t   mask = cap - 1;
  size_t   idx  = (size_t)(h & mask);

  // temporary storage for Robin Hood swapping
  const char *cur_key = key;
  void       *cur_val = val;
  uint64_t    cur_h   = h;

  for (;;) {
    str_slot *s = &slots[idx];

    if (!s->key) {
      // empty — place here
      s->key  = cur_key;
      s->val  = cur_val;
      s->hash = cur_h;
      s->psl  = psl;
      if (is_new) *is_new = true;
      return true;
    }

    if (s->hash == cur_h && strcmp(s->key, cur_key) == 0) {
      // existing key — update value
      if (cur_key != key) {
        // we swapped in a different key; put the original back correctly
        free((void *)cur_key);
      }
      s->val = cur_val;
      if (is_new) *is_new = false;
      return true;
    }

    // Robin Hood: steal from the rich (low PSL) for the poor (high PSL)
    if (s->psl < psl) {
      const char *tmp_key = s->key; void *tmp_val = s->val;
      uint64_t    tmp_h   = s->hash; uint32_t tmp_psl = s->psl;
      s->key = cur_key; s->val = cur_val; s->hash = cur_h; s->psl = psl;
      cur_key = tmp_key; cur_val = tmp_val; cur_h = tmp_h; psl = tmp_psl;
    }

    psl++;
    idx = (idx + 1) & mask;
  }
}

static bool str_map_grow(hash_map *m) {
  size_t    new_cap   = m->cap * 2;
  str_slot *new_slots = calloc(new_cap, sizeof(str_slot));
  if (!new_slots) return false;

  for (size_t i = 0; i < m->cap; i++) {
    str_slot *s = &m->slots[i];
    if (!s->key) continue;
    str_map_insert_slot(new_slots, new_cap, s->key, s->val, s->hash, NULL);
  }

  free(m->slots);
  m->slots = new_slots;
  m->cap   = new_cap;
  return true;
}

void *hash_map_get(hash_map *m, const char *key) {
  uint64_t h    = fnv1a(key);
  size_t   mask = m->cap - 1;
  size_t   idx  = (size_t)(h & mask);
  uint32_t psl  = 0;

  for (;;) {
    str_slot *s = &m->slots[idx];
    if (!s->key) return NULL;
    // NOTE: Early exit when PSL exceeds the slot's PSL — the key cannot be
    // further right under Robin Hood (it would have displaced this slot).
    if (psl > s->psl) return NULL;
    if (s->hash == h && strcmp(s->key, key) == 0) return s->val;
    psl++;
    idx = (idx + 1) & mask;
  }
}

bool hash_map_put(hash_map *m, const char *key, void *val) {
  if (m->len >= m->cap * LOAD_NUM / LOAD_DEN) {
    if (!str_map_grow(m)) return false;
  }

  char *owned = strdup(key);
  if (!owned) return false;

  bool is_new = false;
  str_map_insert_slot(m->slots, m->cap, owned, val, fnv1a(key), &is_new);
  if (is_new) m->len++;
  else free(owned); // key already existed; our copy wasn't used
  return is_new;
}

// NOTE: Robin Hood backward-shift deletion — rather than tombstones, we shift
// subsequent entries back to fill the gap, preserving probe-length invariants.
bool hash_map_del(hash_map *m, const char *key) {
  uint64_t h    = fnv1a(key);
  size_t   mask = m->cap - 1;
  size_t   idx  = (size_t)(h & mask);
  uint32_t psl  = 0;

  for (;;) {
    str_slot *s = &m->slots[idx];
    if (!s->key) return false;
    if (psl > s->psl) return false;
    if (s->hash == h && strcmp(s->key, key) == 0) {
      free((void *)s->key);
      *s = (str_slot){0};
      m->len--;

      // Backward shift
      size_t prev = idx;
      size_t next = (idx + 1) & mask;
      while (m->slots[next].key && m->slots[next].psl > 0) {
        m->slots[prev] = m->slots[next];
        m->slots[prev].psl--;
        m->slots[next] = (str_slot){0};
        prev = next;
        next = (next + 1) & mask;
      }
      return true;
    }
    psl++;
    idx = (idx + 1) & mask;
  }
}

size_t hash_map_len(hash_map *m) { return m->len; }

hash_map_iter *hash_map_iter_new(hash_map *m) {
  hash_map_iter *it = malloc(sizeof(*it));
  if (!it) return NULL;
  it->map = m;
  it->idx = 0;
  return it;
}

bool hash_map_iter_next(hash_map_iter *it, hash_map_entry *out) {
  while (it->idx < it->map->cap) {
    str_slot *s = &it->map->slots[it->idx++];
    if (s->key) { out->key = s->key; out->val = s->val; return true; }
  }
  return false;
}

void hash_map_iter_free(hash_map_iter *it) { free(it); }

// ---------------------------------------------------------------------------
// Integer-keyed map internals
// ---------------------------------------------------------------------------

#define INT_EMPTY UINT64_MAX  // sentinel: key==UINT64_MAX is reserved

typedef struct {
  uint64_t key;
  void    *val;
  uint64_t hash;
  uint32_t psl;
  bool     used;
} int_slot;

struct hash_map_int {
  int_slot *slots;
  size_t    len;
  size_t    cap;
};

// ---------------------------------------------------------------------------
// Integer-keyed map implementation
// ---------------------------------------------------------------------------

hash_map_int *hash_map_int_new(void) {
  hash_map_int *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  m->cap   = INITIAL_CAP;
  m->slots = calloc(m->cap, sizeof(int_slot));
  if (!m->slots) { free(m); return NULL; }
  return m;
}

void hash_map_int_free(hash_map_int *m) {
  if (!m) return;
  free(m->slots);
  free(m);
}

static bool int_map_insert_slot(int_slot *slots, size_t cap,
                                uint64_t key, void *val, uint64_t h,
                                bool *is_new) {
  uint32_t psl  = 0;
  size_t   mask = cap - 1;
  size_t   idx  = (size_t)(h & mask);

  uint64_t cur_key = key;
  void    *cur_val = val;
  uint64_t cur_h   = h;

  for (;;) {
    int_slot *s = &slots[idx];

    if (!s->used) {
      s->key  = cur_key;
      s->val  = cur_val;
      s->hash = cur_h;
      s->psl  = psl;
      s->used = true;
      if (is_new) *is_new = true;
      return true;
    }

    if (s->key == cur_key) {
      s->val = cur_val;
      if (is_new) *is_new = false;
      return true;
    }

    if (s->psl < psl) {
      uint64_t tmp_key = s->key; void *tmp_val = s->val;
      uint64_t tmp_h   = s->hash; uint32_t tmp_psl = s->psl;
      s->key = cur_key; s->val = cur_val; s->hash = cur_h; s->psl = psl;
      cur_key = tmp_key; cur_val = tmp_val; cur_h = tmp_h; psl = tmp_psl;
    }

    psl++;
    idx = (idx + 1) & mask;
  }
}

static bool int_map_grow(hash_map_int *m) {
  size_t    new_cap   = m->cap * 2;
  int_slot *new_slots = calloc(new_cap, sizeof(int_slot));
  if (!new_slots) return false;

  for (size_t i = 0; i < m->cap; i++) {
    int_slot *s = &m->slots[i];
    if (!s->used) continue;
    int_map_insert_slot(new_slots, new_cap, s->key, s->val, s->hash, NULL);
  }

  free(m->slots);
  m->slots = new_slots;
  m->cap   = new_cap;
  return true;
}

void *hash_map_int_get(hash_map_int *m, uint64_t key) {
  uint64_t h    = fib_hash(key);
  size_t   mask = m->cap - 1;
  size_t   idx  = (size_t)(h & mask);
  uint32_t psl  = 0;

  for (;;) {
    int_slot *s = &m->slots[idx];
    if (!s->used) return NULL;
    if (psl > s->psl) return NULL;
    if (s->key == key) return s->val;
    psl++;
    idx = (idx + 1) & mask;
  }
}

bool hash_map_int_put(hash_map_int *m, uint64_t key, void *val) {
  if (m->len >= m->cap * LOAD_NUM / LOAD_DEN) {
    if (!int_map_grow(m)) return false;
  }

  bool is_new = false;
  int_map_insert_slot(m->slots, m->cap, key, val, fib_hash(key), &is_new);
  if (is_new) m->len++;
  return is_new;
}

bool hash_map_int_del(hash_map_int *m, uint64_t key) {
  uint64_t h    = fib_hash(key);
  size_t   mask = m->cap - 1;
  size_t   idx  = (size_t)(h & mask);
  uint32_t psl  = 0;

  for (;;) {
    int_slot *s = &m->slots[idx];
    if (!s->used) return false;
    if (psl > s->psl) return false;
    if (s->key == key) {
      *s = (int_slot){0};
      m->len--;

      size_t prev = idx;
      size_t next = (idx + 1) & mask;
      while (m->slots[next].used && m->slots[next].psl > 0) {
        m->slots[prev] = m->slots[next];
        m->slots[prev].psl--;
        m->slots[next] = (int_slot){0};
        prev = next;
        next = (next + 1) & mask;
      }
      return true;
    }
    psl++;
    idx = (idx + 1) & mask;
  }
}

size_t hash_map_int_len(hash_map_int *m) { return m->len; }
