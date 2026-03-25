#define _POSIX_C_SOURCE 200809L

#include "core/cache.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>

// ---------------------------------------------------------------------------
// TODO: upgrade hot/warm eviction from CLOCK to CLOCK-Pro.
//
// CLOCK-Pro distinguishes three page types:
//   hot  — frequently accessed; needs two consecutive non-refs to demote
//   cold — recently loaded but not re-accessed; evicted on first non-ref
//   test — metadata-only ghost entry for recently evicted pages; if re-accessed
//          before the test entry expires, the page is promoted directly to hot
// The clock hand sweeps a circular list. Hot pages have a higher residence
// requirement than cold, which adapts the cache between recency and frequency
// automatically without LRU's O(n) overhead.
//
// Current implementation: simple CLOCK with reference bits (one circular
// buffer per tier). A referenced entry has its ref bit cleared on first pass;
// an unreferenced entry is evicted. Hot evictees are demoted to the warm
// (compressed) tier. This is correct, adaptive, and sufficient as a starting
// point.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define CACHE_DEFAULT_HOT_BYTES   ((size_t)4 * 1024 * 1024 * 1024)   // 4 GB
#define CACHE_DEFAULT_WARM_BYTES  ((size_t)1 * 1024 * 1024 * 1024)   // 1 GB
#define CACHE_DEFAULT_COLD_BYTES  ((size_t)50 * 1024 * 1024 * 1024)  // 50 GB
#define CACHE_DEFAULT_IO_THREADS  4

// Initial capacity of each hash table (must be power of two)
#define HOT_INIT_CAP   256
#define WARM_INIT_CAP  128

// ---------------------------------------------------------------------------
// Key hashing
// ---------------------------------------------------------------------------

static uint64_t key_hash(chunk_key k) {
  // Combine fields with FNV-1a-inspired mixing
  uint64_t h = 14695981039346656037ULL;
  h ^= (uint64_t)(uint32_t)k.level; h *= 1099511628211ULL;
  h ^= (uint64_t)k.iz;              h *= 1099511628211ULL;
  h ^= (uint64_t)k.iy;              h *= 1099511628211ULL;
  h ^= (uint64_t)k.ix;              h *= 1099511628211ULL;
  return h;
}

static bool key_eq(chunk_key a, chunk_key b) {
  return a.level == b.level && a.iz == b.iz && a.iy == b.iy && a.ix == b.ix;
}

// ---------------------------------------------------------------------------
// Hot tier: open-addressing hash table + CLOCK eviction ring
// ---------------------------------------------------------------------------

typedef struct {
  chunk_key  key;
  chunk_data data;     // owned copy
  bool       occupied;
  bool       pinned;
  bool       ref;      // CLOCK reference bit
} hot_entry;

typedef struct {
  hot_entry *entries;
  size_t     cap;       // power-of-two capacity
  size_t     count;
  size_t     bytes;     // total decompressed bytes currently held
  size_t     hand;      // CLOCK hand index
} hot_tier;

static void hot_init(hot_tier *t, size_t cap) {
  t->entries = calloc(cap, sizeof(hot_entry));
  REQUIRE(t->entries, "hot_init: calloc failed");
  t->cap   = cap;
  t->count = 0;
  t->bytes = 0;
  t->hand  = 0;
}

static void hot_destroy(hot_tier *t) {
  for (size_t i = 0; i < t->cap; i++) {
    if (t->entries[i].occupied && t->entries[i].data.data)
      free(t->entries[i].data.data);
  }
  free(t->entries);
}

// Find slot index for key; returns SIZE_MAX if not found.
static size_t hot_find(const hot_tier *t, chunk_key key) {
  size_t mask = t->cap - 1;
  size_t idx  = (size_t)key_hash(key) & mask;
  for (size_t i = 0; i < t->cap; i++) {
    size_t s = (idx + i) & mask;
    if (!t->entries[s].occupied) return SIZE_MAX;
    if (key_eq(t->entries[s].key, key)) return s;
  }
  return SIZE_MAX;
}

// Evict one unpinned entry via CLOCK. On success sets *out_key and *out_data
// (caller adopts the data) and returns true. Returns false if all pinned.
static bool hot_clock_evict_one(hot_tier *t, chunk_key *out_key, chunk_data *out_data) {
  size_t start = t->hand;
  for (size_t i = 0; i < t->cap * 2; i++) {
    size_t s = (start + i) % t->cap;
    hot_entry *e = &t->entries[s];
    if (!e->occupied || e->pinned) continue;
    if (e->ref) {
      e->ref = false;  // give a second chance
      continue;
    }
    *out_key  = e->key;
    *out_data = e->data;   // transfer ownership
    t->bytes -= e->data.size;
    t->count--;
    *e = (hot_entry){0};
    t->hand = (s + 1) % t->cap;
    return true;
  }
  return false;
}

// Grow the table (rehash into double capacity).
static void hot_grow(hot_tier *t) {
  size_t new_cap = t->cap * 2;
  hot_entry *new_entries = calloc(new_cap, sizeof(hot_entry));
  REQUIRE(new_entries, "hot_grow: calloc failed");
  size_t mask = new_cap - 1;
  for (size_t i = 0; i < t->cap; i++) {
    hot_entry *src = &t->entries[i];
    if (!src->occupied) continue;
    size_t idx = (size_t)key_hash(src->key) & mask;
    for (size_t j = 0; j < new_cap; j++) {
      size_t s = (idx + j) & mask;
      if (!new_entries[s].occupied) { new_entries[s] = *src; break; }
    }
  }
  free(t->entries);
  t->entries = new_entries;
  t->cap     = new_cap;
  t->hand    = t->hand % new_cap;
}

// ---------------------------------------------------------------------------
// Warm tier: compressed chunks in RAM
// ---------------------------------------------------------------------------

typedef struct {
  chunk_key  key;
  uint8_t   *compressed;
  size_t     compressed_size;
  chunk_data meta;      // shape/elem_size only; meta.data == NULL
  bool       occupied;
  bool       ref;
} warm_entry;

typedef struct {
  warm_entry *entries;
  size_t      cap;
  size_t      count;
  size_t      bytes;    // total compressed bytes
  size_t      hand;
} warm_tier;

static void warm_init(warm_tier *t, size_t cap) {
  t->entries = calloc(cap, sizeof(warm_entry));
  REQUIRE(t->entries, "warm_init: calloc failed");
  t->cap = cap; t->count = 0; t->bytes = 0; t->hand = 0;
}

static void warm_destroy(warm_tier *t) {
  for (size_t i = 0; i < t->cap; i++) {
    if (t->entries[i].occupied) free(t->entries[i].compressed);
  }
  free(t->entries);
}

static size_t warm_find(const warm_tier *t, chunk_key key) {
  size_t mask = t->cap - 1;
  size_t idx  = (size_t)key_hash(key) & mask;
  for (size_t i = 0; i < t->cap; i++) {
    size_t s = (idx + i) & mask;
    if (!t->entries[s].occupied) return SIZE_MAX;
    if (key_eq(t->entries[s].key, key)) return s;
  }
  return SIZE_MAX;
}

static void warm_clock_evict_one(warm_tier *t) {
  size_t start = t->hand;
  for (size_t i = 0; i < t->cap * 2; i++) {
    size_t s = (start + i) % t->cap;
    warm_entry *e = &t->entries[s];
    if (!e->occupied) continue;
    if (e->ref) { e->ref = false; continue; }
    t->bytes -= e->compressed_size;
    t->count--;
    free(e->compressed);
    *e = (warm_entry){0};
    t->hand = (s + 1) % t->cap;
    return;
  }
}

static void warm_grow(warm_tier *t) {
  size_t new_cap = t->cap * 2;
  warm_entry *ne = calloc(new_cap, sizeof(warm_entry));
  REQUIRE(ne, "warm_grow: calloc failed");
  size_t mask = new_cap - 1;
  for (size_t i = 0; i < t->cap; i++) {
    warm_entry *src = &t->entries[i];
    if (!src->occupied) continue;
    size_t idx = (size_t)key_hash(src->key) & mask;
    for (size_t j = 0; j < new_cap; j++) {
      size_t s = (idx + j) & mask;
      if (!ne[s].occupied) { ne[s] = *src; break; }
    }
  }
  free(t->entries);
  t->entries = ne;
  t->cap = new_cap;
  t->hand = t->hand % new_cap;
}

// Insert compressed bytes into warm tier (compressed is adopted).
static void warm_insert(warm_tier *t, chunk_key key, uint8_t *compressed,
                        size_t compressed_size, chunk_data meta, size_t max_bytes) {
  if (t->count * 4 >= t->cap * 3) warm_grow(t);
  while (t->bytes + compressed_size > max_bytes)
    warm_clock_evict_one(t);

  size_t mask = t->cap - 1;
  size_t idx  = (size_t)key_hash(key) & mask;
  for (size_t i = 0; i < t->cap; i++) {
    size_t s = (idx + i) & mask;
    if (!t->entries[s].occupied) {
      t->entries[s] = (warm_entry){
        .key = key, .compressed = compressed, .compressed_size = compressed_size,
        .meta = meta, .occupied = true, .ref = true
      };
      t->count++;
      t->bytes += compressed_size;
      return;
    }
    if (key_eq(t->entries[s].key, key)) {
      t->bytes -= t->entries[s].compressed_size;
      free(t->entries[s].compressed);
      t->entries[s].compressed      = compressed;
      t->entries[s].compressed_size = compressed_size;
      t->entries[s].meta            = meta;
      t->entries[s].ref             = true;
      t->bytes += compressed_size;
      return;
    }
  }
  free(compressed);
}

// ---------------------------------------------------------------------------
// Compression helpers
// Simple identity codec — TODO: integrate blosc2 for real compression.
// ---------------------------------------------------------------------------

static uint8_t *compress_data(const chunk_data *d, size_t *out_size) {
  *out_size = d->size;
  if (!d->size) return NULL;
  uint8_t *buf = malloc(d->size);
  if (buf) memcpy(buf, d->data, d->size);
  return buf;
}

static chunk_data decompress_data(const warm_entry *e) {
  chunk_data d = e->meta;
  d.data = malloc(e->compressed_size);
  REQUIRE(d.data, "decompress_data: malloc failed");
  memcpy(d.data, e->compressed, e->compressed_size);
  d.size = e->compressed_size;
  return d;
}

// ---------------------------------------------------------------------------
// Pinned map: simple linked list (coarsest level has few chunks)
// ---------------------------------------------------------------------------

typedef struct pin_node {
  chunk_key   key;
  chunk_data *data;   // not owned — caller retains ownership
  struct pin_node *next;
} pin_node;

// ---------------------------------------------------------------------------
// Prefetch queue: ring buffer of keys
// ---------------------------------------------------------------------------

#define PREFETCH_QUEUE_CAP 256

typedef struct {
  chunk_key  keys[PREFETCH_QUEUE_CAP];
  size_t     head, tail, count;
  pthread_mutex_t mu;
  pthread_cond_t  cv;
  bool       shutdown;
} prefetch_queue;

// ---------------------------------------------------------------------------
// Main cache struct
// ---------------------------------------------------------------------------

struct chunk_cache {
  cache_config    cfg;

  pthread_mutex_t mu;       // protects hot, warm, pinned, stats

  hot_tier        hot;
  warm_tier       warm;
  // cold (disk) and ice (remote) tiers — TODO
  pin_node       *pinned;

  size_t          hits;
  size_t          misses;

  prefetch_queue  pq;
  pthread_t      *io_threads;
  int             n_io_threads;
};

// ---------------------------------------------------------------------------
// chunk_data helpers
// ---------------------------------------------------------------------------

void chunk_data_free(chunk_data *d) {
  if (!d) return;
  free(d->data);
  free(d);
}

static chunk_data *chunk_data_clone(const chunk_data *src) {
  chunk_data *dst = malloc(sizeof(*dst));
  REQUIRE(dst, "chunk_data_clone: malloc failed");
  *dst = *src;
  if (src->data && src->size > 0) {
    dst->data = malloc(src->size);
    REQUIRE(dst->data, "chunk_data_clone: malloc data failed");
    memcpy(dst->data, src->data, src->size);
  } else {
    dst->data = NULL;
  }
  return dst;
}

// ---------------------------------------------------------------------------
// hot_insert: insert into hot tier, demoting evictees to warm
// Called with c->mu held.
// ---------------------------------------------------------------------------

static void hot_insert_with_demotion(hot_tier *hot, warm_tier *warm,
                                     chunk_key key, chunk_data data,
                                     size_t hot_max, size_t warm_max) {
  if (hot->count * 4 >= hot->cap * 3) hot_grow(hot);

  while (hot->bytes + data.size > hot_max) {
    chunk_key  evict_key;
    chunk_data evict_data;
    if (!hot_clock_evict_one(hot, &evict_key, &evict_data)) break;

    // Demote to warm tier via compression
    size_t csz;
    uint8_t *compressed = compress_data(&evict_data, &csz);
    chunk_data meta = evict_data;
    meta.data = NULL;
    free(evict_data.data);

    if (compressed)
      warm_insert(warm, evict_key, compressed, csz, meta, warm_max);
  }

  size_t mask = hot->cap - 1;
  size_t idx  = (size_t)key_hash(key) & mask;
  for (size_t i = 0; i < hot->cap; i++) {
    size_t s = (idx + i) & mask;
    if (!hot->entries[s].occupied) {
      hot->entries[s] = (hot_entry){
        .key = key, .data = data, .occupied = true, .ref = true
      };
      hot->count++;
      hot->bytes += data.size;
      return;
    }
    if (key_eq(hot->entries[s].key, key)) {
      hot->bytes -= hot->entries[s].data.size;
      free(hot->entries[s].data.data);
      hot->entries[s].data = data;
      hot->entries[s].ref  = true;
      hot->bytes += data.size;
      return;
    }
  }
  free(data.data);  // table full after eviction — shouldn't happen
}

// ---------------------------------------------------------------------------
// Prefetch worker
// ---------------------------------------------------------------------------

static void *prefetch_worker(void *arg) {
  chunk_cache *c = arg;
  prefetch_queue *pq = &c->pq;
  for (;;) {
    pthread_mutex_lock(&pq->mu);
    while (!pq->shutdown && pq->count == 0)
      pthread_cond_wait(&pq->cv, &pq->mu);
    if (pq->shutdown && pq->count == 0) {
      pthread_mutex_unlock(&pq->mu);
      break;
    }
    chunk_key key = pq->keys[pq->head];
    pq->head = (pq->head + 1) % PREFETCH_QUEUE_CAP;
    pq->count--;
    pthread_mutex_unlock(&pq->mu);

    // TODO: implement cold/ice tier fetch. For now this is a no-op placeholder.
    (void)key;
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

chunk_cache *cache_new(cache_config cfg) {
  if (!cfg.hot_max_bytes)  cfg.hot_max_bytes  = CACHE_DEFAULT_HOT_BYTES;
  if (!cfg.warm_max_bytes) cfg.warm_max_bytes  = CACHE_DEFAULT_WARM_BYTES;
  if (!cfg.cold_max_bytes) cfg.cold_max_bytes  = CACHE_DEFAULT_COLD_BYTES;
  if (!cfg.io_threads)     cfg.io_threads      = CACHE_DEFAULT_IO_THREADS;

  chunk_cache *c = calloc(1, sizeof(*c));
  REQUIRE(c, "cache_new: calloc failed");
  c->cfg = cfg;

  pthread_mutex_init(&c->mu, NULL);
  hot_init(&c->hot, HOT_INIT_CAP);
  warm_init(&c->warm, WARM_INIT_CAP);

  prefetch_queue *pq = &c->pq;
  pthread_mutex_init(&pq->mu, NULL);
  pthread_cond_init(&pq->cv, NULL);
  pq->head = pq->tail = pq->count = 0;
  pq->shutdown = false;

  int n = cfg.io_threads;
  c->n_io_threads = n;
  c->io_threads = calloc((size_t)n, sizeof(pthread_t));
  REQUIRE(c->io_threads, "cache_new: calloc io_threads failed");
  for (int i = 0; i < n; i++) {
    int rc = pthread_create(&c->io_threads[i], NULL, prefetch_worker, c);
    REQUIRE(rc == 0, "cache_new: pthread_create failed: %d", rc);
  }

  return c;
}

void cache_free(chunk_cache *c) {
  if (!c) return;

  pthread_mutex_lock(&c->pq.mu);
  c->pq.shutdown = true;
  pthread_cond_broadcast(&c->pq.cv);
  pthread_mutex_unlock(&c->pq.mu);
  for (int i = 0; i < c->n_io_threads; i++)
    pthread_join(c->io_threads[i], NULL);
  free(c->io_threads);
  pthread_cond_destroy(&c->pq.cv);
  pthread_mutex_destroy(&c->pq.mu);

  pin_node *pn = c->pinned;
  while (pn) {
    pin_node *next = pn->next;
    free(pn);
    pn = next;
  }

  hot_destroy(&c->hot);
  warm_destroy(&c->warm);
  pthread_mutex_destroy(&c->mu);
  free(c);
}

// ---------------------------------------------------------------------------
// Pinned lookup (must be called under c->mu)
// ---------------------------------------------------------------------------

static chunk_data *pinned_find(const chunk_cache *c, chunk_key key) {
  for (const pin_node *n = c->pinned; n; n = n->next) {
    if (key_eq(n->key, key)) return n->data;
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// cache_pin
// ---------------------------------------------------------------------------

void cache_pin(chunk_cache *c, chunk_key key, chunk_data *data) {
  REQUIRE(c && data, "cache_pin: null argument");
  pin_node *n = malloc(sizeof(*n));
  REQUIRE(n, "cache_pin: malloc failed");
  n->key  = key;
  n->data = data;

  pthread_mutex_lock(&c->mu);
  for (pin_node *existing = c->pinned; existing; existing = existing->next) {
    if (key_eq(existing->key, key)) {
      pthread_mutex_unlock(&c->mu);
      free(n);
      return;
    }
  }
  n->next   = c->pinned;
  c->pinned = n;

  size_t slot = hot_find(&c->hot, key);
  if (slot != SIZE_MAX) c->hot.entries[slot].pinned = true;
  pthread_mutex_unlock(&c->mu);
}

// ---------------------------------------------------------------------------
// cache_put
// ---------------------------------------------------------------------------

void cache_put(chunk_cache *c, chunk_key key, chunk_data *data) {
  REQUIRE(c && data, "cache_put: null argument");

  chunk_data owned = {
    .size      = data->size,
    .elem_size = data->elem_size,
  };
  memcpy(owned.shape, data->shape, sizeof(data->shape));
  owned.data = malloc(data->size);
  REQUIRE(owned.data, "cache_put: malloc failed");
  memcpy(owned.data, data->data, data->size);

  pthread_mutex_lock(&c->mu);
  bool is_pinned = (pinned_find(c, key) != NULL);
  hot_insert_with_demotion(&c->hot, &c->warm, key, owned,
                           c->cfg.hot_max_bytes, c->cfg.warm_max_bytes);
  if (is_pinned) {
    size_t slot = hot_find(&c->hot, key);
    if (slot != SIZE_MAX) c->hot.entries[slot].pinned = true;
  }
  pthread_mutex_unlock(&c->mu);
}

// ---------------------------------------------------------------------------
// cache_get
// ---------------------------------------------------------------------------

chunk_data *cache_get(chunk_cache *c, chunk_key key) {
  REQUIRE(c, "cache_get: null cache");

  pthread_mutex_lock(&c->mu);

  // 1. Check pinned
  chunk_data *pinned = pinned_find(c, key);
  if (pinned) {
    c->hits++;
    chunk_data *result = chunk_data_clone(pinned);
    pthread_mutex_unlock(&c->mu);
    return result;
  }

  // 2. Check hot tier
  size_t slot = hot_find(&c->hot, key);
  if (slot != SIZE_MAX) {
    c->hot.entries[slot].ref = true;
    c->hits++;
    chunk_data *result = chunk_data_clone(&c->hot.entries[slot].data);
    pthread_mutex_unlock(&c->mu);
    return result;
  }

  // 3. Check warm tier — decompress and promote to hot
  size_t wslot = warm_find(&c->warm, key);
  if (wslot != SIZE_MAX) {
    c->warm.entries[wslot].ref = true;
    c->hits++;
    chunk_data decompressed = decompress_data(&c->warm.entries[wslot]);

    // Clone for caller before promotion may evict something
    chunk_data *ret = chunk_data_clone(&decompressed);

    // Promote to hot (adopt decompressed)
    hot_insert_with_demotion(&c->hot, &c->warm, key, decompressed,
                             c->cfg.hot_max_bytes, c->cfg.warm_max_bytes);

    pthread_mutex_unlock(&c->mu);
    return ret;
  }

  c->misses++;
  pthread_mutex_unlock(&c->mu);
  return NULL;
}

// ---------------------------------------------------------------------------
// cache_get_best
// ---------------------------------------------------------------------------

cache_best_result cache_get_best(chunk_cache *c, chunk_key key, int coarsest_level) {
  REQUIRE(c, "cache_get_best: null cache");

  for (int lvl = key.level; lvl <= coarsest_level; lvl++) {
    int steps = lvl - key.level;
    chunk_key k = {
      .level = lvl,
      .iz = key.iz >> steps,
      .iy = key.iy >> steps,
      .ix = key.ix >> steps,
    };
    chunk_data *d = cache_get(c, k);
    if (d) return (cache_best_result){ .data = d, .actual_level = lvl };
  }

  return (cache_best_result){ .data = NULL, .actual_level = -1 };
}

// ---------------------------------------------------------------------------
// cache_get_blocking
// ---------------------------------------------------------------------------

chunk_data *cache_get_blocking(chunk_cache *c, chunk_key key, int timeout_ms) {
  REQUIRE(c, "cache_get_blocking: null cache");

  chunk_data *d = cache_get(c, key);
  if (d) return d;

  // TODO: integrate with cold/ice tier fetch via io_threads.
  // Poll with 1ms sleep until timeout.
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);

  while (true) {
    d = cache_get(c, key);
    if (d) return d;

    if (timeout_ms >= 0) {
      clock_gettime(CLOCK_MONOTONIC, &now);
      long elapsed_ms = (now.tv_sec - start.tv_sec) * 1000L
                      + (now.tv_nsec - start.tv_nsec) / 1000000L;
      if (elapsed_ms >= timeout_ms) break;
    }

    struct timespec req = { .tv_sec = 0, .tv_nsec = 1000000L };
    nanosleep(&req, NULL);
  }

  return NULL;
}

// ---------------------------------------------------------------------------
// cache_prefetch
// ---------------------------------------------------------------------------

void cache_prefetch(chunk_cache *c, chunk_key key) {
  REQUIRE(c, "cache_prefetch: null cache");

  prefetch_queue *pq = &c->pq;
  pthread_mutex_lock(&pq->mu);
  if (!pq->shutdown && pq->count < PREFETCH_QUEUE_CAP) {
    pq->keys[pq->tail] = key;
    pq->tail = (pq->tail + 1) % PREFETCH_QUEUE_CAP;
    pq->count++;
    pthread_cond_signal(&pq->cv);
  }
  pthread_mutex_unlock(&pq->mu);
}

// ---------------------------------------------------------------------------
// Level-granular eviction (compress4d pyramid support)
// ---------------------------------------------------------------------------

// Evict all hot+warm entries at a specific pyramid level.
// Pinned entries are skipped (coarsest level is pinned by convention).
void cache_evict_level(chunk_cache *c, int level) {
  REQUIRE(c, "cache_evict_level: null cache");
  pthread_mutex_lock(&c->mu);
  for (size_t i = 0; i < c->hot.cap; i++) {
    hot_entry *e = &c->hot.entries[i];
    if (!e->occupied || e->pinned || e->key.level != level) continue;
    c->hot.bytes -= e->data.size;
    c->hot.count--;
    free(e->data.data);
    *e = (hot_entry){0};
  }
  for (size_t i = 0; i < c->warm.cap; i++) {
    warm_entry *e = &c->warm.entries[i];
    if (!e->occupied || e->key.level != level) continue;
    c->warm.bytes -= e->compressed_size;
    c->warm.count--;
    free(e->compressed);
    *e = (warm_entry){0};
  }
  pthread_mutex_unlock(&c->mu);
}

// Evict finest levels first (level 0 = 1x is finest, level 4 = 32x is coarsest).
// Stops once at least target_free_bytes of hot-tier memory has been freed.
void cache_evict_finest_first(chunk_cache *c, size_t target_free_bytes) {
  REQUIRE(c, "cache_evict_finest_first: null cache");
  size_t freed = 0;
  // NOTE: level 0 is full-resolution (biggest), level 4 is 32x downsampled (smallest).
  // Evict from finest (0) upward; stop early once target met.
  for (int lvl = 0; lvl <= 4 && freed < target_free_bytes; lvl++) {
    pthread_mutex_lock(&c->mu);
    for (size_t i = 0; i < c->hot.cap && freed < target_free_bytes; i++) {
      hot_entry *e = &c->hot.entries[i];
      if (!e->occupied || e->pinned || e->key.level != lvl) continue;
      freed += e->data.size;
      c->hot.bytes -= e->data.size;
      c->hot.count--;
      free(e->data.data);
      *e = (hot_entry){0};
    }
    pthread_mutex_unlock(&c->mu);
  }
}

// Evict (finest first) until hot tier is under budget_bytes.
void cache_evict_to_budget(chunk_cache *c, size_t budget_bytes) {
  REQUIRE(c, "cache_evict_to_budget: null cache");
  size_t current = cache_hot_bytes(c);
  if (current <= budget_bytes) return;
  cache_evict_finest_first(c, current - budget_bytes);
}

// Return bytes used by chunks at a specific level across hot+warm tiers.
size_t cache_level_bytes(const chunk_cache *c, int level) {
  REQUIRE(c, "cache_level_bytes: null cache");
  pthread_mutex_lock((pthread_mutex_t *)&c->mu);
  size_t total = 0;
  for (size_t i = 0; i < c->hot.cap; i++) {
    const hot_entry *e = &c->hot.entries[i];
    if (e->occupied && e->key.level == level) total += e->data.size;
  }
  for (size_t i = 0; i < c->warm.cap; i++) {
    const warm_entry *e = &c->warm.entries[i];
    if (e->occupied && e->key.level == level) total += e->compressed_size;
  }
  pthread_mutex_unlock((pthread_mutex_t *)&c->mu);
  return total;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

size_t cache_hot_bytes(const chunk_cache *c) {
  REQUIRE(c, "cache_hot_bytes: null cache");
  pthread_mutex_lock((pthread_mutex_t *)&c->mu);
  size_t b = c->hot.bytes;
  pthread_mutex_unlock((pthread_mutex_t *)&c->mu);
  return b;
}

size_t cache_warm_bytes(const chunk_cache *c) {
  REQUIRE(c, "cache_warm_bytes: null cache");
  pthread_mutex_lock((pthread_mutex_t *)&c->mu);
  size_t b = c->warm.bytes;
  pthread_mutex_unlock((pthread_mutex_t *)&c->mu);
  return b;
}

size_t cache_hits(const chunk_cache *c) {
  REQUIRE(c, "cache_hits: null cache");
  pthread_mutex_lock((pthread_mutex_t *)&c->mu);
  size_t h = c->hits;
  pthread_mutex_unlock((pthread_mutex_t *)&c->mu);
  return h;
}

size_t cache_misses(const chunk_cache *c) {
  REQUIRE(c, "cache_misses: null cache");
  pthread_mutex_lock((pthread_mutex_t *)&c->mu);
  size_t m = c->misses;
  pthread_mutex_unlock((pthread_mutex_t *)&c->mu);
  return m;
}
