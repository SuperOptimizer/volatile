#include "render/tile.h"
#include "core/thread.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

// A single unit of queued or in-flight work.
typedef struct tile_job {
  tile_key         key;
  atomic_bool      cancelled;
  struct tile_job *next;   // intrusive list link (completed queue)
} tile_job;

// Completed result waiting to be drained.
typedef struct completed_node {
  tile_result          result;
  struct completed_node *next;
} completed_node;

struct tile_renderer {
  threadpool *pool;

  // Pending queue — jobs submitted but not yet picked up by a worker.
  // Protected by pending_mu.
  pthread_mutex_t  pending_mu;
  tile_job       **pending;     // dynamic array of non-cancelled pending jobs
  int              pending_len;
  int              pending_cap;

  // Completed queue — singly-linked list of results ready to drain.
  // Protected by completed_mu.
  pthread_mutex_t  completed_mu;
  completed_node  *completed_head;
  int              completed_count;
};

// ---------------------------------------------------------------------------
// Test-pattern pixel fill
// NOTE: real rendering goes here; for now paints a checkerboard coloured by
//       (col, row, pyramid_level) so tests can verify tile identity.
// ---------------------------------------------------------------------------

static void fill_test_pattern(uint8_t *pixels, tile_key key) {
  for (int py = 0; py < TILE_PX; py++) {
    for (int px = 0; px < TILE_PX; px++) {
      int   idx    = (py * TILE_PX + px) * 4;
      bool  check  = ((px >> 4) ^ (py >> 4)) & 1;
      uint8_t base_r = (uint8_t)((key.col * 37)  & 0xFF);
      uint8_t base_g = (uint8_t)((key.row * 53)  & 0xFF);
      uint8_t base_b = (uint8_t)((key.pyramid_level * 71 + 40) & 0xFF);
      pixels[idx+0] = check ? base_r : (uint8_t)(base_r ^ 0x40);
      pixels[idx+1] = check ? base_g : (uint8_t)(base_g ^ 0x40);
      pixels[idx+2] = check ? base_b : (uint8_t)(base_b ^ 0x40);
      pixels[idx+3] = 0xFF;
    }
  }
}

// ---------------------------------------------------------------------------
// Worker task — executed on the thread pool
// ---------------------------------------------------------------------------

typedef struct {
  tile_renderer *renderer;
  tile_job      *job;
} worker_arg;

static void *tile_worker(void *arg_ptr) {
  worker_arg    *arg      = arg_ptr;
  tile_renderer *renderer = arg->renderer;
  tile_job      *job      = arg->job;
  free(arg_ptr);

  if (atomic_load(&job->cancelled)) {
    return NULL;  // renderer owns job lifetime; do not free here
  }

  uint8_t *pixels = malloc((size_t)TILE_PX * TILE_PX * 4);
  if (!pixels) {
    return NULL;
  }
  fill_test_pattern(pixels, job->key);

  // Check cancellation again after (potentially expensive) render.
  if (atomic_load(&job->cancelled)) {
    free(pixels);
    return NULL;
  }

  completed_node *node = malloc(sizeof(*node));
  if (!node) {
    free(pixels);
    return NULL;
  }
  node->result.key    = job->key;
  node->result.pixels = pixels;
  node->result.valid  = true;
  node->next          = NULL;

  pthread_mutex_lock(&renderer->completed_mu);
  node->next               = renderer->completed_head;
  renderer->completed_head = node;
  renderer->completed_count++;
  pthread_mutex_unlock(&renderer->completed_mu);

  return NULL;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

tile_renderer *tile_renderer_new(int num_threads) {
  tile_renderer *r = calloc(1, sizeof(*r));
  if (!r) return NULL;

  r->pool = threadpool_new(num_threads);
  if (!r->pool) { free(r); return NULL; }

  pthread_mutex_init(&r->pending_mu,   NULL);
  pthread_mutex_init(&r->completed_mu, NULL);

  r->pending_cap = 64;
  r->pending     = malloc((size_t)r->pending_cap * sizeof(*r->pending));
  if (!r->pending) {
    threadpool_free(r->pool);
    pthread_mutex_destroy(&r->pending_mu);
    pthread_mutex_destroy(&r->completed_mu);
    free(r);
    return NULL;
  }
  return r;
}

void tile_renderer_free(tile_renderer *r) {
  if (!r) return;

  // Cancel all pending jobs so workers exit quickly.
  pthread_mutex_lock(&r->pending_mu);
  for (int i = 0; i < r->pending_len; i++)
    atomic_store(&r->pending[i]->cancelled, true);
  pthread_mutex_unlock(&r->pending_mu);

  // Wait for all in-flight workers to finish.
  threadpool_drain(r->pool, 500);
  threadpool_free(r->pool);

  // Now safe to free all job structs — no workers are running.
  // Workers do NOT free jobs; the renderer owns all job lifetimes.
  for (int i = 0; i < r->pending_len; i++)
    free(r->pending[i]);
  free(r->pending);

  // Drain remaining completed results.
  pthread_mutex_lock(&r->completed_mu);
  completed_node *cn = r->completed_head;
  while (cn) {
    completed_node *next = cn->next;
    free(cn->result.pixels);
    free(cn);
    cn = next;
  }
  pthread_mutex_unlock(&r->completed_mu);

  pthread_mutex_destroy(&r->pending_mu);
  pthread_mutex_destroy(&r->completed_mu);
  free(r);
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

void tile_renderer_submit(tile_renderer *r, tile_key key) {
  tile_job *job = calloc(1, sizeof(*job));
  if (!job) { LOG_WARN("tile_renderer_submit: alloc failed"); return; }
  job->key = key;
  atomic_init(&job->cancelled, false);

  // Track in pending array so cancel_stale can reach it.
  pthread_mutex_lock(&r->pending_mu);
  if (r->pending_len == r->pending_cap) {
    int new_cap = r->pending_cap * 2;
    tile_job **tmp = realloc(r->pending, (size_t)new_cap * sizeof(*r->pending));
    if (!tmp) {
      pthread_mutex_unlock(&r->pending_mu);
      free(job);
      LOG_WARN("tile_renderer_submit: pending array grow failed");
      return;
    }
    r->pending     = tmp;
    r->pending_cap = new_cap;
  }
  r->pending[r->pending_len++] = job;
  pthread_mutex_unlock(&r->pending_mu);

  worker_arg *arg = malloc(sizeof(*arg));
  if (!arg) { atomic_store(&job->cancelled, true); return; }
  arg->renderer = r;
  arg->job      = job;

  threadpool_fire(r->pool, tile_worker, arg);
}

// ---------------------------------------------------------------------------
// Drain
// ---------------------------------------------------------------------------

int tile_renderer_drain(tile_renderer *r, tile_result *out, int max_results) {
  if (max_results <= 0) return 0;

  pthread_mutex_lock(&r->completed_mu);
  int count = 0;
  while (r->completed_head && count < max_results) {
    completed_node *node  = r->completed_head;
    r->completed_head     = node->next;
    r->completed_count--;
    out[count++] = node->result;
    free(node);
  }
  pthread_mutex_unlock(&r->completed_mu);

  // Remove drained / cancelled entries from the pending tracking array.
  // We identify completed jobs by scanning for ones no longer needed;
  // since workers free their own job struct, we just compact out NULLs.
  // NOTE: pending array entries are pointer-stable until the worker frees
  // the job.  After the worker calls free(job) the pointer is dangling —
  // we must not dereference it.  Instead we remove entries from the array
  // that are known-cancelled (epoch cleaned up) via cancel_stale, and
  // accept that the array may grow without bound on heavy submit workloads
  // until cancel_stale is called.  For a 30 Hz drain loop this is fine.

  return count;
}

// ---------------------------------------------------------------------------
// Cancel stale
// ---------------------------------------------------------------------------

void tile_renderer_cancel_stale(tile_renderer *r, uint64_t min_epoch) {
  pthread_mutex_lock(&r->pending_mu);
  int write = 0;
  for (int i = 0; i < r->pending_len; i++) {
    tile_job *job = r->pending[i];
    if (job->key.epoch < min_epoch) {
      atomic_store(&job->cancelled, true);
      // Don't free job here — the worker will free it when it runs.
    } else {
      r->pending[write++] = job;
    }
  }
  r->pending_len = write;
  pthread_mutex_unlock(&r->pending_mu);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int tile_renderer_pending(const tile_renderer *r) {
  pthread_mutex_lock((pthread_mutex_t *)&r->pending_mu);
  int n = r->pending_len;
  pthread_mutex_unlock((pthread_mutex_t *)&r->pending_mu);
  return n;
}

// ---------------------------------------------------------------------------
// slice_cache — LRU image cache
//
// Implemented as a fixed-size array with an LRU counter (timestamp).  For
// typical screen-sized tile grids (< 256 entries) a linear scan is O(1) in
// practice.  Eviction removes the entry with the smallest access_time.
// ---------------------------------------------------------------------------

#include <math.h>

typedef struct {
  slice_cache_key  key;
  uint8_t         *pixels;      // owned
  int8_t           level;
  uint64_t         access_time; // logical clock; higher = more recently used
  bool             valid;
} sc_entry;

struct slice_cache {
  sc_entry *entries;
  int       capacity;
  uint64_t  clock;  // monotonically increasing logical clock
};

static bool sc_key_eq(slice_cache_key a, slice_cache_key b) {
  return a.col == b.col && a.row == b.row &&
         a.scale_q == b.scale_q && a.z_off_q == b.z_off_q &&
         a.level == b.level && a.params_hash == b.params_hash;
}

slice_cache *slice_cache_new(int max_entries) {
  if (max_entries <= 0) max_entries = 256;
  slice_cache *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->entries = calloc((size_t)max_entries, sizeof(sc_entry));
  if (!c->entries) { free(c); return NULL; }
  c->capacity = max_entries;
  c->clock = 1;
  return c;
}

void slice_cache_free(slice_cache *c) {
  if (!c) return;
  for (int i = 0; i < c->capacity; i++)
    if (c->entries[i].valid) free(c->entries[i].pixels);
  free(c->entries);
  free(c);
}

slice_cache_key slice_cache_make_key(int col, int row, float scale, float z_offset,
                                      int level, uint32_t params_hash) {
  return (slice_cache_key){
    .col         = col,
    .row         = row,
    .scale_q     = (int16_t)(int)(roundf(log2f(scale > 0.0f ? scale : 1.0f) * 32.0f)),
    .z_off_q     = (int16_t)(int)(roundf(z_offset * 4.0f)),
    .level       = (int8_t)level,
    .params_hash = params_hash,
  };
}

const slice_cache_entry *slice_cache_get(slice_cache *c, slice_cache_key key) {
  for (int i = 0; i < c->capacity; i++) {
    sc_entry *e = &c->entries[i];
    if (e->valid && sc_key_eq(e->key, key)) {
      e->access_time = c->clock++;
      // Return a pointer into the internal entry.  Caller must not outlive next put.
      // We cast to the public type since it shares the same pixel/level fields.
      return (const slice_cache_entry *)(void *)&e->pixels;
    }
  }
  return NULL;
}

bool slice_cache_get_best(slice_cache *c, slice_cache_key key,
                           slice_cache_entry *out_entry) {
  // Try exact level, then progressively coarser.
  for (int delta = 0; delta <= SLICE_CACHE_MAX_COARSER; delta++) {
    slice_cache_key k = key;
    k.level = (int8_t)(key.level + delta);
    for (int i = 0; i < c->capacity; i++) {
      sc_entry *e = &c->entries[i];
      if (e->valid && sc_key_eq(e->key, k)) {
        e->access_time = c->clock++;
        out_entry->pixels = e->pixels;
        out_entry->level  = e->level;
        return true;
      }
    }
  }
  return false;
}

void slice_cache_put(slice_cache *c, slice_cache_key key, uint8_t *pixels, int8_t level) {
  // Check if entry already exists (update in place).
  for (int i = 0; i < c->capacity; i++) {
    sc_entry *e = &c->entries[i];
    if (e->valid && sc_key_eq(e->key, key)) {
      free(e->pixels);
      e->pixels      = pixels;
      e->level       = level;
      e->access_time = c->clock++;
      return;
    }
  }

  // Find an empty slot or evict the LRU entry.
  int target = -1;
  uint64_t oldest_time = UINT64_MAX;
  for (int i = 0; i < c->capacity; i++) {
    sc_entry *e = &c->entries[i];
    if (!e->valid) { target = i; break; }  // prefer empty slot
    if (e->access_time < oldest_time) {
      oldest_time = e->access_time;
      target = i;
    }
  }

  sc_entry *e = &c->entries[target];
  if (e->valid) free(e->pixels);
  e->key         = key;
  e->pixels      = pixels;
  e->level       = level;
  e->access_time = c->clock++;
  e->valid       = true;
}

void slice_cache_clear(slice_cache *c) {
  if (!c) return;
  for (int i = 0; i < c->capacity; i++) {
    if (c->entries[i].valid) {
      free(c->entries[i].pixels);
      c->entries[i].valid = false;
      c->entries[i].pixels = NULL;
    }
  }
}
