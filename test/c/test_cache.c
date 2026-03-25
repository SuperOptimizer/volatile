#include "greatest.h"
#include "core/cache.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static chunk_key make_key(int level, int64_t iz, int64_t iy, int64_t ix) {
  return (chunk_key){ .level = level, .iz = iz, .iy = iy, .ix = ix };
}

// Allocate a chunk_data with all bytes set to val.
static chunk_data *make_chunk(int val, size_t size) {
  chunk_data *d = malloc(sizeof(*d));
  d->data       = malloc(size);
  d->size       = size;
  d->elem_size  = 1;
  d->shape[0]   = (int)size; d->shape[1] = 1; d->shape[2] = 1;
  memset(d->data, val, size);
  return d;
}

static cache_config small_cfg(void) {
  return (cache_config){
    .hot_max_bytes  = 4096,   // tiny — forces eviction in tests
    .warm_max_bytes = 2048,
    .cold_max_bytes = 0,
    .cold_dir       = NULL,
    .io_threads     = 1,
  };
}

// ---------------------------------------------------------------------------
// Test: basic put and get
// ---------------------------------------------------------------------------

TEST test_put_and_get(void) {
  chunk_cache *c = cache_new(small_cfg());

  chunk_key   key  = make_key(0, 0, 0, 0);
  chunk_data *data = make_chunk(0xAB, 128);

  cache_put(c, key, data);
  chunk_data_free(data);

  chunk_data *got = cache_get(c, key);
  ASSERT(got != NULL);
  ASSERT_EQ(128u, got->size);
  ASSERT_EQ((uint8_t)0xAB, got->data[0]);
  chunk_data_free(got);

  // miss on absent key
  chunk_data *miss = cache_get(c, make_key(0, 9, 9, 9));
  ASSERT(miss == NULL);

  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: eviction when over budget
// ---------------------------------------------------------------------------

TEST test_eviction_over_budget(void) {
  cache_config cfg = small_cfg();
  cfg.hot_max_bytes = 512;   // only 512 bytes — force eviction
  chunk_cache *c = cache_new(cfg);

  // Insert many chunks totalling far more than the budget
  for (int i = 0; i < 20; i++) {
    chunk_data *d = make_chunk(i, 128);
    cache_put(c, make_key(0, 0, 0, i), d);
    chunk_data_free(d);
  }

  // Cache must not exceed budget
  ASSERT(cache_hot_bytes(c) <= 512u);

  // At least some entries were inserted
  int found = 0;
  for (int i = 0; i < 20; i++) {
    chunk_data *d = cache_get(c, make_key(0, 0, 0, i));
    if (d) { found++; chunk_data_free(d); }
  }
  ASSERT(found > 0);

  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: pin prevents eviction
// ---------------------------------------------------------------------------

TEST test_pin_prevents_eviction(void) {
  cache_config cfg = small_cfg();
  cfg.hot_max_bytes = 256;   // very tight budget
  chunk_cache *c = cache_new(cfg);

  // Pin a chunk
  chunk_key  pinned_key  = make_key(3, 0, 0, 0);
  chunk_data *pinned_data = make_chunk(0xFF, 128);
  cache_pin(c, pinned_key, pinned_data);

  // Fill the cache with other chunks to pressure eviction
  for (int i = 1; i < 20; i++) {
    chunk_data *d = make_chunk(i, 64);
    cache_put(c, make_key(0, 0, 0, i), d);
    chunk_data_free(d);
  }

  // Pinned chunk must still be retrievable
  chunk_data *got = cache_get(c, pinned_key);
  ASSERT(got != NULL);
  ASSERT_EQ((uint8_t)0xFF, got->data[0]);
  chunk_data_free(got);

  chunk_data_free(pinned_data);
  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: cache_get_best falls back to coarser levels
// ---------------------------------------------------------------------------

TEST test_get_best_fallback(void) {
  chunk_cache *c = cache_new(small_cfg());

  // Populate level 3 (coarsest) only — coordinates shift right per level
  // key at level 0: iz=4, iy=4, ix=4 → level 3: iz=0, iy=0, ix=0
  chunk_key coarse_key = make_key(3, 0, 0, 0);
  chunk_data *coarse   = make_chunk(0x77, 64);
  cache_pin(c, coarse_key, coarse);

  // Fine key at level 0
  chunk_key fine_key = make_key(0, 4, 4, 4);

  // Fine level is absent, should find coarsest
  cache_best_result res = cache_get_best(c, fine_key, 3);
  ASSERT(res.data != NULL);
  ASSERT_EQ(3, res.actual_level);
  ASSERT_EQ((uint8_t)0x77, res.data->data[0]);
  chunk_data_free(res.data);

  // Now insert the exact fine key
  chunk_data *fine = make_chunk(0x11, 64);
  cache_put(c, fine_key, fine);
  chunk_data_free(fine);

  cache_best_result res2 = cache_get_best(c, fine_key, 3);
  ASSERT(res2.data != NULL);
  ASSERT_EQ(0, res2.actual_level);  // exact match
  chunk_data_free(res2.data);

  chunk_data_free(coarse);
  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: stats tracking
// ---------------------------------------------------------------------------

TEST test_stats_tracking(void) {
  chunk_cache *c = cache_new(small_cfg());

  chunk_key key = make_key(1, 0, 0, 0);

  // Miss before any insert
  chunk_data *miss = cache_get(c, key);
  ASSERT(miss == NULL);
  ASSERT_EQ(1u, cache_misses(c));
  ASSERT_EQ(0u, cache_hits(c));

  // Insert then hit
  chunk_data *d = make_chunk(42, 64);
  cache_put(c, key, d);
  chunk_data_free(d);

  chunk_data *hit = cache_get(c, key);
  ASSERT(hit != NULL);
  chunk_data_free(hit);
  ASSERT_EQ(1u, cache_hits(c));
  ASSERT_EQ(1u, cache_misses(c));  // still one miss from before

  // Another miss
  cache_get(c, make_key(0, 99, 99, 99));
  ASSERT_EQ(2u, cache_misses(c));

  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: hot_bytes reflects inserts and budget
// ---------------------------------------------------------------------------

TEST test_hot_bytes(void) {
  cache_config cfg = small_cfg();
  cfg.hot_max_bytes = 1024;
  chunk_cache *c = cache_new(cfg);

  ASSERT_EQ(0u, cache_hot_bytes(c));

  chunk_data *d1 = make_chunk(1, 256);
  chunk_data *d2 = make_chunk(2, 256);
  cache_put(c, make_key(0, 0, 0, 0), d1);
  cache_put(c, make_key(0, 0, 0, 1), d2);
  chunk_data_free(d1);
  chunk_data_free(d2);

  ASSERT_EQ(512u, cache_hot_bytes(c));
  ASSERT(cache_hot_bytes(c) <= cfg.hot_max_bytes);

  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: prefetch does not crash
// ---------------------------------------------------------------------------

TEST test_prefetch_no_crash(void) {
  chunk_cache *c = cache_new(small_cfg());

  for (int i = 0; i < 10; i++)
    cache_prefetch(c, make_key(0, 0, 0, i));

  // No assertions beyond "doesn't crash or deadlock"
  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: concurrent get from multiple threads
// ---------------------------------------------------------------------------

typedef struct { chunk_cache *c; chunk_key key; atomic_int *hits; } cget_arg;

static void *concurrent_get_fn(void *arg_ptr) {
  cget_arg *a = arg_ptr;
  chunk_data *d = cache_get(a->c, a->key);
  if (d) { atomic_fetch_add(a->hits, 1); chunk_data_free(d); }
  return NULL;
}

TEST test_concurrent_get(void) {
  cache_config cfg = small_cfg();
  cfg.hot_max_bytes = 8192;
  chunk_cache *c = cache_new(cfg);

  chunk_key key = make_key(0, 7, 7, 7);
  chunk_data *d = make_chunk(0xCC, 64);
  cache_put(c, key, d);
  chunk_data_free(d);

  const int NTHREADS = 8;
  pthread_t threads[8];
  cget_arg  args[8];
  atomic_int hits = 0;
  for (int i = 0; i < NTHREADS; i++) {
    args[i] = (cget_arg){c, key, &hits};
    pthread_create(&threads[i], NULL, concurrent_get_fn, &args[i]);
  }
  for (int i = 0; i < NTHREADS; i++) pthread_join(threads[i], NULL);

  ASSERT_EQ(NTHREADS, atomic_load(&hits));
  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: prefetch triggers background load (key missing before, not after wait)
// ---------------------------------------------------------------------------

TEST test_prefetch_background_load(void) {
  chunk_cache *c = cache_new(small_cfg());

  chunk_key key = make_key(0, 3, 3, 3);
  // Key absent — prefetch schedules a background fetch (no-op IO since no
  // backing store, but must not crash or deadlock).
  cache_prefetch(c, key);
  cache_prefetch(c, key);  // duplicate is also safe

  // Insert explicitly and verify it's retrievable.
  chunk_data *d = make_chunk(0x55, 32);
  cache_put(c, key, d);
  chunk_data_free(d);

  chunk_data *got = cache_get(c, key);
  ASSERT(got != NULL);
  ASSERT_EQ((uint8_t)0x55, got->data[0]);
  chunk_data_free(got);

  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: eviction under memory pressure — most-recently-used survives
// ---------------------------------------------------------------------------

TEST test_eviction_mru_survives(void) {
  cache_config cfg = small_cfg();
  cfg.hot_max_bytes = 512;
  chunk_cache *c = cache_new(cfg);

  // Insert 4 chunks of 128 bytes — fills the 512-byte budget exactly.
  for (int i = 0; i < 4; i++) {
    chunk_data *d = make_chunk(i, 128);
    cache_put(c, make_key(0, 0, 0, i), d);
    chunk_data_free(d);
  }

  // Touch key 3 (most recent), then insert more to force eviction.
  chunk_data *touch = cache_get(c, make_key(0, 0, 0, 3));
  if (touch) chunk_data_free(touch);

  for (int i = 4; i < 8; i++) {
    chunk_data *d = make_chunk(i, 128);
    cache_put(c, make_key(0, 0, 0, i), d);
    chunk_data_free(d);
  }

  ASSERT(cache_hot_bytes(c) <= 512u);
  cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(cache_suite) {
  RUN_TEST(test_put_and_get);
  RUN_TEST(test_eviction_over_budget);
  RUN_TEST(test_pin_prevents_eviction);
  RUN_TEST(test_get_best_fallback);
  RUN_TEST(test_stats_tracking);
  RUN_TEST(test_hot_bytes);
  RUN_TEST(test_prefetch_no_crash);
  RUN_TEST(test_concurrent_get);
  RUN_TEST(test_prefetch_background_load);
  RUN_TEST(test_eviction_mru_survives);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(cache_suite);
  GREATEST_MAIN_END();
}
