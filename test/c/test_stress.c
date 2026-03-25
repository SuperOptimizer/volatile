// Stress tests and edge cases across core modules.
// Each test must complete in <1 second.

#include "greatest.h"

#include "core/hash.h"
#include "core/thread.h"
#include "core/json.h"
#include "core/chunk.h"
#include "core/cache.h"
#include "core/compress4d.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Simple LCG for deterministic pseudo-random data
static uint32_t lcg_next(uint32_t *s) {
  *s = *s * 1664525u + 1013904223u;
  return *s;
}

// ---------------------------------------------------------------------------
// Hash map stress
// ---------------------------------------------------------------------------

TEST test_hash_map_100k(void) {
  hash_map *m = hash_map_new();
  ASSERT(m);

  char key[32];
  const int N = 100000;

  // Insert N entries (values are just the index cast to pointer)
  for (int i = 0; i < N; i++) {
    snprintf(key, sizeof(key), "k%d", i);
    hash_map_put(m, key, (void *)(uintptr_t)(i + 1));
  }
  ASSERT_EQ((size_t)N, hash_map_len(m));

  // Verify all retrievable
  for (int i = 0; i < N; i++) {
    snprintf(key, sizeof(key), "k%d", i);
    void *v = hash_map_get(m, key);
    ASSERT_EQ((uintptr_t)(i + 1), (uintptr_t)v);
  }

  // Delete even-indexed half
  int deleted = 0;
  for (int i = 0; i < N; i += 2) {
    snprintf(key, sizeof(key), "k%d", i);
    if (hash_map_del(m, key)) deleted++;
  }
  ASSERT_EQ(N / 2, deleted);
  ASSERT_EQ((size_t)(N - N / 2), hash_map_len(m));

  // Verify odd-indexed entries still present, even-indexed gone
  for (int i = 0; i < N; i++) {
    snprintf(key, sizeof(key), "k%d", i);
    void *v = hash_map_get(m, key);
    if (i % 2 == 0) {
      ASSERT_EQ(NULL, v);
    } else {
      ASSERT_EQ((uintptr_t)(i + 1), (uintptr_t)v);
    }
  }

  hash_map_free(m);
  PASS();
}

SUITE(suite_hash_stress) {
  RUN_TEST(test_hash_map_100k);
}

// ---------------------------------------------------------------------------
// Thread pool stress
// ---------------------------------------------------------------------------

static atomic_int g_task_counter = 0;

static void *increment_task(void *arg) {
  (void)arg;
  atomic_fetch_add(&g_task_counter, 1);
  return NULL;
}

TEST test_threadpool_1000_tasks(void) {
  atomic_store(&g_task_counter, 0);

  threadpool *pool = threadpool_new(4);
  ASSERT(pool);

  const int N = 1000;
  for (int i = 0; i < N; i++) {
    threadpool_fire(pool, increment_task, NULL);
  }

  threadpool_drain(pool, 5000);
  threadpool_free(pool);

  ASSERT_EQ(N, atomic_load(&g_task_counter));
  PASS();
}

SUITE(suite_thread_stress) {
  RUN_TEST(test_threadpool_1000_tasks);
}

// ---------------------------------------------------------------------------
// JSON stress
// ---------------------------------------------------------------------------

TEST test_json_deep_nesting(void) {
  // Build a 100-level nested array: [[[...[1]...]]]
  const int depth = 100;
  // Each level needs "[ " prefix and "]" suffix
  size_t buf_size = (size_t)depth * 2 + 4;
  char *buf = malloc(buf_size);
  ASSERT(buf);

  size_t pos = 0;
  for (int i = 0; i < depth; i++) buf[pos++] = '[';
  buf[pos++] = '1';
  for (int i = 0; i < depth; i++) buf[pos++] = ']';
  buf[pos] = '\0';

  json_value *v = json_parse(buf);
  free(buf);
  ASSERT(v);

  // Descend to the innermost value
  const json_value *cur = v;
  for (int i = 0; i < depth; i++) {
    ASSERT_EQ(JSON_ARRAY, json_typeof(cur));
    ASSERT_EQ((size_t)1, json_array_len(cur));
    cur = json_array_get(cur, 0);
    ASSERT(cur);
  }
  ASSERT_EQ(JSON_NUMBER, json_typeof(cur));
  ASSERT_EQ(1.0, json_get_number(cur, 0.0));

  json_free(v);
  PASS();
}

TEST test_json_long_string(void) {
  const size_t str_len = 10000;
  // Build: "\"" + 'A'*str_len + "\""
  char *buf = malloc(str_len + 3);
  ASSERT(buf);
  buf[0] = '"';
  memset(buf + 1, 'A', str_len);
  buf[str_len + 1] = '"';
  buf[str_len + 2] = '\0';

  json_value *v = json_parse(buf);
  free(buf);
  ASSERT(v);
  ASSERT_EQ(JSON_STRING, json_typeof(v));
  const char *s = json_get_str(v);
  ASSERT(s);
  ASSERT_EQ(str_len, strlen(s));
  json_free(v);
  PASS();
}

SUITE(suite_json_stress) {
  RUN_TEST(test_json_deep_nesting);
  RUN_TEST(test_json_long_string);
}

// ---------------------------------------------------------------------------
// Chunk stress
// ---------------------------------------------------------------------------

TEST test_chunk_large_corners(void) {
  // 100x100x100 volume with 32^3 chunks to keep memory reasonable
  int64_t shape[3]       = {100, 100, 100};
  int64_t chunk_shape[3] = {32, 32, 32};

  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));
  ASSERT(a);

  // Set the 8 corners
  int64_t corners[8][3] = {
    {0, 0, 0}, {0, 0, 99}, {0, 99, 0}, {0, 99, 99},
    {99, 0, 0}, {99, 0, 99}, {99, 99, 0}, {99, 99, 99},
  };
  float expected[8];
  for (int i = 0; i < 8; i++) {
    expected[i] = (float)(i + 1) * 3.14f;
    chunked_array_set_f32(a, corners[i], expected[i]);
  }

  // Verify
  for (int i = 0; i < 8; i++) {
    float got = chunked_array_get_f32(a, corners[i]);
    ASSERT_EQ(expected[i], got);
  }

  chunked_array_free(a);
  PASS();
}

SUITE(suite_chunk_stress) {
  RUN_TEST(test_chunk_large_corners);
}

// ---------------------------------------------------------------------------
// Cache stress
// ---------------------------------------------------------------------------

TEST test_cache_eviction(void) {
  // Budget: 512 bytes hot. Insert chunks of 64 bytes each -> eviction after 8.
  cache_config cfg = {
    .hot_max_bytes  = 512,
    .warm_max_bytes = 0,
    .cold_max_bytes = 0,
    .cold_dir       = NULL,
    .io_threads     = 0,
  };
  chunk_cache *c = cache_new(cfg);
  ASSERT(c);

  const int N = 20;
  const size_t chunk_sz = 64;

  for (int i = 0; i < N; i++) {
    chunk_data *d = malloc(sizeof(chunk_data));
    ASSERT(d);
    d->data = malloc(chunk_sz);
    ASSERT(d->data);
    memset(d->data, (unsigned char)i, chunk_sz);
    d->size = chunk_sz;
    d->shape[0] = 4; d->shape[1] = 4; d->shape[2] = 4;
    d->elem_size = 1;
    chunk_key key = {.level = 0, .iz = i, .iy = 0, .ix = 0};
    cache_put(c, key, d);
  }

  // After inserting 20 * 64 = 1280 bytes into a 512-byte budget,
  // hot tier must respect its memory limit.
  size_t hot = cache_hot_bytes(c);
  ASSERT(hot <= 512);

  // Eviction occurred: not all 20 * 64 bytes remain in hot.
  ASSERT(hot < (size_t)N * chunk_sz);

  cache_free(c);
  PASS();
}

SUITE(suite_cache_stress) {
  RUN_TEST(test_cache_eviction);
}

// ---------------------------------------------------------------------------
// Compress4d roundtrip
// ---------------------------------------------------------------------------

TEST test_compress4d_roundtrip_1mb(void) {
  const size_t N = 1024 * 1024 / sizeof(float);  // 256K floats = 1MB
  float *orig = malloc(N * sizeof(float));
  float *recovered = malloc(N * sizeof(float));
  ASSERT(orig && recovered);

  // Fill with pseudo-random data in [-1, 1]
  uint32_t seed = 0xdeadbeef;
  for (size_t i = 0; i < N; i++) {
    orig[i] = (float)(int32_t)lcg_next(&seed) / (float)INT32_MAX;
  }

  float scale = 1.0f / 127.0f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(orig, N, scale, &enc_len);
  ASSERT(enc);
  ASSERT(enc_len > 0);

  bool ok = compress4d_decode_residual(enc, enc_len, N, scale, recovered);
  ASSERT(ok);

  // Verify exact match (encode/decode is deterministic and lossless within quant)
  // The quantisation error is at most scale = 1/127, so check within tolerance
  float max_err = 0.0f;
  for (size_t i = 0; i < N; i++) {
    float err = orig[i] - recovered[i];
    if (err < 0) err = -err;
    if (err > max_err) max_err = err;
  }
  ASSERT(max_err <= scale * 1.01f);

  free(enc);
  free(orig);
  free(recovered);
  PASS();
}

SUITE(suite_compress4d_stress) {
  RUN_TEST(test_compress4d_roundtrip_1mb);
}

// ---------------------------------------------------------------------------
// Integer hash map: insert/delete 10K, verify survivors
// ---------------------------------------------------------------------------

TEST test_hash_int_insert_delete_10k(void) {
  hash_map_int *m = hash_map_int_new();
  ASSERT(m);

  const int N = 10000;

  // Insert N entries
  for (int i = 0; i < N; i++) {
    hash_map_int_put(m, (uint64_t)i, (void *)(uintptr_t)(i + 1));
  }
  ASSERT_EQ((size_t)N, hash_map_int_len(m));

  // Snapshot: read all values into a local array so we can verify after deletes
  uintptr_t *snap = malloc((size_t)N * sizeof(uintptr_t));
  ASSERT(snap);
  for (int i = 0; i < N; i++) {
    void *v = hash_map_int_get(m, (uint64_t)i);
    snap[i] = (uintptr_t)v;
    ASSERT_EQ((uintptr_t)(i + 1), snap[i]);
  }

  // Delete odd-indexed entries
  for (int i = 1; i < N; i += 2) {
    hash_map_int_del(m, (uint64_t)i);
  }
  ASSERT_EQ((size_t)(N / 2), hash_map_int_len(m));

  // Even entries must still be present and unchanged ("old snapshot" still valid)
  for (int i = 0; i < N; i += 2) {
    void *v = hash_map_int_get(m, (uint64_t)i);
    ASSERT_EQ(snap[i], (uintptr_t)v);
  }
  // Odd entries must be gone
  for (int i = 1; i < N; i += 2) {
    void *v = hash_map_int_get(m, (uint64_t)i);
    ASSERT_EQ(NULL, v);
  }

  free(snap);
  hash_map_int_free(m);
  PASS();
}

SUITE(suite_hamt_stress) {
  RUN_TEST(test_hash_int_insert_delete_10k);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(suite_hash_stress);
  RUN_SUITE(suite_thread_stress);
  RUN_SUITE(suite_json_stress);
  RUN_SUITE(suite_chunk_stress);
  RUN_SUITE(suite_cache_stress);
  RUN_SUITE(suite_compress4d_stress);
  RUN_SUITE(suite_hamt_stress);
  GREATEST_MAIN_END();
}
