#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct chunk_cache chunk_cache;

typedef struct {
  size_t hot_max_bytes;     // max decompressed RAM (default 4GB)
  size_t warm_max_bytes;    // max compressed RAM (default 1GB)
  size_t cold_max_bytes;    // max disk cache (default 50GB)
  const char *cold_dir;     // disk cache directory (NULL to disable)
  int io_threads;           // number of IO threads (default 4)
} cache_config;

// chunk key: identifies a chunk in the pyramid
typedef struct {
  int level;
  int64_t iz, iy, ix;
} chunk_key;

// chunk data: the decompressed result
typedef struct {
  uint8_t *data;
  size_t size;
  int shape[3];      // dimensions of this chunk
  int elem_size;     // bytes per element
} chunk_data;

void chunk_data_free(chunk_data *d);

// cache lifecycle
chunk_cache *cache_new(cache_config cfg);
void cache_free(chunk_cache *c);

// non-blocking get from hot+warm tiers. returns NULL on miss.
chunk_data *cache_get(chunk_cache *c, chunk_key key);

// get best available: searches from requested level up to coarsest.
// returns {data, actual_level}. coarsest level is always pinned.
typedef struct { chunk_data *data; int actual_level; } cache_best_result;
cache_best_result cache_get_best(chunk_cache *c, chunk_key key, int coarsest_level);

// blocking get: promotes through all tiers if needed. may fetch from disk/network.
chunk_data *cache_get_blocking(chunk_cache *c, chunk_key key, int timeout_ms);

// schedule background fetch
void cache_prefetch(chunk_cache *c, chunk_key key);

// pin a chunk (never evict). used for coarsest pyramid level.
void cache_pin(chunk_cache *c, chunk_key key, chunk_data *data);

// insert data into hot tier
void cache_put(chunk_cache *c, chunk_key key, chunk_data *data);

// stats
size_t cache_hot_bytes(const chunk_cache *c);
size_t cache_warm_bytes(const chunk_cache *c);
size_t cache_hits(const chunk_cache *c);
size_t cache_misses(const chunk_cache *c);

// level-granular eviction (compress4d pyramid support)
// evict all cached chunks at a specific pyramid level
void   cache_evict_level(chunk_cache *c, int level);
// evict finest levels first (level 0 first, level 4 last), freeing at least target_free_bytes
void   cache_evict_finest_first(chunk_cache *c, size_t target_free_bytes);
// evict until hot tier is under budget_bytes
void   cache_evict_to_budget(chunk_cache *c, size_t budget_bytes);
// get total memory used by chunks at a specific pyramid level (hot+warm)
size_t cache_level_bytes(const chunk_cache *c, int level);
