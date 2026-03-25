#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct vol_mirror vol_mirror;
typedef struct volume volume;

typedef struct {
  const char *remote_url;      // http://... or s3://...
  const char *local_cache_dir; // where to store cached data (default ~/.cache/volatile/)
  bool auto_rechunk;           // rechunk to optimal local chunk size on cache
  bool auto_compress4d;        // recompress blosc chunks to compress4d locally
  int64_t max_cache_bytes;     // max disk cache size (0 = default 50GB)
  int prefetch_radius;         // neighboring chunks to prefetch (0 = default 2)
} mirror_config;

// Create a mirrored volume: reads from remote, caches locally.
vol_mirror *vol_mirror_new(mirror_config cfg);
void vol_mirror_free(vol_mirror *m);

// Get the volume handle (reads from cache or remote transparently).
volume *vol_mirror_volume(vol_mirror *m);

// Force-download an entire level to local cache.
bool vol_mirror_cache_level(vol_mirror *m, int level);

// Rechunk the locally cached data into a new local zarr with a different chunk shape.
bool vol_mirror_rechunk(vol_mirror *m, const int64_t *new_chunk_shape);

// Recompress local cache from blosc to compress4d residuals.
bool vol_mirror_recompress(vol_mirror *m);

// Stats
size_t vol_mirror_cached_bytes(const vol_mirror *m);
float  vol_mirror_cache_hit_rate(const vol_mirror *m);
int    vol_mirror_chunks_cached(const vol_mirror *m);
int    vol_mirror_chunks_total(const vol_mirror *m, int level);
