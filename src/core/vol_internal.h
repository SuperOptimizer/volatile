#pragma once
// Internal shared definitions for vol.c and vol_write.c.
// Not part of the public API — do not include from outside core/.

#include "core/vol.h"
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define VOL_MAX_LEVELS          16
#define VOL_PATH_MAX            1024
#define SHARD_INDEX_ENTRY_BYTES 16
#define SHARD_EMPTY_ENTRY       UINT64_C(0xFFFFFFFFFFFFFFFF)

struct volume {
  char          path[VOL_PATH_MAX];
  vol_source_t  source;
  int           num_levels;
  zarr_level_meta levels[VOL_MAX_LEVELS];
};

// Shared utility: compute number of inner chunks per shard and optionally fill
// nchunks_per_dim[0..ndim-1].
size_t vol_shard_nchunks(const zarr_level_meta *m, int64_t *nchunks_per_dim);

// Shared utility: flat inner-chunk index from local coords within a shard.
size_t vol_inner_chunk_flat_idx(const zarr_level_meta *m, const int64_t *inner_coords);

// Shared utility: read entire file into malloc'd buffer (caller frees). Sets *size.
uint8_t *vol_read_file_bytes(const char *path, size_t *size);

// Shared utility: write little-endian uint64 to p.
static inline void vol_write_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) { p[i] = (uint8_t)(v & 0xff); v >>= 8; }
}

// Shared utility: read little-endian uint64 from p.
static inline uint64_t vol_read_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
  return v;
}

// Shared: dtype to v2 string ("|u1" etc.)
const char *vol_dtype_to_v2str(int dtype);

// Shared: dtype to v3 string ("uint8" etc.)
const char *vol_dtype_to_v3str(int dtype);

// Shared: element size in bytes for dtype
size_t vol_dtype_elem_size(int dtype);

// Shared: recursively create directory path (like mkdir -p)
bool vol_mkdirp(const char *path);
