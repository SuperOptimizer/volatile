#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// gridstore — file-per-chunk disk storage (COLD tier backend for cache.c)
//
// Chunks are stored as flat binary files under <path>/<z>/<y>/<x>.bin.
// The chunk_shape is informational only (used to validate coords).
// ---------------------------------------------------------------------------

typedef struct gridstore gridstore;

// Open or create a gridstore rooted at `path`. Creates the directory if absent.
// chunk_shape[3]: expected shape of each chunk (z,y,x); used for sanity checks.
gridstore *gridstore_new(const char *path, const int64_t chunk_shape[3]);

void gridstore_free(gridstore *g);

// Write `len` raw bytes for the chunk at `coords[3]` (z,y,x).
// Overwrites any existing chunk. Returns false on I/O error.
bool gridstore_write(gridstore *g, const int64_t *coords, const void *data, size_t len);

// Read the chunk at `coords[3]`. Returns malloc'd buffer; caller must free.
// *out_len is set to the byte count. Returns NULL if chunk absent or error.
uint8_t *gridstore_read(const gridstore *g, const int64_t *coords, size_t *out_len);

// Return true if the chunk file for `coords[3]` exists on disk.
bool gridstore_exists(const gridstore *g, const int64_t *coords);

// Return the total number of chunk files currently stored.
int gridstore_count(const gridstore *g);
