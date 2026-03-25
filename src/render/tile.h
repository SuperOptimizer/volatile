#pragma once
#include <stdint.h>
#include <stdbool.h>

#define TILE_PX 256  // pixels per tile (width and height)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef struct {
  int      col, row;       // tile grid position
  int      pyramid_level;  // LOD level (0 = full res)
  uint64_t epoch;          // camera epoch when submitted
} tile_key;

typedef struct {
  tile_key key;
  uint8_t *pixels;  // TILE_PX * TILE_PX * 4 bytes, RGBA, caller must free
  bool     valid;
} tile_result;

typedef struct tile_renderer tile_renderer;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create a renderer backed by a thread pool of num_threads workers (0 = auto).
tile_renderer *tile_renderer_new(int num_threads);
void           tile_renderer_free(tile_renderer *r);

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

// Submit a tile for background rendering (non-blocking).
// Ignores duplicate keys already pending for the same epoch.
void tile_renderer_submit(tile_renderer *r, tile_key key);

// Drain completed tiles into out[0..max_results-1].
// Caller owns tile_result.pixels (must free each non-NULL pointer).
// Returns number of results written. Call from main thread at ~30 Hz.
int tile_renderer_drain(tile_renderer *r, tile_result *out, int max_results);

// Cancel all pending tiles whose epoch < min_epoch.
void tile_renderer_cancel_stale(tile_renderer *r, uint64_t min_epoch);

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

int tile_renderer_pending(const tile_renderer *r);
