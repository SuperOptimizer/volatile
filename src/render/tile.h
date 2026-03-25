#pragma once
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#define TILE_PX 256  // pixels per tile (width and height)

// Prefetch buffer: 1 tile beyond the visible viewport in each direction.
#define TILE_PREFETCH_BUFFER 1

// Zoom-settle delay: ticks at 30 Hz before triggering full-res re-render.
// 200ms / 33ms ≈ 6 ticks.
#define ZOOM_SETTLE_TICKS 6

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
// priority: higher = rendered sooner. Use pyramid_level as priority so coarser
// tiles (larger level number) render first as progressive placeholders.
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

// ---------------------------------------------------------------------------
// slice_cache — LRU image cache keyed by (col, row, scale_q, z_off_q, level)
//
// Mirrors VC3D's SliceCache.  Keys are quantised so that small camera wobbles
// hit the same bucket.  getBest searches up to MAX_COARSER_LEVELS coarser
// entries to provide a placeholder while fine tiles render.
// ---------------------------------------------------------------------------

#define SLICE_CACHE_MAX_COARSER 8   // how many coarser LODs to try on miss

typedef struct {
  int      col, row;    // tile grid position
  int16_t  scale_q;     // log2(scale) * 32, quantised to 1/32 stops
  int16_t  z_off_q;     // z_offset * 4, quantised to 0.25 units
  int8_t   level;       // pyramid level (0 = finest)
  uint32_t params_hash; // hash of render params (window/level/cmap)
} slice_cache_key;

typedef struct {
  uint8_t *pixels;  // TILE_PX*TILE_PX*4 RGBA, owned by the cache
  int8_t   level;   // actual level of the cached pixels
} slice_cache_entry;

typedef struct slice_cache slice_cache;

// max_entries: maximum number of tiles held in the LRU.
slice_cache *slice_cache_new(int max_entries);
void         slice_cache_free(slice_cache *c);

// Build a key from camera state.
slice_cache_key slice_cache_make_key(int col, int row, float scale, float z_offset,
                                      int level, uint32_t params_hash);

// Exact lookup. Returns pointer to internal entry (valid until next put/evict),
// or NULL on miss.
const slice_cache_entry *slice_cache_get(slice_cache *c, slice_cache_key key);

// Best-available lookup: returns exact match or best coarser fallback.
// out_entry: set to a copy of the found entry (pixels pointer owned by cache).
// Returns true if anything found. out_entry->level reflects actual cached level.
bool slice_cache_get_best(slice_cache *c, slice_cache_key key,
                           slice_cache_entry *out_entry);

// Insert (or replace) a tile. Takes ownership of pixels.
void slice_cache_put(slice_cache *c, slice_cache_key key, uint8_t *pixels, int8_t level);

// Invalidate all entries.
void slice_cache_clear(slice_cache *c);
