#include "viewer.h"
#include "core/vol.h"
#include "render/cmap.h"
#include "render/overlay.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Viewer state
// ---------------------------------------------------------------------------

// Desired level tracked per-tile in the scene so we know when a slot has a
// coarser placeholder vs the final render.
#define TILE_META_MAX 1024

typedef struct {
  int      col, row;
  int8_t   current_level; // level of pixels currently displayed (-1 = empty)
  uint64_t epoch;         // epoch when slot was last updated
} tile_meta;

struct slice_viewer {
  viewer_config        cfg;
  tile_renderer       *renderer;
  volume              *vol;
  const overlay_list  *overlays;

  // Slice cache: LRU pixmap store keyed by (col, row, scale_q, z_off_q, level)
  slice_cache         *scache;

  // Per-tile display metadata (for progressive-refinement tracking)
  tile_meta            meta[TILE_META_MAX];
  int                  meta_count;

  // Zoom-settle countdown.  When user zooms, we set zoom_settle_ticks to
  // ZOOM_SETTLE_TICKS and count down on each viewer_tick() call.  When it
  // hits 0 we trigger a full-res re-render at the settled zoom level.
  int                  zoom_settle_ticks; // > 0 while settling
  bool                 zoom_dirty;        // zoom changed since last settle
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

slice_viewer *viewer_new(viewer_config cfg, tile_renderer *renderer) {
  slice_viewer *v = calloc(1, sizeof(slice_viewer));
  if (!v) return NULL;
  v->cfg = cfg;
  v->renderer = renderer;
  v->scache = slice_cache_new(512);  // 512 tiles ≈ ~128 MB at 256px RGBA
  if (!v->scache) { free(v); return NULL; }
  if (v->cfg.camera.scale == 0.0f) camera_init(&v->cfg.camera);
  return v;
}

void viewer_free(slice_viewer *v) {
  if (!v) return;
  slice_cache_free(v->scache);
  free(v);
}

// ---------------------------------------------------------------------------
// Camera wrappers
// ---------------------------------------------------------------------------

void viewer_pan(slice_viewer *v, float dx, float dy) {
  camera_pan(&v->cfg.camera, dx, dy);
}

void viewer_zoom(slice_viewer *v, float factor, float cx, float cy) {
  viewport vp = { .screen_w = (int)(cx * 2.0f), .screen_h = (int)(cy * 2.0f), .tile_size = TILE_PX };
  camera_zoom(&v->cfg.camera, &vp, factor, cx, cy);
  // Arm zoom-settle: arm again even if already armed so rapid zooming
  // doesn't trigger full-res until the user pauses.
  v->zoom_settle_ticks = ZOOM_SETTLE_TICKS;
  v->zoom_dirty = true;
}

void viewer_scroll_slice(slice_viewer *v, float delta) {
  camera_step_z(&v->cfg.camera, delta);
}

void viewer_set_volume(slice_viewer *v, volume *vol) {
  v->vol = vol;
  // New volume: clear slice cache so stale images don't persist.
  slice_cache_clear(v->scache);
}

void viewer_set_overlays(slice_viewer *v, const overlay_list *overlays) {
  v->overlays = overlays;
}

// ---------------------------------------------------------------------------
// Tile metadata helpers
// ---------------------------------------------------------------------------

static tile_meta *meta_find(slice_viewer *v, int col, int row) {
  for (int i = 0; i < v->meta_count; i++) {
    if (v->meta[i].col == col && v->meta[i].row == row) return &v->meta[i];
  }
  return NULL;
}

static tile_meta *meta_get_or_create(slice_viewer *v, int col, int row) {
  tile_meta *m = meta_find(v, col, row);
  if (m) return m;
  if (v->meta_count < TILE_META_MAX) {
    m = &v->meta[v->meta_count++];
    m->col           = col;
    m->row           = row;
    m->current_level = -1;
    m->epoch         = 0;
    return m;
  }
  // Array full — evict slot 0 (simple FIFO; meta is only used for refinement
  // decisions so eviction just causes one extra re-submit, which is fine).
  memmove(&v->meta[0], &v->meta[1], (size_t)(TILE_META_MAX - 1) * sizeof(tile_meta));
  m = &v->meta[TILE_META_MAX - 1];
  m->col           = col;
  m->row           = row;
  m->current_level = -1;
  m->epoch         = 0;
  return m;
}

// ---------------------------------------------------------------------------
// params_hash — cheap hash of render params for cache key
// ---------------------------------------------------------------------------

static uint32_t params_hash(const slice_viewer *v) {
  // Mix window, level, cmap_id, view_axis into a 32-bit hash.
  uint32_t h = (uint32_t)v->cfg.cmap_id * 2654435761u;
  h ^= (uint32_t)(v->cfg.view_axis + 1) * 0x9e3779b9u;
  // Quantise window/level to 1/16 for the hash (avoids floating-point equality issues)
  h ^= (uint32_t)(int)(v->cfg.window * 16.0f) * 0x6c62272eu;
  h ^= (uint32_t)(int)(v->cfg.level  * 16.0f) * 0x1b873593u;
  return h;
}

// ---------------------------------------------------------------------------
// Synchronous tile rasterization from volume
// ---------------------------------------------------------------------------

static void rasterize_tile(slice_viewer *v, tile_key key,
                            float tile_u0, float tile_v0, float su_per_px,
                            uint8_t *out) {
  if (!v->vol) {
    memset(out, 128, (size_t)TILE_PX * TILE_PX * 4);
    for (int i = 3; i < TILE_PX * TILE_PX * 4; i += 4) out[i] = 255;
    return;
  }

  int level  = key.pyramid_level;
  float z    = v->cfg.camera.z_offset;

  int64_t shape[8] = {0};
  vol_shape(v->vol, level, shape);

  float W = (float)shape[2];
  float H = (float)shape[1];
  float D = (float)shape[0];

  float win = v->cfg.window;
  float lvl = v->cfg.level;
  float lo  = lvl - win * 0.5f;
  float inv = (win > 0.0f) ? (1.0f / win) : 0.0f;

  for (int py = 0; py < TILE_PX; py++) {
    float v_surf = tile_v0 + (float)py * su_per_px;
    for (int px = 0; px < TILE_PX; px++) {
      float u_surf = tile_u0 + (float)px * su_per_px;

      float sz, sy, sx;
      switch (v->cfg.view_axis) {
        case 1:  sz = v_surf * D; sy = z;         sx = u_surf * W; break;
        case 2:  sz = v_surf * D; sy = u_surf * H; sx = z;         break;
        default: sz = z;         sy = v_surf * H; sx = u_surf * W; break;
      }

      float sample = vol_sample(v->vol, level, sz, sy, sx);

      float t = (sample - lo) * inv;
      if (t < 0.0f) t = 0.0f;
      else if (t > 1.0f) t = 1.0f;

      cmap_rgb rgb = cmap_apply((cmap_id)v->cfg.cmap_id, (double)t);

      int idx = (py * TILE_PX + px) * 4;
      out[idx + 0] = rgb.r;
      out[idx + 1] = rgb.g;
      out[idx + 2] = rgb.b;
      out[idx + 3] = 255;
    }
  }
}

// ---------------------------------------------------------------------------
// Submit a tile for async render (coarser fallback first)
//
// We submit two tile_key requests:
//   1. coarse (level+1 or level+2) — high priority, renders fast as placeholder
//   2. fine   (level)              — lower priority, delivers final quality
// The renderer's FIFO order means coarser tiles will generally complete first.
// ---------------------------------------------------------------------------

static void submit_with_coarse_fallback(slice_viewer *v, tile_key key) {
  if (!v->renderer) return;
  int max_level = v->vol ? vol_num_levels(v->vol) - 1 : 3;

  // Submit coarsest-first so placeholders appear quickly.
  for (int delta = 2; delta >= 0; delta--) {
    int lvl = key.pyramid_level + delta;
    if (lvl > max_level) continue;
    tile_key k = key;
    k.pyramid_level = lvl;
    tile_renderer_submit(v->renderer, k);
  }
}

// ---------------------------------------------------------------------------
// viewer_render
// ---------------------------------------------------------------------------

void viewer_render(slice_viewer *v, uint8_t *pixels, int width, int height) {
  if (!pixels || width <= 0 || height <= 0) return;

  viewer_camera *cam = &v->cfg.camera;
  uint32_t ph = params_hash(v);
  int desired_level = (v->vol)
    ? camera_calc_pyramid_level(cam, vol_num_levels(v->vol))
    : 0;

  // --- 1. Drain completed async tiles into slice_cache ---
  if (v->renderer) {
    tile_result results[32];
    int n = tile_renderer_drain(v->renderer, results, 32);
    for (int i = 0; i < n; i++) {
      if (!results[i].valid) { free(results[i].pixels); continue; }
      tile_key rk = results[i].key;
      // Discard stale epochs.
      if (rk.epoch < cam->epoch) { free(results[i].pixels); continue; }

      slice_cache_key ck = slice_cache_make_key(rk.col, rk.row, cam->scale,
                                                 cam->z_offset, rk.pyramid_level, ph);
      // slice_cache_put takes ownership of pixels.
      slice_cache_put(v->scache, ck, results[i].pixels, (int8_t)rk.pyramid_level);

      // Update per-tile metadata: accept if finer than what's shown.
      tile_meta *m = meta_get_or_create(v, rk.col, rk.row);
      if (rk.epoch >= m->epoch &&
          (m->current_level < 0 || rk.pyramid_level <= m->current_level)) {
        m->current_level = (int8_t)rk.pyramid_level;
        m->epoch         = rk.epoch;
      }
    }
  }

  // --- 2. Compute tile grid (visible + prefetch buffer) ---
  float su_per_px = (cam->scale > 0.0f) ? (1.0f / cam->scale) : 1.0f;
  float u0 = cam->center.x - (float)(width  / 2) * su_per_px;
  float v0 = cam->center.y - (float)(height / 2) * su_per_px;
  float tile_su = (float)TILE_PX * su_per_px;

  int col0 = (int)floorf(u0 / tile_su) - TILE_PREFETCH_BUFFER;
  int row0 = (int)floorf(v0 / tile_su) - TILE_PREFETCH_BUFFER;
  int col1 = (int)floorf((u0 + (float)width  * su_per_px) / tile_su) + TILE_PREFETCH_BUFFER;
  int row1 = (int)floorf((v0 + (float)height * su_per_px) / tile_su) + TILE_PREFETCH_BUFFER;

  memset(pixels, 0, (size_t)width * height * 4);

  for (int row = row0; row <= row1; row++) {
    for (int col = col0; col <= col1; col++) {
      float tile_u0 = (float)col * tile_su;
      float tile_v0 = (float)row * tile_su;
      float scr_x   = (tile_u0 - u0) / su_per_px;
      float scr_y   = (tile_v0 - v0) / su_per_px;
      int   px0     = (int)floorf(scr_x);
      int   py0     = (int)floorf(scr_y);

      // Only blit visible tiles (prefetch tiles are submitted but not blitted)
      bool visible = (px0 < width && py0 < height &&
                      px0 + TILE_PX > 0 && py0 + TILE_PX > 0);

      // --- 3. Slice-cache lookup (exact, then coarser fallback) ---
      slice_cache_key ck = slice_cache_make_key(col, row, cam->scale,
                                                 cam->z_offset, desired_level, ph);
      slice_cache_entry best = {0};
      bool has_cached = slice_cache_get_best(v->scache, ck, &best);

      // Determine if we need to submit a (re-)render.
      // Submit when: no cache entry, OR cached level is coarser than desired.
      tile_meta *m = meta_get_or_create(v, col, row);
      bool need_fine = !has_cached ||
                       best.level > desired_level ||
                       m->epoch < cam->epoch;

      if (need_fine) {
        tile_key key = {
          .col           = col,
          .row           = row,
          .pyramid_level = desired_level,
          .epoch         = cam->epoch,
        };
        submit_with_coarse_fallback(v, key);
        if (v->renderer)
          tile_renderer_cancel_stale(v->renderer, cam->epoch);
      }

      if (!visible) continue;

      const uint8_t *src_pixels = NULL;

      if (has_cached && best.pixels) {
        // Use cached pixels (may be coarser-than-desired placeholder).
        src_pixels = best.pixels;
      } else {
        // --- 4. Synchronous fallback: rasterize immediately ---
        // Only for visible tiles with no cached data at all.
        uint8_t *tile_px = malloc((size_t)TILE_PX * TILE_PX * 4);
        if (!tile_px) continue;
        tile_key key = {
          .col           = col,
          .row           = row,
          .pyramid_level = desired_level,
          .epoch         = cam->epoch,
        };
        rasterize_tile(v, key, tile_u0, tile_v0, su_per_px, tile_px);

        // Store in slice-cache (owns the memory now).
        slice_cache_put(v->scache, ck, tile_px, (int8_t)desired_level);
        // Get back the pointer we just inserted.
        slice_cache_entry inserted = {0};
        slice_cache_get_best(v->scache, ck, &inserted);
        src_pixels = inserted.pixels;
        if (!src_pixels) { continue; }  // shouldn't happen
      }

      // --- 5. Blit tile into output buffer ---
      for (int ty = 0; ty < TILE_PX; ty++) {
        int sy = py0 + ty;
        if (sy < 0 || sy >= height) continue;
        for (int tx = 0; tx < TILE_PX; tx++) {
          int sx = px0 + tx;
          if (sx < 0 || sx >= width) continue;
          int src = (ty * TILE_PX + tx) * 4;
          int dst = (sy * width + sx) * 4;
          pixels[dst + 0] = src_pixels[src + 0];
          pixels[dst + 1] = src_pixels[src + 1];
          pixels[dst + 2] = src_pixels[src + 2];
          pixels[dst + 3] = src_pixels[src + 3];
        }
      }
    }
  }

  // Draw overlays on top.
  if (v->overlays) {
    overlay_render(v->overlays, pixels, width, height);
  }
}

// ---------------------------------------------------------------------------
// viewer_tick — call at ~30 Hz to drive zoom-settle and progressive refinement
//
// Returns true if more work is expected (caller should keep ticking).
// ---------------------------------------------------------------------------

bool viewer_tick(slice_viewer *v) {
  if (!v) return false;
  bool more_work = false;

  // Zoom-settle countdown.
  if (v->zoom_settle_ticks > 0) {
    v->zoom_settle_ticks--;
    more_work = true;
    if (v->zoom_settle_ticks == 0 && v->zoom_dirty) {
      v->zoom_dirty = false;
      // Settled: bump epoch to trigger full-res re-render at the settled scale.
      camera_invalidate(&v->cfg.camera);
      slice_cache_clear(v->scache);  // old-scale tiles are no longer useful
    }
  }

  // Progressive refinement: re-submit tiles that have a coarser placeholder
  // but could be refined now.
  if (v->renderer) {
    more_work = more_work || (tile_renderer_pending(v->renderer) > 0);
  }

  return more_work;
}

// ---------------------------------------------------------------------------
// Coordinate transform
// ---------------------------------------------------------------------------

vec3f viewer_screen_to_world(const slice_viewer *v, float sx, float sy) {
  viewer_camera *cam = (viewer_camera *)&v->cfg.camera;
  float su_per_px = (cam->scale > 0.0f) ? (1.0f / cam->scale) : 1.0f;

  float u  = cam->center.x + (sx - 0.0f) * su_per_px;
  float vv = cam->center.y + (sy - 0.0f) * su_per_px;
  float z  = cam->z_offset;

  switch (v->cfg.view_axis) {
    case 1: return (vec3f){ u, z, vv };
    case 2: return (vec3f){ z, u, vv };
    default: return (vec3f){ u, vv, z };
  }
}

float viewer_current_slice(const slice_viewer *v) {
  return v->cfg.camera.z_offset;
}

int viewer_current_level(const slice_viewer *v) {
  if (!v->vol) return 0;
  return camera_calc_pyramid_level(&v->cfg.camera, vol_num_levels(v->vol));
}

int viewer_get_axis(const slice_viewer *v) {
  return v->cfg.view_axis;
}
