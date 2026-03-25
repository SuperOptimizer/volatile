#include "viewer.h"
#include "core/vol.h"
#include "render/cmap.h"
#include "render/overlay.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Tile cache entry
// ---------------------------------------------------------------------------

#define TILE_CACHE_MAX 256

typedef struct {
  tile_key   key;
  uint8_t   *pixels;   // TILE_PX * TILE_PX * 4, RGBA
  bool       valid;
} cached_tile;

// ---------------------------------------------------------------------------
// Viewer state
// ---------------------------------------------------------------------------

struct slice_viewer {
  viewer_config        cfg;
  tile_renderer       *renderer;
  volume              *vol;
  const overlay_list  *overlays;

  // tile cache (simple linear scan; enough for a screenful of tiles)
  cached_tile cache[TILE_CACHE_MAX];
  int         cache_count;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

slice_viewer *viewer_new(viewer_config cfg, tile_renderer *renderer) {
  slice_viewer *v = calloc(1, sizeof(slice_viewer));
  if (!v) return NULL;
  v->cfg = cfg;
  v->renderer = renderer;
  if (v->cfg.camera.scale == 0.0f) camera_init(&v->cfg.camera);
  return v;
}

void viewer_free(slice_viewer *v) {
  if (!v) return;
  for (int i = 0; i < v->cache_count; i++) {
    free(v->cache[i].pixels);
  }
  free(v);
}

// ---------------------------------------------------------------------------
// Camera wrappers
// ---------------------------------------------------------------------------

void viewer_pan(slice_viewer *v, float dx, float dy) {
  camera_pan(&v->cfg.camera, dx, dy);
}

void viewer_zoom(slice_viewer *v, float factor, float cx, float cy) {
  // Build a minimal viewport from the zoom point — caller provides screen coords
  // but we don't store screen dimensions here; use cx/cy as the pivot offset directly.
  viewport vp = { .screen_w = (int)(cx * 2.0f), .screen_h = (int)(cy * 2.0f), .tile_size = TILE_PX };
  camera_zoom(&v->cfg.camera, &vp, factor, cx, cy);
}

void viewer_scroll_slice(slice_viewer *v, float delta) {
  camera_step_z(&v->cfg.camera, delta);
}

void viewer_set_volume(slice_viewer *v, volume *vol) {
  v->vol = vol;
}

void viewer_set_overlays(slice_viewer *v, const overlay_list *overlays) {
  v->overlays = overlays;
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

static bool tile_key_eq(tile_key a, tile_key b) {
  return a.col == b.col && a.row == b.row &&
         a.pyramid_level == b.pyramid_level && a.epoch == b.epoch;
}

static cached_tile *cache_find(slice_viewer *v, tile_key key) {
  for (int i = 0; i < v->cache_count; i++) {
    if (v->cache[i].valid && tile_key_eq(v->cache[i].key, key)) return &v->cache[i];
  }
  return NULL;
}

// Evict least-recently-inserted if cache full, insert at end
static cached_tile *cache_insert(slice_viewer *v, tile_key key) {
  if (v->cache_count < TILE_CACHE_MAX) {
    cached_tile *e = &v->cache[v->cache_count++];
    free(e->pixels);
    e->key   = key;
    e->pixels = NULL;
    e->valid  = false;
    return e;
  }
  // Evict slot 0, shift down
  free(v->cache[0].pixels);
  memmove(&v->cache[0], &v->cache[1], (size_t)(TILE_CACHE_MAX - 1) * sizeof(cached_tile));
  cached_tile *e = &v->cache[TILE_CACHE_MAX - 1];
  e->key   = key;
  e->pixels = NULL;
  e->valid  = false;
  return e;
}

// ---------------------------------------------------------------------------
// Synchronous tile rasterization from volume
// Fills a TILE_PX*TILE_PX*4 RGBA buffer by sampling the volume.
// tile_u0/v0: surface-space origin of this tile; tile_su: surface units per pixel
// ---------------------------------------------------------------------------

static void rasterize_tile(slice_viewer *v, tile_key key,
                            float tile_u0, float tile_v0, float su_per_px,
                            uint8_t *out) {
  if (!v->vol) {
    // No volume: fill with mid-grey
    memset(out, 128, (size_t)TILE_PX * TILE_PX * 4);
    for (int i = 3; i < TILE_PX * TILE_PX * 4; i += 4) out[i] = 255;
    return;
  }

  int level  = key.pyramid_level;
  float z    = v->cfg.camera.z_offset;

  // shape at this level
  int64_t shape[8] = {0};
  vol_shape(v->vol, level, shape);

  // shape layout depends on view_axis:
  // axis=0 (XY): shape ~ [depth, height, width] -> sample(z, v*H, u*W)
  // axis=1 (XZ): shape ~ [depth, height, width] -> sample(v*D, z, u*W)
  // axis=2 (YZ): shape ~ [depth, height, width] -> sample(v*D, u*H, z)
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

      // window/level -> [0,1]
      float t = (sample - lo) * inv;
      if (t < 0.0f) t = 0.0f;
      else if (t > 1.0f) t = 1.0f;

      // colormap
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
// viewer_render
// ---------------------------------------------------------------------------

void viewer_render(slice_viewer *v, uint8_t *pixels, int width, int height) {
  if (!pixels || width <= 0 || height <= 0) return;

  // Drain any completed async tiles into cache
  if (v->renderer) {
    tile_result results[32];
    int n = tile_renderer_drain(v->renderer, results, 32);
    for (int i = 0; i < n; i++) {
      if (!results[i].valid) { free(results[i].pixels); continue; }
      cached_tile *e = cache_find(v, results[i].key);
      if (!e) e = cache_insert(v, results[i].key);
      free(e->pixels);
      e->pixels = results[i].pixels;  // takes ownership
      e->valid  = true;
    }
  }

  // Compute tile grid covering [0,width] x [0,height]
  viewer_camera *cam = &v->cfg.camera;
  int level = (v->vol) ? camera_calc_pyramid_level(cam, vol_num_levels(v->vol)) : 0;

  // surface units per screen pixel = 1/scale
  float su_per_px = (cam->scale > 0.0f) ? (1.0f / cam->scale) : 1.0f;

  // surface coord of screen (0,0): center is at (width/2, height/2)
  float u0 = cam->center.x - (float)(width  / 2) * su_per_px;
  float v0 = cam->center.y - (float)(height / 2) * su_per_px;

  // tile size in surface units
  float tile_su = (float)TILE_PX * su_per_px;

  // first tile grid index
  int col0 = (int)floorf(u0 / tile_su);
  int row0 = (int)floorf(v0 / tile_su);
  int col1 = (int)floorf((u0 + (float)width  * su_per_px) / tile_su);
  int row1 = (int)floorf((v0 + (float)height * su_per_px) / tile_su);

  // clear output
  memset(pixels, 0, (size_t)width * height * 4);

  for (int row = row0; row <= row1; row++) {
    for (int col = col0; col <= col1; col++) {

      tile_key key = { .col = col, .row = row, .pyramid_level = level, .epoch = cam->epoch };

      // surface-space origin of this tile
      float tile_u0 = (float)col * tile_su;
      float tile_v0 = (float)row * tile_su;

      // screen pixel where tile starts
      float scr_x = (tile_u0 - u0) / su_per_px;
      float scr_y = (tile_v0 - v0) / su_per_px;
      int   px0   = (int)floorf(scr_x);
      int   py0   = (int)floorf(scr_y);

      // look for cached tile
      cached_tile *ct = cache_find(v, key);
      if (!ct || !ct->valid) {
        // not cached yet — submit async and rasterize synchronously as fallback
        if (v->renderer) tile_renderer_submit(v->renderer, key);

        // allocate + rasterize
        uint8_t *tile_px = malloc((size_t)TILE_PX * TILE_PX * 4);
        if (!tile_px) continue;
        rasterize_tile(v, key, tile_u0, tile_v0, su_per_px, tile_px);

        // store in cache for future frames
        cached_tile *e = cache_insert(v, key);
        e->pixels = tile_px;
        e->valid  = true;
        ct = e;
      }

      // blit tile into output buffer, clipped to [0,width)x[0,height)
      for (int ty = 0; ty < TILE_PX; ty++) {
        int sy = py0 + ty;
        if (sy < 0 || sy >= height) continue;
        for (int tx = 0; tx < TILE_PX; tx++) {
          int sx = px0 + tx;
          if (sx < 0 || sx >= width) continue;
          int src = (ty * TILE_PX + tx) * 4;
          int dst = (sy * width + sx) * 4;
          pixels[dst + 0] = ct->pixels[src + 0];
          pixels[dst + 1] = ct->pixels[src + 1];
          pixels[dst + 2] = ct->pixels[src + 2];
          pixels[dst + 3] = ct->pixels[src + 3];
        }
      }
    }
  }

  // Draw overlays on top
  if (v->overlays) {
    overlay_render(v->overlays, pixels, width, height);
  }
}

// ---------------------------------------------------------------------------
// Coordinate transform
// ---------------------------------------------------------------------------

vec3f viewer_screen_to_world(const slice_viewer *v, float sx, float sy) {
  viewer_camera *cam = (viewer_camera *)&v->cfg.camera;
  float su_per_px = (cam->scale > 0.0f) ? (1.0f / cam->scale) : 1.0f;

  // surface coords (approximate; no viewport width/height here)
  float u = cam->center.x + (sx - 0.0f) * su_per_px;
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
