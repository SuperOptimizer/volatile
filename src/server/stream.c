#define _POSIX_C_SOURCE 200809L

#include "server/stream.h"
#include "core/vol.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct chunk_streamer {
  volume  *vol;

  // Region in level-0 voxel space
  int64_t z0, y0, x0;
  int64_t z1, y1, x1;

  int num_levels;      // vol_num_levels(vol)
  int cur_level;       // current pyramid level (starts at num_levels-1)

  // Chunk-grid cursor within the current level
  int64_t cz, cy, cx;                 // current chunk coords
  int64_t cz_max, cy_max, cx_max;     // inclusive upper bounds for current level
  int64_t cz_min, cy_min, cx_min;     // inclusive lower bounds for current level

  bool done;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Given level-0 voxel coordinate v, return the corresponding voxel coordinate
// at pyramid level lvl.  Each level is 2x downsampled in each axis.
static int64_t scale_down(int64_t v, int lvl) {
  for (int i = 0; i < lvl; i++) v = (v + 1) / 2;
  return v;
}

// Compute chunk-grid bounds for the requested region at a given level.
static void compute_chunk_bounds(const chunk_streamer *s, int level,
                                  int64_t *cz0, int64_t *cy0, int64_t *cx0,
                                  int64_t *cz1, int64_t *cy1, int64_t *cx1) {
  const zarr_level_meta *m = vol_level_meta(s->vol, level);

  // Region in this level's voxel space
  int64_t lz0 = scale_down(s->z0, level);
  int64_t ly0 = scale_down(s->y0, level);
  int64_t lx0 = scale_down(s->x0, level);
  int64_t lz1 = scale_down(s->z1 - 1, level);   // inclusive end voxel
  int64_t ly1 = scale_down(s->y1 - 1, level);
  int64_t lx1 = scale_down(s->x1 - 1, level);

  // Clamp to volume shape
  int64_t vol_z = m->shape[0], vol_y = m->shape[1], vol_x = m->shape[2];
  if (lz0 < 0) lz0 = 0;
  if (ly0 < 0) ly0 = 0;
  if (lx0 < 0) lx0 = 0;
  if (lz1 >= vol_z) lz1 = vol_z - 1;
  if (ly1 >= vol_y) ly1 = vol_y - 1;
  if (lx1 >= vol_x) lx1 = vol_x - 1;

  int64_t csz = m->chunk_shape[0];
  int64_t csy = m->chunk_shape[1];
  int64_t csx = m->chunk_shape[2];

  *cz0 = lz0 / csz;  *cz1 = lz1 / csz;
  *cy0 = ly0 / csy;  *cy1 = ly1 / csy;
  *cx0 = lx0 / csx;  *cx1 = lx1 / csx;
}

// Advance the level cursor to the next level that has at least one chunk in
// the region.  Returns false if no more levels.
static bool advance_to_next_level(chunk_streamer *s) {
  while (s->cur_level >= 0) {
    int64_t cz0, cy0, cx0, cz1, cy1, cx1;
    compute_chunk_bounds(s, s->cur_level, &cz0, &cy0, &cx0, &cz1, &cy1, &cx1);

    if (cz0 <= cz1 && cy0 <= cy1 && cx0 <= cx1) {
      s->cz_min = cz0; s->cy_min = cy0; s->cx_min = cx0;
      s->cz_max = cz1; s->cy_max = cy1; s->cx_max = cx1;
      s->cz = cz0; s->cy = cy0; s->cx = cx0;
      return true;
    }
    s->cur_level--;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

chunk_streamer *chunk_streamer_new(volume *vol,
                                   int64_t z0, int64_t y0, int64_t x0,
                                   int64_t z1, int64_t y1, int64_t x1) {
  if (!vol) return NULL;

  int nlvl = vol_num_levels(vol);
  if (nlvl <= 0) return NULL;

  chunk_streamer *s = calloc(1, sizeof(*s));
  if (!s) return NULL;

  s->vol        = vol;
  s->z0 = z0; s->y0 = y0; s->x0 = x0;
  s->z1 = z1; s->y1 = y1; s->x1 = x1;
  s->num_levels = nlvl;
  s->cur_level  = nlvl - 1;   // start at coarsest
  s->done       = false;

  if (!advance_to_next_level(s)) s->done = true;
  return s;
}

void chunk_streamer_free(chunk_streamer *s) {
  free(s);
}

int chunk_streamer_num_levels(const chunk_streamer *s) {
  return s ? s->num_levels : 0;
}

int chunk_streamer_current_level(const chunk_streamer *s) {
  return s ? s->cur_level : -1;
}

bool chunk_streamer_next(chunk_streamer *s, stream_packet *out) {
  if (!s || s->done || !out) return false;

  // Read the current chunk from the volume
  int64_t coords[3] = { s->cz, s->cy, s->cx };
  size_t  raw_size  = 0;
  uint8_t *raw = vol_read_chunk(s->vol, s->cur_level, coords, &raw_size);

  // vol_read_chunk returns NULL for missing chunks; emit a zero-byte sentinel
  // so the client knows this chunk is absent (sparse volumes).
  if (!raw) {
    raw      = malloc(1);   // non-NULL sentinel
    raw_size = 0;
    if (!raw) { s->done = true; return false; }
  }

  out->data  = raw;
  out->size  = raw_size;
  out->level = s->cur_level;

  // Advance cursor: X first, then Y, then Z
  bool level_done = false;
  s->cx++;
  if (s->cx > s->cx_max) {
    s->cx = s->cx_min;
    s->cy++;
    if (s->cy > s->cy_max) {
      s->cy = s->cy_min;
      s->cz++;
      if (s->cz > s->cz_max) {
        level_done = true;
      }
    }
  }

  out->is_last_for_level = level_done;

  if (level_done) {
    s->cur_level--;
    if (!advance_to_next_level(s)) s->done = true;
  }

  return true;
}
