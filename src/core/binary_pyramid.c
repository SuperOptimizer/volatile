#include "core/binary_pyramid.h"
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Each pyramid level stores a bit-packed 3D array of size
// ceil(d/2^l) × ceil(h/2^l) × ceil(w/2^l).
// We use uint8_t[] with 1 bit per voxel, packed in x-major order.
// ---------------------------------------------------------------------------

#define MAX_LEVELS 16

typedef struct {
  int     d, h, w;    // dimensions at this level
  uint8_t *bits;      // bit array: index = z*h*w + y*w + x; size = ceil(d*h*w/8)
  size_t  nbytes;
} pyr_level;

struct binary_pyramid {
  int       d0, h0, w0;    // level-0 dimensions
  int       n_levels;
  pyr_level levels[MAX_LEVELS];
};

// ---------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------

static inline void bit_set(uint8_t *bits, int idx, bool v) {
  int byte = idx >> 3, bit = idx & 7;
  if (v) bits[byte] |=  (uint8_t)(1u << bit);
  else   bits[byte] &= ~(uint8_t)(1u << bit);
}

static inline bool bit_get(const uint8_t *bits, int idx) {
  return (bits[idx >> 3] >> (idx & 7)) & 1u;
}

static inline int clamp0(int v) { return v < 0 ? 0 : v; }
static inline int clamph(int v, int hi) { return v >= hi ? hi - 1 : v; }

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

binary_pyramid *binary_pyramid_new(int d, int h, int w) {
  if (d < 1 || h < 1 || w < 1) return NULL;
  binary_pyramid *p = calloc(1, sizeof(*p));
  if (!p) return NULL;

  p->d0 = d; p->h0 = h; p->w0 = w;
  int ld = d, lh = h, lw = w;
  for (int l = 0; l < MAX_LEVELS; l++) {
    p->levels[l].d = ld; p->levels[l].h = lh; p->levels[l].w = lw;
    size_t n = (size_t)ld * (size_t)lh * (size_t)lw;
    p->levels[l].nbytes = (n + 7) / 8;
    p->levels[l].bits   = calloc(p->levels[l].nbytes, 1);
    if (!p->levels[l].bits) { binary_pyramid_free(p); return NULL; }
    p->n_levels = l + 1;
    if (ld == 1 && lh == 1 && lw == 1) break;
    ld = (ld + 1) / 2;
    lh = (lh + 1) / 2;
    lw = (lw + 1) / 2;
  }
  return p;
}

void binary_pyramid_free(binary_pyramid *p) {
  if (!p) return;
  for (int l = 0; l < p->n_levels; l++) free(p->levels[l].bits);
  free(p);
}

// ---------------------------------------------------------------------------
// Set / Get (level 0 only; then propagate up)
// ---------------------------------------------------------------------------

void binary_pyramid_set(binary_pyramid *p, int z, int y, int x, bool value) {
  if (!p) return;
  if (z < 0 || z >= p->d0 || y < 0 || y >= p->h0 || x < 0 || x >= p->w0) return;

  // set at level 0
  pyr_level *l0 = &p->levels[0];
  int idx0 = z * l0->h * l0->w + y * l0->w + x;
  bool old = bit_get(l0->bits, idx0);
  if (old == value) return;  // no change, pyramid already consistent
  bit_set(l0->bits, idx0, value);

  // propagate upward: parent cell at level lv+1 is OR of its 8 children
  int cz = z, cy = y, cx = x;
  for (int lv = 0; lv + 1 < p->n_levels; lv++) {
    int pz = cz / 2, py = cy / 2, px = cx / 2;
    pyr_level *par = &p->levels[lv + 1];
    int pidx = pz * par->h * par->w + py * par->w + px;
    // compute OR of all 8 children
    pyr_level *ch = &p->levels[lv];
    bool any = false;
    for (int dz = 0; dz <= 1 && !any; dz++)
      for (int dy = 0; dy <= 1 && !any; dy++)
        for (int dx = 0; dx <= 1 && !any; dx++) {
          int ciz = pz*2+dz, ciy = py*2+dy, cix = px*2+dx;
          if (ciz >= ch->d || ciy >= ch->h || cix >= ch->w) continue;
          int cidx = ciz * ch->h * ch->w + ciy * ch->w + cix;
          if (bit_get(ch->bits, cidx)) any = true;
        }
    bool prev_par = bit_get(par->bits, pidx);
    if (prev_par == any) break;  // no change at this level → stop
    bit_set(par->bits, pidx, any);
    cz = pz; cy = py; cx = px;
  }
}

bool binary_pyramid_get(const binary_pyramid *p, int z, int y, int x) {
  if (!p) return false;
  if (z < 0 || z >= p->d0 || y < 0 || y >= p->h0 || x < 0 || x >= p->w0) return false;
  const pyr_level *l0 = &p->levels[0];
  int idx = z * l0->h * l0->w + y * l0->w + x;
  return bit_get(l0->bits, idx);
}

// ---------------------------------------------------------------------------
// any_in_region — use pyramid to skip empty regions quickly
// ---------------------------------------------------------------------------

// Recursive helper: check level lv for parent cell (pz,py,px) covering the
// region [rz0,rz1) × [ry0,ry1) × [rx0,rx1) at level 0.
// cell_z0/y0/x0: level-0 origin of this cell's coverage.
static bool any_rec(const binary_pyramid *p, int lv,
                    int pz, int py, int px,
                    int rz0, int ry0, int rx0,
                    int rz1, int ry1, int rx1) {
  const pyr_level *pl = &p->levels[lv];
  int idx = pz * pl->h * pl->w + py * pl->w + px;
  if (!bit_get(pl->bits, idx)) return false;  // coarse cell empty → skip
  if (lv == 0) return true;                   // leaf hit

  // Check children
  for (int dz = 0; dz <= 1; dz++) {
    int cz = pz * 2 + dz;
    if (cz >= p->levels[lv-1].d) continue;
    for (int dy = 0; dy <= 1; dy++) {
      int cy = py * 2 + dy;
      if (cy >= p->levels[lv-1].h) continue;
      for (int dx = 0; dx <= 1; dx++) {
        int cx = px * 2 + dx;
        if (cx >= p->levels[lv-1].w) continue;
        // child covers level-0 region [cz<<(lv-1), ...) etc.
        // quick overlap: child cell at level lv-1 covers 2^(lv-1) voxels each dim
        int scale = 1 << (lv - 1);
        int cz0 = cz * scale, cy0 = cy * scale, cx0 = cx * scale;
        int cz1 = cz0 + scale, cy1 = cy0 + scale, cx1 = cx0 + scale;
        if (cz1 <= rz0 || cz0 >= rz1) continue;
        if (cy1 <= ry0 || cy0 >= ry1) continue;
        if (cx1 <= rx0 || cx0 >= rx1) continue;
        if (any_rec(p, lv-1, cz, cy, cx, rz0, ry0, rx0, rz1, ry1, rx1))
          return true;
      }
    }
  }
  return false;
}

bool binary_pyramid_any_in_region(const binary_pyramid *p,
                                  int z0, int y0, int x0,
                                  int z1, int y1, int x1) {
  if (!p) return false;
  z0 = clamp0(z0); y0 = clamp0(y0); x0 = clamp0(x0);
  z1 = clamph(z1, p->d0 + 1); y1 = clamph(y1, p->h0 + 1); x1 = clamph(x1, p->w0 + 1);
  if (z0 >= z1 || y0 >= y1 || x0 >= x1) return false;

  // walk top level
  int top = p->n_levels - 1;
  const pyr_level *tl = &p->levels[top];
  for (int pz = 0; pz < tl->d; pz++)
    for (int py = 0; py < tl->h; py++)
      for (int px = 0; px < tl->w; px++)
        if (any_rec(p, top, pz, py, px, z0, y0, x0, z1, y1, x1))
          return true;
  return false;
}

// ---------------------------------------------------------------------------
// count — popcount at level 0
// ---------------------------------------------------------------------------

int binary_pyramid_count(const binary_pyramid *p) {
  if (!p) return 0;
  const pyr_level *l0 = &p->levels[0];
  int cnt = 0;
  for (size_t i = 0; i < l0->nbytes; i++) cnt += __builtin_popcount(l0->bits[i]);
  return cnt;
}
