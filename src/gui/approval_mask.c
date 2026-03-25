#include "approval_mask.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Undo snapshot: per-cell {flat_index, old_value, new_value}
// ---------------------------------------------------------------------------

#define UNDO_MAX 1000

typedef struct {
  int     *indices;
  uint8_t *old_vals;
  uint8_t *new_vals;
  int      count;
} undo_snap;

static void snap_free(undo_snap *s) {
  free(s->indices);  free(s->old_vals);  free(s->new_vals);
  s->indices = NULL; s->old_vals = NULL; s->new_vals = NULL;
  s->count = 0;
}

// ---------------------------------------------------------------------------
// approval_mask struct
// ---------------------------------------------------------------------------

struct approval_mask {
  int      rows, cols;
  uint8_t *grid;          // rows*cols: 0=unpainted 1=approved 2=rejected

  undo_snap ring[UNDO_MAX];
  int head;   // slot where next snapshot will be written
  int count;  // valid undo snapshots behind head
  int redo;   // valid redo snapshots ahead of head
};

approval_mask *approval_mask_new(int rows, int cols) {
  assert(rows > 0 && cols > 0);
  approval_mask *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  m->rows = rows;
  m->cols = cols;
  m->grid = calloc((size_t)(rows * cols), 1);
  if (!m->grid) { free(m); return NULL; }
  return m;
}

void approval_mask_free(approval_mask *m) {
  if (!m) return;
  free(m->grid);
  for (int i = 0; i < UNDO_MAX; i++) snap_free(&m->ring[i]);
  free(m);
}

// ---------------------------------------------------------------------------
// Scratch buffer for collecting changed cells during a single paint op
// ---------------------------------------------------------------------------

typedef struct {
  int     *indices;
  uint8_t *old_vals;
  uint8_t *new_vals;
  int      count, cap;
} cbuf;

static bool cbuf_push(cbuf *b, int idx, uint8_t old_val, uint8_t new_val) {
  if (b->count == b->cap) {
    int nc = b->cap ? b->cap * 2 : 64;
    int     *ni = realloc(b->indices,  (size_t)nc * sizeof(int));
    uint8_t *no = realloc(b->old_vals, (size_t)nc);
    uint8_t *nn = realloc(b->new_vals, (size_t)nc);
    if (!ni || !no || !nn) { free(ni); free(no); free(nn); return false; }
    b->indices  = ni;
    b->old_vals = no;
    b->new_vals = nn;
    b->cap = nc;
  }
  b->indices[b->count]  = idx;
  b->old_vals[b->count] = old_val;
  b->new_vals[b->count] = new_val;
  b->count++;
  return true;
}

// Commit cbuf to ring; transfers ownership of arrays into the snapshot.
static void push_snapshot(approval_mask *m, cbuf *b) {
  if (b->count == 0) return;

  // Discard any redo history beyond current position
  for (int i = 0; i < m->redo; i++) {
    snap_free(&m->ring[(m->head + i) % UNDO_MAX]);
  }
  m->redo = 0;

  // If ring is full, drop oldest entry
  if (m->count == UNDO_MAX) {
    int oldest = (m->head - m->count + UNDO_MAX) % UNDO_MAX;
    snap_free(&m->ring[oldest]);
    m->count--;
  }

  undo_snap *s = &m->ring[m->head];
  snap_free(s);
  s->indices  = b->indices;
  s->old_vals = b->old_vals;
  s->new_vals = b->new_vals;
  s->count    = b->count;

  b->indices = NULL; b->old_vals = NULL; b->new_vals = NULL;
  b->count = 0; b->cap = 0;

  m->head = (m->head + 1) % UNDO_MAX;
  m->count++;
}

// ---------------------------------------------------------------------------
// Paint / erase (circular brush in grid coords)
// ---------------------------------------------------------------------------

static void do_paint(approval_mask *m, float u, float v, float radius, uint8_t val) {
  assert(m);
  int ri = (int)ceilf(radius);
  int cr = (int)roundf(v);
  int cc = (int)roundf(u);

  int r0 = cr - ri; if (r0 < 0) r0 = 0;
  int r1 = cr + ri; if (r1 >= m->rows) r1 = m->rows - 1;
  int c0 = cc - ri; if (c0 < 0) c0 = 0;
  int c1 = cc + ri; if (c1 >= m->cols) c1 = m->cols - 1;

  cbuf b = {0};

  for (int r = r0; r <= r1; r++) {
    for (int c = c0; c <= c1; c++) {
      float dr = (float)r - v;
      float dc = (float)c - u;
      if (sqrtf(dr * dr + dc * dc) > radius) continue;

      int idx = r * m->cols + c;
      uint8_t old = m->grid[idx];
      if (old == val) continue;

      cbuf_push(&b, idx, old, val);
      m->grid[idx] = val;
    }
  }

  push_snapshot(m, &b);
  free(b.indices); free(b.old_vals); free(b.new_vals);
}

void approval_mask_paint(approval_mask *m, float u, float v, float radius, bool approved) {
  do_paint(m, u, v, radius, approved ? 1 : 2);
}

void approval_mask_erase(approval_mask *m, float u, float v, float radius) {
  do_paint(m, u, v, radius, 0);
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

bool approval_mask_is_approved(const approval_mask *m, int row, int col) {
  assert(m);
  if (row < 0 || row >= m->rows || col < 0 || col >= m->cols) return false;
  return m->grid[row * m->cols + col] == 1;
}

float approval_mask_coverage(const approval_mask *m) {
  assert(m);
  int total = m->rows * m->cols;
  if (total == 0) return 0.0f;
  int approved = 0;
  for (int i = 0; i < total; i++) {
    if (m->grid[i] == 1) approved++;
  }
  return (float)approved / (float)total;
}

// ---------------------------------------------------------------------------
// Undo / redo
// ---------------------------------------------------------------------------

bool approval_mask_can_undo(const approval_mask *m) { return m && m->count > 0; }
bool approval_mask_can_redo(const approval_mask *m) { return m && m->redo  > 0; }

void approval_mask_undo(approval_mask *m) {
  if (!approval_mask_can_undo(m)) return;
  m->head = (m->head - 1 + UNDO_MAX) % UNDO_MAX;
  m->count--;
  m->redo++;
  undo_snap *s = &m->ring[m->head];
  for (int i = 0; i < s->count; i++)
    m->grid[s->indices[i]] = s->old_vals[i];
}

void approval_mask_redo(approval_mask *m) {
  if (!approval_mask_can_redo(m)) return;
  undo_snap *s = &m->ring[m->head];
  for (int i = 0; i < s->count; i++)
    m->grid[s->indices[i]] = s->new_vals[i];
  m->head = (m->head + 1) % UNDO_MAX;
  m->count++;
  m->redo--;
}

// ---------------------------------------------------------------------------
// Save / load — raw binary: magic(u32) + rows(i32) + cols(i32) + grid bytes
// ---------------------------------------------------------------------------

#define AMASK_MAGIC 0x414D534Bu  // "AMSK"

bool approval_mask_save(const approval_mask *m, const char *path) {
  assert(m && path);
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  uint32_t magic = AMASK_MAGIC;
  int32_t  rows  = (int32_t)m->rows;
  int32_t  cols  = (int32_t)m->cols;
  size_t   n     = (size_t)(m->rows * m->cols);
  bool ok = fwrite(&magic, 4, 1, f) == 1
         && fwrite(&rows,  4, 1, f) == 1
         && fwrite(&cols,  4, 1, f) == 1
         && fwrite(m->grid, 1, n, f) == n;
  fclose(f);
  return ok;
}

approval_mask *approval_mask_load(const char *path) {
  assert(path);
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  uint32_t magic = 0;
  int32_t  rows = 0, cols = 0;
  if (fread(&magic, 4, 1, f) != 1 || magic != AMASK_MAGIC ||
      fread(&rows,  4, 1, f) != 1 || rows  <= 0 ||
      fread(&cols,  4, 1, f) != 1 || cols  <= 0) {
    fclose(f); return NULL;
  }
  approval_mask *m = approval_mask_new((int)rows, (int)cols);
  if (!m) { fclose(f); return NULL; }
  size_t n = (size_t)(rows * cols);
  if (fread(m->grid, 1, n, f) != n) {
    approval_mask_free(m); fclose(f); return NULL;
  }
  fclose(f);
  return m;
}

// ---------------------------------------------------------------------------
// Overlay: approved=green, rejected=red, unpainted=transparent (RGBA8)
// Nearest-neighbour scale from grid dimensions to (width x height).
// ---------------------------------------------------------------------------

void approval_mask_to_overlay(const approval_mask *m, uint8_t *rgba, int width, int height) {
  assert(m && rgba && width > 0 && height > 0);
  for (int py = 0; py < height; py++) {
    int row = (m->rows > 1)
      ? (int)((float)py / (float)(height - 1) * (float)(m->rows - 1) + 0.5f)
      : 0;
    if (row < 0) row = 0;
    if (row >= m->rows) row = m->rows - 1;
    for (int px = 0; px < width; px++) {
      int col = (m->cols > 1)
        ? (int)((float)px / (float)(width - 1) * (float)(m->cols - 1) + 0.5f)
        : 0;
      if (col < 0) col = 0;
      if (col >= m->cols) col = m->cols - 1;
      uint8_t *p = rgba + (py * width + px) * 4;
      uint8_t  v = m->grid[row * m->cols + col];
      if (v == 1) {
        p[0] = 0;   p[1] = 200; p[2] = 0;   p[3] = 180;
      } else if (v == 2) {
        p[0] = 200; p[1] = 0;   p[2] = 0;   p[3] = 180;
      } else {
        p[0] = 0;   p[1] = 0;   p[2] = 0;   p[3] = 0;
      }
    }
  }
}
