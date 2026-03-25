#define _POSIX_C_SOURCE 200809L

#include "cli_inpaint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void inpaint_usage(void) {
  puts("usage: volatile inpaint <image> --mask <mask> --output <out>");
  puts("                        [--radius N]");
  puts("");
  puts("  Fill masked (hole) regions using Telea fast marching inpainting.");
  puts("  image   input image (PPM P6 format, uint8 RGB)");
  puts("  --mask  binary mask image (PGM P5, nonzero = hole to fill)");
  puts("  --radius  influence radius for each border pixel (default: 5)");
}

// ---------------------------------------------------------------------------
// Minimal PPM/PGM I/O
// ---------------------------------------------------------------------------

static uint8_t *read_ppm(const char *path, int *w, int *h, int *ch) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char magic[3] = {0};
  if (fscanf(f, "%2s", magic) != 1) { fclose(f); return NULL; }
  int is_rgb = (strcmp(magic, "P6") == 0);
  int is_gray = (strcmp(magic, "P5") == 0);
  if (!is_rgb && !is_gray) { fclose(f); return NULL; }
  int width, height, maxval;
  if (fscanf(f, " %d %d %d", &width, &height, &maxval) != 3) { fclose(f); return NULL; }
  fgetc(f);  // consume single whitespace after maxval
  int channels = is_rgb ? 3 : 1;
  size_t nbytes = (size_t)width * height * channels;
  uint8_t *data = malloc(nbytes);
  if (!data) { fclose(f); return NULL; }
  if (fread(data, 1, nbytes, f) != nbytes) { free(data); fclose(f); return NULL; }
  fclose(f);
  *w = width; *h = height; *ch = channels;
  return data;
}

static bool write_ppm(const char *path, const uint8_t *data, int w, int h, int ch) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fprintf(f, "%s\n%d %d\n255\n", ch == 3 ? "P6" : "P5", w, h);
  fwrite(data, 1, (size_t)w * (size_t)h * (size_t)ch, f);
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Telea fast-marching inpainting (2D, per-channel)
// ---------------------------------------------------------------------------

// Priority queue node for fast marching (min-heap on distance).
typedef struct { float dist; int idx; } pq_node;

static void pq_push(pq_node *heap, int *size, pq_node node) {
  int i = (*size)++;
  heap[i] = node;
  while (i > 0) {
    int parent = (i - 1) / 2;
    if (heap[parent].dist <= heap[i].dist) break;
    pq_node tmp = heap[parent]; heap[parent] = heap[i]; heap[i] = tmp;
    i = parent;
  }
}

static pq_node pq_pop(pq_node *heap, int *size) {
  pq_node top = heap[0];
  heap[0] = heap[--(*size)];
  int i = 0;
  for (;;) {
    int l = 2*i+1, r = 2*i+2, s = i;
    if (l < *size && heap[l].dist < heap[s].dist) s = l;
    if (r < *size && heap[r].dist < heap[s].dist) s = r;
    if (s == i) break;
    pq_node tmp = heap[s]; heap[s] = heap[i]; heap[i] = tmp;
    i = s;
  }
  return top;
}

// State flags.
#define ST_KNOWN   0
#define ST_BAND    1
#define ST_UNKNOWN 2

static void telea_inpaint(uint8_t *img, const uint8_t *mask,
                          int w, int h, int ch, int radius) {
  size_t n = (size_t)w * (size_t)h;
  uint8_t *state = calloc(n, 1);   // ST_KNOWN / ST_BAND / ST_UNKNOWN
  float   *dist  = malloc(n * sizeof(float));
  pq_node *heap  = malloc(n * sizeof(pq_node));
  if (!state || !dist || !heap) { free(state); free(dist); free(heap); return; }

  int heap_size = 0;

  // Initialise: unknown = masked; band = border of mask; known = rest.
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      int idx = y * w + x;
      if (mask[idx]) {
        state[idx] = ST_UNKNOWN;
        dist[idx]  = 1e9f;
      } else {
        state[idx] = ST_KNOWN;
        dist[idx]  = 0.0f;
      }
    }
  }

  // Find band: known pixels adjacent to unknown ones.
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      if (state[y * w + x] != ST_KNOWN) continue;
      int dx[4] = {-1, 1,  0, 0};
      int dy[4] = { 0, 0, -1, 1};
      for (int d = 0; d < 4; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        if (state[ny * w + nx] == ST_UNKNOWN) {
          state[y * w + x] = ST_BAND;
          dist[y * w + x]  = 0.5f;
          pq_push(heap, &heap_size, (pq_node){0.5f, y * w + x});
          break;
        }
      }
    }
  }

  // Fast marching: expand the known region into unknown.
  while (heap_size > 0) {
    pq_node top = pq_pop(heap, &heap_size);
    int idx = top.idx;
    if (state[idx] == ST_KNOWN) continue;
    state[idx] = ST_KNOWN;

    int cy = idx / w, cx = idx % w;

    // Fill this pixel by weighted average of known neighbors within radius.
    float sum_v[3] = {0};
    float total_w = 0.0f;

    for (int dy2 = -radius; dy2 <= radius; dy2++) {
      for (int dx2 = -radius; dx2 <= radius; dx2++) {
        if (dx2*dx2 + dy2*dy2 > radius*radius) continue;
        int nx = cx + dx2, ny = cy + dy2;
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        int nidx = ny * w + nx;
        if (state[nidx] != ST_KNOWN) continue;

        float d2 = (float)(dx2*dx2 + dy2*dy2);
        if (d2 < 0.001f) d2 = 0.001f;
        // Weight: inversely proportional to distance squared.
        float wt = 1.0f / d2;
        total_w += wt;
        for (int c = 0; c < ch; c++)
          sum_v[c] += wt * (float)img[nidx * ch + c];
      }
    }

    if (total_w > 0.0f) {
      for (int c = 0; c < ch; c++) {
        float v = sum_v[c] / total_w;
        img[idx * ch + c] = (v < 0) ? 0 : (v > 255) ? 255 : (uint8_t)v;
      }
    }

    // Propagate to unknown neighbors.
    int dx4[4] = {-1, 1,  0, 0};
    int dy4[4] = { 0, 0, -1, 1};
    for (int d = 0; d < 4; d++) {
      int nx = cx + dx4[d], ny = cy + dy4[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      int nidx = ny * w + nx;
      if (state[nidx] != ST_UNKNOWN) continue;
      float nd = dist[idx] + 1.0f;
      if (nd < dist[nidx]) {
        dist[nidx]  = nd;
        state[nidx] = ST_BAND;
        pq_push(heap, &heap_size, (pq_node){nd, nidx});
      }
    }
  }

  free(state); free(dist); free(heap);
}

// ---------------------------------------------------------------------------
// cmd_inpaint
// ---------------------------------------------------------------------------

int cmd_inpaint(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    inpaint_usage();
    return argc < 1 ? 1 : 0;
  }

  const char *img_path  = argv[0];
  const char *mask_path = NULL;
  const char *out_path  = NULL;
  int         radius    = 5;

  for (int i = 1; i < argc; i++) {
    if      (strcmp(argv[i], "--mask")   == 0 && i+1 < argc) mask_path = argv[++i];
    else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) out_path  = argv[++i];
    else if (strcmp(argv[i], "--radius") == 0 && i+1 < argc) radius    = atoi(argv[++i]);
  }

  if (!mask_path) { fputs("error: --mask required\n", stderr); return 1; }
  if (!out_path)  { fputs("error: --output required\n", stderr); return 1; }

  int iw, ih, ich;
  uint8_t *img = read_ppm(img_path, &iw, &ih, &ich);
  if (!img) { fprintf(stderr, "error: cannot read image: %s\n", img_path); return 1; }

  int mw, mh, mch;
  uint8_t *mask_data = read_ppm(mask_path, &mw, &mh, &mch);
  if (!mask_data) {
    fprintf(stderr, "error: cannot read mask: %s\n", mask_path);
    free(img); return 1;
  }

  if (mw != iw || mh != ih) {
    fputs("error: image and mask dimensions must match\n", stderr);
    free(img); free(mask_data); return 1;
  }

  // Build flat binary mask (1 = hole).
  uint8_t *flat_mask = malloc((size_t)iw * (size_t)ih);
  if (!flat_mask) { free(img); free(mask_data); fputs("error: oom\n", stderr); return 1; }
  for (int i = 0; i < iw * ih; i++)
    flat_mask[i] = mask_data[i * mch] ? 1 : 0;
  free(mask_data);

  telea_inpaint(img, flat_mask, iw, ih, ich, radius);

  bool ok = write_ppm(out_path, img, iw, ih, ich);
  if (!ok) fprintf(stderr, "error: failed to write: %s\n", out_path);
  else     printf("inpainted %dx%d → %s\n", iw, ih, out_path);

  free(flat_mask);
  free(img);
  return ok ? 0 : 1;
}
