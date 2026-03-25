#include "overlay.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// Internal item representation
// ---------------------------------------------------------------------------

typedef enum {
  ITEM_POINT,
  ITEM_LINE,
  ITEM_RECT,
  ITEM_CIRCLE,
  ITEM_TEXT,
} item_type;

typedef struct {
  item_type type;
  uint8_t r, g, b;
  union {
    struct { float x, y, radius; }                    point;
    struct { float x0, y0, x1, y1, thickness; }       line;
    struct { float x, y, w, h; }                      rect;
    struct { float cx, cy, radius; }                  circle;
    struct { float x, y; char text[64]; }             text;
  };
} overlay_item;

struct overlay_list {
  overlay_item *items;
  int           count;
  int           cap;
};

// ---------------------------------------------------------------------------
// List management
// ---------------------------------------------------------------------------

overlay_list *overlay_list_new(void) {
  overlay_list *l = calloc(1, sizeof(*l));
  if (!l) return NULL;
  l->cap = 16;
  l->items = malloc((size_t)l->cap * sizeof(overlay_item));
  if (!l->items) { free(l); return NULL; }
  return l;
}

void overlay_list_free(overlay_list *l) {
  if (!l) return;
  free(l->items);
  free(l);
}

void overlay_list_clear(overlay_list *l) {
  if (l) l->count = 0;
}

int overlay_count(const overlay_list *l) {
  return l ? l->count : 0;
}

static overlay_item *push_item(overlay_list *l) {
  if (l->count == l->cap) {
    int new_cap = l->cap * 2;
    overlay_item *buf = realloc(l->items, (size_t)new_cap * sizeof(overlay_item));
    if (!buf) return NULL;
    l->items = buf;
    l->cap = new_cap;
  }
  return &l->items[l->count++];
}

// ---------------------------------------------------------------------------
// Add helpers
// ---------------------------------------------------------------------------

void overlay_add_point(overlay_list *l, float x, float y, uint8_t r, uint8_t g, uint8_t b, float radius) {
  assert(l);
  overlay_item *it = push_item(l);
  if (!it) return;
  it->type = ITEM_POINT; it->r = r; it->g = g; it->b = b;
  it->point = (typeof(it->point)){x, y, radius};
}

void overlay_add_line(overlay_list *l, float x0, float y0, float x1, float y1,
                      uint8_t r, uint8_t g, uint8_t b, float thickness) {
  assert(l);
  overlay_item *it = push_item(l);
  if (!it) return;
  it->type = ITEM_LINE; it->r = r; it->g = g; it->b = b;
  it->line = (typeof(it->line)){x0, y0, x1, y1, thickness};
}

void overlay_add_rect(overlay_list *l, float x, float y, float w, float h, uint8_t r, uint8_t g, uint8_t b) {
  assert(l);
  overlay_item *it = push_item(l);
  if (!it) return;
  it->type = ITEM_RECT; it->r = r; it->g = g; it->b = b;
  it->rect = (typeof(it->rect)){x, y, w, h};
}

void overlay_add_circle(overlay_list *l, float cx, float cy, float radius, uint8_t r, uint8_t g, uint8_t b) {
  assert(l);
  overlay_item *it = push_item(l);
  if (!it) return;
  it->type = ITEM_CIRCLE; it->r = r; it->g = g; it->b = b;
  it->circle = (typeof(it->circle)){cx, cy, radius};
}

void overlay_add_text(overlay_list *l, float x, float y, const char *text, uint8_t r, uint8_t g, uint8_t b) {
  assert(l && text);
  overlay_item *it = push_item(l);
  if (!it) return;
  it->type = ITEM_TEXT; it->r = r; it->g = g; it->b = b;
  it->text.x = x; it->text.y = y;
  strncpy(it->text.text, text, sizeof(it->text.text) - 1);
  it->text.text[sizeof(it->text.text) - 1] = '\0';
}

// ---------------------------------------------------------------------------
// Rasterization helpers
// ---------------------------------------------------------------------------

// Blend src color with alpha onto RGBA pixel buffer at (px, py).
static inline void blend_pixel(uint8_t *pixels, int width, int height,
                                int px, int py, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (px < 0 || px >= width || py < 0 || py >= height) return;
  uint8_t *dst = pixels + (py * width + px) * 4;
  uint32_t sa = a;
  uint32_t da = 255 - sa;
  dst[0] = (uint8_t)((r * sa + dst[0] * da) / 255);
  dst[1] = (uint8_t)((g * sa + dst[1] * da) / 255);
  dst[2] = (uint8_t)((b * sa + dst[2] * da) / 255);
  dst[3] = (uint8_t)(sa + (dst[3] * da) / 255);
}

// Filled circle via midpoint / Bresenham circle scan-fill.
static void raster_filled_circle(uint8_t *pixels, int width, int height,
                                 int cx, int cy, int radius, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (radius <= 0) {
    blend_pixel(pixels, width, height, cx, cy, r, g, b, a);
    return;
  }
  for (int dy = -radius; dy <= radius; dy++) {
    int dx_max = (int)sqrtf((float)(radius * radius - dy * dy));
    for (int dx = -dx_max; dx <= dx_max; dx++) {
      blend_pixel(pixels, width, height, cx + dx, cy + dy, r, g, b, a);
    }
  }
}

// Bresenham line between integer endpoints; thickness handled by drawing
// parallel filled-circle stamps (thick) or single pixels (thin).
static void raster_line(uint8_t *pixels, int width, int height,
                        int x0, int y0, int x1, int y1,
                        uint8_t r, uint8_t g, uint8_t b, uint8_t a, int half_thick) {
  int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
  int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
  int err = dx + dy;

  for (;;) {
    raster_filled_circle(pixels, width, height, x0, y0, half_thick, r, g, b, a);
    if (x0 == x1 && y0 == y1) break;
    int e2 = 2 * err;
    if (e2 >= dy) { err += dy; x0 += sx; }
    if (e2 <= dx) { err += dx; y0 += sy; }
  }
}

// Filled rectangle (4-pixel border + hollow interior for clarity as a bounding box).
static void raster_rect_outline(uint8_t *pixels, int width, int height,
                                int x0, int y0, int x1, int y1,
                                uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  // top / bottom edges
  for (int x = x0; x <= x1; x++) {
    blend_pixel(pixels, width, height, x, y0, r, g, b, a);
    blend_pixel(pixels, width, height, x, y1, r, g, b, a);
  }
  // left / right edges
  for (int y = y0 + 1; y < y1; y++) {
    blend_pixel(pixels, width, height, x0, y, r, g, b, a);
    blend_pixel(pixels, width, height, x1, y, r, g, b, a);
  }
}

// Bresenham circle outline.
static void raster_circle_outline(uint8_t *pixels, int width, int height,
                                  int cx, int cy, int radius,
                                  uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (radius <= 0) { blend_pixel(pixels, width, height, cx, cy, r, g, b, a); return; }
  int x = radius, y = 0, err = 0;
  while (x >= y) {
    blend_pixel(pixels, width, height, cx + x, cy + y, r, g, b, a);
    blend_pixel(pixels, width, height, cx + y, cy + x, r, g, b, a);
    blend_pixel(pixels, width, height, cx - y, cy + x, r, g, b, a);
    blend_pixel(pixels, width, height, cx - x, cy + y, r, g, b, a);
    blend_pixel(pixels, width, height, cx - x, cy - y, r, g, b, a);
    blend_pixel(pixels, width, height, cx - y, cy - x, r, g, b, a);
    blend_pixel(pixels, width, height, cx + y, cy - x, r, g, b, a);
    blend_pixel(pixels, width, height, cx + x, cy - y, r, g, b, a);
    y++;
    if (err <= 0) { err += 2 * y + 1; }
    else          { x--; err += 2 * (y - x) + 1; }
  }
}

// Text placeholder: draw a small 3x5 cross at the anchor point.
static void raster_text_marker(uint8_t *pixels, int width, int height,
                               int x, int y, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  for (int dx = -2; dx <= 2; dx++) blend_pixel(pixels, width, height, x + dx, y, r, g, b, a);
  for (int dy = -2; dy <= 2; dy++) blend_pixel(pixels, width, height, x, y + dy, r, g, b, a);
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void overlay_render(const overlay_list *l, uint8_t *pixels, int width, int height) {
  assert(l && pixels && width > 0 && height > 0);
  for (int i = 0; i < l->count; i++) {
    const overlay_item *it = &l->items[i];
    uint8_t r = it->r, g = it->g, b = it->b, a = 255;

    switch (it->type) {
      case ITEM_POINT: {
        int cx = (int)roundf(it->point.x);
        int cy = (int)roundf(it->point.y);
        int rad = (int)roundf(it->point.radius);
        raster_filled_circle(pixels, width, height, cx, cy, rad, r, g, b, a);
        break;
      }
      case ITEM_LINE: {
        int x0 = (int)roundf(it->line.x0), y0 = (int)roundf(it->line.y0);
        int x1 = (int)roundf(it->line.x1), y1 = (int)roundf(it->line.y1);
        int ht = (int)roundf(it->line.thickness * 0.5f);
        raster_line(pixels, width, height, x0, y0, x1, y1, r, g, b, a, ht);
        break;
      }
      case ITEM_RECT: {
        int x0 = (int)roundf(it->rect.x);
        int y0 = (int)roundf(it->rect.y);
        int x1 = (int)roundf(it->rect.x + it->rect.w);
        int y1 = (int)roundf(it->rect.y + it->rect.h);
        raster_rect_outline(pixels, width, height, x0, y0, x1, y1, r, g, b, a);
        break;
      }
      case ITEM_CIRCLE: {
        int cx  = (int)roundf(it->circle.cx);
        int cy  = (int)roundf(it->circle.cy);
        int rad = (int)roundf(it->circle.radius);
        raster_circle_outline(pixels, width, height, cx, cy, rad, r, g, b, a);
        break;
      }
      case ITEM_TEXT: {
        int x = (int)roundf(it->text.x);
        int y = (int)roundf(it->text.y);
        raster_text_marker(pixels, width, height, x, y, r, g, b, a);
        break;
      }
    }
  }
}
