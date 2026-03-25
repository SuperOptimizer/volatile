#pragma once
#include <stdint.h>

// ---------------------------------------------------------------------------
// Overlay types
// ---------------------------------------------------------------------------

typedef enum {
  OVERLAY_SEGMENTATION,  // mesh contour lines on slice
  OVERLAY_POINTS,        // point markers
  OVERLAY_LINES,         // polylines
  OVERLAY_BBOX,          // bounding box rectangles
  OVERLAY_VECTORS,       // vector field arrows
  OVERLAY_TEXT,          // text labels
} overlay_type;

typedef struct { float x, y; uint8_t r, g, b, a; } overlay_point;
typedef struct { overlay_point p0, p1; float thickness; } overlay_line;
typedef struct { float x, y, w, h; uint8_t r, g, b, a; } overlay_rect;

// ---------------------------------------------------------------------------
// Overlay list
// ---------------------------------------------------------------------------

typedef struct overlay_list overlay_list;

overlay_list *overlay_list_new(void);
void          overlay_list_free(overlay_list *l);
void          overlay_list_clear(overlay_list *l);

void overlay_add_point(overlay_list *l, float x, float y, uint8_t r, uint8_t g, uint8_t b, float radius);
void overlay_add_line(overlay_list *l, float x0, float y0, float x1, float y1,
                      uint8_t r, uint8_t g, uint8_t b, float thickness);
void overlay_add_rect(overlay_list *l, float x, float y, float w, float h, uint8_t r, uint8_t g, uint8_t b);
void overlay_add_circle(overlay_list *l, float cx, float cy, float radius, uint8_t r, uint8_t g, uint8_t b);
void overlay_add_text(overlay_list *l, float x, float y, const char *text, uint8_t r, uint8_t g, uint8_t b);

// Render all overlays onto an RGBA pixel buffer (stride = width*4).
void overlay_render(const overlay_list *l, uint8_t *pixels, int width, int height);

int overlay_count(const overlay_list *l);
