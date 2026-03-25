#include "gui/drawing.h"

#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

#define MAX_UNDO 32

// A snapshot of the pixel buffer saved before each committed operation.
typedef struct {
  uint8_t *pixels;  // owned copy, width*height*4 bytes
} snapshot;

struct drawing_canvas {
  int      width, height;
  uint8_t *pixels;          // RGBA, width*height*4 bytes (current state)
  uint8_t *scratch;         // working copy for shape previews

  // Stroke state
  bool        stroking;
  draw_params stroke_params;
  float       last_x, last_y;

  // Shape preview state
  bool        shaping;
  draw_tool   shape_tool;
  draw_params shape_params;
  float       shape_x0, shape_y0;

  // Undo / redo ring
  snapshot undo_stack[MAX_UNDO];
  int      undo_top;   // index of the next slot to write into
  int      undo_count; // how many valid snapshots are below undo_top
  snapshot redo_stack[MAX_UNDO];
  int      redo_count;
};

static int canvas_bytes(const drawing_canvas *c) {
  return c->width * c->height * 4;
}

// Save current pixels as undo snapshot and clear redo stack.
static void push_undo(drawing_canvas *c) {
  int idx = c->undo_top % MAX_UNDO;
  free(c->undo_stack[idx].pixels);
  c->undo_stack[idx].pixels = malloc((size_t)canvas_bytes(c));
  if (!c->undo_stack[idx].pixels) return;
  memcpy(c->undo_stack[idx].pixels, c->pixels, (size_t)canvas_bytes(c));
  c->undo_top++;
  if (c->undo_count < MAX_UNDO) c->undo_count++;
  // clear redo
  for (int i = 0; i < c->redo_count; i++) {
    free(c->redo_stack[i].pixels);
    c->redo_stack[i].pixels = NULL;
  }
  c->redo_count = 0;
}

// Paint a filled circle of radius r centered at (cx, cy) onto buf.
static void paint_circle(uint8_t *buf, int w, int h,
                         float cx, float cy, float r,
                         const uint8_t color[4], bool erase) {
  int x0 = (int)(cx - r - 1), x1 = (int)(cx + r + 1);
  int y0 = (int)(cy - r - 1), y1 = (int)(cy + r + 1);
  if (x0 < 0) x0 = 0;  if (x1 >= w) x1 = w - 1;
  if (y0 < 0) y0 = 0;  if (y1 >= h) y1 = h - 1;
  float r2 = r * r;
  for (int y = y0; y <= y1; y++) {
    for (int x = x0; x <= x1; x++) {
      float dx = x - cx, dy = y - cy;
      if (dx*dx + dy*dy <= r2) {
        uint8_t *p = buf + (y * w + x) * 4;
        if (erase) {
          p[0] = p[1] = p[2] = p[3] = 0;
        } else {
          p[0] = color[0]; p[1] = color[1];
          p[2] = color[2]; p[3] = color[3];
        }
      }
    }
  }
}

// Draw a thick line from (x0,y0) to (x1,y1) by walking and splatting circles.
static void paint_line_thick(uint8_t *buf, int w, int h,
                             float x0, float y0, float x1, float y1,
                             float r, const uint8_t color[4], bool erase) {
  float dx = x1 - x0, dy = y1 - y0;
  float len = sqrtf(dx*dx + dy*dy);
  int steps = (int)(len / (r * 0.5f)) + 1;
  for (int i = 0; i <= steps; i++) {
    float t = (steps > 0) ? (float)i / (float)steps : 0.0f;
    paint_circle(buf, w, h, x0 + t*dx, y0 + t*dy, r, color, erase);
  }
}

// Draw an axis-aligned rectangle outline with given line thickness.
static void paint_rect_outline(uint8_t *buf, int w, int h,
                               float x0, float y0, float x1, float y1,
                               float thickness, const uint8_t color[4]) {
  float r = thickness * 0.5f;
  paint_line_thick(buf, w, h, x0, y0, x1, y0, r, color, false);
  paint_line_thick(buf, w, h, x1, y0, x1, y1, r, color, false);
  paint_line_thick(buf, w, h, x1, y1, x0, y1, r, color, false);
  paint_line_thick(buf, w, h, x0, y1, x0, y0, r, color, false);
}

// Draw a circle outline.
static void paint_circle_outline(uint8_t *buf, int w, int h,
                                 float cx, float cy, float radius,
                                 float thickness, const uint8_t color[4]) {
  int steps = (int)(2.0f * (float)M_PI * radius / 1.5f) + 8;
  float r = thickness * 0.5f;
  for (int i = 0; i < steps; i++) {
    float angle = 2.0f * (float)M_PI * (float)i / (float)steps;
    float px = cx + cosf(angle) * radius;
    float py = cy + sinf(angle) * radius;
    paint_circle(buf, w, h, px, py, r, color, false);
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

drawing_canvas *drawing_canvas_new(int width, int height) {
  drawing_canvas *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  c->width  = width;
  c->height = height;
  c->pixels  = calloc((size_t)(width * height * 4), 1);
  c->scratch = calloc((size_t)(width * height * 4), 1);
  if (!c->pixels || !c->scratch) { drawing_canvas_free(c); return NULL; }
  return c;
}

void drawing_canvas_free(drawing_canvas *c) {
  if (!c) return;
  free(c->pixels);
  free(c->scratch);
  for (int i = 0; i < MAX_UNDO; i++) free(c->undo_stack[i].pixels);
  for (int i = 0; i < MAX_UNDO; i++) free(c->redo_stack[i].pixels);
  free(c);
}

// ---------------------------------------------------------------------------
// Stroke (freehand / eraser)
// ---------------------------------------------------------------------------

void drawing_begin_stroke(drawing_canvas *c, float x, float y,
                          const draw_params *params) {
  if (!c || !params) return;
  push_undo(c);
  c->stroking      = true;
  c->stroke_params = *params;
  c->last_x = x;
  c->last_y = y;
  bool erase = (params->tool == DRAW_ERASER);
  paint_circle(c->pixels, c->width, c->height,
               x, y, params->brush_radius, params->color, erase);
}

void drawing_continue_stroke(drawing_canvas *c, float x, float y) {
  if (!c || !c->stroking) return;
  bool erase = (c->stroke_params.tool == DRAW_ERASER);
  paint_line_thick(c->pixels, c->width, c->height,
                   c->last_x, c->last_y, x, y,
                   c->stroke_params.brush_radius,
                   c->stroke_params.color, erase);
  c->last_x = x;
  c->last_y = y;
}

void drawing_end_stroke(drawing_canvas *c) {
  if (!c) return;
  c->stroking = false;
}

// ---------------------------------------------------------------------------
// Shape (rect / circle preview then commit)
// ---------------------------------------------------------------------------

void drawing_begin_shape(drawing_canvas *c, float x0, float y0,
                         draw_tool tool, const draw_params *params) {
  if (!c || !params) return;
  c->shaping      = true;
  c->shape_tool   = tool;
  c->shape_params = *params;
  c->shape_x0     = x0;
  c->shape_y0     = y0;
  // Copy current pixels into scratch so we can preview on top.
  memcpy(c->scratch, c->pixels, (size_t)canvas_bytes(c));
}

void drawing_update_shape(drawing_canvas *c, float x1, float y1) {
  if (!c || !c->shaping) return;
  // Reset scratch then draw preview.
  memcpy(c->scratch, c->pixels, (size_t)canvas_bytes(c));
  float t = c->shape_params.line_width > 0 ? c->shape_params.line_width : 2.0f;
  if (c->shape_tool == DRAW_RECT) {
    paint_rect_outline(c->scratch, c->width, c->height,
                       c->shape_x0, c->shape_y0, x1, y1,
                       t, c->shape_params.color);
  } else if (c->shape_tool == DRAW_CIRCLE) {
    float dx = x1 - c->shape_x0, dy = y1 - c->shape_y0;
    float radius = sqrtf(dx*dx + dy*dy);
    paint_circle_outline(c->scratch, c->width, c->height,
                         c->shape_x0, c->shape_y0, radius,
                         t, c->shape_params.color);
  }
}

void drawing_finish_shape(drawing_canvas *c) {
  if (!c || !c->shaping) return;
  push_undo(c);
  // Commit scratch -> pixels.
  memcpy(c->pixels, c->scratch, (size_t)canvas_bytes(c));
  c->shaping = false;
}

// ---------------------------------------------------------------------------
// Undo / redo
// ---------------------------------------------------------------------------

void drawing_undo(drawing_canvas *c) {
  if (!c || c->undo_count == 0) return;
  // Push current state to redo.
  if (c->redo_count < MAX_UNDO) {
    int ri = c->redo_count;
    free(c->redo_stack[ri].pixels);
    c->redo_stack[ri].pixels = malloc((size_t)canvas_bytes(c));
    if (c->redo_stack[ri].pixels) {
      memcpy(c->redo_stack[ri].pixels, c->pixels, (size_t)canvas_bytes(c));
      c->redo_count++;
    }
  }
  c->undo_top--;
  c->undo_count--;
  int idx = ((c->undo_top % MAX_UNDO) + MAX_UNDO) % MAX_UNDO;
  if (c->undo_stack[idx].pixels)
    memcpy(c->pixels, c->undo_stack[idx].pixels, (size_t)canvas_bytes(c));
  else
    memset(c->pixels, 0, (size_t)canvas_bytes(c));
}

void drawing_redo(drawing_canvas *c) {
  if (!c || c->redo_count == 0) return;
  push_undo(c);
  c->undo_count--; // push_undo incremented it, adjust so redo pops correctly
  c->redo_count--;
  snapshot *snap = &c->redo_stack[c->redo_count];
  if (snap->pixels) {
    memcpy(c->pixels, snap->pixels, (size_t)canvas_bytes(c));
    free(snap->pixels);
    snap->pixels = NULL;
  }
}

// ---------------------------------------------------------------------------
// Read-back
// ---------------------------------------------------------------------------

const uint8_t *drawing_get_pixels(const drawing_canvas *c) {
  return c ? c->pixels : NULL;
}

void drawing_clear(drawing_canvas *c) {
  if (!c) return;
  push_undo(c);
  memset(c->pixels, 0, (size_t)canvas_bytes(c));
}

void drawing_export_mask(const drawing_canvas *c, uint8_t *mask_out) {
  if (!c || !mask_out) return;
  int n = c->width * c->height;
  for (int i = 0; i < n; i++)
    mask_out[i] = c->pixels[i * 4 + 3] ? 255 : 0;
}
