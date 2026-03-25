#pragma once
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Drawing tools
// ---------------------------------------------------------------------------

typedef enum {
  DRAW_FREEHAND,
  DRAW_LINE,
  DRAW_RECT,
  DRAW_CIRCLE,
  DRAW_ERASER,
} draw_tool;

typedef struct {
  draw_tool tool;
  float     brush_radius;
  uint8_t   color[4];   // RGBA
  float     line_width;
} draw_params;

typedef struct drawing_canvas drawing_canvas;

// Lifecycle
drawing_canvas *drawing_canvas_new(int width, int height);
void            drawing_canvas_free(drawing_canvas *c);

// Freehand / eraser strokes (mouse-drag)
void drawing_begin_stroke(drawing_canvas *c, float x, float y, const draw_params *params);
void drawing_continue_stroke(drawing_canvas *c, float x, float y);
void drawing_end_stroke(drawing_canvas *c);

// Rect / circle (click-drag preview, then commit)
void drawing_begin_shape(drawing_canvas *c, float x0, float y0, draw_tool tool, const draw_params *params);
void drawing_update_shape(drawing_canvas *c, float x1, float y1);
void drawing_finish_shape(drawing_canvas *c);

// Undo / redo
void drawing_undo(drawing_canvas *c);
void drawing_redo(drawing_canvas *c);

// Read-back
const uint8_t *drawing_get_pixels(const drawing_canvas *c);

// Clear the canvas
void drawing_clear(drawing_canvas *c);

// Export single-channel mask (0 = empty, 255 = drawn); caller supplies buffer
// of width*height bytes.
void drawing_export_mask(const drawing_canvas *c, uint8_t *mask_out);
