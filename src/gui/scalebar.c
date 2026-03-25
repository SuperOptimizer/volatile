// scalebar.c — physical-distance ruler for viewer panels
//
// NK_IMPLEMENTATION is owned by app.c. Include nuklear.h declaration-only here.

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include "nuklear.h"

#include "gui/scalebar.h"
#include <string.h>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define DEFAULT_MAX_PX 120
#define BAR_HEIGHT_PX   6

struct scalebar {
  float voxel_um;   // µm per voxel at current pyramid level
};

scalebar *scalebar_new(float voxel_size_um) {
  scalebar *s = calloc(1, sizeof(*s));
  if (!s) return NULL;
  s->voxel_um = (voxel_size_um > 0.0f) ? voxel_size_um : 1.0f;
  return s;
}

void scalebar_free(scalebar *s) { free(s); }

void scalebar_set_voxel_size(scalebar *s, float voxel_size_um) {
  if (!s || voxel_size_um <= 0.0f) return;
  s->voxel_um = voxel_size_um;
}

// Pick a nice scale value: largest 1/2/5 × 10^n that fits in max_px.
// Returns the nice distance in µm and sets *bar_px to its pixel width.
static float nice_scale(float um_per_px, int max_px, int *bar_px) {
  float max_um = (float)max_px * um_per_px;

  // Magnitude of max_um
  float mag = powf(10.0f, floorf(log10f(max_um)));
  float steps[] = {1.0f, 2.0f, 5.0f, 10.0f};
  float chosen = mag;
  for (int i = 0; i < 4; i++) {
    float candidate = steps[i] * mag;
    if (candidate <= max_um) chosen = candidate;
  }

  *bar_px = (int)(chosen / um_per_px + 0.5f);
  return chosen;
}

// Format µm value as a label: prefer mm when >= 1000 µm.
static void fmt_um(char *buf, size_t n, float um) {
  if (um >= 1000.0f)
    snprintf(buf, n, "%.3g mm", um / 1000.0f);
  else
    snprintf(buf, n, "%.4g µm", um);
}

void scalebar_render(scalebar *s, struct nk_context *ctx,
                     float zoom, int bar_width_px) {
  if (!s || !ctx) return;
  if (bar_width_px <= 0) bar_width_px = DEFAULT_MAX_PX;
  if (zoom <= 0.0f)      zoom = 1.0f;

  // µm represented by one screen pixel
  float um_per_px = s->voxel_um / zoom;

  int bar_px = 0;
  float bar_um = nice_scale(um_per_px, bar_width_px, &bar_px);

  char label[32];
  fmt_um(label, sizeof(label), bar_um);

  // Allocate a thin canvas row for the bar graphic + label below it.
  struct nk_rect bounds;
  nk_layout_row_dynamic(ctx, (float)(BAR_HEIGHT_PX + 14), 1);
  bounds = nk_widget_bounds(ctx);

  // Draw the horizontal bar via canvas commands.
  struct nk_command_buffer *canvas = nk_window_get_canvas(ctx);
  struct nk_color white = {255, 255, 255, 220};
  float bx = bounds.x;
  float by = bounds.y;
  nk_fill_rect(canvas, nk_rect(bx, by, (float)bar_px, (float)BAR_HEIGHT_PX),
               0.0f, white);

  // Tick marks at each end
  nk_fill_rect(canvas, nk_rect(bx, by, 2.0f, (float)(BAR_HEIGHT_PX + 3)),
               0.0f, white);
  nk_fill_rect(canvas,
               nk_rect(bx + (float)bar_px - 2.0f, by,
                       2.0f, (float)(BAR_HEIGHT_PX + 3)),
               0.0f, white);

  // Label centred under the bar
  nk_draw_text(canvas,
               nk_rect(bx, by + (float)BAR_HEIGHT_PX + 2.0f,
                       (float)bar_px + 40.0f, 14.0f),
               label, (int)strlen(label),
               ctx->style.font, nk_rgba(0, 0, 0, 0), white);

  // Consume the widget space so Nuklear layout advances.
  nk_widget(&bounds, ctx);
}
