// statusbar.c — bottom-of-window info strip (cursor, voxel, zoom, fps, mem)
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

#include "gui/statusbar.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define DEFAULT_HEIGHT 22

struct statusbar {
  float  x, y, z;
  float  voxel_value;
  float  zoom;
  int    pyramid_level;
  float  fps;
  size_t mem_bytes;
};

statusbar *statusbar_new(void) {
  return calloc(1, sizeof(statusbar));
}

void statusbar_free(statusbar *s) {
  free(s);
}

void statusbar_update(statusbar *s,
                      float x, float y, float z,
                      float voxel_value,
                      float zoom,
                      int   pyramid_level,
                      float fps,
                      size_t mem_bytes) {
  if (!s) return;
  s->x             = x;
  s->y             = y;
  s->z             = z;
  s->voxel_value   = voxel_value;
  s->zoom          = zoom;
  s->pyramid_level = pyramid_level;
  s->fps           = fps;
  s->mem_bytes     = mem_bytes;
}

// Format resident memory as "123 MB" or "1.2 GB".
static void fmt_mem(char *buf, size_t bufsz, size_t bytes) {
  if (bytes >= (size_t)1 << 30)
    snprintf(buf, bufsz, "%.1f GB", (double)bytes / (1 << 30));
  else if (bytes >= (size_t)1 << 20)
    snprintf(buf, bufsz, "%zu MB", bytes >> 20);
  else
    snprintf(buf, bufsz, "%zu KB", bytes >> 10);
}

void statusbar_render(statusbar *s, struct nk_context *ctx, int height) {
  if (!s || !ctx) return;
  if (height <= 0) height = DEFAULT_HEIGHT;

  char pos[48], val[24], zoom[24], lvl[16], fps[16], mem[20];

  snprintf(pos,  sizeof(pos),  "xyz: %.1f, %.1f, %.1f", s->x, s->y, s->z);
  snprintf(val,  sizeof(val),  "val: %.4g",              s->voxel_value);
  snprintf(zoom, sizeof(zoom), "zoom: %.2fx",            s->zoom);
  snprintf(lvl,  sizeof(lvl),  "L%d",                   s->pyramid_level);
  snprintf(fps,  sizeof(fps),  "%.0f fps",               s->fps);
  fmt_mem(mem, sizeof(mem), s->mem_bytes);

  // Six equal-width columns across the full row.
  nk_layout_row_dynamic(ctx, (float)height, 6);
  nk_label(ctx, pos,  NK_TEXT_LEFT);
  nk_label(ctx, val,  NK_TEXT_CENTERED);
  nk_label(ctx, zoom, NK_TEXT_CENTERED);
  nk_label(ctx, lvl,  NK_TEXT_CENTERED);
  nk_label(ctx, fps,  NK_TEXT_RIGHT);
  nk_label(ctx, mem,  NK_TEXT_RIGHT);
}
