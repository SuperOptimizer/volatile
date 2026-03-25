#pragma once
#include <stdint.h>

struct nk_context;

// ---------------------------------------------------------------------------
// statusbar — bottom-of-window info strip, updated every frame.
// Shows: cursor position, voxel value, zoom, pyramid level, FPS, memory.
// ---------------------------------------------------------------------------

typedef struct statusbar statusbar;

statusbar *statusbar_new(void);
void       statusbar_free(statusbar *s);

// Call each frame before rendering to push latest values.
void statusbar_update(statusbar *s,
                      float x, float y, float z,  // cursor world position
                      float voxel_value,
                      float zoom,
                      int   pyramid_level,
                      float fps,
                      size_t mem_bytes);           // resident memory

// Render a full-width strip at the bottom of the current Nuklear window.
// height: row height in pixels (pass 0 for default 22).
void statusbar_render(statusbar *s, struct nk_context *ctx, int height);
