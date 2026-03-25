// WIDGET TYPE: CONTENT — call inside an nk_begin/nk_end block.
#pragma once
#include <stdbool.h>
#include <stdint.h>

// Forward-declare nk_context to avoid pulling nuklear.h into consumers.
struct nk_context;

// ---------------------------------------------------------------------------
// window_range_state — window/level contrast control state
// Port of VC3D WindowRangeWidget (Qt) -> plain C + Nuklear.
//
// low/high are normalised to [0,1]; window and level are derived:
//   window = high - low
//   level  = (low + high) / 2
// ---------------------------------------------------------------------------

typedef struct {
  float low, high;      // current window endpoints [0,1]
  float window, level;  // derived: window=high-low, level=(low+high)/2
  bool  auto_range;     // when true, window_range_auto() drives low/high
  int   cmap_id;        // index into cmap_id enum (render/cmap.h)
} window_range_state;

// Initialise with full range [0,1], grayscale colormap, auto_range=false.
void window_range_init(window_range_state *s);

// Set low/high explicitly; clamps and recomputes window/level.
void window_range_set(window_range_state *s, float low, float high);

// Auto-fit: set low/high from a percentile-clipped data range.
// p2 and p98 are the 2nd and 98th percentile values in the same units as
// data_min/data_max (all normalised to [0,1] by the caller).
void window_range_auto(window_range_state *s,
                       float data_min, float data_max,
                       float p2, float p98);

// Render as a Nuklear widget: dual range slider + colormap combo.
// Returns true if any value changed this frame.
bool window_range_render(window_range_state *s, struct nk_context *ctx);

// Apply windowing to an 8-bit pixel value.
// Maps input [0,255] through the current low/high window -> output [0,255].
uint8_t window_range_apply(const window_range_state *s, uint8_t value);
