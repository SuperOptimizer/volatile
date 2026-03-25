#pragma once
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef struct {
  uint8_t r, g, b;
} cmap_rgb;

typedef enum {
  CMAP_GRAYSCALE = 0,
  CMAP_VIRIDIS,
  CMAP_MAGMA,
  CMAP_INFERNO,
  CMAP_PLASMA,
  CMAP_HOT,
  CMAP_COOL,
  CMAP_BONE,
  CMAP_JET,
  CMAP_TURBO,
  CMAP_COUNT,
} cmap_id;

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

// Map a normalised value in [0,1] through the given colormap.
// Values outside [0,1] are clamped.
cmap_rgb cmap_apply(cmap_id id, double value);

// Batch version: apply cmap to each element of `values` (length `n`),
// writing results into `out`.
void cmap_apply_buf(cmap_id id, const double *values, cmap_rgb *out, size_t n);

// Human-readable name of a colormap ("grayscale", "viridis", …).
// Returns NULL for out-of-range id.
const char *cmap_name(cmap_id id);

// Number of available colormaps (always == CMAP_COUNT).
int cmap_count(void);
