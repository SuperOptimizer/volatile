#pragma once

struct nk_context;

// ---------------------------------------------------------------------------
// scalebar — physical-distance ruler drawn in a viewer panel corner.
// Auto-scales to nice numbers (1/2/5 × 10^n µm).
// ---------------------------------------------------------------------------

typedef struct scalebar scalebar;

// voxel_size_um: physical size of one voxel in micrometres.
scalebar *scalebar_new(float voxel_size_um);
void      scalebar_free(scalebar *s);

// Update voxel size (e.g. when pyramid level changes the effective resolution).
void scalebar_set_voxel_size(scalebar *s, float voxel_size_um);

// Render a scale bar in the current Nuklear layout row.
// zoom: current viewer zoom (pixels per voxel).
// bar_width_px: maximum pixel width for the bar (pass 0 for default 120).
void scalebar_render(scalebar *s, struct nk_context *ctx,
                     float zoom, int bar_width_px);
