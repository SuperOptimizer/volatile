#pragma once
#include "core/geom.h"
#include "gui/seg.h"
#include "gui/seg_growth.h"
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// seg_panels — all VC3D segmentation panels stacked in the right dock
//
// Panels (each collapsible):
//  1. SegmentationHeaderRow        — name, ID, status indicator
//  2. SegmentationEditingPanel     — radius/sigma sliders, edit mode
//  3. SegmentationGrowthPanel      — method, direction, generations, step
//  4. SegmentationCorrectionsPanel — correction point list, add/remove
//  5. SegmentationApprovalMaskPanel— brush size, paint/erase, coverage
//  6. SegmentationCustomParamsPanel— JSON parameter editor
//  7. SegmentationCellReoptPanel   — cell reoptimization parameters
//  8. SegmentationDirectionFieldPanel — direction field toggle
//  9. SegmentationLasagnaPanel     — external optimization service
// 10. SegmentationNeuralTracerPanel— neural tracer service controls
// ---------------------------------------------------------------------------

typedef struct seg_panels seg_panels;

seg_panels *seg_panels_new(void);
void        seg_panels_free(seg_panels *p);

// Render all panels stacked vertically.  active_surface may be NULL.
void seg_panels_render(seg_panels *p, struct nk_context *ctx,
                       quad_surface *active_surface);

// Query current UI state
seg_tool_params  seg_panels_get_tool_params(const seg_panels *p);
growth_params    seg_panels_get_growth_params(const seg_panels *p);
