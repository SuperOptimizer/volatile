#pragma once
#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// approval_mask — per-surface painted approval/rejection bitmap
// Values: 0=unpainted, 1=approved, 2=rejected
// ---------------------------------------------------------------------------

typedef struct approval_mask approval_mask;

approval_mask *approval_mask_new(int rows, int cols);
void           approval_mask_free(approval_mask *m);

// Paint approved/rejected region with circular brush (u,v in grid coords)
void approval_mask_paint(approval_mask *m, float u, float v, float radius, bool approved);
void approval_mask_erase(approval_mask *m, float u, float v, float radius);

// Query
bool  approval_mask_is_approved(const approval_mask *m, int row, int col);
float approval_mask_coverage(const approval_mask *m);  // fraction of cells approved

// Undo/redo (ring buffer, max 1000 snapshots)
void approval_mask_undo(approval_mask *m);
void approval_mask_redo(approval_mask *m);
bool approval_mask_can_undo(const approval_mask *m);
bool approval_mask_can_redo(const approval_mask *m);

// Persistence
bool           approval_mask_save(const approval_mask *m, const char *path);
approval_mask *approval_mask_load(const char *path);

// Render overlay: approved=green, rejected=red, unpainted=transparent (RGBA8)
void approval_mask_to_overlay(const approval_mask *m, uint8_t *rgba, int width, int height);
