#pragma once
#include "core/math.h"
#include "render/overlay.h"
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Seed point manager — tracks 3D starting points for segmentations.
// ---------------------------------------------------------------------------

typedef struct seed_manager seed_manager;

seed_manager *seed_mgr_new(void);
void          seed_mgr_free(seed_manager *m);

// Add a point; returns a non-negative seed id.
int   seed_mgr_add(seed_manager *m, vec3f point);

// Remove by id; returns true if found and removed.
bool  seed_mgr_remove(seed_manager *m, int id);

// Retrieve a point by id (returns zero-vector if not found).
vec3f seed_mgr_get(const seed_manager *m, int id);

// Number of currently stored seeds.
int   seed_mgr_count(const seed_manager *m);

// Generate overlay markers (circles) for all seeds.
void  seed_mgr_to_overlay(const seed_manager *m, overlay_list *out,
                           float marker_radius);
