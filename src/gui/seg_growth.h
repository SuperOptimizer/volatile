#pragma once
#include "core/geom.h"
#include "core/vol.h"
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Segmentation growth — port of VC3D SegmentationGrower
// ---------------------------------------------------------------------------

typedef enum {
  GROWTH_TRACER,        // follow volume gradients (intensity peak along normal)
  GROWTH_EXTRAPOLATION, // linear extrapolation of boundary vertex positions
  GROWTH_CORRECTIONS,   // anchor points that constrain the surface
} growth_method;

typedef enum {
  GROWTH_DIR_ALL,
  GROWTH_DIR_UP,
  GROWTH_DIR_DOWN,
  GROWTH_DIR_LEFT,
  GROWTH_DIR_RIGHT,
} growth_direction;

typedef struct {
  growth_method    method;
  growth_direction direction;
  int              generations;    // number of generations to grow
  float            step_size;      // step size along normal (tracer) or per gen (extrap)
  float            straightness_weight;  // 2D/3D straightness constraint
  float            distance_weight;      // distance preservation weight
} growth_params;

typedef struct seg_grower seg_grower;

// lifecycle
seg_grower   *seg_grower_new(volume *vol, quad_surface *seed);
void          seg_grower_free(seg_grower *g);

// grow one generation (non-blocking, runs in thread pool)
bool          seg_grower_step(seg_grower *g, const growth_params *params);

// get the current surface (caller must NOT free; owned by grower)
quad_surface *seg_grower_surface(seg_grower *g);

// check if growth is running
bool          seg_grower_busy(const seg_grower *g);

// add a correction point: UV coordinates + 3D target position
void          seg_grower_add_correction(seg_grower *g, float u, float v, vec3f target);

// ---------------------------------------------------------------------------
// Advanced growth — port of villa's GrowSurface algorithm
// ---------------------------------------------------------------------------

typedef struct {
  float straightness_2d;    // 2D straightness constraint weight
  float straightness_3d;    // 3D straightness constraint weight
  float distance_weight;    // distance preservation weight
  float z_location_weight;  // Z-axis anchoring weight
  float jitter_amount;      // deterministic jitter for robustness
  int   search_steps;       // number of normal-search samples
  float search_radius;      // search range along normal (voxels)
  bool  use_direction_field;// guide growth with structure tensor
  bool  use_corrections;    // honor correction anchor points
  int   max_generations;    // stop after N generations (0 = unlimited)
} advanced_growth_params;

// Multi-generation growth with per-vertex cost-function optimization.
// Blocks until all generations complete; safe to call from any thread.
bool seg_grower_grow_advanced(seg_grower *g, const advanced_growth_params *params);

// Set exclusion surfaces — new positions within exclusion_radius of any
// point on these surfaces will be rejected and snapped back.
void seg_grower_set_exclusion_surfaces(seg_grower *g,
                                       quad_surface **others, int count);

// Deterministic jitter — SplitMix64 PRNG used by villa for reproducibility.
uint64_t splitmix64(uint64_t *state);
