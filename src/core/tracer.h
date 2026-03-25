#pragma once
#include "core/geom.h"
#include "core/vol.h"
#include <stdbool.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// tracer — surface growth via fringe BFS + cost-function candidate selection
//
// Port of villa's GrowPatch algorithm.  For each fringe (boundary) vertex,
// N candidates are sampled along the local surface normal.  The candidate with
// the lowest weighted cost (straightness_2d + straightness_3d + distance +
// z_location) is selected.  SplitMix64 deterministic jitter adds robustness.
// ---------------------------------------------------------------------------

typedef enum {
  GROWTH_ALL    = 0,   // expand in all four grid directions
  GROWTH_ROW    = 1,   // expand along row axis only
  GROWTH_COL    = 2,   // expand along col axis only
} growth_direction;

typedef struct {
  float straightness_2d;      // weight for 2D grid-distance straightness (default 0.7)
  float straightness_3d;      // weight for 3D normal-direction straightness (default 4.0)
  float distance_weight;      // weight for inter-point distance regularization (default 1.0)
  float z_location_weight;    // weight for z (depth) position smoothness (default 0.1)
  float search_radius;        // how far along the normal to search (voxels, default 5.0)
  int   search_steps;         // number of candidate steps (default 16)
  float jitter;               // SplitMix64 jitter magnitude (default 0.05)
  bool  use_direction_field;  // if set, use direction-field volume for guidance
  bool  use_edt;              // if set, use EDT volume for distance-based cost
  bool  use_neural;           // reserved: use neural tracer guidance
  float falloff_sigma;        // gaussian falloff sigma for cost blending (default 2.0)
} tracer_params;

typedef struct tracer tracer;

tracer       *tracer_new(volume *vol);
void          tracer_free(tracer *t);

// Attach optional guidance volumes.
void          tracer_set_direction_field(tracer *t, volume *dir_vol);
void          tracer_set_edt(tracer *t, volume *edt_vol);

// Add a surface to the exclusion list — candidates overlapping it are rejected.
void          tracer_add_exclusion(tracer *t, const quad_surface *other);

// Grow one patch outward from seed by `generations` BFS steps.
// Returns a new quad_surface (caller owns) or NULL on error.
quad_surface *tracer_grow_patch(tracer *t, const quad_surface *seed,
                                const tracer_params *params,
                                int generations, growth_direction dir);

// Cost for a single candidate position at grid (row,col).
// Lower = better.  Components: 2D straightness, 3D normal alignment,
// distance regularization, z-location smoothness, EDT term (if enabled).
float         tracer_cost(const tracer *t, const quad_surface *surf,
                          int row, int col, vec3f candidate,
                          const tracer_params *params);

// Returns true if pos is within threshold voxels of any exclusion surface vertex.
bool          tracer_check_overlap(const tracer *t, vec3f pos, float threshold);

// Default params (all weights as in villa defaults).
tracer_params tracer_params_default(void);
