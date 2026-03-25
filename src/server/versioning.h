#pragma once
#include "core/geom.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct surface_history surface_history;

// Open (or create) versioning store for surface_id at db_path.
// Use ":memory:" for tests.
surface_history *surface_history_new(int64_t surface_id, const char *db_path);
void             surface_history_free(surface_history *h);

// Snapshot current surface state. Returns new version_id, or -1 on error.
int64_t surface_history_commit(surface_history *h, int user_id,
                               const char *message,
                               const quad_surface *surface);

// Version metadata returned by surface_history_list.
typedef struct {
  int64_t version_id;
  int     user_id;
  int64_t timestamp;       // Unix time
  char    message[256];
} version_info;

// Fill out[] with up to max_versions entries, newest first.
// Returns number of entries written, or -1 on error.
int surface_history_list(surface_history *h, version_info *out,
                         int max_versions);

// Deserialise and return the surface at version_id. Caller owns result.
// Returns NULL if version not found or on error.
quad_surface *surface_history_checkout(surface_history *h, int64_t version_id);

// Compute per-vertex displacement between two versions.
// Returns heap-allocated float array of length (*out_count * 3) — one vec3
// per vertex — or NULL on error.  Caller must free.
float *surface_history_diff(surface_history *h, int64_t v1, int64_t v2,
                            int *out_count);

// Enable periodic auto-save. After this call, surface_history_autosave_tick
// must be called each frame (or timer tick) with the current surface.
void surface_history_enable_autosave(surface_history *h, int interval_seconds);

// Call periodically when autosave is enabled. Commits if the surface has
// changed since the last commit and the interval has elapsed.
void surface_history_autosave_tick(surface_history *h,
                                   const quad_surface *surface);
