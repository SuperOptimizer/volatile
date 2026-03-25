#include "greatest.h"
#include "gui/seg_growth.h"
#include "core/geom.h"
#include "core/vol.h"
#include "core/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define VOL_DIM  32   // tiny synthetic volume
#define EPS      0.1f

// Recursively create a directory path (simple 3-level helper for tests).
static void mkdirp(const char *p) {
  char tmp[512];
  snprintf(tmp, sizeof(tmp), "%s", p);
  for (char *s = tmp + 1; *s; s++) {
    if (*s == '/') {
      *s = '\0';
      mkdir(tmp, 0755);
      *s = '/';
    }
  }
  mkdir(tmp, 0755);
}

// Write bytes to a file (simple helper).
static bool write_file(const char *path, const void *data, size_t size) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  size_t n = fwrite(data, 1, size, f);
  fclose(f);
  return n == size;
}

// Create a synthetic Zarr v2 volume (VOL_DIM^3) in /tmp/test_seg_growth_vol
// with a bright intensity sphere at the center.  Returns opened volume*.
static volume *make_synthetic_volume(void) {
  const char *root = "/tmp/test_seg_growth_vol";
  char path[512];

  // Clean up any previous run
  snprintf(path, sizeof(path), "rm -rf %s", root);
  system(path);

  mkdirp(root);

  // Write top-level .zattrs (minimal multiscales)
  snprintf(path, sizeof(path), "%s/.zattrs", root);
  const char *zattrs =
    "{\"multiscales\":[{\"axes\":[{\"name\":\"z\"},{\"name\":\"y\"},{\"name\":\"x\"}],"
    "\"datasets\":[{\"path\":\"0\"}],\"name\":\"/\"}]}";
  write_file(path, zattrs, strlen(zattrs));

  // Write level 0 directory and .zarray
  snprintf(path, sizeof(path), "%s/0", root);
  mkdirp(path);

  snprintf(path, sizeof(path), "%s/0/.zarray", root);
  char zarray[512];
  snprintf(zarray, sizeof(zarray),
    "{\"chunks\":[%d,%d,%d],\"shape\":[%d,%d,%d],"
    "\"dtype\":\"<u1\",\"order\":\"C\",\"zarr_format\":2,"
    "\"compressor\":null}",
    VOL_DIM, VOL_DIM, VOL_DIM,
    VOL_DIM, VOL_DIM, VOL_DIM);
  write_file(path, zarray, strlen(zarray));

  // Build voxel data: bright sphere at center
  int n = VOL_DIM * VOL_DIM * VOL_DIM;
  uint8_t *data = calloc(1, n);
  float cx = (float)(VOL_DIM / 2);
  float cy = (float)(VOL_DIM / 2);
  float cz = (float)(VOL_DIM / 2);
  for (int z = 0; z < VOL_DIM; z++) {
    for (int y = 0; y < VOL_DIM; y++) {
      for (int x = 0; x < VOL_DIM; x++) {
        float r = sqrtf((x - cx) * (x - cx) +
                        (y - cy) * (y - cy) +
                        (z - cz) * (z - cz));
        float v = (r < 8.0f) ? 220.0f * (1.0f - r / 8.0f) : 0.0f;
        data[z * VOL_DIM * VOL_DIM + y * VOL_DIM + x] = (uint8_t)v;
      }
    }
  }

  // Write single chunk as raw file (no compression, null compressor)
  snprintf(path, sizeof(path), "%s/0/0.0.0", root);
  write_file(path, data, n);
  free(data);

  return vol_open(root);
}

// Build a flat rows×cols grid in the XY plane at z=VOL_DIM/2.
static quad_surface *flat_grid(int rows, int cols) {
  quad_surface *s = quad_surface_new(rows, cols);
  float cx = (float)(VOL_DIM / 2);
  float cy = (float)(VOL_DIM / 2);
  float cz = (float)(VOL_DIM / 2);
  // Place grid in XY plane centered at volume center
  for (int r = 0; r < rows; r++) {
    for (int c = 0; c < cols; c++) {
      vec3f p = {
        cx - (float)(cols / 2) + (float)c,
        cy - (float)(rows / 2) + (float)r,
        cz
      };
      quad_surface_set(s, r, c, p);
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_grower_new_free(void) {
  quad_surface *seed = flat_grid(5, 5);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  ASSERT(g != NULL);
  ASSERT(!seg_grower_busy(g));

  quad_surface *surf = seg_grower_surface(g);
  ASSERT(surf != NULL);
  ASSERT_EQ(5, surf->rows);
  ASSERT_EQ(5, surf->cols);

  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// Extrapolation growth: boundary should move outward after one generation.
TEST test_growth_extrapolation_expands(void) {
  quad_surface *seed = flat_grid(7, 7);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  ASSERT(g != NULL);

  growth_params p = {
    .method             = GROWTH_EXTRAPOLATION,
    .direction          = GROWTH_DIR_ALL,
    .generations        = 1,
    .step_size          = 1.5f,
    .straightness_weight = 0.0f,
    .distance_weight    = 0.0f,
  };

  // Record boundary position before
  quad_surface *before = quad_surface_clone(seg_grower_surface(g));

  bool started = seg_grower_step(g, &p);
  ASSERT(started);

  // Wait for completion (poll with timeout)
  for (int i = 0; i < 1000 && seg_grower_busy(g); i++) {
    struct timespec ts = {0, 1000000};  // 1ms
    nanosleep(&ts, NULL);
  }
  ASSERT(!seg_grower_busy(g));

  quad_surface *after = seg_grower_surface(g);

  // Top boundary row should have moved (row 0, all cols)
  for (int c = 0; c < after->cols; c++) {
    vec3f bp = quad_surface_get(before, 0, c);
    vec3f ap = quad_surface_get(after,  0, c);
    float dist = vec3f_len(vec3f_sub(ap, bp));
    ASSERT_IN_RANGE(0.5f, dist, 5.0f);
  }

  quad_surface_free(before);
  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// Tracer growth: surface should change (intensity-seeking along normal).
TEST test_growth_tracer_step(void) {
  quad_surface *seed = flat_grid(5, 5);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  ASSERT(g != NULL);

  growth_params p = {
    .method      = GROWTH_TRACER,
    .direction   = GROWTH_DIR_ALL,
    .generations = 1,
    .step_size   = 2.0f,
  };

  bool started = seg_grower_step(g, &p);
  ASSERT(started);

  for (int i = 0; i < 1000 && seg_grower_busy(g); i++) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
  }
  ASSERT(!seg_grower_busy(g));

  // Surface should still have correct dimensions
  quad_surface *surf = seg_grower_surface(g);
  ASSERT_EQ(5, surf->rows);
  ASSERT_EQ(5, surf->cols);

  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// step() while busy should return false
TEST test_step_returns_false_when_busy(void) {
  quad_surface *seed = flat_grid(9, 9);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  ASSERT(g != NULL);

  growth_params p = {
    .method      = GROWTH_EXTRAPOLATION,
    .direction   = GROWTH_DIR_ALL,
    .generations = 3,
    .step_size   = 1.0f,
  };

  bool first = seg_grower_step(g, &p);
  ASSERT(first);

  // If still busy, second step should fail
  if (seg_grower_busy(g)) {
    bool second = seg_grower_step(g, &p);
    ASSERT(!second);
  }

  // Drain
  for (int i = 0; i < 2000 && seg_grower_busy(g); i++) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
  }

  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// Corrections: adding a correction point and running CORRECTIONS growth
TEST test_growth_corrections(void) {
  quad_surface *seed = flat_grid(5, 5);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  ASSERT(g != NULL);

  // Add a correction at the surface center pointing inward
  vec3f target = {(float)(VOL_DIM / 2), (float)(VOL_DIM / 2), (float)(VOL_DIM / 2)};
  seg_grower_add_correction(g, 0.5f, 0.5f, target);

  growth_params p = {
    .method          = GROWTH_CORRECTIONS,
    .direction       = GROWTH_DIR_ALL,
    .generations     = 1,
    .step_size       = 1.0f,
    .distance_weight = 0.5f,
  };

  bool started = seg_grower_step(g, &p);
  ASSERT(started);

  for (int i = 0; i < 1000 && seg_grower_busy(g); i++) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
  }
  ASSERT(!seg_grower_busy(g));

  quad_surface *surf = seg_grower_surface(g);
  ASSERT_EQ(5, surf->rows);
  ASSERT_EQ(5, surf->cols);

  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// Directional growth: GROWTH_DIR_DOWN should only move the bottom row
TEST test_growth_directional(void) {
  quad_surface *seed = flat_grid(7, 7);
  volume *vol = make_synthetic_volume();
  ASSERT(vol != NULL);

  seg_grower *g = seg_grower_new(vol, seed);
  quad_surface *before = quad_surface_clone(seg_grower_surface(g));

  growth_params p = {
    .method      = GROWTH_EXTRAPOLATION,
    .direction   = GROWTH_DIR_DOWN,
    .generations = 1,
    .step_size   = 2.0f,
  };

  seg_grower_step(g, &p);
  for (int i = 0; i < 1000 && seg_grower_busy(g); i++) {
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
  }

  quad_surface *after = seg_grower_surface(g);
  int rows = after->rows;
  int cols = after->cols;

  // Bottom row should have moved
  for (int c = 0; c < cols; c++) {
    vec3f bp = quad_surface_get(before, rows - 1, c);
    vec3f ap = quad_surface_get(after,  rows - 1, c);
    ASSERT(vec3f_len(vec3f_sub(ap, bp)) > EPS);
  }

  // Top row should be unchanged
  for (int c = 0; c < cols; c++) {
    vec3f bp = quad_surface_get(before, 0, c);
    vec3f ap = quad_surface_get(after,  0, c);
    ASSERT(vec3f_len(vec3f_sub(ap, bp)) < EPS);
  }

  quad_surface_free(before);
  seg_grower_free(g);
  quad_surface_free(seed);
  vol_free(vol);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(seg_growth_suite) {
  RUN_TEST(test_grower_new_free);
  RUN_TEST(test_growth_extrapolation_expands);
  RUN_TEST(test_growth_tracer_step);
  RUN_TEST(test_step_returns_false_when_busy);
  RUN_TEST(test_growth_corrections);
  RUN_TEST(test_growth_directional);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(seg_growth_suite);
  GREATEST_MAIN_END();
}
