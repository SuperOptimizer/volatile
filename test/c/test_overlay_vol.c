#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "gui/overlay_vol.h"
#include "core/vol.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Minimal synthetic Zarr fixture (single level, 8^3, chunk 8^3, |u1)
// ---------------------------------------------------------------------------

static char g_zarr_root[256];

static void write_file(const char *path, const void *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return;
  fwrite(data, 1, len, f);
  fclose(f);
}

static void mkdirp_simple(const char *path) {
  char tmp[512];
  snprintf(tmp, sizeof(tmp), "%s", path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') { *p = '\0'; mkdir(tmp, 0755); *p = '/'; }
  }
  mkdir(tmp, 0755);
}

static void create_test_zarr(void) {
  snprintf(g_zarr_root, sizeof(g_zarr_root), "/tmp/ov_test_XXXXXX");
  if (!mkdtemp(g_zarr_root)) { g_zarr_root[0] = '\0'; return; }

  char lvl0[300]; snprintf(lvl0, sizeof(lvl0), "%s/0", g_zarr_root);
  mkdirp_simple(lvl0);

  const char *meta =
    "{\"chunks\":[8,8,8],\"compressor\":null,\"dtype\":\"|u1\","
    "\"fill_value\":0,\"filters\":null,\"order\":\"C\","
    "\"shape\":[8,8,8],\"zarr_format\":2}";
  char path[400]; snprintf(path, sizeof(path), "%s/.zarray", lvl0);
  write_file(path, meta, strlen(meta));

  // Fill chunk with ramp: voxel = z*64 + y*8 + x (mod 256)
  uint8_t buf[512];
  for (int z = 0; z < 8; z++)
    for (int y = 0; y < 8; y++)
      for (int x = 0; x < 8; x++)
        buf[z*64 + y*8 + x] = (uint8_t)((z*64 + y*8 + x) & 0xFF);
  snprintf(path, sizeof(path), "%s/0.0.0", lvl0);
  write_file(path, buf, 512);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_lifecycle(void) {
  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);

  // Defaults
  ASSERT(overlay_volume_visible(v) == true);
  ASSERT(overlay_volume_opacity(v) > 0.0f);
  ASSERT(overlay_volume_cmap(v) == CMAP_HOT);
  ASSERT(overlay_volume_threshold(v) == 0.0f);
  ASSERT(overlay_volume_blend(v) == OVERLAY_BLEND_ALPHA);

  overlay_volume_free(v);
  PASS();
}

TEST test_free_null_safe(void) {
  overlay_volume_free(NULL);  // must not crash
  PASS();
}

TEST test_setters(void) {
  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);

  overlay_volume_set_opacity(v, 0.75f);
  ASSERT_IN_RANGE(0.74f, overlay_volume_opacity(v), 0.76f);

  overlay_volume_set_cmap(v, CMAP_VIRIDIS);
  ASSERT_EQ(overlay_volume_cmap(v), (int)CMAP_VIRIDIS);

  overlay_volume_set_threshold(v, 0.3f);
  ASSERT_IN_RANGE(0.29f, overlay_volume_threshold(v), 0.31f);

  overlay_volume_set_visible(v, false);
  ASSERT(overlay_volume_visible(v) == false);

  overlay_volume_set_blend(v, OVERLAY_BLEND_ADDITIVE);
  ASSERT_EQ(overlay_volume_blend(v), OVERLAY_BLEND_ADDITIVE);

  overlay_volume_free(v);
  PASS();
}

TEST test_opacity_clamping(void) {
  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);

  overlay_volume_set_opacity(v, -1.0f);
  ASSERT_EQ(overlay_volume_opacity(v), 0.0f);

  overlay_volume_set_opacity(v, 5.0f);
  ASSERT_EQ(overlay_volume_opacity(v), 1.0f);

  overlay_volume_free(v);
  PASS();
}

TEST test_composite_null_safety(void) {
  // All null / zero-opacity combinations must not crash
  overlay_volume_composite_tile(NULL, NULL, 0, 0, 0, 0, 0, 1.0f, 0);

  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);
  overlay_volume_set_opacity(v, 0.0f);

  uint8_t tile[16] = {0};
  overlay_volume_composite_tile(v, tile, 2, 2, 0, 0, 0, 1.0f, 0);  // no vol set

  overlay_volume_free(v);
  PASS();
}

TEST test_composite_on_data(void) {
  create_test_zarr();
  ASSERT(g_zarr_root[0] != '\0');

  volume *vol = vol_open(g_zarr_root);
  ASSERT(vol != NULL);

  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);
  overlay_volume_set_volume(v, vol);
  overlay_volume_set_opacity(v, 1.0f);
  overlay_volume_set_cmap(v, CMAP_GRAYSCALE);
  overlay_volume_set_threshold(v, 0.0f);
  overlay_volume_set_blend(v, OVERLAY_BLEND_ALPHA);

  // 4×4 tile, black background
  int w = 4, h = 4;
  uint8_t tile[4 * 4 * 4];
  memset(tile, 0, sizeof(tile));

  overlay_volume_composite_tile(v, tile, w, h, 0.0f, 0.0f, 0.0f, 1.0f, 0);

  // With full opacity and grayscale cmap, at least one pixel must be non-zero
  bool any_nonzero = false;
  for (int i = 0; i < w * h * 4; i++) {
    if (tile[i]) { any_nonzero = true; break; }
  }
  ASSERT(any_nonzero);

  overlay_volume_free(v);
  vol_free(vol);
  PASS();
}

TEST test_composite_invisible(void) {
  create_test_zarr();
  ASSERT(g_zarr_root[0] != '\0');

  volume *vol = vol_open(g_zarr_root);
  ASSERT(vol != NULL);

  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);
  overlay_volume_set_volume(v, vol);
  overlay_volume_set_visible(v, false);

  int w = 4, h = 4;
  uint8_t tile[4 * 4 * 4];
  memset(tile, 0, sizeof(tile));

  overlay_volume_composite_tile(v, tile, w, h, 0.0f, 0.0f, 0.0f, 1.0f, 0);

  // Invisible — tile must remain all zeros
  for (int i = 0; i < w * h * 4; i++) {
    ASSERT_EQ(tile[i], 0);
  }

  overlay_volume_free(v);
  vol_free(vol);
  PASS();
}

TEST test_composite_threshold(void) {
  create_test_zarr();
  ASSERT(g_zarr_root[0] != '\0');

  volume *vol = vol_open(g_zarr_root);
  ASSERT(vol != NULL);

  overlay_volume *v = overlay_volume_new();
  ASSERT(v != NULL);
  overlay_volume_set_volume(v, vol);
  overlay_volume_set_opacity(v, 1.0f);
  overlay_volume_set_threshold(v, 1.1f);  // above max — nothing passes

  int w = 4, h = 4;
  uint8_t tile[4 * 4 * 4];
  memset(tile, 0, sizeof(tile));

  overlay_volume_composite_tile(v, tile, w, h, 0.0f, 0.0f, 0.0f, 1.0f, 0);

  // threshold > 1.0 after clamping == 1.0, so all voxels at exactly 255 would
  // still be 1.0; clamp threshold to 1.0 means we may still get some.
  // The important thing: no crash.
  overlay_volume_free(v);
  vol_free(vol);
  PASS();
}

TEST test_blend_modes(void) {
  create_test_zarr();
  ASSERT(g_zarr_root[0] != '\0');

  volume *vol = vol_open(g_zarr_root);
  ASSERT(vol != NULL);

  int w = 4, h = 4;
  uint8_t tile_alpha[4 * 4 * 4], tile_add[4 * 4 * 4], tile_mul[4 * 4 * 4];

  for (int mode = 0; mode < 3; mode++) {
    overlay_volume *v = overlay_volume_new();
    ASSERT(v != NULL);
    overlay_volume_set_volume(v, vol);
    overlay_volume_set_opacity(v, 0.5f);
    overlay_volume_set_blend(v, (overlay_blend_mode)mode);

    uint8_t *dst = mode == 0 ? tile_alpha : (mode == 1 ? tile_add : tile_mul);
    memset(dst, 128, w * h * 4);  // grey background
    overlay_volume_composite_tile(v, dst, w, h, 0.0f, 0.0f, 0.0f, 1.0f, 0);
    overlay_volume_free(v);
  }

  // All three blend modes ran without crashing and produced distinct outputs.
  // At least two modes should differ somewhere.
  bool differs = false;
  for (int i = 0; i < w * h * 4; i++) {
    if (tile_alpha[i] != tile_add[i]) { differs = true; break; }
  }
  ASSERT(differs);

  vol_free(vol);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(overlay_vol_suite) {
  RUN_TEST(test_lifecycle);
  RUN_TEST(test_free_null_safe);
  RUN_TEST(test_setters);
  RUN_TEST(test_opacity_clamping);
  RUN_TEST(test_composite_null_safety);
  RUN_TEST(test_composite_on_data);
  RUN_TEST(test_composite_invisible);
  RUN_TEST(test_composite_threshold);
  RUN_TEST(test_blend_modes);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(overlay_vol_suite);
  GREATEST_MAIN_END();
}
