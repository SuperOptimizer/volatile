#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "server/stream.h"
#include "core/vol.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Synthetic Zarr v2 fixture
//
// Layout:
//   <root>/0/.zarray  — 16x16x16, chunks 8x8x8, dtype |u1, no compressor
//   <root>/0/0.0.0    — raw 8^3 = 512 bytes chunk
//   <root>/0/1.0.0    — raw 8^3 = 512 bytes chunk
//   <root>/0/0.1.0    — ...
//   ... (8 chunks total at level 0)
//   <root>/1/.zarray  — 8x8x8, chunks 8x8x8 (one chunk)
//   <root>/1/0.0.0    — raw 8^3 = 512 bytes
//   <root>/2/.zarray  — 4x4x4, chunks 4x4x4 (one chunk)
//   <root>/2/0.0.0    — raw 4^3 = 64 bytes
// ---------------------------------------------------------------------------

static void write_file(const char *path, const void *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return;
  fwrite(data, 1, len, f);
  fclose(f);
}

static void mkdirp(const char *path) {
  char tmp[512];
  snprintf(tmp, sizeof(tmp), "%s", path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      mkdir(tmp, 0755);
      *p = '/';
    }
  }
  mkdir(tmp, 0755);
}

static char g_zarr_root[256];

// Build a .zarray JSON for a given shape / chunk shape.
static void write_zarray(const char *dir, int sz, int csz) {
  char meta[512];
  snprintf(meta, sizeof(meta),
    "{\"chunks\":[%d,%d,%d],\"compressor\":null,\"dtype\":\"|u1\","
    "\"fill_value\":0,\"filters\":null,\"order\":\"C\","
    "\"shape\":[%d,%d,%d],\"zarr_format\":2}",
    csz, csz, csz, sz, sz, sz);
  char path[512];
  snprintf(path, sizeof(path), "%s/.zarray", dir);
  write_file(path, meta, strlen(meta));
}

// Write one raw chunk file filled with a distinguishing byte pattern.
static void write_chunk(const char *dir, const char *key, int size, uint8_t fill) {
  uint8_t *buf = malloc((size_t)size);
  if (!buf) return;
  memset(buf, fill, (size_t)size);
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, key);
  write_file(path, buf, (size_t)size);
  free(buf);
}

// Build the synthetic 3-level Zarr in /tmp and store path in g_zarr_root.
static void create_synthetic_zarr(void) {
  snprintf(g_zarr_root, sizeof(g_zarr_root), "/tmp/volatile_test_zarr_XXXXXX");
  // mkdtemp in place
  if (!mkdtemp(g_zarr_root)) {
    g_zarr_root[0] = '\0';
    return;
  }

  // level 0: 16^3, chunk 8^3 → 2x2x2 = 8 chunks
  char lvl0[256]; snprintf(lvl0, sizeof(lvl0), "%s/0", g_zarr_root);
  mkdirp(lvl0);
  write_zarray(lvl0, 16, 8);
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 2; y++)
      for (int x = 0; x < 2; x++) {
        char key[32]; snprintf(key, sizeof(key), "%d.%d.%d", z, y, x);
        write_chunk(lvl0, key, 512, (uint8_t)(z*4 + y*2 + x + 1));
      }

  // level 1: 8^3, chunk 8^3 → 1 chunk
  char lvl1[256]; snprintf(lvl1, sizeof(lvl1), "%s/1", g_zarr_root);
  mkdirp(lvl1);
  write_zarray(lvl1, 8, 8);
  write_chunk(lvl1, "0.0.0", 512, 42);

  // level 2: 4^3, chunk 4^3 → 1 chunk (64 bytes)
  char lvl2[256]; snprintf(lvl2, sizeof(lvl2), "%s/2", g_zarr_root);
  mkdirp(lvl2);
  write_zarray(lvl2, 4, 4);
  write_chunk(lvl2, "0.0.0", 64, 7);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_open_synthetic_vol(void) {
  create_synthetic_zarr();
  ASSERT(g_zarr_root[0] != '\0');

  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);
  ASSERT_EQ(vol_num_levels(v), 3);

  vol_free(v);
  PASS();
}

TEST test_streamer_coarsest_first(void) {
  ASSERT(g_zarr_root[0] != '\0');
  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);

  // Request the full volume region
  chunk_streamer *s = chunk_streamer_new(v, 0,0,0, 16,16,16);
  ASSERT(s != NULL);
  ASSERT_EQ(chunk_streamer_num_levels(s), 3);

  // First packet must be from the coarsest level (level 2)
  stream_packet pkt;
  bool ok = chunk_streamer_next(s, &pkt);
  ASSERT(ok);
  ASSERT_EQ(pkt.level, 2);
  free(pkt.data);

  chunk_streamer_free(s);
  vol_free(v);
  PASS();
}

TEST test_streamer_level_ordering(void) {
  ASSERT(g_zarr_root[0] != '\0');
  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);

  chunk_streamer *s = chunk_streamer_new(v, 0,0,0, 16,16,16);
  ASSERT(s != NULL);

  int prev_level = INT32_MAX;
  int packets = 0;
  stream_packet pkt;

  while (chunk_streamer_next(s, &pkt)) {
    // Level must be monotonically non-increasing across packets
    ASSERT(pkt.level <= prev_level);
    prev_level = pkt.level;
    packets++;
    free(pkt.data);
  }

  // 3 levels: level2=1 chunk, level1=1 chunk, level0=8 chunks → 10 total
  ASSERT_EQ(packets, 10);

  chunk_streamer_free(s);
  vol_free(v);
  PASS();
}

TEST test_streamer_sizes_increase_per_level(void) {
  ASSERT(g_zarr_root[0] != '\0');
  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);

  chunk_streamer *s = chunk_streamer_new(v, 0,0,0, 16,16,16);
  ASSERT(s != NULL);

  // Collect per-level total bytes
  size_t level_bytes[3] = {0, 0, 0};
  stream_packet pkt;
  while (chunk_streamer_next(s, &pkt)) {
    if (pkt.level >= 0 && pkt.level < 3)
      level_bytes[pkt.level] += pkt.size;
    free(pkt.data);
  }

  // level0 (8 chunks×512 bytes) > level1 (1 chunk×512) ≥ level2 (1 chunk×64)
  ASSERT(level_bytes[0] > level_bytes[1]);
  ASSERT(level_bytes[1] >= level_bytes[2]);

  chunk_streamer_free(s);
  vol_free(v);
  PASS();
}

TEST test_streamer_is_last_for_level(void) {
  ASSERT(g_zarr_root[0] != '\0');
  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);

  chunk_streamer *s = chunk_streamer_new(v, 0,0,0, 16,16,16);
  ASSERT(s != NULL);

  int last_flags = 0;
  stream_packet pkt;
  while (chunk_streamer_next(s, &pkt)) {
    if (pkt.is_last_for_level) last_flags++;
    free(pkt.data);
  }

  // One is_last_for_level=true per level
  ASSERT_EQ(last_flags, 3);

  chunk_streamer_free(s);
  vol_free(v);
  PASS();
}

TEST test_streamer_subregion(void) {
  ASSERT(g_zarr_root[0] != '\0');
  volume *v = vol_open(g_zarr_root);
  ASSERT(v != NULL);

  // Request only the first octant: [0,8)^3 → at level 0 only 1 chunk (0.0.0)
  chunk_streamer *s = chunk_streamer_new(v, 0,0,0, 8,8,8);
  ASSERT(s != NULL);

  int level0_packets = 0;
  stream_packet pkt;
  while (chunk_streamer_next(s, &pkt)) {
    if (pkt.level == 0) level0_packets++;
    free(pkt.data);
  }
  ASSERT_EQ(level0_packets, 1);

  chunk_streamer_free(s);
  vol_free(v);
  PASS();
}

TEST test_streamer_null_vol(void) {
  chunk_streamer *s = chunk_streamer_new(NULL, 0,0,0, 16,16,16);
  ASSERT(s == NULL);
  PASS();
}

TEST test_streamer_free_null_safe(void) {
  chunk_streamer_free(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(stream_suite) {
  RUN_TEST(test_open_synthetic_vol);
  RUN_TEST(test_streamer_coarsest_first);
  RUN_TEST(test_streamer_level_ordering);
  RUN_TEST(test_streamer_sizes_increase_per_level);
  RUN_TEST(test_streamer_is_last_for_level);
  RUN_TEST(test_streamer_subregion);
  RUN_TEST(test_streamer_null_vol);
  RUN_TEST(test_streamer_free_null_safe);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  log_set_level(LOG_WARN);  // silence vol_open INFO noise during tests
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(stream_suite);
  GREATEST_MAIN_END();
}
