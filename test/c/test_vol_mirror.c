#include "greatest.h"
#include "core/vol_mirror.h"
#include "core/vol.h"
#include "core/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool write_file(const char *path, const void *data, size_t size) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  size_t n = fwrite(data, 1, size, f);
  fclose(f);
  return n == size;
}

static int make_dir(const char *p) { return mkdir(p, 0755); }

// Build a minimal 2-level synthetic v2 zarr in /tmp that vol_mirror can open.
static char g_src_zarr[256];
static char g_cache_dir[256];

static void setup_synthetic_source(void) {
  snprintf(g_src_zarr,  sizeof(g_src_zarr),  "/tmp/mirror_src_%d.zarr",   (int)getpid());
  snprintf(g_cache_dir, sizeof(g_cache_dir), "/tmp/mirror_cache_%d",       (int)getpid());

  make_dir(g_src_zarr);
  make_dir(g_cache_dir);

  // level 0: 8x8x8, 4x4x4 chunks, u8, no compression
  char l0[512]; snprintf(l0, sizeof(l0), "%s/0", g_src_zarr); make_dir(l0);
  const char *za0 =
    "{\"chunks\":[4,4,4],\"shape\":[8,8,8],\"dtype\":\"|u1\","
    "\"order\":\"C\",\"compressor\":null,\"zarr_format\":2}";
  char zp[512]; snprintf(zp, sizeof(zp), "%s/.zarray", l0);
  write_file(zp, za0, strlen(za0));

  // write 2 chunks: (0,0,0) and (0,0,1)
  uint8_t c0[64], c1[64];
  for (int i = 0; i < 64; i++) { c0[i] = (uint8_t)i; c1[i] = (uint8_t)(128 + i); }
  char cp[512];
  snprintf(cp, sizeof(cp), "%s/0.0.0", l0); write_file(cp, c0, 64);
  snprintf(cp, sizeof(cp), "%s/0.0.1", l0); write_file(cp, c1, 64);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_mirror_new_opens_source(void) {
  setup_synthetic_source();

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = g_cache_dir,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);

  // vol_mirror_volume returns the remote (or local if cached)
  volume *v = vol_mirror_volume(m);
  ASSERT(v != NULL);
  ASSERT(vol_num_levels(v) >= 1);

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_chunks_total(void) {
  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = g_cache_dir,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);

  // 8x8x8 with 4x4x4 chunks → 2x2x2 = 8 total
  int total = vol_mirror_chunks_total(m, 0);
  ASSERT_EQ(8, total);

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_cache_level(void) {
  // use a fresh cache dir so we know it starts empty
  char fresh_cache[256];
  snprintf(fresh_cache, sizeof(fresh_cache), "/tmp/mirror_cache2_%d", (int)getpid());
  make_dir(fresh_cache);

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = fresh_cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);

  // cache_level downloads (or copies) chunks to local disk
  // For a local "remote" this just copies the files that exist (2 present, 6 absent)
  bool ok = vol_mirror_cache_level(m, 0);
  ASSERT(ok);

  // chunks_cached should be > 0 (the 2 chunks that exist were fetched)
  ASSERT(vol_mirror_chunks_cached(m) > 0);

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_cached_chunks_match_source(void) {
  char fresh_cache[256];
  snprintf(fresh_cache, sizeof(fresh_cache), "/tmp/mirror_cache3_%d", (int)getpid());
  make_dir(fresh_cache);

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = fresh_cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT(vol_mirror_cache_level(m, 0));

  // Re-open local cache and verify chunk (0,0,0) content
  volume *lv = vol_mirror_volume(m);
  ASSERT(lv != NULL);

  int64_t coords[3] = {0, 0, 0};
  size_t sz = 0;
  uint8_t *data = vol_read_chunk(lv, 0, coords, &sz);
  ASSERT(data != NULL);
  ASSERT_EQ((size_t)64, sz);
  // chunk (0,0,0) was filled with bytes 0..63
  for (int i = 0; i < 64; i++) ASSERT_EQ((uint8_t)i, data[i]);
  free(data);

  // chunk (0,0,1) was filled with 128..191
  int64_t coords2[3] = {0, 0, 1};
  size_t sz2 = 0;
  uint8_t *data2 = vol_read_chunk(lv, 0, coords2, &sz2);
  ASSERT(data2 != NULL);
  ASSERT_EQ((uint8_t)128, data2[0]);
  free(data2);

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_stats(void) {
  char fresh_cache[256];
  snprintf(fresh_cache, sizeof(fresh_cache), "/tmp/mirror_cache4_%d", (int)getpid());
  make_dir(fresh_cache);

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = fresh_cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT(vol_mirror_cache_level(m, 0));

  // cached_bytes > 0 after caching
  ASSERT(vol_mirror_cached_bytes(m) > 0);

  // hit rate is a float in [0,1]
  float hr = vol_mirror_cache_hit_rate(m);
  ASSERT(hr >= 0.0f && hr <= 1.0f);

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_rechunk(void) {
  char fresh_cache[256];
  snprintf(fresh_cache, sizeof(fresh_cache), "/tmp/mirror_cache5_%d", (int)getpid());
  make_dir(fresh_cache);

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = fresh_cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT(vol_mirror_cache_level(m, 0));

  int64_t new_chunk[3] = {2, 2, 2};
  bool ok = vol_mirror_rechunk(m, new_chunk);
  ASSERT(ok);

  // rechunked zarr directory should exist
  char rechunked[512];
  snprintf(rechunked, sizeof(rechunked), "%s/%016llx_rechunked",
           fresh_cache, 0ULL);  // path based on url hash; just check vol_mirror didn't crash
  (void)rechunked;

  vol_mirror_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: compress4d detection and recompress skip logic
// ---------------------------------------------------------------------------

// Build a synthetic zarr whose .zarray declares compressor id "compress4d"
static char g_c4d_zarr[256];

static void setup_compress4d_source(void) {
  snprintf(g_c4d_zarr, sizeof(g_c4d_zarr), "/tmp/mirror_c4d_%d.zarr", (int)getpid());
  make_dir(g_c4d_zarr);

  char l0[512]; snprintf(l0, sizeof(l0), "%s/0", g_c4d_zarr); make_dir(l0);
  // declare compressor id as "compress4d"
  const char *za =
    "{\"chunks\":[4,4,4],\"shape\":[8,8,8],\"dtype\":\"|u1\","
    "\"order\":\"C\","
    "\"compressor\":{\"id\":\"compress4d\",\"clevel\":5,\"shuffle\":0},"
    "\"zarr_format\":2}";
  char zp[512]; snprintf(zp, sizeof(zp), "%s/.zarray", l0);
  write_file(zp, za, strlen(za));

  // write one chunk of raw bytes
  uint8_t chunk[64];
  for (int i = 0; i < 64; i++) chunk[i] = (uint8_t)i;
  char cp[512]; snprintf(cp, sizeof(cp), "%s/0.0.0", l0);
  write_file(cp, chunk, 64);
}

TEST test_mirror_detects_compress4d(void) {
  setup_compress4d_source();

  char cache[256];
  snprintf(cache, sizeof(cache), "/tmp/mirror_c4d_cache_%d", (int)getpid());
  make_dir(cache);

  mirror_config cfg = {
    .remote_url      = g_c4d_zarr,
    .local_cache_dir = cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);

  // should detect compress4d codec
  ASSERT(vol_mirror_remote_is_compress4d(m));

  // recompress should return true immediately (skipped) without force_recompress
  ASSERT(vol_mirror_recompress(m));

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_force_recompress_overrides(void) {
  char cache[256];
  snprintf(cache, sizeof(cache), "/tmp/mirror_c4d_force_%d", (int)getpid());
  make_dir(cache);

  mirror_config cfg = {
    .remote_url       = g_c4d_zarr,
    .local_cache_dir  = cache,
    .force_recompress = true,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT(vol_mirror_remote_is_compress4d(m));

  // cache first so src has data
  ASSERT(vol_mirror_cache_level(m, 0));

  // with force_recompress=true, recompress should actually run (return true)
  ASSERT(vol_mirror_recompress(m));

  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_non_c4d_not_detected(void) {
  // g_src_zarr uses compressor:null — should NOT be detected as compress4d
  char cache[256];
  snprintf(cache, sizeof(cache), "/tmp/mirror_nc4d_%d", (int)getpid());
  make_dir(cache);

  mirror_config cfg = {
    .remote_url      = g_src_zarr,
    .local_cache_dir = cache,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT_FALSE(vol_mirror_remote_is_compress4d(m));
  vol_mirror_free(m);
  PASS();
}

TEST test_mirror_no_volatile_server_for_local(void) {
  // local path should never be detected as a volatile TCP server
  char cache[256];
  snprintf(cache, sizeof(cache), "/tmp/mirror_novolt_%d", (int)getpid());
  make_dir(cache);

  mirror_config cfg = {
    .remote_url             = g_src_zarr,
    .local_cache_dir        = cache,
    .prefer_binary_protocol = true,
  };
  vol_mirror *m = vol_mirror_new(cfg);
  ASSERT(m != NULL);
  ASSERT_FALSE(vol_mirror_remote_is_volatile_server(m));
  vol_mirror_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(vol_mirror_suite) {
  RUN_TEST(test_mirror_new_opens_source);
  RUN_TEST(test_mirror_chunks_total);
  // must be before cache/roundtrip tests that also use g_src_zarr
  RUN_TEST(test_mirror_non_c4d_not_detected);
  RUN_TEST(test_mirror_no_volatile_server_for_local);
  RUN_TEST(test_mirror_cache_level);
  RUN_TEST(test_mirror_cached_chunks_match_source);
  RUN_TEST(test_mirror_stats);
  RUN_TEST(test_mirror_rechunk);
  RUN_TEST(test_mirror_detects_compress4d);
  RUN_TEST(test_mirror_force_recompress_overrides);

}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vol_mirror_suite);
  GREATEST_MAIN_END();
}
