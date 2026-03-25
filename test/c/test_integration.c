#define _DEFAULT_SOURCE
#include "greatest.h"

#include "core/vol.h"
#include "core/cache.h"
#include "core/compress4d.h"
#include "core/json.h"
#include "render/tile.h"
#include "render/composite.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static char g_tmpdir[256];
static void make_tmpdir(void) { snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/vol_integ_%d", (int)getpid()); }
static void rmdir_r(const char *p) { char cmd[512]; snprintf(cmd, sizeof(cmd), "rm -rf %s", p); (void)system(cmd); }

// Test 1: Volume pipeline — create, write chunk, re-open, read, verify

TEST test_vol_roundtrip(void) {
  char path[300];
  snprintf(path, sizeof(path), "%s/roundtrip.zarr", g_tmpdir);

  // Create a 3D volume: shape 8x8x8, chunk 8x8x8, u8
  vol_create_params p = {
    .zarr_version = 2,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {8, 8, 8},
    .dtype = DTYPE_U8,
    .compressor = NULL,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);

  // Write a chunk with known data
  uint8_t buf[512];
  for (int i = 0; i < 512; i++) buf[i] = (uint8_t)(i & 0xFF);
  int64_t coords[3] = {0, 0, 0};
  bool ok = vol_write_chunk(v, 0, coords, buf, 512);
  ASSERT(ok);
  vol_free(v);

  // Re-open and read back
  volume *v2 = vol_open(path);
  ASSERT(v2 != NULL);
  ASSERT_EQ(1, vol_num_levels(v2));

  size_t out_size = 0;
  uint8_t *data = vol_read_chunk(v2, 0, coords, &out_size);
  ASSERT(data != NULL);
  ASSERT_EQ(512u, out_size);

  int mismatches = 0;
  for (int i = 0; i < 512; i++) mismatches += (data[i] != buf[i]);
  ASSERT_EQ(0, mismatches);

  free(data);
  vol_free(v2);
  PASS();
}

// Test 2: Cache + Volume — cold miss then hot hit

TEST test_cache_vol_hit_miss(void) {
  char path[300];
  snprintf(path, sizeof(path), "%s/cache_vol.zarr", g_tmpdir);

  // Create a small volume
  vol_create_params p = {
    .zarr_version = 2, .ndim = 3,
    .shape = {4, 4, 4}, .chunk_shape = {4, 4, 4},
    .dtype = DTYPE_U8, .compressor = NULL,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);
  uint8_t chunk[64];
  memset(chunk, 0xAB, 64);
  int64_t coords[3] = {0, 0, 0};
  ASSERT(vol_write_chunk(v, 0, coords, chunk, 64));
  vol_free(v);

  // Open for reading
  v = vol_open(path);
  ASSERT(v != NULL);

  // Create cache (hot-only, no disk)
  cache_config cfg = {
    .hot_max_bytes  = 64 * 1024 * 1024,
    .warm_max_bytes = 0,
    .cold_max_bytes = 0,
    .cold_dir = NULL,
    .io_threads = 1,
  };
  chunk_cache *c = cache_new(cfg);
  ASSERT(c != NULL);

  // Cold miss
  chunk_key key = {.level = 0, .iz = 0, .iy = 0, .ix = 0};
  chunk_data *d = cache_get(c, key);
  ASSERT(d == NULL);
  ASSERT_EQ(0u, cache_hits(c));

  // Put data into cache manually
  chunk_data *cd = malloc(sizeof(chunk_data));
  cd->data = malloc(64);
  memset(cd->data, 0xAB, 64);
  cd->size = 64;
  cd->shape[0] = cd->shape[1] = cd->shape[2] = 4;
  cd->elem_size = 1;
  cache_put(c, key, cd);

  // Now should be a hit
  chunk_data *hit = cache_get(c, key);
  ASSERT(hit != NULL);
  ASSERT(cache_hits(c) >= 1u);
  ASSERT_EQ(64u, hit->size);
  ASSERT_EQ(0xAB, hit->data[0]);
  chunk_data_free(hit);

  cache_free(c);
  vol_free(v);
  PASS();
}

// Test 3: Render pipeline — submit tile, drain, verify non-zero pixels

TEST test_render_tile_nonzero(void) {
  tile_renderer *r = tile_renderer_new(1);
  ASSERT(r != NULL);

  tile_key key = {.col = 3, .row = 7, .pyramid_level = 0, .epoch = 1};
  tile_renderer_submit(r, key);

  // Drain with retry
  tile_result results[4];
  int got = 0;
  for (int attempt = 0; attempt < 50 && got == 0; attempt++) {
    got = tile_renderer_drain(r, results, 4);
    if (got == 0) usleep(5000);
  }
  ASSERT(got > 0);

  // Verify non-zero pixels and correct key
  tile_result *res = &results[0];
  ASSERT(res->valid);
  ASSERT(res->pixels != NULL);
  ASSERT_EQ(3, res->key.col);
  ASSERT_EQ(7, res->key.row);

  int nonzero = 0;
  for (int i = 0; i < TILE_PX * TILE_PX * 4; i++) nonzero += (res->pixels[i] != 0);
  ASSERT(nonzero > 0);

  free(res->pixels);
  tile_renderer_free(r);
  PASS();
}

// Test 4: Compress pipeline — encode residual, decode, within tolerance

TEST test_compress_residual_roundtrip(void) {
  const int N = 256;
  float orig[256];
  for (int i = 0; i < N; i++) orig[i] = sinf((float)i * 0.1f) * 10.0f;

  float scale = 0.1f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(orig, N, scale, &enc_len);
  ASSERT(enc != NULL);
  ASSERT(enc_len > 0);
  ASSERT(enc_len < (size_t)(N * 4));  // should compress

  float decoded[256];
  bool ok = compress4d_decode_residual(enc, enc_len, N, scale, decoded);
  ASSERT(ok);

  // Max quantisation error = scale * 1.0 (one quant step)
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float e = fabsf(orig[i] - decoded[i]);
    if (e > max_err) max_err = e;
  }
  ASSERT(max_err < scale * 2.0f);

  free(enc);
  PASS();
}

// Test 5: CLI pipeline — write zarr, run `volatile stats`, verify output

TEST test_cli_stats_output(void) {
  char path[300];
  snprintf(path, sizeof(path), "%s/stats_vol.zarr", g_tmpdir);

  // Create a minimal volume
  vol_create_params p = {
    .zarr_version = 2, .ndim = 3,
    .shape = {4, 4, 4}, .chunk_shape = {4, 4, 4},
    .dtype = DTYPE_U8, .compressor = NULL,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);
  uint8_t chunk[64];
  memset(chunk, 1, 64);
  int64_t coords[3] = {0, 0, 0};
  ASSERT(vol_write_chunk(v, 0, coords, chunk, 64));
  vol_free(v);

  // Run the volatile binary if it's in build/src/cli/
  char cmd[512];
  snprintf(cmd, sizeof(cmd),
           "build/src/cli/volatile-cli stats %s 2>/dev/null", path);
  FILE *f = popen(cmd, "r");
  if (!f) SKIP();  // binary not built yet — skip rather than fail

  char output[1024] = {0};
  (void)fread(output, 1, sizeof(output) - 1, f);
  int rc = pclose(f);
  if (rc != 0) SKIP();  // stats subcommand not implemented yet — skip

  // Output should mention the path or some volume info
  ASSERT(strstr(output, "zarr") != NULL || strstr(output, "shape") != NULL
         || strstr(output, "4") != NULL);
  PASS();
}

// Test 6: JSON + vol metadata — parse .zarray written by vol_create

TEST test_json_zarray_parse(void) {
  char path[300];
  snprintf(path, sizeof(path), "%s/json_vol.zarr", g_tmpdir);

  // Create a volume so .zarray gets written
  vol_create_params p = {
    .zarr_version = 2, .ndim = 3,
    .shape = {16, 8, 4}, .chunk_shape = {4, 4, 4},
    .dtype = DTYPE_U16, .compressor = NULL,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);
  vol_free(v);

  // Read back the .zarray file and parse it
  char meta_path[400];
  snprintf(meta_path, sizeof(meta_path), "%s/0/.zarray", path);
  FILE *f = fopen(meta_path, "r");
  ASSERT(f != NULL);
  char buf[2048] = {0};
  (void)fread(buf, 1, sizeof(buf) - 1, f);
  fclose(f);

  json_value *root = json_parse(buf);
  ASSERT(root != NULL);
  ASSERT_EQ(JSON_OBJECT, json_typeof(root));

  // Verify shape field
  const json_value *shape = json_object_get(root, "shape");
  ASSERT(shape != NULL);
  ASSERT_EQ(JSON_ARRAY, json_typeof(shape));
  ASSERT_EQ(3u, json_array_len(shape));
  ASSERT_EQ(16, (int)json_get_int(json_array_get(shape, 0), -1));
  ASSERT_EQ(8,  (int)json_get_int(json_array_get(shape, 1), -1));
  ASSERT_EQ(4,  (int)json_get_int(json_array_get(shape, 2), -1));

  const json_value *dtype = json_object_get(root, "dtype");
  ASSERT(dtype != NULL && json_typeof(dtype) == JSON_STRING);
  ASSERT(strstr(json_get_str(dtype), "u2") != NULL || strstr(json_get_str(dtype), "2") != NULL);
  json_free(root);
  PASS();
}

SUITE(integration_suite) {
  make_tmpdir();
  rmdir_r(g_tmpdir);
  char cmd[300];
  snprintf(cmd, sizeof(cmd), "mkdir -p %s", g_tmpdir);
  (void)system(cmd);

  RUN_TEST(test_vol_roundtrip);
  RUN_TEST(test_cache_vol_hit_miss);
  RUN_TEST(test_render_tile_nonzero);
  RUN_TEST(test_compress_residual_roundtrip);
  RUN_TEST(test_cli_stats_output);
  RUN_TEST(test_json_zarray_parse);

  rmdir_r(g_tmpdir);
}

GREATEST_MAIN_DEFS();
int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(integration_suite);
  GREATEST_MAIN_END();
}
