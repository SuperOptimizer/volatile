#define _POSIX_C_SOURCE 200809L

#include "greatest.h"

// Pull in the subcommand implementations directly.
// cmd_convert/rechunk/stats all use volatile_core, which is linked by the test runner.
#include "cli/cli_convert.h"
#include "cli/cli_stats.h"

#include "core/vol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool dir_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static bool file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

// Create a minimal zarr store at path with a single uint8 chunk.
// Layout: path/0/.zarray  +  path/0/0.0.0 (64 bytes of known data).
static bool make_test_zarr(const char *path) {
  // Create directories
  char buf[512];
  snprintf(buf, sizeof(buf), "mkdir -p '%s/0'", path);
  if (system(buf) != 0) return false;

  // Write .zarray
  snprintf(buf, sizeof(buf), "%s/0/.zarray", path);
  FILE *f = fopen(buf, "w");
  if (!f) return false;
  fputs("{\n"
        "  \"zarr_format\": 2,\n"
        "  \"shape\": [4, 4, 4],\n"
        "  \"chunks\": [4, 4, 4],\n"
        "  \"dtype\": \"|u1\",\n"
        "  \"order\": \"C\",\n"
        "  \"compressor\": null,\n"
        "  \"fill_value\": 0,\n"
        "  \"filters\": null\n"
        "}\n", f);
  fclose(f);

  // Write .zgroup
  snprintf(buf, sizeof(buf), "%s/.zgroup", path);
  f = fopen(buf, "w"); if (!f) return false;
  fputs("{\"zarr_format\":2}\n", f); fclose(f);

  // Write chunk 0.0.0 — 64 bytes with values 0..63
  snprintf(buf, sizeof(buf), "%s/0/0.0.0", path);
  f = fopen(buf, "wb"); if (!f) return false;
  for (int i = 0; i < 64; i++) { uint8_t v = (uint8_t)i; fwrite(&v, 1, 1, f); }
  fclose(f);

  return true;
}

static void rm_rf(const char *path) {
  char buf[512];
  snprintf(buf, sizeof(buf), "rm -rf '%s'", path);
  system(buf);
}

// ---------------------------------------------------------------------------
// Tests: convert --help
// ---------------------------------------------------------------------------

TEST test_convert_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_convert(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_rechunk_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_rechunk(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_stats_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_stats(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: convert no args prints usage
// ---------------------------------------------------------------------------

TEST test_convert_no_args(void) {
  int rc = cmd_convert(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

TEST test_rechunk_no_args(void) {
  int rc = cmd_rechunk(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

TEST test_stats_no_args(void) {
  int rc = cmd_stats(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: stats on synthetic zarr
// ---------------------------------------------------------------------------

// Test stats by opening the zarr directly and verifying vol_read_chunk gives
// expected data, which is the underlying logic that cmd_stats exercises.
// We avoid redirecting stdout since that's fragile in a test harness.
TEST test_stats_synthetic_zarr(void) {
  const char *zarr_path = "/tmp/volatile_test_stats_zarr";
  rm_rf(zarr_path);
  ASSERT(make_test_zarr(zarr_path));

  // Verify vol_open can read the zarr we created.
  volume *v = vol_open(zarr_path);
  ASSERT(v != NULL);
  ASSERT_EQ(1, vol_num_levels(v));

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT_EQ(3, m->ndim);
  ASSERT_EQ(4, m->shape[0]);

  // Read the single chunk and verify voxel values 0..63.
  int64_t coords[3] = {0, 0, 0};
  size_t sz;
  uint8_t *raw = vol_read_chunk(v, 0, coords, &sz);
  ASSERT(raw != NULL);
  ASSERT_EQ(64u, sz);
  for (int i = 0; i < 64; i++) ASSERT_EQ((uint8_t)i, raw[i]);
  free(raw);

  vol_free(v);
  rm_rf(zarr_path);

  // Also run cmd_stats and verify it returns 0 (output goes to stdout/stderr).
  const char *zarr_path2 = "/tmp/volatile_test_stats_zarr2";
  rm_rf(zarr_path2);
  ASSERT(make_test_zarr(zarr_path2));
  char *args[] = { (char *)zarr_path2 };
  int rc = cmd_stats(1, args);
  ASSERT_EQ(0, rc);
  rm_rf(zarr_path2);

  PASS();
}

// ---------------------------------------------------------------------------
// Tests: convert zarr → zarr copies chunk data
// ---------------------------------------------------------------------------

TEST test_convert_zarr_to_zarr(void) {
  const char *in_path  = "/tmp/volatile_test_convert_in";
  const char *out_path = "/tmp/volatile_test_convert_out";
  rm_rf(in_path);
  rm_rf(out_path);
  ASSERT(make_test_zarr(in_path));

  char *args[] = { (char *)in_path, (char *)out_path };
  int rc = cmd_convert(2, args);
  ASSERT_EQ(0, rc);

  // Output zarr must have .zarray and at least one chunk.
  ASSERT(dir_exists(out_path));
  char buf[512];
  snprintf(buf, sizeof(buf), "%s/0/.zarray", out_path);
  ASSERT(file_exists(buf));
  snprintf(buf, sizeof(buf), "%s/0/0.0.0", out_path);
  ASSERT(file_exists(buf));

  // Chunk content must match input.
  snprintf(buf, sizeof(buf), "%s/0/0.0.0", in_path);
  FILE *fi = fopen(buf, "rb");
  snprintf(buf, sizeof(buf), "%s/0/0.0.0", out_path);
  FILE *fo = fopen(buf, "rb");
  ASSERT(fi && fo);
  uint8_t bi[64], bo[64];
  size_t ni = fread(bi, 1, 64, fi);
  size_t no = fread(bo, 1, 64, fo);
  fclose(fi); fclose(fo);
  ASSERT_EQ(ni, no);
  ASSERT_EQ(0, memcmp(bi, bo, ni));

  rm_rf(in_path);
  rm_rf(out_path);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: convert bad format
// ---------------------------------------------------------------------------

TEST test_convert_bad_format(void) {
  char *args[] = { "input.zarr", "output.zarr", "--format", "hdf5" };
  int rc = cmd_convert(4, args);
  ASSERT_EQ(1, rc);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: rechunk changes chunk size
// ---------------------------------------------------------------------------

TEST test_rechunk_produces_output(void) {
  const char *in_path  = "/tmp/volatile_test_rechunk_in";
  const char *out_path = "/tmp/volatile_test_rechunk_out";
  rm_rf(in_path);
  rm_rf(out_path);
  ASSERT(make_test_zarr(in_path));

  char *args[] = { (char *)in_path, "--output", (char *)out_path, "--chunk-size", "2,2,2" };
  int rc = cmd_rechunk(5, args);
  ASSERT_EQ(0, rc);

  ASSERT(dir_exists(out_path));
  char zarray[512];
  snprintf(zarray, sizeof(zarray), "%s/0/.zarray", out_path);
  ASSERT(file_exists(zarray));

  // 4x4x4 volume with 2x2x2 chunks → 8 chunks expected.
  int chunk_count = 0;
  for (int z = 0; z < 2; z++)
    for (int y = 0; y < 2; y++)
      for (int x = 0; x < 2; x++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/0/%d.%d.%d", out_path, z, y, x);
        if (file_exists(path)) chunk_count++;
      }
  ASSERT_EQ(8, chunk_count);

  rm_rf(in_path);
  rm_rf(out_path);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(cli_convert_suite) {
  RUN_TEST(test_convert_help);
  RUN_TEST(test_rechunk_help);
  RUN_TEST(test_stats_help);
  RUN_TEST(test_convert_no_args);
  RUN_TEST(test_rechunk_no_args);
  RUN_TEST(test_stats_no_args);
  RUN_TEST(test_convert_bad_format);
  RUN_TEST(test_convert_zarr_to_zarr);
  RUN_TEST(test_stats_synthetic_zarr);
  RUN_TEST(test_rechunk_produces_output);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(cli_convert_suite);
  GREATEST_MAIN_END();
}
