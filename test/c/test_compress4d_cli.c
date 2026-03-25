#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "cli/cli_compress.h"
#include "core/compress4d.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static char *make_tmpdir(void) {
  char tmpl[] = "/tmp/test_c4d_XXXXXX";
  char *d = mkdtemp(tmpl);
  if (!d) return NULL;
  char *out = malloc(64);
  snprintf(out, 64, "%s", d);
  return out;
}

static void rm_tmpdir(const char *p) {
  if (!p) return;
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "rm -rf '%s'", p);
  system(cmd);
}

static bool mkdirs_p(const char *path) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", path);
  for (char *p = buf + 1; *p; p++) {
    if (*p != '/') continue;
    *p = '\0';
    if (mkdir(buf, 0755) != 0 && errno != EEXIST) { *p = '/'; return false; }
    *p = '/';
  }
  return mkdir(buf, 0755) == 0 || errno == EEXIST;
}

static bool write_bytes(const char *path, const void *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, len, f) == len;
  fclose(f);
  return ok;
}

// Create a synthetic single-level zarr with one float32 chunk.
// Volume: 8x8x8 (one chunk, uncompressed float32).
// Values: v[z*64 + y*8 + x] = (float)(z*64 + y*8 + x)
#define SYNTH_DIM 8
#define SYNTH_N   (SYNTH_DIM * SYNTH_DIM * SYNTH_DIM)  // 512

static bool make_synthetic_zarr(const char *zarr_dir, float *out_orig) {
  char dir0[512];
  snprintf(dir0, sizeof(dir0), "%s/0", zarr_dir);
  if (!mkdirs_p(dir0)) return false;

  // .zarray
  char zarray_path[512];
  snprintf(zarray_path, sizeof(zarray_path), "%s/.zarray", dir0);
  FILE *f = fopen(zarray_path, "w");
  if (!f) return false;
  fprintf(f,
    "{\n  \"zarr_format\": 2,\n"
    "  \"shape\": [%d, %d, %d],\n"
    "  \"chunks\": [%d, %d, %d],\n"
    "  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
    "  \"compressor\": null,\n  \"fill_value\": 0,\n"
    "  \"filters\": null\n}\n",
    SYNTH_DIM, SYNTH_DIM, SYNTH_DIM,
    SYNTH_DIM, SYNTH_DIM, SYNTH_DIM);
  fclose(f);

  // Build raw float data
  float *data = malloc(SYNTH_N * sizeof(float));
  if (!data) return false;
  for (int i = 0; i < SYNTH_N; i++) data[i] = (float)i;
  if (out_orig) memcpy(out_orig, data, SYNTH_N * sizeof(float));

  // chunk file: 0.0.0
  char chunk_path[512];
  snprintf(chunk_path, sizeof(chunk_path), "%s/0.0.0", dir0);
  bool ok = write_bytes(chunk_path, data, SYNTH_N * sizeof(float));
  free(data);
  return ok;
}

// Read a raw float file.
static float *read_float_file(const char *path, size_t *out_n) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) { fclose(f); return NULL; }
  float *buf = malloc((size_t)sz);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  *out_n = n / sizeof(float);
  return buf;
}

// ---------------------------------------------------------------------------
// Tests: compress4d_encode_residual / compress4d_decode_residual roundtrip
// These test the codec directly (same API used by cmd_compress4d internally).
// ---------------------------------------------------------------------------

TEST test_codec_roundtrip_64cube(void) {
  // Synthetic 64^3 volume: ramp + sine
  const int N = 64 * 64 * 64;
  float *orig = malloc((size_t)N * sizeof(float));
  ASSERT(orig != NULL);
  for (int i = 0; i < N; i++)
    orig[i] = (float)i * 0.01f + sinf((float)i * 0.001f) * 50.0f;

  float quality = 1.0f;  // quantisation step
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(orig, (size_t)N, quality, &enc_len);
  ASSERT(enc != NULL);
  ASSERT(enc_len > 0);
  // Should compress to less than raw
  ASSERT(enc_len < (size_t)N * sizeof(float));

  float *decoded = malloc((size_t)N * sizeof(float));
  ASSERT(decoded != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, (size_t)N, quality, decoded));

  // Max error should not exceed 1 quantisation step
  float max_err = 0.0f;
  for (int i = 0; i < N; i++) {
    float e = fabsf(decoded[i] - orig[i]);
    if (e > max_err) max_err = e;
  }
  ASSERT(max_err <= quality + 1e-3f);

  free(orig); free(enc); free(decoded);
  PASS();
}

TEST test_codec_roundtrip_fine_quality(void) {
  const int N = 512;
  float *orig = malloc((size_t)N * sizeof(float));
  ASSERT(orig != NULL);
  for (int i = 0; i < N; i++) orig[i] = (float)i * 0.5f;

  float quality = 0.1f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(orig, (size_t)N, quality, &enc_len);
  ASSERT(enc != NULL);

  float *decoded = malloc((size_t)N * sizeof(float));
  ASSERT(decoded != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, (size_t)N, quality, decoded));

  for (int i = 0; i < N; i++)
    ASSERT(fabsf(decoded[i] - orig[i]) <= quality + 1e-4f);

  free(orig); free(enc); free(decoded);
  PASS();
}

TEST test_codec_compression_ratio(void) {
  // Smooth ramp: highly compressible.
  const int N = 1024;
  float *orig = malloc((size_t)N * sizeof(float));
  ASSERT(orig != NULL);
  for (int i = 0; i < N; i++) orig[i] = (float)i;

  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(orig, (size_t)N, 1.0f, &enc_len);
  ASSERT(enc != NULL);
  // A smooth ramp should compress meaningfully.
  ASSERT(enc_len < (size_t)N * sizeof(float));

  free(orig); free(enc);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: cmd_compress4d end-to-end (zarr -> .c4d)
// ---------------------------------------------------------------------------

static char *g_zarr_dir = NULL;
static float g_orig[SYNTH_N];

static void setup_zarr(void *unused) {
  (void)unused;
  g_zarr_dir = make_tmpdir();
  if (g_zarr_dir)
    make_synthetic_zarr(g_zarr_dir, g_orig);
}

static void teardown_zarr(void *unused) {
  (void)unused;
  rm_tmpdir(g_zarr_dir);
  free(g_zarr_dir);
  g_zarr_dir = NULL;
}

TEST test_compress4d_creates_file(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/out.c4d", g_zarr_dir);

  char *argv[] = { g_zarr_dir, "--output", c4d_path, "--quality", "1.0" };
  int rc = cmd_compress4d(5, argv);
  ASSERT_EQ(0, rc);

  // File should exist and have the magic header
  FILE *f = fopen(c4d_path, "rb");
  ASSERT(f != NULL);
  char magic[4];
  ASSERT_EQ(4u, fread(magic, 1, 4, f));
  ASSERT_MEM_EQ("C4DV", magic, 4);
  fclose(f);
  PASS();
}

TEST test_compress4d_info(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/info.c4d", g_zarr_dir);

  // Create .c4d first
  char *enc_argv[] = { g_zarr_dir, "--output", c4d_path };
  ASSERT_EQ(0, cmd_compress4d(3, enc_argv));

  // cmd_compress4d_info should return 0 without crash
  char *info_argv[] = { c4d_path };
  int rc = cmd_compress4d_info(1, info_argv);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_decompress4d_roundtrip(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256], out_zarr[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/rt.c4d",    g_zarr_dir);
  snprintf(out_zarr, sizeof(out_zarr), "%s/rt_out",     g_zarr_dir);

  // Encode with quality 0.5 (tight)
  char *enc_argv[] = { g_zarr_dir, "--output", c4d_path,
                       "--quality", "0.5" };
  ASSERT_EQ(0, cmd_compress4d(5, enc_argv));

  // Decode
  char *dec_argv[] = { c4d_path, "--output", out_zarr };
  ASSERT_EQ(0, cmd_decompress4d(3, dec_argv));

  // Read decoded chunk file: <out_zarr>/0/0.0.0
  char chunk_path[320];
  snprintf(chunk_path, sizeof(chunk_path), "%s/0/0.0.0", out_zarr);
  size_t n = 0;
  float *decoded = read_float_file(chunk_path, &n);
  ASSERT(decoded != NULL);
  ASSERT_EQ((size_t)SYNTH_N, n);

  // Check max error is within quality bound
  float max_err = 0.0f;
  for (int i = 0; i < SYNTH_N; i++) {
    float e = fabsf(decoded[i] - g_orig[i]);
    if (e > max_err) max_err = e;
  }
  // quality=0.5 so max error should be <= 0.5 + epsilon
  ASSERT(max_err <= 0.5f + 0.01f);

  free(decoded);
  PASS();
}

TEST test_decompress4d_zarray_written(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256], out_zarr[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/za.c4d",  g_zarr_dir);
  snprintf(out_zarr, sizeof(out_zarr), "%s/za_out",   g_zarr_dir);

  char *enc_argv[] = { g_zarr_dir, "--output", c4d_path };
  ASSERT_EQ(0, cmd_compress4d(3, enc_argv));
  char *dec_argv[] = { c4d_path, "--output", out_zarr };
  ASSERT_EQ(0, cmd_decompress4d(3, dec_argv));

  // .zarray should exist for level 0
  char zarray[320];
  snprintf(zarray, sizeof(zarray), "%s/0/.zarray", out_zarr);
  FILE *f = fopen(zarray, "r");
  ASSERT(f != NULL);
  char buf[64];
  size_t n = fread(buf, 1, sizeof(buf)-1, f);
  buf[n] = '\0';
  fclose(f);
  // Should start with '{'
  ASSERT_EQ('{', buf[0]);
  PASS();
}

TEST test_compress4d_with_stats_flag(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/stats.c4d", g_zarr_dir);

  char *argv[] = { g_zarr_dir, "--output", c4d_path, "--stats" };
  int rc = cmd_compress4d(4, argv);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_compress4d_with_verify_flag(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/ver.c4d", g_zarr_dir);

  char *argv[] = { g_zarr_dir, "--output", c4d_path,
                   "--verify", "--stats" };
  int rc = cmd_compress4d(5, argv);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_compress4d_streaming_flag(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/stream.c4d", g_zarr_dir);

  char *argv[] = { g_zarr_dir, "--output", c4d_path, "--streaming" };
  int rc = cmd_compress4d(4, argv);
  ASSERT_EQ(0, rc);

  // Verify file is a valid c4d
  FILE *f = fopen(c4d_path, "rb");
  ASSERT(f != NULL);
  char magic[4];
  ASSERT_EQ(4u, fread(magic, 1, 4, f));
  ASSERT_MEM_EQ("C4DV", magic, 4);
  fclose(f);
  PASS();
}

TEST test_compress4d_help_no_crash(void) {
  char *argv[] = { "--help" };
  int rc = cmd_compress4d(1, argv);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_decompress4d_level_filter(void) {
  ASSERT(g_zarr_dir != NULL);

  char c4d_path[256], out_zarr[256];
  snprintf(c4d_path, sizeof(c4d_path), "%s/lvl.c4d",  g_zarr_dir);
  snprintf(out_zarr, sizeof(out_zarr), "%s/lvl_out",   g_zarr_dir);

  char *enc_argv[] = { g_zarr_dir, "--output", c4d_path };
  ASSERT_EQ(0, cmd_compress4d(3, enc_argv));

  // Decode only level 0 (there's only one level anyway, so this just tests
  // the flag doesn't break anything)
  char *dec_argv[] = { c4d_path, "--output", out_zarr, "--level", "0" };
  int rc = cmd_decompress4d(5, dec_argv);
  ASSERT_EQ(0, rc);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(codec_suite) {
  RUN_TEST(test_codec_roundtrip_64cube);
  RUN_TEST(test_codec_roundtrip_fine_quality);
  RUN_TEST(test_codec_compression_ratio);
}

SUITE(cli_suite) {
  SET_SETUP(setup_zarr, NULL);
  SET_TEARDOWN(teardown_zarr, NULL);

  RUN_TEST(test_compress4d_creates_file);
  RUN_TEST(test_compress4d_info);
  RUN_TEST(test_decompress4d_roundtrip);
  RUN_TEST(test_decompress4d_zarray_written);
  RUN_TEST(test_compress4d_with_stats_flag);
  RUN_TEST(test_compress4d_with_verify_flag);
  RUN_TEST(test_compress4d_streaming_flag);
  RUN_TEST(test_compress4d_help_no_crash);
  RUN_TEST(test_decompress4d_level_filter);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(codec_suite);
  RUN_SUITE(cli_suite);
  GREATEST_MAIN_END();
}
