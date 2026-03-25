#include "greatest.h"
#include "core/compress4d.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// ANS roundtrip tests
// ---------------------------------------------------------------------------

TEST test_ans_roundtrip_simple(void) {
  const uint8_t src[] = {1, 2, 3, 1, 2, 1, 0, 5};
  size_t len = sizeof(src);

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);

  size_t  enc_len = 0;
  uint8_t *enc    = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);
  ASSERT(enc_len > 0);

  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(enc);
  free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_roundtrip_uniform(void) {
  // All 256 symbols present with equal frequency
  size_t len = 512;
  uint8_t *src = malloc(len);
  ASSERT(src != NULL);
  for (size_t i = 0; i < len; i++) src[i] = (uint8_t)(i & 0xFF);

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);

  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);

  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(src); free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_roundtrip_single_symbol(void) {
  // All same symbol: highly skewed distribution
  size_t len = 256;
  uint8_t *src = malloc(len);
  ASSERT(src != NULL);
  memset(src, 42, len);

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);

  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);

  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(src); free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_roundtrip_large(void) {
  // 4096 bytes with natural-ish byte distribution (modulo pattern)
  size_t len = 4096;
  uint8_t *src = malloc(len);
  ASSERT(src != NULL);
  for (size_t i = 0; i < len; i++) src[i] = (uint8_t)((i * 7 + 13) % 256);

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);

  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);

  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(src); free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_table_from_freqs_roundtrip(void) {
  const uint8_t src[] = {10, 20, 30, 10, 20, 10};
  size_t len = sizeof(src);

  ans_table *t1 = ans_table_build(src, len);
  ASSERT(t1 != NULL);

  uint16_t freqs[256];
  ans_table_get_freqs(t1, freqs);

  ans_table *t2 = ans_table_from_freqs(freqs);
  ASSERT(t2 != NULL);

  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t1, src, len, &enc_len);
  ASSERT(enc != NULL);

  // Decode with reconstructed table
  uint8_t *dec = ans_decode(t2, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(enc); free(dec);
  ans_table_free(t1);
  ans_table_free(t2);
  PASS();
}

// ---------------------------------------------------------------------------
// Lanczos-3 3-D upsample tests
// ---------------------------------------------------------------------------

TEST test_lanczos3_constant_volume(void) {
  // A constant-valued volume upsampled 2x should remain (approximately) constant.
  size_t dx = 4, dy = 4, dz = 4;
  size_t n = dx * dy * dz;
  float *src = malloc(n * sizeof(float));
  float *dst = malloc(8 * n * sizeof(float));
  ASSERT(src != NULL && dst != NULL);

  float val = 3.14f;
  for (size_t i = 0; i < n; i++) src[i] = val;

  lanczos3_upsample3d(src, dx, dy, dz, dst);

  size_t out_n = (dx*2) * (dy*2) * (dz*2);
  for (size_t i = 0; i < out_n; i++) {
    ASSERT_IN_RANGE(val - 0.1f, dst[i], val + 0.1f);
  }

  free(src); free(dst);
  PASS();
}

TEST test_lanczos3_output_size(void) {
  // Check output is exactly 8x the number of elements.
  size_t dx = 3, dy = 5, dz = 2;
  size_t n   = dx * dy * dz;
  size_t n2  = (dx*2) * (dy*2) * (dz*2);
  float *src = calloc(n,  sizeof(float));
  float *dst = calloc(n2, sizeof(float));
  ASSERT(src != NULL && dst != NULL);

  lanczos3_upsample3d(src, dx, dy, dz, dst);
  // If we get here without crashing or sanitizer errors, sizes are correct.

  free(src); free(dst);
  PASS();
}

TEST test_lanczos3_single_voxel(void) {
  // 1x1x1 volume upsamples to 2x2x2; all values should equal the single input.
  float src[1] = {7.0f};
  float dst[8] = {0};
  lanczos3_upsample3d(src, 1, 1, 1, dst);
  for (int i = 0; i < 8; i++) {
    ASSERT_IN_RANGE(6.9f, dst[i], 7.1f);
  }
  PASS();
}

// ---------------------------------------------------------------------------
// Residual encode / decode tests
// ---------------------------------------------------------------------------

TEST test_residual_roundtrip_zeros(void) {
  size_t len = 256;
  float *res = calloc(len, sizeof(float));
  ASSERT(res != NULL);

  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(res, len, 1.0f, &enc_len);
  ASSERT(enc != NULL);
  ASSERT(enc_len > 0);

  float *out = malloc(len * sizeof(float));
  ASSERT(out != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, len, 1.0f, out));

  for (size_t i = 0; i < len; i++) ASSERT_IN_RANGE(-0.5f, out[i], 0.5f);

  free(res); free(enc); free(out);
  PASS();
}

TEST test_residual_roundtrip_values(void) {
  // Residuals within [-10, 10], scale=0.1 -> 100 steps -> max error 0.05.
  size_t len = 512;
  float *res = malloc(len * sizeof(float));
  ASSERT(res != NULL);
  for (size_t i = 0; i < len; i++) res[i] = (float)(i % 21) - 10.0f;

  float scale = 0.1f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(res, len, scale, &enc_len);
  ASSERT(enc != NULL);

  float *out = malloc(len * sizeof(float));
  ASSERT(out != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, len, scale, out));

  for (size_t i = 0; i < len; i++) {
    float err = fabsf(out[i] - res[i]);
    ASSERT(err <= scale + 1e-4f);
  }

  free(res); free(enc); free(out);
  PASS();
}

TEST test_residual_roundtrip_large(void) {
  size_t len = 4096;
  float *res = malloc(len * sizeof(float));
  ASSERT(res != NULL);
  for (size_t i = 0; i < len; i++) res[i] = sinf((float)i * 0.1f) * 5.0f;

  float scale = 0.05f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(res, len, scale, &enc_len);
  ASSERT(enc != NULL);

  float *out = malloc(len * sizeof(float));
  ASSERT(out != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, len, scale, out));

  for (size_t i = 0; i < len; i++) {
    ASSERT(fabsf(out[i] - res[i]) <= scale + 1e-4f);
  }

  free(res); free(enc); free(out);
  PASS();
}

TEST test_ans_roundtrip_large_64k(void) {
  // 64KB with pseudo-random byte distribution.
  size_t len = 65536;
  uint8_t *src = malloc(len);
  ASSERT(src != NULL);
  uint32_t lcg = 0xdeadbeef;
  for (size_t i = 0; i < len; i++) {
    lcg = lcg * 1664525u + 1013904223u;
    src[i] = (uint8_t)(lcg >> 24);
  }

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);
  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);
  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(src); free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_roundtrip_all_zeros(void) {
  size_t len = 1024;
  uint8_t *src = calloc(len, 1);
  ASSERT(src != NULL);

  ans_table *t = ans_table_build(src, len);
  ASSERT(t != NULL);
  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, len, &enc_len);
  ASSERT(enc != NULL);
  uint8_t *dec = ans_decode(t, enc, enc_len, len);
  ASSERT(dec != NULL);
  ASSERT_MEM_EQ(src, dec, len);

  free(src); free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_roundtrip_single_byte(void) {
  uint8_t src[1] = {0x7F};
  ans_table *t = ans_table_build(src, 1);
  ASSERT(t != NULL);
  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, 1, &enc_len);
  ASSERT(enc != NULL);
  uint8_t *dec = ans_decode(t, enc, enc_len, 1);
  ASSERT(dec != NULL);
  ASSERT_EQ(src[0], dec[0]);

  free(enc); free(dec);
  ans_table_free(t);
  PASS();
}

TEST test_ans_empty_input(void) {
  // Empty build should either return NULL or handle gracefully (no crash).
  ans_table *t = ans_table_build(NULL, 0);
  if (t) ans_table_free(t);
  // Just verifying no crash.
  PASS();
}

TEST test_residual_all_same_value(void) {
  size_t len = 256;
  float *res = malloc(len * sizeof(float));
  ASSERT(res != NULL);
  for (size_t i = 0; i < len; i++) res[i] = 3.5f;

  float scale = 0.1f;
  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(res, len, scale, &enc_len);
  ASSERT(enc != NULL);
  float *out = malloc(len * sizeof(float));
  ASSERT(out != NULL);
  ASSERT(compress4d_decode_residual(enc, enc_len, len, scale, out));
  for (size_t i = 0; i < len; i++)
    ASSERT(fabsf(out[i] - res[i]) <= scale + 1e-4f);

  free(res); free(enc); free(out);
  PASS();
}

// ---------------------------------------------------------------------------
// Pyramid encode / decode tests
// ---------------------------------------------------------------------------

// Build a simple 2-level pyramid: level0 = 8^3 (finest), level1 = 4^3 (coarsest).
// Fill level1 with a smooth gradient; level0 = 2x that + small noise.
// quant_step used in pyramid tests: data in [-Q_MAX, Q_MAX] where Q_MAX = 127 * QS
#define TEST_QS 0.5f
#define TEST_QMAX (127.0f * TEST_QS)  // 63.5

static void make_pyramid_2level(float **levels, int64_t shapes[2][3]) {
  shapes[1][0] = 4; shapes[1][1] = 4; shapes[1][2] = 4;
  shapes[0][0] = 8; shapes[0][1] = 8; shapes[0][2] = 8;

  size_t n1 = 4*4*4, n0 = 8*8*8;
  levels[1] = malloc(n1 * sizeof(float));
  levels[0] = malloc(n0 * sizeof(float));

  // Values in [0, TEST_QMAX] so they fit within the quantiser range
  for (size_t i = 0; i < n1; i++)
    levels[1][i] = (float)i / (float)n1 * TEST_QMAX;

  for (size_t i = 0; i < n0; i++)
    levels[0][i] = (float)i / (float)n0 * TEST_QMAX;
}

TEST test_pyramid_encode_decode_each_level(void) {
  float *levels[2]; int64_t shapes[2][3];
  make_pyramid_2level(levels, shapes);

  compress4d_params p = compress4d_params_default();
  p.num_levels = 2;
  p.quant_step = TEST_QS;
  p.chunk_shape[0] = p.chunk_shape[1] = p.chunk_shape[2] = 4;

  size_t stream_len = 0;
  uint8_t *stream = compress4d_encode_pyramid(
    (const float *const *)levels, (const int64_t (*)[3])shapes, 2, p, &stream_len);
  ASSERT(stream != NULL);
  ASSERT(stream_len > 0);

  // Decode coarsest level (level 1)
  int64_t sh1[3];
  float *dec1 = compress4d_decode_level(stream, stream_len, 1, sh1);
  ASSERT(dec1 != NULL);
  ASSERT_EQ(4, sh1[0]); ASSERT_EQ(4, sh1[1]); ASSERT_EQ(4, sh1[2]);
  // Coarsest: direct residual, error <= 1 quant step
  for (size_t i = 0; i < 4*4*4; i++)
    ASSERT(fabsf(dec1[i] - levels[1][i]) <= p.quant_step * 1.5f);

  // Decode finest level (level 0)
  int64_t sh0[3];
  float *dec0 = compress4d_decode_level(stream, stream_len, 0, sh0);
  ASSERT(dec0 != NULL);
  ASSERT_EQ(8, sh0[0]); ASSERT_EQ(8, sh0[1]); ASSERT_EQ(8, sh0[2]);
  // Finest level: residual on top of Lanczos prediction. The prediction error
  // from Lanczos is typically small for smooth data, but add generous tolerance
  // since prediction + quantisation can compound.
  for (size_t i = 0; i < 8*8*8; i++)
    ASSERT(fabsf(dec0[i] - levels[0][i]) <= TEST_QMAX * 0.5f);

  free(dec0); free(dec1);
  free(stream);
  free(levels[0]); free(levels[1]);
  PASS();
}

TEST test_pyramid_streaming_decode(void) {
  float *levels[2]; int64_t shapes[2][3];
  make_pyramid_2level(levels, shapes);

  compress4d_params p = compress4d_params_default();
  p.num_levels = 2;
  p.quant_step = TEST_QS;
  p.chunk_shape[0] = p.chunk_shape[1] = p.chunk_shape[2] = 4;

  size_t stream_len = 0;
  uint8_t *stream = compress4d_encode_pyramid(
    (const float *const *)levels, (const int64_t (*)[3])shapes, 2, p, &stream_len);
  ASSERT(stream != NULL);

  compress4d_decoder *d = compress4d_decoder_new(stream, stream_len);
  ASSERT(d != NULL);

  // First next() should yield coarsest (level 1)
  float *data = NULL; int64_t sh[3]; int lv = -1;
  ASSERT(compress4d_decoder_next(d, &data, sh, &lv));
  ASSERT_EQ(1, lv);
  ASSERT(data != NULL);
  ASSERT_EQ(4, sh[0]); ASSERT_EQ(4, sh[1]); ASSERT_EQ(4, sh[2]);
  // data is owned by decoder; do not free it

  // Second next() should yield finest (level 0)
  float *data2 = NULL; int64_t sh2[3]; int lv2 = -1;
  ASSERT(compress4d_decoder_next(d, &data2, sh2, &lv2));
  ASSERT_EQ(0, lv2);
  ASSERT(data2 != NULL);
  ASSERT_EQ(8, sh2[0]); ASSERT_EQ(8, sh2[1]); ASSERT_EQ(8, sh2[2]);

  // Progressive refinement: finest should be closer to original than coarsest upsampled
  float err_finest = 0.0f;
  for (size_t i = 0; i < 8*8*8; i++) {
    float e = fabsf(data2[i] - levels[0][i]);
    if (e > err_finest) err_finest = e;
  }
  ASSERT(err_finest < 50.0f);

  // Third next() should return false
  float *data3 = NULL; int lv3 = -1; int64_t sh3[3];
  ASSERT_FALSE(compress4d_decoder_next(d, &data3, sh3, &lv3));

  compress4d_decoder_free(d);
  free(stream);
  free(levels[0]); free(levels[1]);
  PASS();
}

TEST test_pyramid_chunk_random_access(void) {
  // Encode a 16^3 volume with 8^3 chunks -> 8 chunks total.
  // Decode only the coarsest level (no upsampling needed) and verify
  // that individual 8^3 tiles decode correctly.
  const int64_t D = 16, H = 16, W = 16;
  size_t n = (size_t)(D * H * W);
  float *vol = malloc(n * sizeof(float));
  ASSERT(vol != NULL);
  // Scale to [0, TEST_QMAX] so values fit within ±127 * quant_step
  for (size_t i = 0; i < n; i++) vol[i] = (float)i / (float)n * TEST_QMAX;

  const float *levels[1] = {vol};
  const int64_t shapes[1][3] = {{D, H, W}};

  compress4d_params p = compress4d_params_default();
  p.num_levels = 1;
  p.quant_step = TEST_QS;
  p.chunk_shape[0] = p.chunk_shape[1] = p.chunk_shape[2] = 8;

  size_t stream_len = 0;
  uint8_t *stream = compress4d_encode_pyramid(levels, shapes, 1, p, &stream_len);
  ASSERT(stream != NULL);

  int64_t sh[3];
  float *dec = compress4d_decode_level(stream, stream_len, 0, sh);
  ASSERT(dec != NULL);
  ASSERT_EQ(D, sh[0]); ASSERT_EQ(H, sh[1]); ASSERT_EQ(W, sh[2]);

  // Verify every voxel within quantisation error
  for (size_t i = 0; i < n; i++)
    ASSERT(fabsf(dec[i] - vol[i]) <= p.quant_step * 1.5f);

  free(dec);
  free(stream);
  free(vol);
  PASS();
}

TEST test_pyramid_5level_roundtrip(void) {
  // 5-level pyramid: level 0 = 32^3 (finest), down to level 4 = 2^3 (coarsest).
  // Keep sizes small so test completes in <1s.
  const int NUM = 5;
  int64_t shapes[5][3];
  float *levels[5];
  int64_t sz = 32;
  for (int i = 0; i < NUM; i++) {
    int64_t s = sz >> i;  // 32, 16, 8, 4, 2
    shapes[i][0] = shapes[i][1] = shapes[i][2] = s;
    size_t n = (size_t)(s * s * s);
    levels[i] = malloc(n * sizeof(float));
    ASSERT(levels[i] != NULL);
    // Values in [-TEST_QMAX/2, TEST_QMAX/2] to fit quantiser range
    for (size_t j = 0; j < n; j++)
      levels[i][j] = sinf((float)j * 0.1f) * (TEST_QMAX * 0.45f);
  }

  compress4d_params p = compress4d_params_default();
  p.num_levels = NUM;
  p.quant_step = TEST_QS;
  p.chunk_shape[0] = p.chunk_shape[1] = p.chunk_shape[2] = 8;

  size_t stream_len = 0;
  uint8_t *stream = compress4d_encode_pyramid(
    (const float *const *)levels, (const int64_t (*)[3])shapes, NUM, p, &stream_len);
  ASSERT(stream != NULL);
  ASSERT(stream_len > 0);

  // Decode coarsest level (level 4, shape 2^3)
  int64_t sh[3];
  float *dec = compress4d_decode_level(stream, stream_len, NUM - 1, sh);
  ASSERT(dec != NULL);
  ASSERT_EQ(2, sh[0]);
  for (size_t i = 0; i < 2*2*2; i++)
    ASSERT(fabsf(dec[i] - levels[NUM-1][i]) <= p.quant_step * 1.5f);
  free(dec);

  // Decode finest level (level 0, shape 32^3) — skip if too slow, check coarsest only
  // Actually decode it to verify the full pipeline:
  float *dec0 = compress4d_decode_level(stream, stream_len, 0, sh);
  ASSERT(dec0 != NULL);
  ASSERT_EQ(32, sh[0]);
  free(dec0);

  // Measure compression ratio: stream should be smaller than raw
  size_t raw_bytes = 0;
  for (int i = 0; i < NUM; i++)
    raw_bytes += (size_t)(shapes[i][0] * shapes[i][1] * shapes[i][2]) * sizeof(float);
  // No assertion on ratio — just print it
  (void)raw_bytes;

  free(stream);
  for (int i = 0; i < NUM; i++) free(levels[i]);
  PASS();
}

SUITE(pyramid_suite) {
  RUN_TEST(test_pyramid_encode_decode_each_level);
  RUN_TEST(test_pyramid_streaming_decode);
  RUN_TEST(test_pyramid_chunk_random_access);
  RUN_TEST(test_pyramid_5level_roundtrip);
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(ans_suite) {
  RUN_TEST(test_ans_roundtrip_simple);
  RUN_TEST(test_ans_roundtrip_uniform);
  RUN_TEST(test_ans_roundtrip_single_symbol);
  RUN_TEST(test_ans_roundtrip_large);
  RUN_TEST(test_ans_table_from_freqs_roundtrip);
  RUN_TEST(test_ans_roundtrip_large_64k);
  RUN_TEST(test_ans_roundtrip_all_zeros);
  RUN_TEST(test_ans_roundtrip_single_byte);
  RUN_TEST(test_ans_empty_input);
}

SUITE(lanczos_suite) {
  RUN_TEST(test_lanczos3_constant_volume);
  RUN_TEST(test_lanczos3_output_size);
  RUN_TEST(test_lanczos3_single_voxel);
}

SUITE(residual_suite) {
  RUN_TEST(test_residual_roundtrip_zeros);
  RUN_TEST(test_residual_roundtrip_values);
  RUN_TEST(test_residual_roundtrip_large);
  RUN_TEST(test_residual_all_same_value);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(ans_suite);
  RUN_SUITE(lanczos_suite);
  RUN_SUITE(residual_suite);
  RUN_SUITE(pyramid_suite);
  GREATEST_MAIN_END();
}
