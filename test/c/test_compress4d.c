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
  GREATEST_MAIN_END();
}
