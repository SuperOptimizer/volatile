/* test_compress4d_zarr.c — tests for compress4d zarr v3 codec plugin */
#include "greatest.h"
#include "core/compress4d_zarr.h"
#include "core/compress4d.h"
#include "core/json.h"
#include "core/vol.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <errno.h>

/* -------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */

static float *make_floats(size_t n, float base) {
  float *f = malloc(n * sizeof(float));
  if (!f) return NULL;
  for (size_t i = 0; i < n; i++) f[i] = base + (float)i * 0.01f;
  return f;
}

/* -------------------------------------------------------------------------
 * 1. register: idempotent, no crash
 * --------------------------------------------------------------------- */

TEST test_register_idempotent(void) {
  compress4d_zarr_register();
  compress4d_zarr_register();
  compress4d_zarr_register();
  PASS();
}

/* -------------------------------------------------------------------------
 * 2. encode then decode: round-trip, output matches input within tolerance
 * --------------------------------------------------------------------- */

TEST test_encode_decode_roundtrip(void) {
  const size_t n = 64;
  float *src = make_floats(n, 0.5f);
  ASSERT(src != NULL);

  size_t enc_len = 0;
  uint8_t *enc = compress4d_zarr_encode((const uint8_t *)src, n * sizeof(float),
                                         &enc_len, NULL);
  ASSERT(enc != NULL);
  ASSERT(enc_len > 0);

  size_t dec_len = 0;
  uint8_t *dec = compress4d_zarr_decode(enc, enc_len, &dec_len, NULL);
  ASSERT(dec != NULL);
  ASSERT_EQ(dec_len, n * sizeof(float));

  float *fout = (float *)dec;
  for (size_t i = 0; i < n; i++) {
    float diff = fabsf(fout[i] - src[i]);
    /* default quality=1 → scale=0.01 → quant step=0.01; allow 2× margin */
    ASSERT_IN_RANGE(0.0f, diff, 0.02f);
  }

  free(src); free(enc); free(dec);
  PASS();
}

/* -------------------------------------------------------------------------
 * 3. encode then decode with quality=0.5 config
 * --------------------------------------------------------------------- */

TEST test_encode_decode_quality_config(void) {
  const size_t n = 16;
  float *src = make_floats(n, 1.0f);
  ASSERT(src != NULL);

  json_value *cfg = json_parse("{\"quality\": 0.5, \"levels\": 5}");
  ASSERT(cfg != NULL);

  size_t enc_len = 0;
  uint8_t *enc = compress4d_zarr_encode((const uint8_t *)src, n * sizeof(float),
                                         &enc_len, cfg);
  ASSERT(enc != NULL);

  size_t dec_len = 0;
  uint8_t *dec = compress4d_zarr_decode(enc, enc_len, &dec_len, cfg);
  ASSERT(dec != NULL);
  ASSERT_EQ(dec_len, n * sizeof(float));

  json_free(cfg);
  free(src); free(enc); free(dec);
  PASS();
}

/* -------------------------------------------------------------------------
 * 4. decode bad magic → NULL
 * --------------------------------------------------------------------- */

TEST test_decode_bad_magic(void) {
  uint8_t garbage[32] = "BADMAGIC_____________________";
  size_t out_len = 0;
  uint8_t *r = compress4d_zarr_decode(garbage, sizeof(garbage), &out_len, NULL);
  ASSERT_EQ(r, NULL);
  PASS();
}

/* -------------------------------------------------------------------------
 * 5. decode truncated header → NULL
 * --------------------------------------------------------------------- */

TEST test_decode_truncated(void) {
  uint8_t short_buf[8] = {0};
  memcpy(short_buf, "C4DR", 4);  // magic present but no header body
  size_t out_len = 0;
  uint8_t *r = compress4d_zarr_decode(short_buf, sizeof(short_buf), &out_len, NULL);
  ASSERT_EQ(r, NULL);
  PASS();
}

/* -------------------------------------------------------------------------
 * 6. encode non-float-aligned length → NULL
 * --------------------------------------------------------------------- */

TEST test_encode_unaligned_len(void) {
  uint8_t buf[7] = {0};
  size_t out_len = 0;
  uint8_t *r = compress4d_zarr_encode(buf, 7, &out_len, NULL);
  ASSERT_EQ(r, NULL);
  PASS();
}

/* -------------------------------------------------------------------------
 * 7. vol.c parses zarr.json with compress4d codec
 * --------------------------------------------------------------------- */

static bool write_text(const char *path, const char *text) {
  FILE *f = fopen(path, "w");
  if (!f) return false;
  fputs(text, f);
  fclose(f);
  return true;
}

static bool mkdir_p(const char *path) {
  return mkdir(path, 0755) == 0 || errno == EEXIST;
}

/* We don't actually need errno if we use the return values carefully */
#include <errno.h>

TEST test_vol_parses_compress4d_codec(void) {
  /* Build a minimal zarr v3 directory in /tmp */
  const char *base = "/tmp/test_c4d_zarr";
  const char *lvl0 = "/tmp/test_c4d_zarr/0";
  mkdir_p(base);
  mkdir_p(lvl0);

  const char *zarr_json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [8, 8, 8],"
    "  \"data_type\": \"float32\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [8, 8, 8]}"
    "  },"
    "  \"codecs\": ["
    "    {\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}},"
    "    {\"name\": \"compress4d\", \"configuration\": {\"quality\": 1.0, \"levels\": 5}}"
    "  ]"
    "}";

  char path[256];
  snprintf(path, sizeof(path), "%s/zarr.json", lvl0);
  ASSERT(write_text(path, zarr_json));

  volume *v = vol_open(base);
  ASSERT(v != NULL);
  ASSERT_EQ(vol_num_levels(v), 1);

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT_EQ(m->zarr_version, 3);
  ASSERT_STR_EQ(m->compressor_id, "compress4d");

  vol_free(v);
  PASS();
}

/* -------------------------------------------------------------------------
 * 8. vol_read_chunk decodes compress4d chunk
 * --------------------------------------------------------------------- */

TEST test_vol_read_chunk_compress4d(void) {
  const char *base = "/tmp/test_c4d_zarr_rw";
  const char *lvl0 = "/tmp/test_c4d_zarr_rw/0";
  const char *c_dir = "/tmp/test_c4d_zarr_rw/0/c";
  mkdir_p(base);
  mkdir_p(lvl0);
  mkdir_p(c_dir);
  mkdir_p("/tmp/test_c4d_zarr_rw/0/c/0");
  mkdir_p("/tmp/test_c4d_zarr_rw/0/c/0/0");

  const char *zarr_json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [4, 4, 4],"
    "  \"data_type\": \"float32\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [4, 4, 4]}"
    "  },"
    "  \"codecs\": ["
    "    {\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}},"
    "    {\"name\": \"compress4d\", \"configuration\": {\"quality\": 1.0, \"levels\": 3}}"
    "  ]"
    "}";

  char path[256];
  snprintf(path, sizeof(path), "%s/zarr.json", lvl0);
  ASSERT(write_text(path, zarr_json));

  /* encode a 4×4×4 float32 chunk (64 floats); values in [0.1,0.73] fit scale=0.01 range */
  const size_t n = 64;
  float *src = make_floats(n, 0.1f);
  ASSERT(src != NULL);

  size_t enc_len = 0;
  uint8_t *enc = compress4d_zarr_encode((const uint8_t *)src, n * sizeof(float),
                                         &enc_len, NULL);
  ASSERT(enc != NULL);

  /* write chunk file at c/0/0/0 */
  char cpath[256];
  snprintf(cpath, sizeof(cpath), "%s/0/0/0", c_dir);
  FILE *cf = fopen(cpath, "wb");
  ASSERT(cf != NULL);
  fwrite(enc, 1, enc_len, cf);
  fclose(cf);
  free(enc);

  volume *v = vol_open(base);
  ASSERT(v != NULL);

  int64_t coords[3] = {0, 0, 0};
  size_t out_size = 0;
  uint8_t *chunk = vol_read_chunk(v, 0, coords, &out_size);
  ASSERT(chunk != NULL);
  ASSERT_EQ(out_size, n * sizeof(float));

  float *fout = (float *)chunk;
  for (size_t i = 0; i < n; i++) {
    float diff = fabsf(fout[i] - src[i]);
    ASSERT_IN_RANGE(0.0f, diff, 0.02f);
  }

  free(chunk); free(src);
  vol_free(v);
  PASS();
}

/* -------------------------------------------------------------------------
 * Suite
 * --------------------------------------------------------------------- */

SUITE(compress4d_zarr_suite) {
  RUN_TEST(test_register_idempotent);
  RUN_TEST(test_encode_decode_roundtrip);
  RUN_TEST(test_encode_decode_quality_config);
  RUN_TEST(test_decode_bad_magic);
  RUN_TEST(test_decode_truncated);
  RUN_TEST(test_encode_unaligned_len);
  RUN_TEST(test_vol_parses_compress4d_codec);
  RUN_TEST(test_vol_read_chunk_compress4d);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(compress4d_zarr_suite);
  GREATEST_MAIN_END();
}
