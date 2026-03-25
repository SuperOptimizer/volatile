// test_real_data.c — integration tests against live Vesuvius Challenge scroll data.
// All tests are skipped if there is no network connectivity.
//
// Target volume:
//   https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/
//       volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr
//
// Level 4 is the smallest scale (shape ~[899,493,506]) and is used for chunk
// tests to keep transfer sizes manageable (~1.6 MB per chunk).
//
// dimension_separator="/" means chunk keys are z/y/x, not z.y.x.

#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "core/vol.h"
#include "core/net.h"

// greatest.h doesn't have SKIP_IF; emulate it.
#define SKIP_IF(cond) do { if (cond) { SKIPm(#cond); } } while (0)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define SCROLL1_URL \
  "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/" \
  "volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr"

// Level 0 .zarray URL (to check network availability)
#define SCROLL1_ZARRAY_URL SCROLL1_URL "/0/.zarray"

// Level 4 .zarray URL (small scale metadata)
#define SCROLL1_L4_ZARRAY_URL SCROLL1_URL "/4/.zarray"

// A specific chunk at level 4: chunk coordinate (0,0,0)
// URL format: <zarr>/4/0/0/0  (dimension_separator="/")
#define SCROLL1_L4_CHUNK_URL SCROLL1_URL "/4/0/0/0"

// Known metadata for level 4 (shape ~[899,493,506], chunks [128,128,128])
#define SCROLL1_L4_CHUNK_Z 128
#define SCROLL1_L4_CHUNK_Y 128
#define SCROLL1_L4_CHUNK_X 128
#define SCROLL1_L4_SHAPE_Z 899
#define SCROLL1_L4_DTYPE   DTYPE_U8

// Timeout for real network requests (5 seconds)
#define NET_TIMEOUT_MS 5000

// ---------------------------------------------------------------------------
// Network availability check
// ---------------------------------------------------------------------------

static bool g_network_checked = false;
static bool g_network_available = false;

static bool check_network(void) {
  if (g_network_checked) return g_network_available;
  g_network_checked = true;

  http_init();
  http_response *r = http_head(SCROLL1_ZARRAY_URL, NET_TIMEOUT_MS);
  if (r && r->status_code == 200) {
    g_network_available = true;
  }
  http_response_free(r);
  return g_network_available;
}

// ---------------------------------------------------------------------------
// Test: fetch .zarray JSON for level 4
// ---------------------------------------------------------------------------

TEST test_fetch_zarray_json(void) {
  SKIP_IF(!check_network());

  http_response *r = http_get(SCROLL1_L4_ZARRAY_URL, NET_TIMEOUT_MS);
  ASSERT_NEQ(NULL, r);
  ASSERT_EQ(200, r->status_code);
  ASSERT_NEQ(NULL, r->data);
  ASSERT(r->size > 0);

  // Should contain zarr metadata fields
  const char *body = (const char *)r->data;
  ASSERT_NEQ(NULL, strstr(body, "\"chunks\""));
  ASSERT_NEQ(NULL, strstr(body, "\"shape\""));
  ASSERT_NEQ(NULL, strstr(body, "\"dtype\""));
  // dimension_separator must be "/" for correct chunk key construction
  ASSERT_NEQ(NULL, strstr(body, "dimension_separator"));

  http_response_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: parse level 4 .zarray into zarr_level_meta
// ---------------------------------------------------------------------------

TEST test_parse_zarray_metadata(void) {
  SKIP_IF(!check_network());

  http_response *r = http_get(SCROLL1_L4_ZARRAY_URL, NET_TIMEOUT_MS);
  ASSERT_NEQ(NULL, r);
  ASSERT_EQ(200, r->status_code);

  zarr_level_meta m;
  bool ok = zarr_parse_zarray((const char *)r->data, &m);
  http_response_free(r);

  ASSERT(ok);
  ASSERT_EQ(3, m.ndim);
  ASSERT_EQ(SCROLL1_L4_DTYPE, m.dtype);
  ASSERT_EQ(SCROLL1_L4_CHUNK_Z, m.chunk_shape[0]);
  ASSERT_EQ(SCROLL1_L4_CHUNK_Y, m.chunk_shape[1]);
  ASSERT_EQ(SCROLL1_L4_CHUNK_X, m.chunk_shape[2]);
  // Shape[0] should be ~899 (downsampled by 2^4 = 16 from level 0 ~14376)
  ASSERT(m.shape[0] >= 800 && m.shape[0] <= 1000);
  ASSERT_EQ(2, m.zarr_version);
  // dimension_separator is "/" so chunk_sep must be '/'
  ASSERT_EQ('/', m.chunk_sep);
  // compressor should be blosc
  ASSERT_STR_EQ("blosc", m.compressor_id);

  PASS();
}

// ---------------------------------------------------------------------------
// Test: fetch level 4 chunk (0,0,0) and decompress
// ---------------------------------------------------------------------------

TEST test_fetch_and_decompress_chunk(void) {
  SKIP_IF(!check_network());

  // Fetch the raw (compressed) chunk data
  http_response *r = http_get(SCROLL1_L4_CHUNK_URL, NET_TIMEOUT_MS);
  ASSERT_NEQ(NULL, r);
  ASSERT_EQ_FMT(200, r->status_code, "%d");

  // Data must be non-empty (blosc-compressed ~1.6 MB)
  ASSERT(r->size > 0);
  ASSERT(r->size < 8 * 1024 * 1024);  // sanity: < 8 MB compressed

  http_response_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: vol_open on the remote URL
// ---------------------------------------------------------------------------

TEST test_vol_open_remote(void) {
  SKIP_IF(!check_network());

  volume *v = vol_open(SCROLL1_URL);
  ASSERT_NEQ(NULL, v);

  // Should discover multiple pyramid levels
  int nlevels = vol_num_levels(v);
  ASSERT(nlevels >= 2);

  // Level 0: full resolution ~[14376, 7888, 8096]
  const zarr_level_meta *m0 = vol_level_meta(v, 0);
  ASSERT_NEQ(NULL, m0);
  ASSERT_EQ(3, m0->ndim);
  ASSERT(m0->shape[0] > 10000);
  ASSERT_EQ(DTYPE_U8, m0->dtype);
  ASSERT_EQ('/', m0->chunk_sep);

  // Level 4 (if present): smaller shape
  if (nlevels > 4) {
    const zarr_level_meta *m4 = vol_level_meta(v, 4);
    ASSERT_NEQ(NULL, m4);
    ASSERT_EQ(3, m4->ndim);
    ASSERT(m4->shape[0] < m0->shape[0]);
    ASSERT_EQ('/', m4->chunk_sep);
  }

  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: vol_read_chunk from level 4 chunk (0,0,0)
// ---------------------------------------------------------------------------

TEST test_vol_read_chunk_remote(void) {
  SKIP_IF(!check_network());

  volume *v = vol_open(SCROLL1_URL);
  ASSERT_NEQ(NULL, v);

  int nlevels = vol_num_levels(v);
  // Use the last level (smallest) for a manageable download
  int level = nlevels - 1;

  int64_t coords[3] = {0, 0, 0};
  size_t out_size = 0;
  uint8_t *data = vol_read_chunk(v, level, coords, &out_size);

  ASSERT_NEQ(NULL, data);
  // Decompressed chunk: 128*128*128 = 2,097,152 bytes for u8
  ASSERT(out_size > 0);
  ASSERT(out_size <= 128 * 128 * 128 * 2);  // at most 2 bytes/voxel

  free(data);
  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: vol_sample at a voxel near the center of level 4
// ---------------------------------------------------------------------------

TEST test_vol_sample_remote(void) {
  SKIP_IF(!check_network());

  volume *v = vol_open(SCROLL1_URL);
  ASSERT_NEQ(NULL, v);

  int nlevels = vol_num_levels(v);
  int level = nlevels - 1;

  const zarr_level_meta *m = vol_level_meta(v, level);
  ASSERT_NEQ(NULL, m);

  // Sample near center — value should be in [0, 255] for u8
  float cz = (float)(m->shape[0] / 4);
  float cy = (float)(m->shape[1] / 4);
  float cx = (float)(m->shape[2] / 4);

  float val = vol_sample(v, level, cz, cy, cx);
  ASSERT(val >= 0.0f && val <= 255.0f);

  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(real_data_suite) {
  RUN_TEST(test_fetch_zarray_json);
  RUN_TEST(test_parse_zarray_metadata);
  RUN_TEST(test_fetch_and_decompress_chunk);
  RUN_TEST(test_vol_open_remote);
  RUN_TEST(test_vol_read_chunk_remote);
  RUN_TEST(test_vol_sample_remote);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(real_data_suite);
  GREATEST_MAIN_END();
}
