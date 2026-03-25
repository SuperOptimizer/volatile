#include "greatest.h"
#include "core/vol.h"
#include "core/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int make_dir(const char *path) {
  return mkdir(path, 0755);
}

// Write bytes to a file, creating it.
static bool write_file(const char *path, const void *data, size_t size) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  size_t n = fwrite(data, 1, size, f);
  fclose(f);
  return n == size;
}

// Declare internal parsers exposed for testing
extern bool zarr_parse_zarray(const char *json_str, zarr_level_meta *out);
extern bool zarr_parse_zarr_json(const char *json_str, zarr_level_meta *out);

// ---------------------------------------------------------------------------
// Test: .zarray JSON parsing
// ---------------------------------------------------------------------------

TEST test_parse_zarray_basic(void) {
  const char *json =
    "{"
    "  \"chunks\": [64, 64, 64],"
    "  \"shape\":  [512, 512, 256],"
    "  \"dtype\":  \"<u1\","
    "  \"order\":  \"C\","
    "  \"compressor\": {"
    "    \"id\":      \"blosc\","
    "    \"cname\":   \"zstd\","
    "    \"clevel\":  5,"
    "    \"shuffle\": 1"
    "  },"
    "  \"zarr_format\": 2"
    "}";

  zarr_level_meta meta;
  bool ok = zarr_parse_zarray(json, &meta);
  ASSERT(ok);

  ASSERT_EQ(3, meta.ndim);
  ASSERT_EQ((int64_t)64,  meta.chunk_shape[0]);
  ASSERT_EQ((int64_t)64,  meta.chunk_shape[1]);
  ASSERT_EQ((int64_t)64,  meta.chunk_shape[2]);
  ASSERT_EQ((int64_t)512, meta.shape[0]);
  ASSERT_EQ((int64_t)512, meta.shape[1]);
  ASSERT_EQ((int64_t)256, meta.shape[2]);
  ASSERT_EQ(DTYPE_U8, meta.dtype);
  ASSERT_EQ('C', meta.order);
  ASSERT_STR_EQ("blosc", meta.compressor_id);
  ASSERT_STR_EQ("zstd",  meta.compressor_cname);
  ASSERT_EQ(5, meta.compressor_clevel);
  ASSERT_EQ(1, meta.compressor_shuffle);

  PASS();
}

TEST test_parse_zarray_u16(void) {
  const char *json =
    "{"
    "  \"chunks\": [32, 32],"
    "  \"shape\":  [1024, 1024],"
    "  \"dtype\":  \"<u2\","
    "  \"order\":  \"C\","
    "  \"compressor\": null,"
    "  \"zarr_format\": 2"
    "}";

  zarr_level_meta meta;
  bool ok = zarr_parse_zarray(json, &meta);
  ASSERT(ok);
  ASSERT_EQ(2, meta.ndim);
  ASSERT_EQ(DTYPE_U16, meta.dtype);
  ASSERT_EQ((int64_t)32, meta.chunk_shape[0]);
  ASSERT_EQ((int64_t)32, meta.chunk_shape[1]);
  // compressor_id should be empty when null compressor
  ASSERT_EQ('\0', meta.compressor_id[0]);

  PASS();
}

TEST test_parse_zarray_fortran_order(void) {
  const char *json =
    "{"
    "  \"chunks\": [16, 16, 16],"
    "  \"shape\":  [64, 64, 64],"
    "  \"dtype\":  \"<f4\","
    "  \"order\":  \"F\","
    "  \"compressor\": null,"
    "  \"zarr_format\": 2"
    "}";

  zarr_level_meta meta;
  bool ok = zarr_parse_zarray(json, &meta);
  ASSERT(ok);
  ASSERT_EQ('F', meta.order);
  ASSERT_EQ(DTYPE_F32, meta.dtype);

  PASS();
}

TEST test_parse_zarray_bad_json(void) {
  zarr_level_meta meta;
  bool ok = zarr_parse_zarray("{not valid json!!!", &meta);
  ASSERT(!ok);
  PASS();
}

TEST test_parse_zarray_missing_chunks(void) {
  // "chunks" key is mandatory
  const char *json = "{\"shape\": [64], \"dtype\": \"<u1\", \"order\": \"C\"}";
  zarr_level_meta meta;
  bool ok = zarr_parse_zarray(json, &meta);
  ASSERT(!ok);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: synthetic .zarr directory on disk
// ---------------------------------------------------------------------------

// Build a tiny zarr in /tmp for integration tests.
// Structure: /tmp/test_vol_XXXX/
//   0/.zarray  (8x8x8 volume, 4x4x4 chunks, u8, no compressor)
//   0/0.0.0    raw chunk bytes
//   0/0.0.1    raw chunk bytes
//   1/.zarray  (4x4x4 volume, 4x4x4 chunks, u8, no compressor)
//   1/0.0.0    raw chunk bytes

static char g_zarr_base[256];

static void setup_synthetic_zarr(void) {
  snprintf(g_zarr_base, sizeof(g_zarr_base), "/tmp/test_vol_%d.zarr", (int)getpid());
  make_dir(g_zarr_base);

  // --- level 0 ---
  char l0[512];
  snprintf(l0, sizeof(l0), "%s/0", g_zarr_base);
  make_dir(l0);

  const char *zarray0 =
    "{"
    "  \"chunks\": [4, 4, 4],"
    "  \"shape\":  [8, 8, 8],"
    "  \"dtype\":  \"|u1\","
    "  \"order\":  \"C\","
    "  \"compressor\": null,"
    "  \"zarr_format\": 2"
    "}";
  char zpath[512];
  snprintf(zpath, sizeof(zpath), "%s/.zarray", l0);
  write_file(zpath, zarray0, strlen(zarray0));

  // chunk 0.0.0 — fill with index value (0..63)
  uint8_t chunk0[64];
  for (int i = 0; i < 64; i++) chunk0[i] = (uint8_t)i;
  char cpath[512];
  snprintf(cpath, sizeof(cpath), "%s/0.0.0", l0);
  write_file(cpath, chunk0, 64);

  // chunk 0.0.1 — fill with 200+index
  uint8_t chunk1[64];
  for (int i = 0; i < 64; i++) chunk1[i] = (uint8_t)(200 + i);
  snprintf(cpath, sizeof(cpath), "%s/0.0.1", l0);
  write_file(cpath, chunk1, 64);

  // --- level 1 ---
  char l1[512];
  snprintf(l1, sizeof(l1), "%s/1", g_zarr_base);
  make_dir(l1);

  const char *zarray1 =
    "{"
    "  \"chunks\": [4, 4, 4],"
    "  \"shape\":  [4, 4, 4],"
    "  \"dtype\":  \"|u1\","
    "  \"order\":  \"C\","
    "  \"compressor\": null,"
    "  \"zarr_format\": 2"
    "}";
  snprintf(zpath, sizeof(zpath), "%s/.zarray", l1);
  write_file(zpath, zarray1, strlen(zarray1));

  uint8_t chunk_l1[64];
  for (int i = 0; i < 64; i++) chunk_l1[i] = (uint8_t)(100 + i);
  snprintf(cpath, sizeof(cpath), "%s/0.0.0", l1);
  write_file(cpath, chunk_l1, 64);
}

TEST test_vol_open_levels(void) {
  setup_synthetic_zarr();

  volume *v = vol_open(g_zarr_base);
  ASSERT(v != NULL);
  ASSERT_EQ(2, vol_num_levels(v));

  ASSERT_FALSE(vol_is_remote(v));
  ASSERT_EQ(VOL_LOCAL, vol_source(v));
  ASSERT_STR_EQ(g_zarr_base, vol_path(v));

  vol_free(v);
  PASS();
}

TEST test_vol_shape(void) {
  volume *v = vol_open(g_zarr_base);
  ASSERT(v != NULL);

  // level 0: shape 8x8x8
  int64_t shape[8] = {0};
  vol_shape(v, 0, shape);
  ASSERT_EQ((int64_t)8, shape[0]);
  ASSERT_EQ((int64_t)8, shape[1]);
  ASSERT_EQ((int64_t)8, shape[2]);

  // level 1: shape 4x4x4
  int64_t shape1[8] = {0};
  vol_shape(v, 1, shape1);
  ASSERT_EQ((int64_t)4, shape1[0]);
  ASSERT_EQ((int64_t)4, shape1[1]);
  ASSERT_EQ((int64_t)4, shape1[2]);

  vol_free(v);
  PASS();
}

TEST test_vol_level_meta(void) {
  volume *v = vol_open(g_zarr_base);
  ASSERT(v != NULL);

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT_EQ(3, m->ndim);
  ASSERT_EQ(DTYPE_U8, m->dtype);
  ASSERT_EQ((int64_t)4, m->chunk_shape[0]);
  ASSERT_EQ('C', m->order);

  // out-of-range returns NULL
  ASSERT_EQ(NULL, vol_level_meta(v, 99));

  vol_free(v);
  PASS();
}

TEST test_vol_read_chunk(void) {
  volume *v = vol_open(g_zarr_base);
  ASSERT(v != NULL);

  // read chunk 0.0.0 at level 0 (no compression, raw bytes)
  int64_t coords[3] = {0, 0, 0};
  size_t sz = 0;
  uint8_t *data = vol_read_chunk(v, 0, coords, &sz);
  ASSERT(data != NULL);
  ASSERT_EQ((size_t)64, sz);

  // verify content: index values 0..63
  for (int i = 0; i < 64; i++) {
    ASSERT_EQ((uint8_t)i, data[i]);
  }
  free(data);

  // read chunk 0.0.1 at level 0
  int64_t coords2[3] = {0, 0, 1};
  size_t sz2 = 0;
  uint8_t *data2 = vol_read_chunk(v, 0, coords2, &sz2);
  ASSERT(data2 != NULL);
  ASSERT_EQ((size_t)64, sz2);
  ASSERT_EQ((uint8_t)200, data2[0]);
  free(data2);

  // missing chunk returns NULL
  int64_t coords3[3] = {1, 1, 1};
  size_t sz3 = 0;
  uint8_t *data3 = vol_read_chunk(v, 0, coords3, &sz3);
  ASSERT_EQ(NULL, data3);

  vol_free(v);
  PASS();
}

TEST test_vol_open_nonexistent(void) {
  volume *v = vol_open("/tmp/does_not_exist_zarr_xyz");
  ASSERT_EQ(NULL, v);
  PASS();
}

TEST test_vol_open_remote_stub(void) {
  // Remote volumes return NULL (not yet implemented)
  volume *v = vol_open("https://example.com/volume.zarr");
  ASSERT_EQ(NULL, v);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: Zarr v3 zarr.json parsing
// ---------------------------------------------------------------------------

TEST test_parse_zarr_json_basic(void) {
  const char *json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [256, 512, 512],"
    "  \"data_type\": \"uint8\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [64, 128, 128]}"
    "  },"
    "  \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}},"
    "  \"codecs\": ["
    "    {\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}},"
    "    {\"name\": \"blosc\", \"configuration\": {\"cname\": \"zstd\", \"clevel\": 5, \"shuffle\": 1}}"
    "  ]"
    "}";

  zarr_level_meta meta;
  bool ok = zarr_parse_zarr_json(json, &meta);
  ASSERT(ok);

  ASSERT_EQ(3, meta.zarr_version);
  ASSERT_EQ(3, meta.ndim);
  ASSERT_EQ((int64_t)256, meta.shape[0]);
  ASSERT_EQ((int64_t)512, meta.shape[1]);
  ASSERT_EQ((int64_t)512, meta.shape[2]);
  ASSERT_EQ((int64_t)64,  meta.chunk_shape[0]);
  ASSERT_EQ((int64_t)128, meta.chunk_shape[1]);
  ASSERT_EQ((int64_t)128, meta.chunk_shape[2]);
  ASSERT_EQ(DTYPE_U8, meta.dtype);
  ASSERT_EQ('C', meta.order);
  ASSERT_STR_EQ("blosc", meta.compressor_id);
  ASSERT_STR_EQ("zstd",  meta.compressor_cname);
  ASSERT_EQ(5, meta.compressor_clevel);
  ASSERT_FALSE(meta.sharded);

  PASS();
}

TEST test_parse_zarr_json_float32(void) {
  const char *json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [1024, 1024],"
    "  \"data_type\": \"float32\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [256, 256]}"
    "  },"
    "  \"codecs\": [{\"name\": \"bytes\"}]"
    "}";

  zarr_level_meta meta;
  ASSERT(zarr_parse_zarr_json(json, &meta));
  ASSERT_EQ(DTYPE_F32, meta.dtype);
  ASSERT_EQ(2, meta.ndim);
  ASSERT_EQ((int64_t)256, meta.chunk_shape[0]);
  PASS();
}

TEST test_parse_zarr_json_wrong_format(void) {
  // zarr_format != 3 must fail
  const char *json =
    "{"
    "  \"zarr_format\": 2,"
    "  \"node_type\": \"array\","
    "  \"shape\": [64],"
    "  \"data_type\": \"uint8\","
    "  \"chunk_grid\": {\"name\": \"regular\", \"configuration\": {\"chunk_shape\": [16]}}"
    "}";
  zarr_level_meta meta;
  ASSERT_FALSE(zarr_parse_zarr_json(json, &meta));
  PASS();
}

TEST test_parse_zarr_json_sharding(void) {
  const char *json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [256, 256, 256],"
    "  \"data_type\": \"uint8\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [128, 128, 128]}"
    "  },"
    "  \"codecs\": [{"
    "    \"name\": \"sharding_indexed\","
    "    \"configuration\": {"
    "      \"chunk_shape\": [32, 32, 32],"
    "      \"codecs\": ["
    "        {\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}},"
    "        {\"name\": \"blosc\", \"configuration\": {\"cname\": \"lz4\", \"clevel\": 3, \"shuffle\": 1}}"
    "      ],"
    "      \"index_codecs\": [{\"name\": \"bytes\"}, {\"name\": \"crc32c\"}]"
    "    }"
    "  }]"
    "}";

  zarr_level_meta meta;
  bool ok = zarr_parse_zarr_json(json, &meta);
  ASSERT(ok);

  ASSERT_EQ(3, meta.zarr_version);
  ASSERT(meta.sharded);
  // shard_shape should be the outer grid size: 128x128x128
  ASSERT_EQ((int64_t)128, meta.shard_shape[0]);
  ASSERT_EQ((int64_t)128, meta.shard_shape[1]);
  ASSERT_EQ((int64_t)128, meta.shard_shape[2]);
  // chunk_shape should be the inner size: 32x32x32
  ASSERT_EQ((int64_t)32, meta.chunk_shape[0]);
  ASSERT_EQ((int64_t)32, meta.chunk_shape[1]);
  ASSERT_EQ((int64_t)32, meta.chunk_shape[2]);
  // inner codec: blosc/lz4
  ASSERT_STR_EQ("blosc", meta.compressor_id);
  ASSERT_STR_EQ("lz4",   meta.compressor_cname);
  ASSERT_EQ(3, meta.compressor_clevel);

  PASS();
}

// ---------------------------------------------------------------------------
// Tests: synthetic Zarr v3 directory
// ---------------------------------------------------------------------------

static char g_zarr3_base[256];

static void setup_synthetic_zarr3(void) {
  snprintf(g_zarr3_base, sizeof(g_zarr3_base), "/tmp/test_vol3_%d.zarr", (int)getpid());
  make_dir(g_zarr3_base);

  // level 0: 8x8x8 volume, 4x4x4 chunks, u8, no compressor
  char l0[512];
  snprintf(l0, sizeof(l0), "%s/0", g_zarr3_base);
  make_dir(l0);

  const char *zarr_json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [8, 8, 8],"
    "  \"data_type\": \"uint8\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [4, 4, 4]}"
    "  },"
    "  \"codecs\": [{\"name\": \"bytes\"}]"
    "}";
  char zpath[512];
  snprintf(zpath, sizeof(zpath), "%s/zarr.json", l0);
  write_file(zpath, zarr_json, strlen(zarr_json));

  // v3 chunk key: c/0/0/0
  char cdir[512];
  snprintf(cdir, sizeof(cdir), "%s/c", l0);
  make_dir(cdir);
  char c0[512]; snprintf(c0, sizeof(c0), "%s/0", cdir); make_dir(c0);
  char c00[512]; snprintf(c00, sizeof(c00), "%s/0", c0); make_dir(c00);

  uint8_t chunk[64];
  for (int i = 0; i < 64; i++) chunk[i] = (uint8_t)(10 + i);
  char cpath[512];
  snprintf(cpath, sizeof(cpath), "%s/0", c00);
  write_file(cpath, chunk, 64);
}

TEST test_vol_v3_open(void) {
  setup_synthetic_zarr3();
  volume *v = vol_open(g_zarr3_base);
  ASSERT(v != NULL);
  ASSERT_EQ(1, vol_num_levels(v));

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT_EQ(3, m->zarr_version);
  ASSERT_EQ(3, m->ndim);
  ASSERT_EQ(DTYPE_U8, m->dtype);
  ASSERT_EQ((int64_t)8, m->shape[0]);
  ASSERT_EQ((int64_t)4, m->chunk_shape[0]);

  vol_free(v);
  PASS();
}

TEST test_vol_v3_shape(void) {
  volume *v = vol_open(g_zarr3_base);
  ASSERT(v != NULL);

  int64_t shape[8] = {0};
  vol_shape(v, 0, shape);
  ASSERT_EQ((int64_t)8, shape[0]);
  ASSERT_EQ((int64_t)8, shape[1]);
  ASSERT_EQ((int64_t)8, shape[2]);

  vol_free(v);
  PASS();
}

TEST test_vol_v3_read_chunk(void) {
  volume *v = vol_open(g_zarr3_base);
  ASSERT(v != NULL);

  int64_t coords[3] = {0, 0, 0};
  size_t sz = 0;
  uint8_t *data = vol_read_chunk(v, 0, coords, &sz);
  ASSERT(data != NULL);
  ASSERT_EQ((size_t)64, sz);
  ASSERT_EQ((uint8_t)10, data[0]);  // first byte = 10
  ASSERT_EQ((uint8_t)73, data[63]); // last byte = 10+63
  free(data);

  // missing chunk
  int64_t coords2[3] = {1, 1, 1};
  size_t sz2 = 0;
  uint8_t *data2 = vol_read_chunk(v, 0, coords2, &sz2);
  ASSERT_EQ(NULL, data2);

  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: synthetic sharded Zarr v3
// ---------------------------------------------------------------------------

// Write little-endian uint64
static void write_le64(uint8_t *p, uint64_t v) {
  for (int i = 0; i < 8; i++) { p[i] = (uint8_t)(v & 0xff); v >>= 8; }
}

// Build a synthetic shard file:
//   shard_shape = 8x8x8, chunk_shape = 4x4x4 → 2x2x2 = 8 inner chunks
//   We write chunks 0 and 3 (indices 0 and 3); rest are marked empty.
//   Raw (uncompressed) chunks. Index at end.
static void build_shard(uint8_t **out_buf, size_t *out_size) {
  // chunk data: 4x4x4 = 64 bytes each
  uint8_t chunk0[64], chunk3[64];
  for (int i = 0; i < 64; i++) { chunk0[i] = (uint8_t)i; chunk3[i] = (uint8_t)(200 + i); }

  size_t index_size = 8 * 16;  // 8 chunks × 16 bytes/entry
  size_t data_size  = 64 + 64; // chunk0 + chunk3
  size_t total      = data_size + index_size;

  uint8_t *buf = calloc(1, total);

  // write chunk0 at offset 0, chunk3 at offset 64
  memcpy(buf + 0,  chunk0, 64);
  memcpy(buf + 64, chunk3, 64);

  // build index at the end
  uint8_t *idx = buf + data_size;
  for (int i = 0; i < 8; i++) {
    // mark all empty initially
    write_le64(idx + i * 16,     UINT64_C(0xFFFFFFFFFFFFFFFF));
    write_le64(idx + i * 16 + 8, UINT64_C(0xFFFFFFFFFFFFFFFF));
  }
  // chunk 0: offset=0, nbytes=64
  write_le64(idx + 0 * 16,     0);
  write_le64(idx + 0 * 16 + 8, 64);
  // chunk 3: offset=64, nbytes=64
  write_le64(idx + 3 * 16,     64);
  write_le64(idx + 3 * 16 + 8, 64);

  *out_buf  = buf;
  *out_size = total;
}

static char g_zarr3_shard_base[256];

static void setup_synthetic_sharded_zarr3(void) {
  snprintf(g_zarr3_shard_base, sizeof(g_zarr3_shard_base),
           "/tmp/test_vol3_shard_%d.zarr", (int)getpid());
  make_dir(g_zarr3_shard_base);

  char l0[512];
  snprintf(l0, sizeof(l0), "%s/0", g_zarr3_shard_base);
  make_dir(l0);

  // shard_shape (outer chunk grid) = 8x8x8, inner chunk = 4x4x4
  const char *zarr_json =
    "{"
    "  \"zarr_format\": 3,"
    "  \"node_type\": \"array\","
    "  \"shape\": [8, 8, 8],"
    "  \"data_type\": \"uint8\","
    "  \"chunk_grid\": {"
    "    \"name\": \"regular\","
    "    \"configuration\": {\"chunk_shape\": [8, 8, 8]}"
    "  },"
    "  \"codecs\": [{"
    "    \"name\": \"sharding_indexed\","
    "    \"configuration\": {"
    "      \"chunk_shape\": [4, 4, 4],"
    "      \"codecs\": [{\"name\": \"bytes\"}],"
    "      \"index_codecs\": [{\"name\": \"bytes\"}]"
    "    }"
    "  }]"
    "}";
  char zpath[512];
  snprintf(zpath, sizeof(zpath), "%s/zarr.json", l0);
  write_file(zpath, zarr_json, strlen(zarr_json));

  // write shard file at c/0/0/0
  char cdir[512];  snprintf(cdir,  sizeof(cdir),  "%s/c",     l0);   make_dir(cdir);
  char c0[512];    snprintf(c0,    sizeof(c0),    "%s/0",     cdir); make_dir(c0);
  char c00[512];   snprintf(c00,   sizeof(c00),   "%s/0",     c0);   make_dir(c00);

  uint8_t *shard_buf = NULL;
  size_t shard_size = 0;
  build_shard(&shard_buf, &shard_size);

  char spath[512];
  snprintf(spath, sizeof(spath), "%s/0", c00);
  write_file(spath, shard_buf, shard_size);
  free(shard_buf);
}

TEST test_vol_sharded_meta(void) {
  setup_synthetic_sharded_zarr3();
  volume *v = vol_open(g_zarr3_shard_base);
  ASSERT(v != NULL);

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT(m->sharded);
  ASSERT_EQ((int64_t)8, m->shard_shape[0]);
  ASSERT_EQ((int64_t)8, m->shard_shape[1]);
  ASSERT_EQ((int64_t)8, m->shard_shape[2]);
  ASSERT_EQ((int64_t)4, m->chunk_shape[0]);
  ASSERT_EQ((int64_t)4, m->chunk_shape[1]);
  ASSERT_EQ((int64_t)4, m->chunk_shape[2]);

  vol_free(v);
  PASS();
}

TEST test_vol_sharded_read_chunk(void) {
  volume *v = vol_open(g_zarr3_shard_base);
  ASSERT(v != NULL);

  // chunk 0 (inner index 0) — should have bytes 0..63
  int64_t coords0[3] = {0, 0, 0};
  size_t sz0 = 0;
  uint8_t *d0 = vol_read_chunk(v, 0, coords0, &sz0);
  ASSERT(d0 != NULL);
  ASSERT_EQ((size_t)64, sz0);
  ASSERT_EQ((uint8_t)0,  d0[0]);
  ASSERT_EQ((uint8_t)63, d0[63]);
  free(d0);

  // chunk 3 (inner index 3 in row-major 2x2x2: coords (0,1,1)) — bytes 200..263%256
  // inner 2x2x2: flat index 3 = z*4 + y*2 + x with nchunks={2,2,2}
  // 3 = 0*4+1*2+1 → local coords (0,1,1)
  int64_t coords3[3] = {0, 1, 1};
  size_t sz3 = 0;
  uint8_t *d3 = vol_read_chunk(v, 0, coords3, &sz3);
  ASSERT(d3 != NULL);
  ASSERT_EQ((size_t)64, sz3);
  ASSERT_EQ((uint8_t)200, d3[0]);
  free(d3);

  // chunk 1 (inner index 1, not present) → NULL
  int64_t coords1[3] = {0, 0, 1};
  size_t sz1 = 0;
  uint8_t *d1 = vol_read_chunk(v, 0, coords1, &sz1);
  ASSERT_EQ(NULL, d1);

  vol_free(v);
  PASS();
}

TEST test_vol_read_shard_all_chunks(void) {
  volume *v = vol_open(g_zarr3_shard_base);
  ASSERT(v != NULL);

  int64_t shard_coords[3] = {0, 0, 0};
  uint8_t **chunks = NULL;
  size_t  *sizes   = NULL;
  size_t   n       = 0;

  int extracted = vol_read_shard(v, 0, shard_coords, &chunks, &sizes, &n);
  ASSERT_EQ(2, extracted);   // only chunks 0 and 3 are present
  ASSERT_EQ((size_t)8, n);   // 2×2×2 = 8 slots total

  // slot 0 present
  ASSERT(chunks[0] != NULL);
  ASSERT_EQ((size_t)64, sizes[0]);
  ASSERT_EQ((uint8_t)0, chunks[0][0]);

  // slot 3 present
  ASSERT(chunks[3] != NULL);
  ASSERT_EQ((uint8_t)200, chunks[3][0]);

  // slot 1 absent
  ASSERT_EQ(NULL, chunks[1]);

  for (size_t i = 0; i < n; i++) free(chunks[i]);
  free(chunks);
  free(sizes);

  vol_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Tests: Write API (vol_create / vol_write_chunk / vol_write_shard /
//                   vol_build_pyramid / vol_finalize)
// ---------------------------------------------------------------------------

static char g_write_base[256];

// Helper: fill a 4x4x4 buffer with a known pattern
static void fill_chunk_pattern(uint8_t *buf, uint8_t base) {
  for (int i = 0; i < 64; i++) buf[i] = (uint8_t)(base + i);
}

TEST test_vol_create_v2(void) {
  snprintf(g_write_base, sizeof(g_write_base), "/tmp/test_write_v2_%d.zarr", (int)getpid());

  vol_create_params p = {
    .zarr_version = 2,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(g_write_base, p);
  ASSERT(v != NULL);
  ASSERT_EQ(1, vol_num_levels(v));

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT(m != NULL);
  ASSERT_EQ(2, m->zarr_version);
  ASSERT_EQ(3, m->ndim);
  ASSERT_EQ((int64_t)8, m->shape[0]);
  ASSERT_EQ((int64_t)4, m->chunk_shape[0]);
  ASSERT_EQ(DTYPE_U8, m->dtype);

  // .zarray file must exist
  char zpath[512];
  snprintf(zpath, sizeof(zpath), "%s/0/.zarray", g_write_base);
  FILE *f = fopen(zpath, "r");
  ASSERT(f != NULL);
  fclose(f);

  vol_free(v);
  PASS();
}

TEST test_vol_create_v3(void) {
  char path[256];
  snprintf(path, sizeof(path), "/tmp/test_write_v3_%d.zarr", (int)getpid());

  vol_create_params p = {
    .zarr_version = 3,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U16,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);

  const zarr_level_meta *m = vol_level_meta(v, 0);
  ASSERT_EQ(3, m->zarr_version);
  ASSERT_EQ(DTYPE_U16, m->dtype);

  // zarr.json file must exist
  char jpath[512];
  snprintf(jpath, sizeof(jpath), "%s/0/zarr.json", path);
  FILE *f = fopen(jpath, "r");
  ASSERT(f != NULL);
  fclose(f);

  vol_free(v);
  PASS();
}

TEST test_vol_write_chunk_roundtrip_v2(void) {
  // reuse g_write_base from test_vol_create_v2 (same pid-based name)
  snprintf(g_write_base, sizeof(g_write_base), "/tmp/test_rtrip_v2_%d.zarr", (int)getpid());

  vol_create_params p = {
    .zarr_version = 2,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(g_write_base, p);
  ASSERT(v != NULL);

  uint8_t chunk_in[64];
  fill_chunk_pattern(chunk_in, 42);

  int64_t coords[3] = {0, 0, 0};
  bool ok = vol_write_chunk(v, 0, coords, chunk_in, sizeof(chunk_in));
  ASSERT(ok);
  vol_free(v);

  // re-open and read back
  volume *v2 = vol_open(g_write_base);
  ASSERT(v2 != NULL);

  size_t sz = 0;
  uint8_t *chunk_out = vol_read_chunk(v2, 0, coords, &sz);
  ASSERT(chunk_out != NULL);
  ASSERT_EQ(sizeof(chunk_in), sz);
  for (int i = 0; i < 64; i++) ASSERT_EQ(chunk_in[i], chunk_out[i]);

  free(chunk_out);
  vol_free(v2);
  PASS();
}

TEST test_vol_write_chunk_roundtrip_v3(void) {
  char path[256];
  snprintf(path, sizeof(path), "/tmp/test_rtrip_v3_%d.zarr", (int)getpid());

  vol_create_params p = {
    .zarr_version = 3,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);

  uint8_t chunk_in[64];
  fill_chunk_pattern(chunk_in, 7);

  int64_t coords[3] = {0, 1, 0};
  ASSERT(vol_write_chunk(v, 0, coords, chunk_in, sizeof(chunk_in)));
  vol_free(v);

  volume *v2 = vol_open(path);
  ASSERT(v2 != NULL);

  size_t sz = 0;
  uint8_t *chunk_out = vol_read_chunk(v2, 0, coords, &sz);
  ASSERT(chunk_out != NULL);
  ASSERT_EQ((size_t)64, sz);
  for (int i = 0; i < 64; i++) ASSERT_EQ(chunk_in[i], chunk_out[i]);

  free(chunk_out);
  vol_free(v2);
  PASS();
}

TEST test_vol_write_shard_roundtrip(void) {
  char path[256];
  snprintf(path, sizeof(path), "/tmp/test_rtrip_shard_%d.zarr", (int)getpid());

  // shard_shape = 8x8x8, chunk_shape = 4x4x4 → 2x2x2 = 8 inner chunks
  vol_create_params p = {
    .zarr_version = 3,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = true,
    .shard_shape = {8, 8, 8},
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);

  // prepare 8 inner chunks (2x2x2); set chunks 0 and 5 non-null
  uint8_t c0[64], c5[64];
  fill_chunk_pattern(c0, 0);
  fill_chunk_pattern(c5, 100);

  const void *chunk_data[8] = { c0, NULL, NULL, NULL, NULL, c5, NULL, NULL };
  size_t chunk_sizes[8]     = { 64,    0,    0,    0,    0,  64,    0,    0 };

  int64_t shard_coords[3] = {0, 0, 0};
  ASSERT(vol_write_shard(v, 0, shard_coords, chunk_data, chunk_sizes, 8));
  vol_free(v);

  // re-open and read back individual chunks
  volume *v2 = vol_open(path);
  ASSERT(v2 != NULL);

  // inner flat index 0 → global coords (0,0,0)
  int64_t coord0[3] = {0, 0, 0};
  size_t sz0 = 0;
  uint8_t *d0 = vol_read_chunk(v2, 0, coord0, &sz0);
  ASSERT(d0 != NULL);
  ASSERT_EQ((size_t)64, sz0);
  ASSERT_EQ(c0[0], d0[0]);
  ASSERT_EQ(c0[63], d0[63]);
  free(d0);

  // inner flat index 5 → 2x2x2 grid: 5 = 1*4+0*2+1 → (1,0,1)
  int64_t coord5[3] = {1, 0, 1};
  size_t sz5 = 0;
  uint8_t *d5 = vol_read_chunk(v2, 0, coord5, &sz5);
  ASSERT(d5 != NULL);
  ASSERT_EQ(c5[0], d5[0]);
  free(d5);

  // inner flat index 1 → absent → NULL
  int64_t coord1[3] = {0, 0, 1};
  size_t sz1 = 0;
  uint8_t *d1 = vol_read_chunk(v2, 0, coord1, &sz1);
  ASSERT_EQ(NULL, d1);

  vol_free(v2);
  PASS();
}

TEST test_vol_build_pyramid(void) {
  char path[256];
  snprintf(path, sizeof(path), "/tmp/test_pyramid_%d.zarr", (int)getpid());

  // create an 8x8x8 volume with 4x4x4 chunks, fill every chunk
  vol_create_params p = {
    .zarr_version = 2,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);

  // write all 8 chunks (2x2x2 chunks)
  for (int cz = 0; cz < 2; cz++) {
    for (int cy = 0; cy < 2; cy++) {
      for (int cx = 0; cx < 2; cx++) {
        uint8_t buf[64];
        // fill with a constant value based on chunk position
        memset(buf, (uint8_t)(40 + cz * 4 + cy * 2 + cx), 64);
        int64_t cc[3] = {cz, cy, cx};
        ASSERT(vol_write_chunk(v, 0, cc, buf, 64));
      }
    }
  }

  // build pyramid (1 additional level from 8x8x8 → 4x4x4)
  ASSERT(vol_build_pyramid(v, 1));
  ASSERT_EQ(2, vol_num_levels(v));

  // level 1 shape must be 4x4x4
  int64_t shape1[8] = {0};
  vol_shape(v, 1, shape1);
  ASSERT_EQ((int64_t)4, shape1[0]);
  ASSERT_EQ((int64_t)4, shape1[1]);
  ASSERT_EQ((int64_t)4, shape1[2]);

  // the single chunk at level 1 must be readable and contain the downsampled mean
  int64_t cc[3] = {0, 0, 0};
  size_t sz = 0;
  uint8_t *data = vol_read_chunk(v, 1, cc, &sz);
  ASSERT(data != NULL);
  // downsampled from constant-value chunks, each element ≥ 40
  ASSERT(data[0] >= 40);
  free(data);

  vol_free(v);
  PASS();
}

TEST test_vol_finalize_writes_zattrs(void) {
  char path[256];
  snprintf(path, sizeof(path), "/tmp/test_finalize_%d.zarr", (int)getpid());

  vol_create_params p = {
    .zarr_version = 2,
    .ndim = 3,
    .shape       = {8, 8, 8},
    .chunk_shape = {4, 4, 4},
    .dtype       = DTYPE_U8,
    .compressor  = NULL,
    .clevel      = 0,
    .sharded     = false,
  };
  volume *v = vol_create(path, p);
  ASSERT(v != NULL);
  ASSERT(vol_finalize(v));
  vol_free(v);

  // .zattrs file must exist and mention "multiscales"
  char zpath[512];
  snprintf(zpath, sizeof(zpath), "%s/.zattrs", path);
  FILE *f = fopen(zpath, "r");
  ASSERT(f != NULL);
  char buf[1024] = {0};
  size_t nr = fread(buf, 1, sizeof(buf) - 1, f);
  (void)nr;
  fclose(f);
  ASSERT(strstr(buf, "multiscales") != NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(vol_suite) {
  // .zarray (v2) parsing
  RUN_TEST(test_parse_zarray_basic);
  RUN_TEST(test_parse_zarray_u16);
  RUN_TEST(test_parse_zarray_fortran_order);
  RUN_TEST(test_parse_zarray_bad_json);
  RUN_TEST(test_parse_zarray_missing_chunks);

  // zarr.json (v3) parsing
  RUN_TEST(test_parse_zarr_json_basic);
  RUN_TEST(test_parse_zarr_json_float32);
  RUN_TEST(test_parse_zarr_json_wrong_format);
  RUN_TEST(test_parse_zarr_json_sharding);

  // v2 disk-backed integration
  RUN_TEST(test_vol_open_levels);
  RUN_TEST(test_vol_shape);
  RUN_TEST(test_vol_level_meta);
  RUN_TEST(test_vol_read_chunk);
  RUN_TEST(test_vol_open_nonexistent);
  RUN_TEST(test_vol_open_remote_stub);

  // v3 disk-backed integration
  RUN_TEST(test_vol_v3_open);
  RUN_TEST(test_vol_v3_shape);
  RUN_TEST(test_vol_v3_read_chunk);

  // sharding
  RUN_TEST(test_vol_sharded_meta);
  RUN_TEST(test_vol_sharded_read_chunk);
  RUN_TEST(test_vol_read_shard_all_chunks);

  // write API
  RUN_TEST(test_vol_create_v2);
  RUN_TEST(test_vol_create_v3);
  RUN_TEST(test_vol_write_chunk_roundtrip_v2);
  RUN_TEST(test_vol_write_chunk_roundtrip_v3);
  RUN_TEST(test_vol_write_shard_roundtrip);
  RUN_TEST(test_vol_build_pyramid);
  RUN_TEST(test_vol_finalize_writes_zattrs);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vol_suite);
  GREATEST_MAIN_END();
}
