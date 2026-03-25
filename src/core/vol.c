#include "core/vol_internal.h"
#include "core/io.h"
#include "core/json.h"
#include "core/log.h"

#include <blosc2.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define VOL_MAX_LEVELS 16
#define VOL_PATH_MAX   1024

// Shard index entry: two little-endian uint64 values (offset, nbytes).
// A value of UINT64_MAX means the chunk is not present (empty/fill-value).
#define SHARD_INDEX_ENTRY_BYTES 16
#define SHARD_EMPTY_ENTRY       UINT64_C(0xFFFFFFFFFFFFFFFF)

// ---------------------------------------------------------------------------
// Internal structure
// ---------------------------------------------------------------------------

struct volume {
  char          path[VOL_PATH_MAX];
  vol_source_t  source;
  int           num_levels;
  zarr_level_meta levels[VOL_MAX_LEVELS];
};

// ---------------------------------------------------------------------------
// Helpers — dtype parsing
// ---------------------------------------------------------------------------

// Map a zarr v2 dtype string (e.g. "|u1", "<u2", "<f4") or v3 data_type name
// ("uint8", "uint16", "float32", "float64") to a dtype_t int. Returns -1 on unknown.
static int parse_dtype(const char *s) {
  if (!s) return -1;
  // v3 long-form names
  if (strcmp(s, "uint8")   == 0) return DTYPE_U8;
  if (strcmp(s, "uint16")  == 0) return DTYPE_U16;
  if (strcmp(s, "float32") == 0) return DTYPE_F32;
  if (strcmp(s, "float64") == 0) return DTYPE_F64;
  // v2 short-form names — skip endian prefix (|, <, >)
  if (*s == '|' || *s == '<' || *s == '>') s++;
  if (strcmp(s, "u1") == 0) return DTYPE_U8;
  if (strcmp(s, "u2") == 0) return DTYPE_U16;
  if (strcmp(s, "f4") == 0) return DTYPE_F32;
  if (strcmp(s, "f8") == 0) return DTYPE_F64;
  return -1;
}

// ---------------------------------------------------------------------------
// .zarray (Zarr v2) JSON parsing
// ---------------------------------------------------------------------------

// Parse a .zarray JSON string into a zarr_level_meta. Returns true on success.
bool zarr_parse_zarray(const char *json_str, zarr_level_meta *out) {
  if (!json_str || !out) return false;
  memset(out, 0, sizeof(*out));

  json_value *root = json_parse(json_str);
  if (!root) return false;

  // "chunks" array
  const json_value *chunks_arr = json_object_get(root, "chunks");
  if (!chunks_arr || json_typeof(chunks_arr) != JSON_ARRAY) {
    json_free(root);
    return false;
  }
  out->ndim = (int)json_array_len(chunks_arr);
  if (out->ndim < 1 || out->ndim > 8) { json_free(root); return false; }
  for (int d = 0; d < out->ndim; d++)
    out->chunk_shape[d] = json_get_int(json_array_get(chunks_arr, (size_t)d), 1);

  // "shape" array
  const json_value *shape_arr = json_object_get(root, "shape");
  if (shape_arr && json_typeof(shape_arr) == JSON_ARRAY) {
    for (int d = 0; d < out->ndim && d < (int)json_array_len(shape_arr); d++)
      out->shape[d] = json_get_int(json_array_get(shape_arr, (size_t)d), 0);
  }

  // "dtype"
  out->dtype = parse_dtype(json_get_str(json_object_get(root, "dtype")));

  // "order"
  const char *order_str = json_get_str(json_object_get(root, "order"));
  out->order = (order_str && order_str[0] == 'F') ? 'F' : 'C';

  // "compressor" object
  const json_value *comp = json_object_get(root, "compressor");
  if (comp && json_typeof(comp) == JSON_OBJECT) {
    const char *cid = json_get_str(json_object_get(comp, "id"));
    if (cid) strncpy(out->compressor_id, cid, sizeof(out->compressor_id) - 1);
    const char *cname = json_get_str(json_object_get(comp, "cname"));
    if (cname) strncpy(out->compressor_cname, cname, sizeof(out->compressor_cname) - 1);
    out->compressor_clevel  = (int)json_get_int(json_object_get(comp, "clevel"), 5);
    out->compressor_shuffle = (int)json_get_int(json_object_get(comp, "shuffle"), 1);
  }

  out->zarr_version = 2;
  json_free(root);
  return true;
}

// ---------------------------------------------------------------------------
// zarr.json (Zarr v3) JSON parsing
// ---------------------------------------------------------------------------

// Walk the v3 "codecs" array and pull out compressor + sharding info.
static void parse_v3_codecs(const json_value *codecs, zarr_level_meta *out) {
  if (!codecs || json_typeof(codecs) != JSON_ARRAY) return;
  size_t n = json_array_len(codecs);
  for (size_t i = 0; i < n; i++) {
    const json_value *codec = json_array_get(codecs, i);
    if (!codec || json_typeof(codec) != JSON_OBJECT) continue;
    const char *name = json_get_str(json_object_get(codec, "name"));
    if (!name) continue;
    const json_value *cfg = json_object_get(codec, "configuration");

    if (strcmp(name, "blosc") == 0 || strcmp(name, "blosc2") == 0) {
      strncpy(out->compressor_id, "blosc", sizeof(out->compressor_id) - 1);
      if (cfg && json_typeof(cfg) == JSON_OBJECT) {
        const char *cname = json_get_str(json_object_get(cfg, "cname"));
        if (cname) strncpy(out->compressor_cname, cname, sizeof(out->compressor_cname) - 1);
        out->compressor_clevel  = (int)json_get_int(json_object_get(cfg, "clevel"), 5);
        out->compressor_shuffle = (int)json_get_int(json_object_get(cfg, "shuffle"), 1);
      }
    } else if (strcmp(name, "zstd") == 0) {
      strncpy(out->compressor_id, "zstd", sizeof(out->compressor_id) - 1);
      if (cfg && json_typeof(cfg) == JSON_OBJECT)
        out->compressor_clevel = (int)json_get_int(json_object_get(cfg, "level"), 5);
    } else if (strcmp(name, "sharding_indexed") == 0) {
      out->sharded = true;
      if (cfg && json_typeof(cfg) == JSON_OBJECT) {
        // inner chunk_shape (the actual data chunk, smaller than the shard)
        // The shard's chunk_shape in the outer grid is already stored in out->chunk_shape.
        // We swap: shard_shape = current chunk_shape (outer), chunk_shape = inner.
        memcpy(out->shard_shape, out->chunk_shape, sizeof(out->shard_shape));
        const json_value *inner = json_object_get(cfg, "chunk_shape");
        if (inner && json_typeof(inner) == JSON_ARRAY) {
          for (int d = 0; d < out->ndim && d < (int)json_array_len(inner); d++)
            out->chunk_shape[d] = json_get_int(json_array_get(inner, (size_t)d), 1);
        }
        // recurse into inner codecs for the actual compressor
        parse_v3_codecs(json_object_get(cfg, "codecs"), out);
      }
    }
  }
}

// Parse a zarr.json (v3) string into a zarr_level_meta. Returns true on success.
bool zarr_parse_zarr_json(const char *json_str, zarr_level_meta *out) {
  if (!json_str || !out) return false;
  memset(out, 0, sizeof(*out));

  json_value *root = json_parse(json_str);
  if (!root) return false;

  // must be zarr_format == 3 and node_type == "array"
  int64_t fmt = json_get_int(json_object_get(root, "zarr_format"), 0);
  if (fmt != 3) { json_free(root); return false; }
  const char *node_type = json_get_str(json_object_get(root, "node_type"));
  if (!node_type || strcmp(node_type, "array") != 0) { json_free(root); return false; }

  // "shape"
  const json_value *shape_arr = json_object_get(root, "shape");
  if (!shape_arr || json_typeof(shape_arr) != JSON_ARRAY) { json_free(root); return false; }
  out->ndim = (int)json_array_len(shape_arr);
  if (out->ndim < 1 || out->ndim > 8) { json_free(root); return false; }
  for (int d = 0; d < out->ndim; d++)
    out->shape[d] = json_get_int(json_array_get(shape_arr, (size_t)d), 0);

  // "data_type"
  out->dtype = parse_dtype(json_get_str(json_object_get(root, "data_type")));

  // "chunk_grid" — only "regular" grids supported
  const json_value *cg = json_object_get(root, "chunk_grid");
  if (cg && json_typeof(cg) == JSON_OBJECT) {
    const char *grid_name = json_get_str(json_object_get(cg, "name"));
    if (grid_name && strcmp(grid_name, "regular") == 0) {
      const json_value *cfg = json_object_get(cg, "configuration");
      if (cfg && json_typeof(cfg) == JSON_OBJECT) {
        const json_value *cs = json_object_get(cfg, "chunk_shape");
        if (cs && json_typeof(cs) == JSON_ARRAY) {
          for (int d = 0; d < out->ndim && d < (int)json_array_len(cs); d++)
            out->chunk_shape[d] = json_get_int(json_array_get(cs, (size_t)d), 1);
        }
      }
    }
  }

  // "chunk_key_encoding" — default separator is "/" for v3; we handle that in chunk_path
  // "dimension_names" — ignored for now
  // "codecs"
  parse_v3_codecs(json_object_get(root, "codecs"), out);

  out->zarr_version = 3;
  out->order = 'C';  // v3 is always C-order (row-major)

  json_free(root);
  return true;
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------

// Read entire file into a malloc'd buffer; caller must free. Sets *size.
static uint8_t *read_file_bytes(const char *path, size_t *size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;

  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long len = ftell(f);
  if (len < 0) { fclose(f); return NULL; }
  rewind(f);

  uint8_t *buf = malloc((size_t)len + 1);
  if (!buf) { fclose(f); return NULL; }

  size_t got = fread(buf, 1, (size_t)len, f);
  fclose(f);

  if ((long)got != len) { free(buf); return NULL; }
  buf[len] = '\0';
  if (size) *size = (size_t)len;
  return buf;
}

// Read len bytes at byte offset from file into a pre-allocated buffer.
// Returns true on success.
static bool read_file_range(const char *path, uint64_t offset, size_t len, uint8_t *buf) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  if (fseek(f, (long)offset, SEEK_SET) != 0) { fclose(f); return false; }
  size_t got = fread(buf, 1, len, f);
  fclose(f);
  return got == len;
}

// ---------------------------------------------------------------------------
// Level discovery
// ---------------------------------------------------------------------------

static bool load_level_v2(const char *base, int level, zarr_level_meta *meta) {
  char path[VOL_PATH_MAX];
  snprintf(path, sizeof(path), "%s/%d/.zarray", base, level);
  size_t sz = 0;
  uint8_t *data = read_file_bytes(path, &sz);
  if (!data) return false;
  bool ok = zarr_parse_zarray((const char *)data, meta);
  free(data);
  return ok;
}

static bool load_level_v3(const char *base, int level, zarr_level_meta *meta) {
  char path[VOL_PATH_MAX];
  snprintf(path, sizeof(path), "%s/%d/zarr.json", base, level);
  size_t sz = 0;
  uint8_t *data = read_file_bytes(path, &sz);
  if (!data) return false;
  bool ok = zarr_parse_zarr_json((const char *)data, meta);
  free(data);
  return ok;
}

// Try v3 first (zarr.json), then fall back to v2 (.zarray).
static bool load_level(const char *base, int level, zarr_level_meta *meta) {
  return load_level_v3(base, level, meta) || load_level_v2(base, level, meta);
}

// ---------------------------------------------------------------------------
// vol_open
// ---------------------------------------------------------------------------

volume *vol_open(const char *path) {
  if (!path) return NULL;

  volume *v = calloc(1, sizeof(volume));
  if (!v) return NULL;

  strncpy(v->path, path, VOL_PATH_MAX - 1);

  if (strncmp(path, "http://", 7) == 0 || strncmp(path, "https://", 8) == 0) {
    v->source = VOL_REMOTE;
  } else {
    v->source = VOL_LOCAL;
  }

  if (v->source == VOL_REMOTE) {
    // TODO: implement remote level discovery via HTTP fetch
    LOG_WARN("remote volumes not yet implemented: %s", path);
    free(v);
    return NULL;
  }

  v->num_levels = 0;
  for (int lvl = 0; lvl < VOL_MAX_LEVELS; lvl++) {
    zarr_level_meta meta;
    if (!load_level(path, lvl, &meta)) break;
    v->levels[v->num_levels++] = meta;
  }

  if (v->num_levels == 0) {
    LOG_WARN("no valid levels found in zarr at %s", path);
    free(v);
    return NULL;
  }

  LOG_INFO("opened volume %s (%d levels, v%d)", path, v->num_levels, v->levels[0].zarr_version);
  return v;
}

void vol_free(volume *v) {
  free(v);
}

// ---------------------------------------------------------------------------
// Metadata accessors
// ---------------------------------------------------------------------------

int vol_num_levels(const volume *v) {
  return v ? v->num_levels : 0;
}

const zarr_level_meta *vol_level_meta(const volume *v, int level) {
  if (!v || level < 0 || level >= v->num_levels) return NULL;
  return &v->levels[level];
}

void vol_shape(const volume *v, int level, int64_t *shape_out) {
  if (!v || !shape_out) return;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m) return;
  for (int d = 0; d < m->ndim; d++) shape_out[d] = m->shape[d];
}

// ---------------------------------------------------------------------------
// Chunk path construction
// ---------------------------------------------------------------------------

// v2: <base>/<level>/iz.iy.ix
// v3: <base>/<level>/c/iz/iy/ix
static void chunk_path(const volume *v, int level, const int64_t *coords, int ndim,
                       char *buf, size_t bufsz) {
  const zarr_level_meta *m = &v->levels[level];
  int used = snprintf(buf, bufsz, "%s/%d", v->path, level);
  if (m->zarr_version == 3) {
    used += snprintf(buf + used, bufsz - (size_t)used, "/c");
    for (int d = 0; d < ndim && used < (int)bufsz - 1; d++)
      used += snprintf(buf + used, bufsz - (size_t)used, "/%lld", (long long)coords[d]);
  } else {
    for (int d = 0; d < ndim && used < (int)bufsz - 1; d++) {
      char sep = (d == 0) ? '/' : '.';
      used += snprintf(buf + used, bufsz - (size_t)used, "%c%lld", sep, (long long)coords[d]);
    }
  }
}

// v3 sharded: shard file is at <base>/<level>/c/sz/sy/sx  (outer grid coords)
static void shard_path(const volume *v, int level, const int64_t *shard_coords, int ndim,
                       char *buf, size_t bufsz) {
  int used = snprintf(buf, bufsz, "%s/%d/c", v->path, level);
  for (int d = 0; d < ndim && used < (int)bufsz - 1; d++)
    used += snprintf(buf + used, bufsz - (size_t)used, "/%lld", (long long)shard_coords[d]);
}

// ---------------------------------------------------------------------------
// Decompression
// ---------------------------------------------------------------------------

static uint8_t *blosc2_decompress_chunk(const uint8_t *src, size_t src_size, size_t *out_size) {
  int32_t nbytes = 0, cbytes = 0, blocksize = 0;
  blosc2_cbuffer_sizes(src, &nbytes, &cbytes, &blocksize);
  if (nbytes <= 0) return NULL;

  uint8_t *dst = malloc((size_t)nbytes);
  if (!dst) return NULL;

  int ret = blosc2_decompress(src, (int32_t)src_size, dst, nbytes);
  if (ret < 0) { free(dst); return NULL; }
  if (out_size) *out_size = (size_t)ret;
  return dst;
}

// Decompress according to the compressor_id stored in the level meta.
// If no compressor is set, copies src as-is.
static uint8_t *decompress_buf(const zarr_level_meta *m, const uint8_t *src, size_t src_size,
                               size_t *out_size) {
  if (m->compressor_id[0] != '\0') {
    // currently only blosc/blosc2 supported at runtime; fall through for others
    if (strncmp(m->compressor_id, "blosc", 5) == 0 || strcmp(m->compressor_id, "zstd") == 0) {
      return blosc2_decompress_chunk(src, src_size, out_size);
    }
  }
  // raw / unrecognised — return a copy
  uint8_t *copy = malloc(src_size);
  if (!copy) return NULL;
  memcpy(copy, src, src_size);
  if (out_size) *out_size = src_size;
  return copy;
}

// ---------------------------------------------------------------------------
// Shard index helpers (Zarr v3 sharding_indexed)
// ---------------------------------------------------------------------------

// The shard index sits at the END of the shard file.
// Layout: N × 16 bytes, each entry = [offset:uint64_le, nbytes:uint64_le].
// N = product of (shard_shape[d] / chunk_shape[d]) for all d.

static uint64_t read_le64(const uint8_t *p) {
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (i * 8);
  return v;
}

// Compute number of inner chunks per shard dimension and total.
static size_t shard_nchunks(const zarr_level_meta *m, int64_t *nchunks_per_dim) {
  size_t total = 1;
  for (int d = 0; d < m->ndim; d++) {
    int64_t n = m->shard_shape[d] / m->chunk_shape[d];
    if (n < 1) n = 1;
    if (nchunks_per_dim) nchunks_per_dim[d] = n;
    total *= (size_t)n;
  }
  return total;
}

// Flat index of an inner chunk (row-major) from its local coords within the shard.
static size_t inner_chunk_flat_idx(const zarr_level_meta *m, const int64_t *inner_coords) {
  int64_t nchunks[8];
  shard_nchunks(m, nchunks);
  size_t idx = 0;
  for (int d = 0; d < m->ndim; d++)
    idx = idx * (size_t)nchunks[d] + (size_t)inner_coords[d];
  return idx;
}

// ---------------------------------------------------------------------------
// vol_read_shard
// ---------------------------------------------------------------------------

int vol_read_shard(const volume *v, int level, const int64_t *shard_coords,
                   uint8_t ***chunks_out, size_t **chunk_sizes_out, size_t *n_chunks_out) {
  if (!v || !shard_coords || !chunks_out || !chunk_sizes_out || !n_chunks_out) return -1;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m || !m->sharded) return -1;

  if (v->source == VOL_REMOTE) {
    LOG_WARN("remote shard read not yet implemented");
    return -1;
  }

  // build shard file path and read it
  char path[VOL_PATH_MAX];
  shard_path(v, level, shard_coords, m->ndim, path, sizeof(path));

  size_t shard_size = 0;
  uint8_t *shard_data = read_file_bytes(path, &shard_size);
  if (!shard_data) return -1;

  // compute expected index size
  int64_t nchunks_per_dim[8];
  size_t n_inner = shard_nchunks(m, nchunks_per_dim);
  size_t index_size = n_inner * SHARD_INDEX_ENTRY_BYTES;

  if (shard_size < index_size) {
    LOG_WARN("shard file too small: %zu < %zu (index)", shard_size, index_size);
    free(shard_data);
    return -1;
  }

  const uint8_t *index = shard_data + shard_size - index_size;

  // allocate output arrays
  uint8_t **chunks = calloc(n_inner, sizeof(uint8_t *));
  size_t  *sizes   = calloc(n_inner, sizeof(size_t));
  if (!chunks || !sizes) { free(chunks); free(sizes); free(shard_data); return -1; }

  int extracted = 0;
  for (size_t ci = 0; ci < n_inner; ci++) {
    uint64_t offset = read_le64(index + ci * SHARD_INDEX_ENTRY_BYTES);
    uint64_t nbytes = read_le64(index + ci * SHARD_INDEX_ENTRY_BYTES + 8);
    if (offset == SHARD_EMPTY_ENTRY || nbytes == 0) continue;
    if (offset + nbytes > shard_size - index_size) {
      LOG_WARN("shard chunk %zu out of bounds (offset=%llu, nbytes=%llu, shard_data_size=%zu)",
               ci, (unsigned long long)offset, (unsigned long long)nbytes,
               shard_size - index_size);
      continue;
    }
    size_t decomp_size = 0;
    uint8_t *decomp = decompress_buf(m, shard_data + offset, (size_t)nbytes, &decomp_size);
    if (!decomp) continue;
    chunks[ci] = decomp;
    sizes[ci]  = decomp_size;
    extracted++;
  }

  free(shard_data);
  *chunks_out      = chunks;
  *chunk_sizes_out = sizes;
  *n_chunks_out    = n_inner;
  return extracted;
}

// ---------------------------------------------------------------------------
// vol_read_chunk — handles both sharded and non-sharded, v2 and v3
// ---------------------------------------------------------------------------

uint8_t *vol_read_chunk(const volume *v, int level, const int64_t *chunk_coords, size_t *out_size) {
  if (!v || !chunk_coords) return NULL;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m) return NULL;

  if (v->source == VOL_REMOTE) {
    LOG_WARN("remote chunk read not yet implemented");
    return NULL;
  }

  if (m->sharded) {
    // compute which shard this chunk belongs to, and its local coords within the shard
    int64_t nchunks_per_dim[8];
    shard_nchunks(m, nchunks_per_dim);

    int64_t shard_coords[8], inner_coords[8];
    for (int d = 0; d < m->ndim; d++) {
      shard_coords[d] = chunk_coords[d] / nchunks_per_dim[d];
      inner_coords[d] = chunk_coords[d] % nchunks_per_dim[d];
    }

    char path[VOL_PATH_MAX];
    shard_path(v, level, shard_coords, m->ndim, path, sizeof(path));

    size_t shard_size = 0;
    uint8_t *shard_data = read_file_bytes(path, &shard_size);
    if (!shard_data) return NULL;

    size_t n_inner = shard_nchunks(m, NULL);
    size_t index_size = n_inner * SHARD_INDEX_ENTRY_BYTES;
    if (shard_size < index_size) { free(shard_data); return NULL; }

    size_t ci = inner_chunk_flat_idx(m, inner_coords);
    const uint8_t *index = shard_data + shard_size - index_size;
    uint64_t offset = read_le64(index + ci * SHARD_INDEX_ENTRY_BYTES);
    uint64_t nbytes = read_le64(index + ci * SHARD_INDEX_ENTRY_BYTES + 8);

    if (offset == SHARD_EMPTY_ENTRY || nbytes == 0) { free(shard_data); return NULL; }
    if (offset + nbytes > shard_size - index_size) { free(shard_data); return NULL; }

    size_t decomp_size = 0;
    uint8_t *decomp = decompress_buf(m, shard_data + offset, (size_t)nbytes, &decomp_size);
    free(shard_data);
    if (out_size) *out_size = decomp_size;
    return decomp;
  }

  // non-sharded: read the chunk file directly
  char path[VOL_PATH_MAX];
  chunk_path(v, level, chunk_coords, m->ndim, path, sizeof(path));

  size_t raw_size = 0;
  uint8_t *raw = read_file_bytes(path, &raw_size);
  if (!raw) return NULL;

  if (m->compressor_id[0] != '\0') {
    size_t decomp_size = 0;
    uint8_t *decomp = blosc2_decompress_chunk(raw, raw_size, &decomp_size);
    free(raw);
    if (!decomp) return NULL;
    if (out_size) *out_size = decomp_size;
    return decomp;
  }

  if (out_size) *out_size = raw_size;
  return raw;
}

// ---------------------------------------------------------------------------
// vol_sample — trilinear interpolation (3D only)
// ---------------------------------------------------------------------------

static float read_elem_float(const uint8_t *chunk_data, size_t flat_idx, int dtype) {
  switch (dtype) {
    case DTYPE_U8:  return (float)((const uint8_t *)chunk_data)[flat_idx];
    case DTYPE_U16: { uint16_t vv; memcpy(&vv, chunk_data + flat_idx * 2, 2); return (float)vv; }
    case DTYPE_F32: { float vv;    memcpy(&vv, chunk_data + flat_idx * 4, 4); return vv; }
    case DTYPE_F64: { double vv;   memcpy(&vv, chunk_data + flat_idx * 8, 8); return (float)vv; }
    default:        return 0.0f;
  }
}

static inline size_t chunk_offset_3d(int64_t lz, int64_t ly, int64_t lx,
                                     int64_t cy, int64_t cx) {
  return (size_t)(lz * cy * cx + ly * cx + lx);
}

static float sample_voxel(const volume *v, int level, int64_t iz, int64_t iy, int64_t ix) {
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m || m->ndim != 3) return 0.0f;
  if (iz < 0 || iz >= m->shape[0]) return 0.0f;
  if (iy < 0 || iy >= m->shape[1]) return 0.0f;
  if (ix < 0 || ix >= m->shape[2]) return 0.0f;

  int64_t coords[3] = { iz / m->chunk_shape[0], iy / m->chunk_shape[1], ix / m->chunk_shape[2] };
  int64_t local[3]  = { iz % m->chunk_shape[0], iy % m->chunk_shape[1], ix % m->chunk_shape[2] };

  size_t chunk_size = 0;
  uint8_t *data = vol_read_chunk(v, level, coords, &chunk_size);
  if (!data) return 0.0f;

  size_t off = chunk_offset_3d(local[0], local[1], local[2], m->chunk_shape[1], m->chunk_shape[2]);
  float val = read_elem_float(data, off, m->dtype);
  free(data);
  return val;
}

float vol_sample(const volume *v, int level, float fz, float fy, float fx) {
  if (!v) return 0.0f;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m || m->ndim != 3) return 0.0f;

  float z0f = fz - 0.5f, y0f = fy - 0.5f, x0f = fx - 0.5f;
  int64_t z0 = (int64_t)z0f, y0 = (int64_t)y0f, x0 = (int64_t)x0f;
  float tz = z0f - (float)z0, ty = y0f - (float)y0, tx = x0f - (float)x0;

  float c000 = sample_voxel(v, level, z0,     y0,     x0    );
  float c001 = sample_voxel(v, level, z0,     y0,     x0 + 1);
  float c010 = sample_voxel(v, level, z0,     y0 + 1, x0    );
  float c011 = sample_voxel(v, level, z0,     y0 + 1, x0 + 1);
  float c100 = sample_voxel(v, level, z0 + 1, y0,     x0    );
  float c101 = sample_voxel(v, level, z0 + 1, y0,     x0 + 1);
  float c110 = sample_voxel(v, level, z0 + 1, y0 + 1, x0    );
  float c111 = sample_voxel(v, level, z0 + 1, y0 + 1, x0 + 1);

  float c00 = c000 * (1.0f - tx) + c001 * tx;
  float c01 = c010 * (1.0f - tx) + c011 * tx;
  float c10 = c100 * (1.0f - tx) + c101 * tx;
  float c11 = c110 * (1.0f - tx) + c111 * tx;
  float c0  = c00  * (1.0f - ty) + c01  * ty;
  float c1  = c10  * (1.0f - ty) + c11  * ty;
  return c0 * (1.0f - tz) + c1 * tz;
}

// ---------------------------------------------------------------------------
// Info accessors
// ---------------------------------------------------------------------------

const char *vol_path(const volume *v)     { return v ? v->path : NULL; }
bool        vol_is_remote(const volume *v) { return v ? (v->source == VOL_REMOTE) : false; }
vol_source_t vol_source(const volume *v)  { return v ? v->source : VOL_LOCAL; }

// ---------------------------------------------------------------------------
// Shared helpers (used by vol_write.c via vol_internal.h)
// ---------------------------------------------------------------------------

size_t vol_shard_nchunks(const zarr_level_meta *m, int64_t *nchunks_per_dim) {
  return shard_nchunks(m, nchunks_per_dim);
}

size_t vol_inner_chunk_flat_idx(const zarr_level_meta *m, const int64_t *inner_coords) {
  return inner_chunk_flat_idx(m, inner_coords);
}

uint8_t *vol_read_file_bytes(const char *path, size_t *size) {
  return read_file_bytes(path, size);
}

const char *vol_dtype_to_v2str(int dtype) {
  switch (dtype) {
    case DTYPE_U8:  return "|u1";
    case DTYPE_U16: return "<u2";
    case DTYPE_F32: return "<f4";
    case DTYPE_F64: return "<f8";
    default:        return "|u1";
  }
}

const char *vol_dtype_to_v3str(int dtype) {
  switch (dtype) {
    case DTYPE_U8:  return "uint8";
    case DTYPE_U16: return "uint16";
    case DTYPE_F32: return "float32";
    case DTYPE_F64: return "float64";
    default:        return "uint8";
  }
}

size_t vol_dtype_elem_size(int dtype) {
  switch (dtype) {
    case DTYPE_U8:  return 1;
    case DTYPE_U16: return 2;
    case DTYPE_F32: return 4;
    case DTYPE_F64: return 8;
    default:        return 1;
  }
}

bool vol_mkdirp(const char *path) {
  char tmp[VOL_PATH_MAX];
  snprintf(tmp, sizeof(tmp), "%s", path);
  size_t len = strlen(tmp);
  if (len > 0 && tmp[len - 1] == '/') tmp[--len] = '\0';
  for (size_t i = 1; i <= len; i++) {
    if (tmp[i] == '/' || tmp[i] == '\0') {
      char save = tmp[i];
      tmp[i] = '\0';
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return false;
      tmp[i] = save;
    }
  }
  return true;
}
