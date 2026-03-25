#include "core/vol_internal.h"
#include "core/log.h"

#include <blosc2.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Compression helpers
// ---------------------------------------------------------------------------

// Compress src_size bytes from src using the compressor specified in meta.
// Returns malloc'd compressed buffer; caller must free. Sets *out_size.
// Falls back to a plain copy if no compressor is configured.
static uint8_t *compress_chunk(const zarr_level_meta *m, const void *src, size_t src_size,
                                size_t *out_size) {
  if (m->compressor_id[0] != '\0' &&
      (strncmp(m->compressor_id, "blosc", 5) == 0 || strcmp(m->compressor_id, "zstd") == 0)) {
    // blosc2_compress needs a dest buffer; over-allocate to be safe
    size_t dest_cap = src_size + BLOSC2_MAX_OVERHEAD;
    uint8_t *dst = malloc(dest_cap);
    if (!dst) return NULL;

    int clevel   = m->compressor_clevel > 0 ? m->compressor_clevel : 5;
    int shuffle  = m->compressor_shuffle;
    int typesize = 1;  // byte shuffle; caller provides raw bytes

    int ret = blosc2_compress(clevel, shuffle, typesize, src, (int32_t)src_size,
                              dst, (int32_t)dest_cap);
    if (ret <= 0) {
      free(dst);
      return NULL;
    }
    // shrink to actual size
    uint8_t *shrunk = realloc(dst, (size_t)ret);
    *out_size = (size_t)ret;
    return shrunk ? shrunk : dst;
  }

  // no compression — copy as-is
  uint8_t *copy = malloc(src_size);
  if (!copy) return NULL;
  memcpy(copy, src, src_size);
  *out_size = src_size;
  return copy;
}

// ---------------------------------------------------------------------------
// Metadata serialisation
// ---------------------------------------------------------------------------

// Write v2 .zarray JSON for the given level meta.
static bool write_zarray(const char *dir, const zarr_level_meta *m) {
  char path[VOL_PATH_MAX];
  snprintf(path, sizeof(path), "%s/.zarray", dir);

  FILE *f = fopen(path, "w");
  if (!f) return false;

  fprintf(f, "{\n");
  fprintf(f, "  \"zarr_format\": 2,\n");

  // shape
  fprintf(f, "  \"shape\": [");
  for (int d = 0; d < m->ndim; d++) fprintf(f, "%s%lld", d ? ", " : "", (long long)m->shape[d]);
  fprintf(f, "],\n");

  // chunks
  fprintf(f, "  \"chunks\": [");
  for (int d = 0; d < m->ndim; d++) fprintf(f, "%s%lld", d ? ", " : "", (long long)m->chunk_shape[d]);
  fprintf(f, "],\n");

  fprintf(f, "  \"dtype\": \"%s\",\n", vol_dtype_to_v2str(m->dtype));
  fprintf(f, "  \"order\": \"C\",\n");
  fprintf(f, "  \"fill_value\": 0,\n");
  fprintf(f, "  \"filters\": null,\n");

  if (m->compressor_id[0] != '\0') {
    fprintf(f, "  \"compressor\": {\n");
    fprintf(f, "    \"id\": \"%s\",\n", m->compressor_id);
    if (m->compressor_cname[0])
      fprintf(f, "    \"cname\": \"%s\",\n", m->compressor_cname);
    fprintf(f, "    \"clevel\": %d,\n", m->compressor_clevel);
    fprintf(f, "    \"shuffle\": %d\n", m->compressor_shuffle);
    fprintf(f, "  }\n");
  } else {
    fprintf(f, "  \"compressor\": null\n");
  }

  fprintf(f, "}\n");
  fclose(f);
  return true;
}

// Write v3 zarr.json for the given level meta.
static bool write_zarr_json(const char *dir, const zarr_level_meta *m) {
  char path[VOL_PATH_MAX];
  snprintf(path, sizeof(path), "%s/zarr.json", dir);

  FILE *f = fopen(path, "w");
  if (!f) return false;

  fprintf(f, "{\n");
  fprintf(f, "  \"zarr_format\": 3,\n");
  fprintf(f, "  \"node_type\": \"array\",\n");

  // shape
  fprintf(f, "  \"shape\": [");
  for (int d = 0; d < m->ndim; d++) fprintf(f, "%s%lld", d ? ", " : "", (long long)m->shape[d]);
  fprintf(f, "],\n");

  fprintf(f, "  \"data_type\": \"%s\",\n", vol_dtype_to_v3str(m->dtype));
  fprintf(f, "  \"chunk_key_encoding\": {\"name\": \"default\", \"configuration\": {\"separator\": \"/\"}},\n");
  fprintf(f, "  \"fill_value\": 0,\n");

  // chunk_grid uses the outer shape (shard_shape if sharded, else chunk_shape)
  const int64_t *outer = m->sharded ? m->shard_shape : m->chunk_shape;
  fprintf(f, "  \"chunk_grid\": {\n");
  fprintf(f, "    \"name\": \"regular\",\n");
  fprintf(f, "    \"configuration\": {\"chunk_shape\": [");
  for (int d = 0; d < m->ndim; d++) fprintf(f, "%s%lld", d ? ", " : "", (long long)outer[d]);
  fprintf(f, "]}\n");
  fprintf(f, "  },\n");

  // codecs
  fprintf(f, "  \"codecs\": [");
  if (m->sharded) {
    fprintf(f, "{\n");
    fprintf(f, "    \"name\": \"sharding_indexed\",\n");
    fprintf(f, "    \"configuration\": {\n");
    fprintf(f, "      \"chunk_shape\": [");
    for (int d = 0; d < m->ndim; d++) fprintf(f, "%s%lld", d ? ", " : "", (long long)m->chunk_shape[d]);
    fprintf(f, "],\n");
    fprintf(f, "      \"codecs\": [{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}");
    if (m->compressor_id[0]) {
      fprintf(f, ", {\"name\": \"%s\", \"configuration\": {\"cname\": \"%s\", \"clevel\": %d, \"shuffle\": %d}}",
              m->compressor_id, m->compressor_cname[0] ? m->compressor_cname : "blosclz",
              m->compressor_clevel, m->compressor_shuffle);
    }
    fprintf(f, "],\n");
    fprintf(f, "      \"index_codecs\": [{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}]\n");
    fprintf(f, "    }\n");
    fprintf(f, "  }");
  } else {
    fprintf(f, "{\"name\": \"bytes\", \"configuration\": {\"endian\": \"little\"}}");
    if (m->compressor_id[0]) {
      fprintf(f, ", {\"name\": \"%s\", \"configuration\": {\"cname\": \"%s\", \"clevel\": %d, \"shuffle\": %d}}",
              m->compressor_id, m->compressor_cname[0] ? m->compressor_cname : "blosclz",
              m->compressor_clevel, m->compressor_shuffle);
    }
  }
  fprintf(f, "]\n");
  fprintf(f, "}\n");
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Chunk / shard path helpers (write side mirrors read side)
// ---------------------------------------------------------------------------

static void write_chunk_path(const volume *v, int level, const int64_t *coords, int ndim,
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

static void write_shard_path(const volume *v, int level, const int64_t *shard_coords, int ndim,
                              char *buf, size_t bufsz) {
  int used = snprintf(buf, bufsz, "%s/%d/c", v->path, level);
  for (int d = 0; d < ndim && used < (int)bufsz - 1; d++)
    used += snprintf(buf + used, bufsz - (size_t)used, "/%lld", (long long)shard_coords[d]);
}

// Ensure all parent directories of path exist.
static bool mkdirs_for_file(const char *filepath) {
  char tmp[VOL_PATH_MAX];
  snprintf(tmp, sizeof(tmp), "%s", filepath);
  // walk backwards to find the last '/'
  char *last_slash = strrchr(tmp, '/');
  if (!last_slash) return true;
  *last_slash = '\0';
  return vol_mkdirp(tmp);
}

// ---------------------------------------------------------------------------
// vol_create
// ---------------------------------------------------------------------------

volume *vol_create(const char *path, vol_create_params p) {
  if (!path || p.ndim < 1 || p.ndim > 8) return NULL;
  if (p.zarr_version != 2 && p.zarr_version != 3) return NULL;

  if (!vol_mkdirp(path)) {
    LOG_WARN("vol_create: cannot create directory %s: %s", path, strerror(errno));
    return NULL;
  }

  volume *v = calloc(1, sizeof(volume));
  if (!v) return NULL;
  strncpy(v->path, path, VOL_PATH_MAX - 1);
  v->source     = VOL_LOCAL;
  v->num_levels = 1;

  // fill level 0 meta from params
  zarr_level_meta *m = &v->levels[0];
  m->zarr_version = p.zarr_version;
  m->ndim         = p.ndim;
  memcpy(m->shape,       p.shape,       sizeof(int64_t) * (size_t)p.ndim);
  memcpy(m->chunk_shape, p.chunk_shape, sizeof(int64_t) * (size_t)p.ndim);
  m->dtype      = (int)p.dtype;
  m->order      = 'C';
  m->chunk_sep  = (p.zarr_version == 2) ? '.' : '/';  // zarr v2 default: '.', v3: '/'

  if (p.compressor && p.compressor[0] != '\0') {
    strncpy(m->compressor_id, p.compressor, sizeof(m->compressor_id) - 1);
    if (strcmp(p.compressor, "blosc") == 0)
      strncpy(m->compressor_cname, "lz4", sizeof(m->compressor_cname) - 1);
    m->compressor_clevel  = p.clevel > 0 ? p.clevel : 5;
    m->compressor_shuffle = 1;  // byte shuffle default
  }

  if (p.sharded && p.zarr_version == 3) {
    m->sharded = true;
    memcpy(m->shard_shape, p.shard_shape, sizeof(int64_t) * (size_t)p.ndim);
  }

  // create level directory and write metadata
  char level_dir[VOL_PATH_MAX];
  snprintf(level_dir, sizeof(level_dir), "%s/0", path);
  if (!vol_mkdirp(level_dir)) { free(v); return NULL; }

  bool meta_ok = (p.zarr_version == 3) ? write_zarr_json(level_dir, m)
                                        : write_zarray(level_dir, m);
  if (!meta_ok) {
    LOG_WARN("vol_create: failed to write metadata in %s", level_dir);
    free(v);
    return NULL;
  }

  LOG_INFO("created volume %s (v%d, %d dims)", path, p.zarr_version, p.ndim);
  return v;
}

// ---------------------------------------------------------------------------
// vol_write_chunk
// ---------------------------------------------------------------------------

bool vol_write_chunk(volume *v, int level, const int64_t *chunk_coords,
                     const void *data, size_t size) {
  if (!v || !chunk_coords || !data || size == 0) return false;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m) return false;

  size_t comp_size = 0;
  uint8_t *comp = compress_chunk(m, data, size, &comp_size);
  if (!comp) return false;

  char path[VOL_PATH_MAX];
  write_chunk_path(v, level, chunk_coords, m->ndim, path, sizeof(path));

  if (!mkdirs_for_file(path)) { free(comp); return false; }

  FILE *f = fopen(path, "wb");
  if (!f) { free(comp); return false; }

  bool ok = fwrite(comp, 1, comp_size, f) == comp_size;
  fclose(f);
  free(comp);
  return ok;
}

// ---------------------------------------------------------------------------
// vol_write_shard
// ---------------------------------------------------------------------------

bool vol_write_shard(volume *v, int level, const int64_t *shard_coords,
                     const void **chunk_data, const size_t *chunk_sizes, int num_chunks) {
  if (!v || !shard_coords || !chunk_data || !chunk_sizes || num_chunks <= 0) return false;
  const zarr_level_meta *m = vol_level_meta(v, level);
  if (!m || !m->sharded) return false;

  size_t n_inner = vol_shard_nchunks(m, NULL);
  size_t index_size = n_inner * SHARD_INDEX_ENTRY_BYTES;

  // first pass: compress all chunks, track their sizes
  uint8_t **comp_bufs  = calloc(n_inner, sizeof(uint8_t *));
  size_t   *comp_sizes = calloc(n_inner, sizeof(size_t));
  if (!comp_bufs || !comp_sizes) { free(comp_bufs); free(comp_sizes); return false; }

  size_t data_total = 0;
  for (int i = 0; i < num_chunks && (size_t)i < n_inner; i++) {
    if (!chunk_data[i] || chunk_sizes[i] == 0) continue;
    comp_bufs[i] = compress_chunk(m, chunk_data[i], chunk_sizes[i], &comp_sizes[i]);
    if (comp_bufs[i]) data_total += comp_sizes[i];
  }

  // build shard: data region followed by index
  uint8_t *shard = malloc(data_total + index_size);
  if (!shard) {
    for (size_t i = 0; i < n_inner; i++) free(comp_bufs[i]);
    free(comp_bufs); free(comp_sizes);
    return false;
  }

  // write index region (initialise all entries as empty)
  uint8_t *idx = shard + data_total;
  for (size_t i = 0; i < n_inner; i++) {
    vol_write_le64(idx + i * SHARD_INDEX_ENTRY_BYTES,     SHARD_EMPTY_ENTRY);
    vol_write_le64(idx + i * SHARD_INDEX_ENTRY_BYTES + 8, SHARD_EMPTY_ENTRY);
  }

  // copy compressed chunks into data region and fill index
  uint64_t offset = 0;
  for (size_t i = 0; i < n_inner; i++) {
    if (!comp_bufs[i]) continue;
    memcpy(shard + offset, comp_bufs[i], comp_sizes[i]);
    vol_write_le64(idx + i * SHARD_INDEX_ENTRY_BYTES,     offset);
    vol_write_le64(idx + i * SHARD_INDEX_ENTRY_BYTES + 8, (uint64_t)comp_sizes[i]);
    offset += (uint64_t)comp_sizes[i];
    free(comp_bufs[i]);
    comp_bufs[i] = NULL;
  }
  free(comp_bufs);
  free(comp_sizes);

  // write shard file
  char path[VOL_PATH_MAX];
  write_shard_path(v, level, shard_coords, m->ndim, path, sizeof(path));
  if (!mkdirs_for_file(path)) { free(shard); return false; }

  FILE *f = fopen(path, "wb");
  if (!f) { free(shard); return false; }
  bool ok = fwrite(shard, 1, data_total + index_size, f) == (data_total + index_size);
  fclose(f);
  free(shard);
  return ok;
}

// ---------------------------------------------------------------------------
// vol_build_pyramid
// ---------------------------------------------------------------------------

// Mean-pool a raw uint8 3D array by 2x in each dimension.
// src_shape = {D, H, W}, dst_shape = {D/2, H/2, W/2} (floor).
static void downsample_u8_3d(const uint8_t *src, int64_t sd, int64_t sh, int64_t sw,
                              uint8_t *dst) {
  int64_t dd = sd / 2, dh = sh / 2, dw = sw / 2;
  for (int64_t z = 0; z < dd; z++) {
    for (int64_t y = 0; y < dh; y++) {
      for (int64_t x = 0; x < dw; x++) {
        // 2x2x2 mean
        uint32_t acc = 0;
        for (int64_t dz = 0; dz < 2; dz++)
          for (int64_t dy = 0; dy < 2; dy++)
            for (int64_t dx = 0; dx < 2; dx++)
              acc += src[((z*2+dz) * sh + (y*2+dy)) * sw + (x*2+dx)];
        dst[(z * dh + y) * dw + x] = (uint8_t)(acc / 8);
      }
    }
  }
}

static void downsample_u16_3d(const uint16_t *src, int64_t sd, int64_t sh, int64_t sw,
                               uint16_t *dst) {
  int64_t dd = sd / 2, dh = sh / 2, dw = sw / 2;
  for (int64_t z = 0; z < dd; z++) {
    for (int64_t y = 0; y < dh; y++) {
      for (int64_t x = 0; x < dw; x++) {
        uint32_t acc = 0;
        for (int64_t dz = 0; dz < 2; dz++)
          for (int64_t dy = 0; dy < 2; dy++)
            for (int64_t dx = 0; dx < 2; dx++)
              acc += src[((z*2+dz) * sh + (y*2+dy)) * sw + (x*2+dx)];
        dst[(z * dh + y) * dw + x] = (uint16_t)(acc / 8);
      }
    }
  }
}

bool vol_build_pyramid(volume *v, int max_levels) {
  if (!v || max_levels < 1) return false;
  const zarr_level_meta *m0 = vol_level_meta(v, 0);
  if (!m0 || m0->ndim != 3) return false;  // only 3D supported

  int dtype = m0->dtype;
  if (dtype != DTYPE_U8 && dtype != DTYPE_U16) {
    LOG_WARN("vol_build_pyramid: only uint8/uint16 supported, got dtype=%d", dtype);
    return false;
  }

  size_t elem = vol_dtype_elem_size(dtype);

  for (int lvl = 1; lvl <= max_levels; lvl++) {
    const zarr_level_meta *prev = vol_level_meta(v, lvl - 1);
    if (!prev) break;

    int64_t sd = prev->shape[0], sh = prev->shape[1], sw = prev->shape[2];
    if (sd < 2 || sh < 2 || sw < 2) break;

    int64_t dd = sd / 2, dh = sh / 2, dw = sw / 2;

    // allocate full level buffers
    size_t src_elems = (size_t)(sd * sh * sw);
    size_t dst_elems = (size_t)(dd * dh * dw);
    uint8_t *src_vol = calloc(src_elems, elem);
    uint8_t *dst_vol = calloc(dst_elems, elem);
    if (!src_vol || !dst_vol) { free(src_vol); free(dst_vol); return false; }

    // read all chunks from level lvl-1 into src_vol
    int64_t nchunks[3] = {
      (sd + prev->chunk_shape[0] - 1) / prev->chunk_shape[0],
      (sh + prev->chunk_shape[1] - 1) / prev->chunk_shape[1],
      (sw + prev->chunk_shape[2] - 1) / prev->chunk_shape[2],
    };
    for (int64_t cz = 0; cz < nchunks[0]; cz++) {
      for (int64_t cy = 0; cy < nchunks[1]; cy++) {
        for (int64_t cx = 0; cx < nchunks[2]; cx++) {
          int64_t cc[3] = {cz, cy, cx};
          size_t chunk_sz = 0;
          uint8_t *chunk = vol_read_chunk(v, lvl - 1, cc, &chunk_sz);
          if (!chunk) continue;

          // scatter into src_vol
          int64_t oz = cz * prev->chunk_shape[0];
          int64_t oy = cy * prev->chunk_shape[1];
          int64_t ox = cx * prev->chunk_shape[2];
          int64_t ez = oz + prev->chunk_shape[0]; if (ez > sd) ez = sd;
          int64_t ey = oy + prev->chunk_shape[1]; if (ey > sh) ey = sh;
          int64_t ex = ox + prev->chunk_shape[2]; if (ex > sw) ex = sw;

          size_t ci = 0;
          for (int64_t z = oz; z < ez; z++)
            for (int64_t y = oy; y < ey; y++)
              for (int64_t x = ox; x < ex; x++) {
                size_t vi = (size_t)((z * sh + y) * sw + x);
                memcpy(src_vol + vi * elem, chunk + ci * elem, elem);
                ci++;
              }
          free(chunk);
        }
      }
    }

    // downsample
    if (dtype == DTYPE_U8)
      downsample_u8_3d(src_vol, sd, sh, sw, dst_vol);
    else
      downsample_u16_3d((uint16_t *)src_vol, sd, sh, sw, (uint16_t *)dst_vol);
    free(src_vol);

    // add new level to volume struct
    if (v->num_levels >= VOL_MAX_LEVELS) { free(dst_vol); break; }
    zarr_level_meta *nm = &v->levels[v->num_levels];
    *nm = *prev;
    nm->shape[0] = dd; nm->shape[1] = dh; nm->shape[2] = dw;

    // create level directory and write metadata
    char level_dir[VOL_PATH_MAX];
    snprintf(level_dir, sizeof(level_dir), "%s/%d", v->path, v->num_levels);
    if (!vol_mkdirp(level_dir)) { free(dst_vol); break; }
    bool meta_ok = (nm->zarr_version == 3) ? write_zarr_json(level_dir, nm)
                                           : write_zarray(level_dir, nm);
    if (!meta_ok) { free(dst_vol); break; }

    v->num_levels++;

    // write all chunks for new level
    int64_t nc[3] = {
      (dd + nm->chunk_shape[0] - 1) / nm->chunk_shape[0],
      (dh + nm->chunk_shape[1] - 1) / nm->chunk_shape[1],
      (dw + nm->chunk_shape[2] - 1) / nm->chunk_shape[2],
    };
    int cur_lvl = v->num_levels - 1;
    size_t chunk_elems = (size_t)(nm->chunk_shape[0] * nm->chunk_shape[1] * nm->chunk_shape[2]);
    uint8_t *chunk_buf = malloc(chunk_elems * elem);
    if (!chunk_buf) { free(dst_vol); break; }

    for (int64_t cz = 0; cz < nc[0]; cz++) {
      for (int64_t cy = 0; cy < nc[1]; cy++) {
        for (int64_t cx = 0; cx < nc[2]; cx++) {
          // gather from dst_vol into chunk_buf
          int64_t oz = cz * nm->chunk_shape[0];
          int64_t oy = cy * nm->chunk_shape[1];
          int64_t ox = cx * nm->chunk_shape[2];
          int64_t ez = oz + nm->chunk_shape[0]; if (ez > dd) ez = dd;
          int64_t ey = oy + nm->chunk_shape[1]; if (ey > dh) ey = dh;
          int64_t ex = ox + nm->chunk_shape[2]; if (ex > dw) ex = dw;

          memset(chunk_buf, 0, chunk_elems * elem);
          size_t ci = 0;
          for (int64_t z = oz; z < ez; z++)
            for (int64_t y = oy; y < ey; y++)
              for (int64_t x = ox; x < ex; x++) {
                size_t vi = (size_t)((z * dh + y) * dw + x);
                memcpy(chunk_buf + ci * elem, dst_vol + vi * elem, elem);
                ci++;
              }

          int64_t cc[3] = {cz, cy, cx};
          vol_write_chunk(v, cur_lvl, cc, chunk_buf, chunk_elems * elem);
        }
      }
    }
    free(chunk_buf);
    free(dst_vol);
  }

  return true;
}

// ---------------------------------------------------------------------------
// vol_finalize — write OME-Zarr .zattrs multiscales
// ---------------------------------------------------------------------------

bool vol_finalize(volume *v) {
  if (!v) return false;

  char path[VOL_PATH_MAX];
  snprintf(path, sizeof(path), "%s/.zattrs", v->path);

  FILE *f = fopen(path, "w");
  if (!f) return false;

  fprintf(f, "{\n");
  fprintf(f, "  \"multiscales\": [{\n");
  fprintf(f, "    \"version\": \"0.4\",\n");
  fprintf(f, "    \"name\": \"\",\n");
  fprintf(f, "    \"axes\": [\n");
  // v0.4 OME-Zarr: axes are named; for 3D assume z/y/x with type "space"
  const zarr_level_meta *m0 = vol_level_meta(v, 0);
  if (m0 && m0->ndim == 3) {
    fprintf(f, "      {\"name\": \"z\", \"type\": \"space\", \"unit\": \"micrometer\"},\n");
    fprintf(f, "      {\"name\": \"y\", \"type\": \"space\", \"unit\": \"micrometer\"},\n");
    fprintf(f, "      {\"name\": \"x\", \"type\": \"space\", \"unit\": \"micrometer\"}\n");
  } else {
    for (int d = 0; m0 && d < m0->ndim; d++) {
      fprintf(f, "      {\"name\": \"d%d\", \"type\": \"space\", \"unit\": \"micrometer\"}%s\n",
              d, (d < m0->ndim - 1) ? "," : "");
    }
  }
  fprintf(f, "    ],\n");
  fprintf(f, "    \"datasets\": [\n");
  for (int lvl = 0; lvl < v->num_levels; lvl++) {
    double scale = (double)(1 << lvl);  // each level is 2^lvl coarser
    fprintf(f, "      {\n");
    fprintf(f, "        \"path\": \"%d\",\n", lvl);
    fprintf(f, "        \"coordinateTransformations\": [{\n");
    fprintf(f, "          \"type\": \"scale\",\n");
    fprintf(f, "          \"scale\": [");
    if (m0 && m0->ndim == 3) fprintf(f, "%.6g, %.6g, %.6g", scale, scale, scale);
    else if (m0) { for (int d = 0; d < m0->ndim; d++) fprintf(f, "%s%.6g", d ? ", " : "", scale); }
    fprintf(f, "]\n        }]\n");
    fprintf(f, "      }%s\n", (lvl < v->num_levels - 1) ? "," : "");
  }
  fprintf(f, "    ]\n");
  fprintf(f, "  }]\n");
  fprintf(f, "}\n");

  fclose(f);
  return true;
}
