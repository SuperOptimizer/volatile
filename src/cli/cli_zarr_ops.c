#define _POSIX_C_SOURCE 200809L

#include "cli_zarr_ops.h"
#include "cli_progress.h"
#include "core/vol.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

// Count total chunks across all dimensions for a level.
static int64_t count_chunks(const zarr_level_meta *m) {
  int64_t total = 1;
  for (int d = 0; d < m->ndim; d++)
    total *= (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
  return total;
}

// Advance an ndim chunk coordinate by 1, returns false when wrapped.
static bool next_coord(int64_t *coords, const int64_t *nchunks, int ndim) {
  for (int d = ndim - 1; d >= 0; d--) {
    if (++coords[d] < nchunks[d]) return true;
    coords[d] = 0;
  }
  return false;
}

// Create output volume mirroring src meta, optionally overriding shape.
static volume *create_output(const char *path, const zarr_level_meta *m,
                             const int64_t *override_shape) {
  vol_create_params p = {
    .zarr_version = m->zarr_version > 0 ? m->zarr_version : 2,
    .ndim         = m->ndim,
    .dtype        = (dtype_t)m->dtype,
    .compressor   = m->compressor_id[0] ? m->compressor_id : NULL,
    .clevel       = m->compressor_clevel > 0 ? m->compressor_clevel : 5,
    .sharded      = false,
  };
  for (int d = 0; d < m->ndim; d++) {
    p.shape[d]       = override_shape ? override_shape[d] : m->shape[d];
    p.chunk_shape[d] = m->chunk_shape[d];
  }
  return vol_create(path, p);
}

// ---------------------------------------------------------------------------
// downsample: 2x mean pooling at level 0 → writes level 1 of a new volume
// ---------------------------------------------------------------------------

int cmd_downsample(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile downsample <zarr> --output <out> [--factor 2]");
    return argc < 1 ? 1 : 0;
  }

  const char *input  = argv[0];
  const char *output = NULL;
  // factor is fixed at 2 for now (mean pooling only)

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) output = argv[++i];
    else if (strcmp(argv[i], "--factor") == 0) ++i; // consume, value ignored
  }

  if (!output) { fputs("error: --output required\n", stderr); return 1; }

  volume *src = vol_open(input);
  if (!src) { fprintf(stderr, "error: cannot open %s\n", input); return 1; }

  const zarr_level_meta *m = vol_level_meta(src, 0);
  if (!m || m->ndim != 3 || (m->dtype != DTYPE_U8 && m->dtype != DTYPE_U16)) {
    fputs("error: downsample requires a 3D uint8 or uint16 volume\n", stderr);
    vol_free(src); return 1;
  }

  // output shape: floor divide each dim by 2
  int64_t out_shape[8];
  for (int d = 0; d < m->ndim; d++) out_shape[d] = m->shape[d] / 2;

  volume *dst = create_output(output, m, out_shape);
  if (!dst) {
    fprintf(stderr, "error: cannot create output %s\n", output);
    vol_free(src); return 1;
  }

  // iterate output chunks, read 2x2x2 neighbourhood from src, mean-pool
  int64_t nchunks[3], total = 1;
  for (int d = 0; d < 3; d++) {
    nchunks[d] = (out_shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
    total *= nchunks[d];
  }

  size_t elem      = (m->dtype == DTYPE_U8) ? 1 : 2;
  size_t chunk_vol = (size_t)(m->chunk_shape[0] * m->chunk_shape[1] * m->chunk_shape[2]);
  uint8_t *buf = malloc(chunk_vol * elem);
  if (!buf) { vol_free(src); vol_free(dst); return 1; }

  int64_t coords[3] = {0};
  int64_t done = 0;
  do {
    cli_progress((int)done, (int)total, "downsample");

    // for each voxel in this output chunk, read the 2x src voxels
    int64_t oz = coords[0] * m->chunk_shape[0];
    int64_t oy = coords[1] * m->chunk_shape[1];
    int64_t ox = coords[2] * m->chunk_shape[2];
    int64_t ez = oz + m->chunk_shape[0]; if (ez > out_shape[0]) ez = out_shape[0];
    int64_t ey = oy + m->chunk_shape[1]; if (ey > out_shape[1]) ey = out_shape[1];
    int64_t ex = ox + m->chunk_shape[2]; if (ex > out_shape[2]) ex = out_shape[2];

    size_t ci = 0;
    for (int64_t z = oz; z < ez; z++) {
      for (int64_t y = oy; y < ey; y++) {
        for (int64_t x = ox; x < ex; x++) {
          // sample 2x2x2 neighbourhood at src coords 2z..2z+1, etc.
          uint32_t acc = 0;
          for (int dz = 0; dz < 2; dz++) for (int dy = 0; dy < 2; dy++) for (int dx = 0; dx < 2; dx++) {
            float v = vol_sample(src, 0, (float)(2*z+dz), (float)(2*y+dy), (float)(2*x+dx));
            acc += (uint32_t)v;
          }
          uint32_t mean = acc / 8;
          if (elem == 1) buf[ci] = (uint8_t)mean;
          else           ((uint16_t *)buf)[ci] = (uint16_t)mean;
          ci++;
        }
      }
    }

    vol_write_chunk(dst, 0, coords, buf, ci * elem);
    done++;
  } while (next_coord(coords, nchunks, 3));

  cli_progress((int)total, (int)total, "downsample");
  free(buf);
  vol_finalize(dst);
  vol_free(dst);
  vol_free(src);
  printf("written: %s\n", output);
  return 0;
}

// ---------------------------------------------------------------------------
// threshold: values outside [low,high] -> 0
// ---------------------------------------------------------------------------

int cmd_threshold(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile threshold <zarr> --output <out> [--low N] [--high N]");
    return argc < 1 ? 1 : 0;
  }

  const char *input  = argv[0];
  const char *output = NULL;
  uint32_t low = 0, high = 65535;

  for (int i = 1; i < argc - 1; i++) {
    if      (strcmp(argv[i], "--output") == 0) output = argv[++i];
    else if (strcmp(argv[i], "--low")    == 0) low    = (uint32_t)atoi(argv[++i]);
    else if (strcmp(argv[i], "--high")   == 0) high   = (uint32_t)atoi(argv[++i]);
  }

  if (!output) { fputs("error: --output required\n", stderr); return 1; }

  volume *src = vol_open(input);
  if (!src) { fprintf(stderr, "error: cannot open %s\n", input); return 1; }

  const zarr_level_meta *m = vol_level_meta(src, 0);
  if (!m) { vol_free(src); return 1; }

  volume *dst = create_output(output, m, NULL);
  if (!dst) { vol_free(src); return 1; }

  int64_t nchunks[8];
  int64_t total = count_chunks(m);
  for (int d = 0; d < m->ndim; d++)
    nchunks[d] = (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];

  size_t elem = (m->dtype == DTYPE_U8) ? 1 : (m->dtype == DTYPE_U16) ? 2 : 4;
  int64_t coords[8] = {0};
  int64_t done = 0;

  do {
    cli_progress((int)done, (int)total, "threshold");
    size_t sz = 0;
    uint8_t *chunk = vol_read_chunk(src, 0, coords, &sz);
    if (chunk) {
      size_t n = sz / elem;
      for (size_t i = 0; i < n; i++) {
        uint32_t v = 0;
        if      (elem == 1) v = chunk[i];
        else if (elem == 2) v = ((uint16_t *)chunk)[i];
        else                v = (uint32_t)((uint32_t *)chunk)[i];
        if (v < low || v > high) {
          if      (elem == 1) chunk[i] = 0;
          else if (elem == 2) ((uint16_t *)chunk)[i] = 0;
          else                ((uint32_t *)chunk)[i] = 0;
        }
      }
      vol_write_chunk(dst, 0, coords, chunk, sz);
      free(chunk);
    }
    done++;
  } while (next_coord(coords, nchunks, m->ndim));

  cli_progress((int)total, (int)total, "threshold");
  vol_finalize(dst);
  vol_free(dst);
  vol_free(src);
  printf("written: %s\n", output);
  return 0;
}

// ---------------------------------------------------------------------------
// merge: element-wise combination of two volumes
// ---------------------------------------------------------------------------

typedef enum { MERGE_MAX, MERGE_ADD, MERGE_MASK } merge_op_t;

int cmd_merge(int argc, char **argv) {
  if (argc < 2 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile merge <zarr1> <zarr2> --output <out> [--op max|add|mask]");
    return argc < 2 ? 1 : 0;
  }

  const char *in1    = argv[0];
  const char *in2    = argv[1];
  const char *output = NULL;
  merge_op_t  op     = MERGE_MAX;

  for (int i = 2; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--op") == 0) {
      ++i;
      if      (strcmp(argv[i], "add")  == 0) op = MERGE_ADD;
      else if (strcmp(argv[i], "mask") == 0) op = MERGE_MASK;
      else                                   op = MERGE_MAX;
    }
  }

  if (!output) { fputs("error: --output required\n", stderr); return 1; }

  volume *v1 = vol_open(in1);
  volume *v2 = vol_open(in2);
  if (!v1 || !v2) {
    fprintf(stderr, "error: cannot open input volumes\n");
    vol_free(v1); vol_free(v2); return 1;
  }

  const zarr_level_meta *m = vol_level_meta(v1, 0);
  if (!m) { vol_free(v1); vol_free(v2); return 1; }

  volume *dst = create_output(output, m, NULL);
  if (!dst) { vol_free(v1); vol_free(v2); return 1; }

  int64_t nchunks[8];
  int64_t total = count_chunks(m);
  for (int d = 0; d < m->ndim; d++)
    nchunks[d] = (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];

  size_t elem = (m->dtype == DTYPE_U8) ? 1 : (m->dtype == DTYPE_U16) ? 2 : 4;
  int64_t coords[8] = {0};
  int64_t done = 0;

  do {
    cli_progress((int)done, (int)total, "merge");
    size_t sz1 = 0, sz2 = 0;
    uint8_t *c1 = vol_read_chunk(v1, 0, coords, &sz1);
    uint8_t *c2 = vol_read_chunk(v2, 0, coords, &sz2);

    if (c1 && c2 && sz1 == sz2) {
      size_t n = sz1 / elem;
      for (size_t i = 0; i < n; i++) {
        if (elem == 1) {
          uint8_t a = c1[i], b = c2[i];
          switch (op) {
            case MERGE_MAX:  c1[i] = a > b ? a : b; break;
            case MERGE_ADD:  c1[i] = (uint8_t)((a + b > 255) ? 255 : a + b); break;
            case MERGE_MASK: c1[i] = b ? a : 0; break;
          }
        } else if (elem == 2) {
          uint16_t a = ((uint16_t *)c1)[i], b = ((uint16_t *)c2)[i];
          switch (op) {
            case MERGE_MAX:  ((uint16_t *)c1)[i] = a > b ? a : b; break;
            case MERGE_ADD:  ((uint16_t *)c1)[i] = (uint16_t)((a + b > 65535) ? 65535 : a + b); break;
            case MERGE_MASK: ((uint16_t *)c1)[i] = b ? a : 0; break;
          }
        }
      }
      vol_write_chunk(dst, 0, coords, c1, sz1);
    } else if (c1) {
      vol_write_chunk(dst, 0, coords, c1, sz1);
    }

    free(c1); free(c2);
    done++;
  } while (next_coord(coords, nchunks, m->ndim));

  cli_progress((int)total, (int)total, "merge");
  vol_finalize(dst);
  vol_free(dst);
  vol_free(v1);
  vol_free(v2);
  printf("written: %s\n", output);
  return 0;
}

// ---------------------------------------------------------------------------
// extract: copy a bbox sub-region into a new volume
// ---------------------------------------------------------------------------

int cmd_extract(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile extract <zarr> --bbox z0,y0,x0,z1,y1,x1 --output <out>");
    return argc < 1 ? 1 : 0;
  }

  const char *input  = argv[0];
  const char *output = NULL;
  int64_t z0 = 0, y0 = 0, x0 = 0, z1 = 0, y1 = 0, x1 = 0;
  bool have_bbox = false;

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--bbox") == 0) {
      long long a, b, c, d, e, f;
      if (sscanf(argv[++i], "%lld,%lld,%lld,%lld,%lld,%lld", &a,&b,&c,&d,&e,&f) == 6) {
        z0 = a; y0 = b; x0 = c; z1 = d; y1 = e; x1 = f;
        have_bbox = true;
      }
    }
  }

  if (!output)    { fputs("error: --output required\n",  stderr); return 1; }
  if (!have_bbox) { fputs("error: --bbox required\n",    stderr); return 1; }
  if (z1 <= z0 || y1 <= y0 || x1 <= x0) {
    fputs("error: bbox end must be > start\n", stderr); return 1;
  }

  volume *src = vol_open(input);
  if (!src) { fprintf(stderr, "error: cannot open %s\n", input); return 1; }

  const zarr_level_meta *m = vol_level_meta(src, 0);
  if (!m || m->ndim != 3) {
    fputs("error: extract requires a 3D volume\n", stderr);
    vol_free(src); return 1;
  }

  int64_t out_shape[3] = { z1 - z0, y1 - y0, x1 - x0 };
  volume *dst = create_output(output, m, out_shape);
  if (!dst) { vol_free(src); return 1; }

  // iterate over output chunk grid
  int64_t nchunks[3], total = 1;
  for (int d = 0; d < 3; d++) {
    nchunks[d] = (out_shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
    total *= nchunks[d];
  }

  size_t elem = (m->dtype == DTYPE_U8) ? 1 : (m->dtype == DTYPE_U16) ? 2 : 4;
  size_t chunk_vol = (size_t)(m->chunk_shape[0] * m->chunk_shape[1] * m->chunk_shape[2]);
  uint8_t *buf = calloc(chunk_vol, elem);
  if (!buf) { vol_free(src); vol_free(dst); return 1; }

  int64_t coords[3] = {0};
  int64_t done = 0;

  do {
    cli_progress((int)done, (int)total, "extract");
    memset(buf, 0, chunk_vol * elem);

    int64_t oz = coords[0] * m->chunk_shape[0];
    int64_t oy = coords[1] * m->chunk_shape[1];
    int64_t ox = coords[2] * m->chunk_shape[2];
    int64_t ez = oz + m->chunk_shape[0]; if (ez > out_shape[0]) ez = out_shape[0];
    int64_t ey = oy + m->chunk_shape[1]; if (ey > out_shape[1]) ey = out_shape[1];
    int64_t ex = ox + m->chunk_shape[2]; if (ex > out_shape[2]) ex = out_shape[2];

    size_t ci = 0;
    for (int64_t z = oz; z < ez; z++) {
      for (int64_t y = oy; y < ey; y++) {
        for (int64_t x = ox; x < ex; x++) {
          float v = vol_sample(src, 0,
                               (float)(z + z0), (float)(y + y0), (float)(x + x0));
          if      (elem == 1) buf[ci] = (uint8_t)v;
          else if (elem == 2) ((uint16_t *)buf)[ci] = (uint16_t)v;
          else                ((uint32_t *)buf)[ci] = (uint32_t)v;
          ci++;
        }
      }
    }

    vol_write_chunk(dst, 0, coords, buf, ci * elem);
    done++;
  } while (next_coord(coords, nchunks, 3));

  cli_progress((int)total, (int)total, "extract");
  free(buf);
  vol_finalize(dst);
  vol_free(dst);
  vol_free(src);
  printf("written: %s\n", output);
  return 0;
}
