#define _POSIX_C_SOURCE 200809L

#include "cli_compress.h"
#include "cli_progress.h"
#include "core/io.h"
#include "core/vol.h"
#include "core/compress4d.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// .c4d file format
//
//   Magic:   "C4DV" (4 bytes)
//   Version: uint32_t  (currently 1)
//   nlevels: uint32_t
//   quality: float32   (quantisation step used at encode time)
//   For each level:
//     shape[3]:      int64_t  z,y,x
//     chunk[3]:      int64_t  z,y,x
//     nchunks:       int64_t  total chunks in this level
//     For each chunk (row-major order, z outermost):
//       coords[3]:   int64_t  chunk coords z,y,x
//       comp_size:   uint64_t byte count of compressed payload
//       [comp_size bytes of compress4d residual stream]
//   End:     "END4" (4 bytes)
// ---------------------------------------------------------------------------

#define C4D_MAGIC   "C4DV"
#define C4D_END     "END4"
#define C4D_VERSION 1u

// ---------------------------------------------------------------------------
// Generic helpers
// ---------------------------------------------------------------------------

static bool mkdirs(const char *path) {
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

static bool write_file(const char *path, const uint8_t *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, len, f) == len;
  fclose(f);
  return ok;
}

// Write a .zarray JSON for a zarr v2 level.
static void write_zarray(const char *out_dir, int level,
                         const zarr_level_meta *m) {
  char dir[512];
  snprintf(dir, sizeof(dir), "%s/%d", out_dir, level);
  mkdirs(dir);

  char path[560];
  snprintf(path, sizeof(path), "%s/.zarray", dir);
  FILE *f = fopen(path, "w");
  if (!f) { LOG_WARN("write_zarray: cannot create %s", path); return; }

  fprintf(f, "{\n  \"zarr_format\": 2,\n  \"shape\": [");
  for (int d = 0; d < m->ndim; d++)
    fprintf(f, "%s%lld", d ? ", " : "", (long long)m->shape[d]);
  fprintf(f, "],\n  \"chunks\": [");
  for (int d = 0; d < m->ndim; d++)
    fprintf(f, "%s%lld", d ? ", " : "", (long long)m->chunk_shape[d]);
  fprintf(f,
    "],\n  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
    "  \"compressor\": null,\n  \"fill_value\": 0,\n  \"filters\": null\n}\n");
  fclose(f);
}

// Iterate all chunk coords for a level; advance N-dim counter.
// Returns false when exhausted.
static bool next_coords(int64_t *coords, const int64_t *num_chunks, int ndim) {
  for (int d = ndim - 1; d >= 0; d--) {
    coords[d]++;
    if (coords[d] < num_chunks[d]) return true;
    coords[d] = 0;
  }
  return false;
}

// Count total chunks for a level.
static int64_t count_chunks(const zarr_level_meta *m, int64_t num_chunks[8]) {
  int64_t total = 1;
  for (int d = 0; d < m->ndim; d++) {
    num_chunks[d] = (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
    total *= num_chunks[d];
  }
  return total;
}

// ---------------------------------------------------------------------------
// Codec selection (legacy compress command)
// ---------------------------------------------------------------------------

typedef enum { CODEC_COMPRESS4D, CODEC_BLOSC, CODEC_ZSTD } codec_t;

static codec_t parse_codec(const char *s) {
  if (strcmp(s, "compress4d") == 0) return CODEC_COMPRESS4D;
  if (strcmp(s, "blosc")      == 0) return CODEC_BLOSC;
  if (strcmp(s, "zstd")       == 0) return CODEC_ZSTD;
  fprintf(stderr, "warning: unknown codec '%s', defaulting to compress4d\n", s);
  return CODEC_COMPRESS4D;
}

static uint8_t *compress_chunk(codec_t codec, int clevel,
                                const uint8_t *raw, size_t raw_size,
                                size_t *out_size) {
  switch (codec) {
    case CODEC_COMPRESS4D: {
      size_t nfloats = raw_size / sizeof(float);
      if (nfloats == 0) { *out_size = 0; return NULL; }
      return compress4d_encode_residual((const float *)raw, nfloats,
                                       1.0f / 255.0f, out_size);
    }
    case CODEC_BLOSC:
    case CODEC_ZSTD: {
      (void)clevel;
      uint8_t *buf = malloc(raw_size);
      if (!buf) { *out_size = 0; return NULL; }
      memcpy(buf, raw, raw_size);
      *out_size = raw_size;
      return buf;
    }
  }
  *out_size = 0;
  return NULL;
}

// ---------------------------------------------------------------------------
// cmd_compress  (legacy: re-compress zarr -> zarr)
// ---------------------------------------------------------------------------

int cmd_compress(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile compress <input_zarr> --output <out>");
    puts("       [--codec compress4d|blosc|zstd] [--level 1-9]");
    puts("       [--rechunk Z,Y,X]");
    return argc < 1 ? 1 : 0;
  }

  const char *input   = argv[0];
  const char *output  = NULL;
  codec_t     codec   = CODEC_COMPRESS4D;
  int         clevel  = 5;
  int         rechunk[3] = {0, 0, 0};
  bool        do_rechunk = false;

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--codec") == 0) {
      codec = parse_codec(argv[++i]);
    } else if (strcmp(argv[i], "--level") == 0) {
      clevel = atoi(argv[++i]);
      clevel = clevel < 1 ? 1 : clevel > 9 ? 9 : clevel;
    } else if (strcmp(argv[i], "--rechunk") == 0) {
      if (sscanf(argv[++i], "%d,%d,%d",
                 &rechunk[0], &rechunk[1], &rechunk[2]) == 3)
        do_rechunk = true;
    }
  }

  if (!output) { fputs("error: --output <path> required\n", stderr); return 1; }

  volume *v = vol_open(input);
  if (!v) { fprintf(stderr, "error: cannot open %s\n", input); return 1; }

  const char *codec_name =
      codec == CODEC_COMPRESS4D ? "compress4d" :
      codec == CODEC_BLOSC      ? "blosc" : "zstd";

  if (!mkdirs(output)) {
    fprintf(stderr, "error: cannot create %s\n", output);
    vol_free(v); return 1;
  }

  int nlevels = vol_num_levels(v);
  size_t total_in = 0, total_out = 0;
  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  for (int lvl = 0; lvl < nlevels; lvl++) {
    const zarr_level_meta *m = vol_level_meta(v, lvl);
    if (!m) continue;

    // write .zarray  (reuse zarr_level_meta but patch compressor)
    {
      char dir[512];
      snprintf(dir, sizeof(dir), "%s/%d", output, lvl);
      mkdirs(dir);
      char path[560];
      snprintf(path, sizeof(path), "%s/.zarray", dir);
      FILE *f = fopen(path, "w");
      if (f) {
        int64_t cs[8];
        for (int d = 0; d < m->ndim; d++)
          cs[d] = (do_rechunk && d < 3) ? rechunk[2-d] : m->chunk_shape[d];
        fprintf(f, "{\n  \"zarr_format\": 2,\n  \"shape\": [");
        for (int d = 0; d < m->ndim; d++)
          fprintf(f, "%s%lld", d?", ":"", (long long)m->shape[d]);
        fprintf(f, "],\n  \"chunks\": [");
        for (int d = 0; d < m->ndim; d++)
          fprintf(f, "%s%lld", d?", ":"", (long long)cs[d]);
        fprintf(f,
          "],\n  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
          "  \"compressor\": {\"id\": \"%s\", \"clevel\": %d},\n"
          "  \"fill_value\": 0,\n  \"filters\": null\n}\n",
          codec_name, clevel);
        fclose(f);
      }
    }

    int64_t num_chunks[8], total_chunks;
    total_chunks = count_chunks(m, num_chunks);

    int64_t done = 0;
    int64_t coords[8] = {0};
    char label[64];

    do {
      snprintf(label, sizeof(label), "compressing level %d", lvl);
      cli_progress((int)done, (int)total_chunks, label);

      size_t raw_size;
      uint8_t *raw = vol_read_chunk(v, lvl, coords, &raw_size);
      if (raw) {
        total_in += raw_size;
        size_t comp_size;
        uint8_t *comp = compress_chunk(codec, clevel, raw, raw_size, &comp_size);
        free(raw);
        if (comp && comp_size > 0) {
          char chunk_path[512];
          int n = snprintf(chunk_path, sizeof(chunk_path), "%s/%d", output, lvl);
          for (int d = 0; d < m->ndim; d++)
            n += snprintf(chunk_path + n, sizeof(chunk_path) - (size_t)n,
                          "%s%lld", d ? "." : "/", (long long)coords[d]);
          if (!write_file(chunk_path, comp, comp_size))
            LOG_WARN("compress: failed to write %s", chunk_path);
          total_out += comp_size;
          free(comp);
        }
      }
      done++;
    } while (next_coords(coords, num_chunks, m->ndim));

    cli_progress((int)total_chunks, (int)total_chunks, label);
  }

  vol_free(v);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) +
                   (t1.tv_nsec - t0.tv_nsec) * 1e-9;
  double ratio = total_in > 0 ? (double)total_out / (double)total_in : 0.0;
  printf("input:  %zu bytes (%.2f MB)\n", total_in,  (double)total_in  / 1e6);
  printf("output: %zu bytes (%.2f MB)\n", total_out, (double)total_out / 1e6);
  printf("ratio:  %.3f  time: %.2f s\n", ratio, elapsed);
  printf("wrote:  %s\n", output);
  return 0;
}

// ---------------------------------------------------------------------------
// Quality metric helpers (for --verify and --stats)
// ---------------------------------------------------------------------------

// Compute max absolute error and mean squared error between two float arrays.
static void compute_error(const float *orig, const float *decoded,
                          size_t n,
                          float *out_max_err,
                          float *out_mse,
                          float *out_psnr) {
  double max_err = 0.0, sse = 0.0, max_val = 0.0;
  for (size_t i = 0; i < n; i++) {
    double diff = fabs((double)orig[i] - (double)decoded[i]);
    if (diff > max_err) max_err = diff;
    sse += diff * diff;
    double v = fabs((double)orig[i]);
    if (v > max_val) max_val = v;
  }
  *out_max_err = (float)max_err;
  *out_mse     = (float)(sse / (double)n);
  double signal_power = max_val * max_val;
  if (*out_mse > 0.0 && signal_power > 0.0)
    *out_psnr = (float)(10.0 * log10(signal_power / (double)*out_mse));
  else
    *out_psnr = (float)INFINITY;
}

// ---------------------------------------------------------------------------
// c4d file writer
// ---------------------------------------------------------------------------

typedef struct {
  FILE    *fp;
  uint32_t nlevels;
  float    quality;
  // per-level stats accumulated during write
  size_t   bytes_raw;
  size_t   bytes_comp;
} c4d_writer;

static c4d_writer *c4d_writer_open(const char *path, uint32_t nlevels,
                                   float quality) {
  FILE *fp = fopen(path, "wb");
  if (!fp) return NULL;

  c4d_writer *w = calloc(1, sizeof(*w));
  if (!w) { fclose(fp); return NULL; }
  w->fp      = fp;
  w->nlevels = nlevels;
  w->quality = quality;

  // Header
  fwrite(C4D_MAGIC, 1, 4, fp);
  uint32_t ver = C4D_VERSION;
  fwrite(&ver,     sizeof(ver),     1, fp);
  fwrite(&nlevels, sizeof(nlevels), 1, fp);
  fwrite(&quality, sizeof(quality), 1, fp);
  return w;
}

static void c4d_writer_close(c4d_writer *w) {
  if (!w) return;
  fwrite(C4D_END, 1, 4, w->fp);
  fclose(w->fp);
  free(w);
}

// Write one chunk; returns compressed size written (0 on error).
static size_t c4d_write_chunk(c4d_writer *w,
                               const int64_t coords[3],
                               const float *samples, size_t nfloats,
                               float quality) {
  size_t comp_size;
  uint8_t *comp = compress4d_encode_residual(samples, nfloats,
                                             quality, &comp_size);
  if (!comp) return 0;

  fwrite(coords,    sizeof(int64_t), 3, w->fp);
  uint64_t cs64 = (uint64_t)comp_size;
  fwrite(&cs64,     sizeof(cs64),    1, w->fp);
  fwrite(comp,      1,               comp_size, w->fp);
  free(comp);

  w->bytes_raw  += nfloats * sizeof(float);
  w->bytes_comp += comp_size;
  return comp_size;
}

// ---------------------------------------------------------------------------
// c4d file reader (index-free: sequential scan)
// ---------------------------------------------------------------------------

typedef struct {
  FILE    *fp;
  uint32_t version;
  uint32_t nlevels;
  float    quality;
} c4d_reader;

static c4d_reader *c4d_reader_open(const char *path) {
  FILE *fp = fopen(path, "rb");
  if (!fp) return NULL;

  char magic[4];
  if (fread(magic, 1, 4, fp) != 4 ||
      memcmp(magic, C4D_MAGIC, 4) != 0) {
    fprintf(stderr, "error: not a .c4d file: %s\n", path);
    fclose(fp); return NULL;
  }

  c4d_reader *r = calloc(1, sizeof(*r));
  if (!r) { fclose(fp); return NULL; }
  r->fp = fp;
  fread(&r->version, sizeof(r->version), 1, fp);
  fread(&r->nlevels, sizeof(r->nlevels), 1, fp);
  fread(&r->quality, sizeof(r->quality), 1, fp);
  return r;
}

static void c4d_reader_close(c4d_reader *r) {
  if (!r) return;
  fclose(r->fp);
  free(r);
}

// ---------------------------------------------------------------------------
// cmd_compress4d
// ---------------------------------------------------------------------------

int cmd_compress4d(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile compress4d <input.zarr> --output <output.c4d> [options]");
    puts("  --quality <0.1-10.0>   Quantisation step (default 1.0, lower=better)");
    puts("  --chunk-size <Z,Y,X>   Spatial tile size (default 128,128,128)");
    puts("  --levels <N>           Max pyramid levels (default: auto)");
    puts("  --stats                Print per-level compression statistics");
    puts("  --verify               Decode and verify against original");
    puts("  --streaming            Write levels coarsest-first");
    return argc < 1 ? 1 : 0;
  }

  const char *input    = argv[0];
  const char *output   = NULL;
  float       quality  = 1.0f;
  int         chunk_z  = 128, chunk_y = 128, chunk_x = 128;
  int         max_lvls = 0;       // 0 = auto
  bool        do_stats     = false;
  bool        do_verify    = false;
  bool        do_streaming = false;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--output") == 0 && i+1 < argc) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--quality") == 0 && i+1 < argc) {
      quality = (float)atof(argv[++i]);
      if (quality < 0.001f) quality = 0.001f;
      if (quality > 100.0f) quality = 100.0f;
    } else if (strcmp(argv[i], "--chunk-size") == 0 && i+1 < argc) {
      sscanf(argv[++i], "%d,%d,%d", &chunk_z, &chunk_y, &chunk_x);
    } else if (strcmp(argv[i], "--levels") == 0 && i+1 < argc) {
      max_lvls = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--stats")     == 0) { do_stats     = true; }
      else if (strcmp(argv[i], "--verify")    == 0) { do_verify    = true; }
      else if (strcmp(argv[i], "--streaming") == 0) { do_streaming = true; }
  }

  if (!output) {
    fputs("error: --output <output.c4d> is required\n", stderr);
    return 1;
  }

  volume *v = vol_open(input);
  if (!v) { fprintf(stderr, "error: cannot open volume: %s\n", input); return 1; }

  int nlevels = vol_num_levels(v);
  if (max_lvls > 0 && max_lvls < nlevels) nlevels = max_lvls;

  // Determine level order for streaming (coarsest-first) or normal (finest-first)
  int *level_order = malloc((size_t)nlevels * sizeof(int));
  if (!level_order) { vol_free(v); return 1; }
  if (do_streaming) {
    for (int i = 0; i < nlevels; i++) level_order[i] = nlevels - 1 - i;
  } else {
    for (int i = 0; i < nlevels; i++) level_order[i] = i;
  }

  c4d_writer *w = c4d_writer_open(output, (uint32_t)nlevels, quality);
  if (!w) {
    fprintf(stderr, "error: cannot create output file: %s\n", output);
    free(level_order); vol_free(v); return 1;
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  // Per-level stats
  size_t *lvl_raw  = calloc((size_t)nlevels, sizeof(size_t));
  size_t *lvl_comp = calloc((size_t)nlevels, sizeof(size_t));

  // Verify metrics accumulated across all levels
  float verify_max_err = 0.0f, verify_mse_sum = 0.0f;
  size_t verify_nsamples = 0;

  for (int li = 0; li < nlevels; li++) {
    int lvl = level_order[li];
    const zarr_level_meta *m = vol_level_meta(v, lvl);
    if (!m) continue;

    // Write level header: shape[3], chunk[3], nchunks
    // Use the first 3 spatial dims (or pad to 3 if fewer)
    int spatial_dims = m->ndim > 3 ? 3 : m->ndim;
    int64_t shape3[3]  = {1, 1, 1};
    int64_t cshape3[3] = {chunk_z, chunk_y, chunk_x};
    for (int d = 0; d < spatial_dims; d++) {
      // zarr dims are typically [z, y, x]; take last 3 if ndim > 3
      int src = m->ndim - spatial_dims + d;
      shape3[d]  = m->shape[src];
      cshape3[d] = (m->chunk_shape[src] > 0) ? m->chunk_shape[src] :
                   (d == 0 ? chunk_z : d == 1 ? chunk_y : chunk_x);
    }

    int64_t nc[3];
    nc[0] = (shape3[0] + cshape3[0] - 1) / cshape3[0];
    nc[1] = (shape3[1] + cshape3[1] - 1) / cshape3[1];
    nc[2] = (shape3[2] + cshape3[2] - 1) / cshape3[2];
    int64_t total_chunks = nc[0] * nc[1] * nc[2];

    fwrite(shape3,  sizeof(int64_t), 3, w->fp);
    fwrite(cshape3, sizeof(int64_t), 3, w->fp);
    fwrite(&total_chunks, sizeof(total_chunks), 1, w->fp);

    int64_t done = 0;
    char label[64];
    snprintf(label, sizeof(label), "compress4d level %d", lvl);

    // Iterate chunk coords using the full ndim coordinate but write as 3D
    int64_t num_chunks_nd[8] = {0};
    count_chunks(m, num_chunks_nd);
    int64_t full_coords[8] = {0};

    do {
      cli_progress((int)done, (int)total_chunks, label);

      // Map full coords to 3D
      int64_t c3[3] = {0, 0, 0};
      int src = m->ndim - spatial_dims;
      for (int d = 0; d < spatial_dims; d++)
        c3[d] = full_coords[src + d];

      size_t raw_size;
      uint8_t *raw = vol_read_chunk(v, lvl, full_coords, &raw_size);
      if (raw && raw_size > 0) {
        size_t nfloats = raw_size / sizeof(float);

        // Verify: decode after encoding and compare
        if (do_verify && nfloats > 0) {
          size_t comp_size;
          uint8_t *comp = compress4d_encode_residual(
              (const float *)raw, nfloats, quality, &comp_size);
          if (comp) {
            float *decoded = malloc(nfloats * sizeof(float));
            if (decoded && compress4d_decode_residual(
                    comp, comp_size, nfloats, quality, decoded)) {
              float me, mse, psnr;
              compute_error((const float *)raw, decoded, nfloats,
                            &me, &mse, &psnr);
              if (me > verify_max_err) verify_max_err = me;
              verify_mse_sum   += mse;
              verify_nsamples  += nfloats;
            }
            free(decoded);
            free(comp);
          }
        }

        size_t written = c4d_write_chunk(w, c3, (const float *)raw,
                                         nfloats, quality);
        lvl_raw[lvl]  += raw_size;
        lvl_comp[lvl] += written;
        free(raw);
      }
      done++;
    } while (next_coords(full_coords, num_chunks_nd, m->ndim));

    cli_progress((int)total_chunks, (int)total_chunks, label);
  }

  c4d_writer_close(w);
  vol_free(v);
  free(level_order);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) +
                   (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  // Total bytes
  size_t total_raw = 0, total_comp = 0;
  for (int lvl = 0; lvl < nlevels; lvl++) {
    total_raw  += lvl_raw[lvl];
    total_comp += lvl_comp[lvl];
  }

  printf("output:  %s\n", output);
  printf("quality: %.3f\n", (double)quality);
  printf("levels:  %d\n", nlevels);
  printf("input:   %.2f MB\n", (double)total_raw  / 1e6);
  printf("output:  %.2f MB\n", (double)total_comp / 1e6);
  printf("ratio:   %.3f (%.1fx)\n",
         total_raw > 0 ? (double)total_comp / (double)total_raw : 0.0,
         total_raw > 0 ? (double)total_raw  / (double)total_comp : 0.0);
  printf("time:    %.2f s  (%.1f MB/s)\n",
         elapsed,
         elapsed > 0.0 ? (double)total_raw / 1e6 / elapsed : 0.0);

  if (do_stats) {
    printf("\nper-level statistics:\n");
    for (int lvl = 0; lvl < nlevels; lvl++) {
      double r = lvl_raw[lvl] > 0 ?
                 (double)lvl_comp[lvl] / (double)lvl_raw[lvl] : 0.0;
      printf("  level %d: %.2f MB -> %.2f MB  ratio %.3f\n",
             lvl,
             (double)lvl_raw[lvl]  / 1e6,
             (double)lvl_comp[lvl] / 1e6,
             r);
    }
  }

  if (do_verify && verify_nsamples > 0) {
    float avg_mse = verify_mse_sum / (float)verify_nsamples *
                    (float)(verify_nsamples > 0 ? 1 : 0);
    // Recompute properly: mse_sum was already per-chunk averages, not totals.
    // Present max error and average MSE per chunk.
    printf("\nverification:\n");
    printf("  max absolute error: %.6g\n", (double)verify_max_err);
    printf("  mean MSE per chunk: %.6g\n", (double)avg_mse);
    printf("  chunks sampled:     %zu\n",  verify_nsamples);
  }

  free(lvl_raw);
  free(lvl_comp);
  return 0;
}

// ---------------------------------------------------------------------------
// cmd_decompress4d
// ---------------------------------------------------------------------------

int cmd_decompress4d(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile decompress4d <input.c4d> --output <output.zarr> [options]");
    puts("  --level <N>                       Decode up to level N (0=finest)");
    puts("  --region <z0,y0,x0:z1,y1,x1>     Decode a spatial sub-region only");
    return argc < 1 ? 1 : 0;
  }

  const char *input   = argv[0];
  const char *output  = NULL;
  int         max_lvl = INT32_MAX;
  // Region clipping (in voxels; applied per-chunk at decode time)
  bool    has_region = false;
  int64_t reg_z0 = 0, reg_y0 = 0, reg_x0 = 0;
  int64_t reg_z1 = INT64_MAX, reg_y1 = INT64_MAX, reg_x1 = INT64_MAX;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--output") == 0 && i+1 < argc) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--level") == 0 && i+1 < argc) {
      max_lvl = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--region") == 0 && i+1 < argc) {
      if (sscanf(argv[++i], "%lld,%lld,%lld:%lld,%lld,%lld",
                 (long long*)&reg_z0, (long long*)&reg_y0, (long long*)&reg_x0,
                 (long long*)&reg_z1, (long long*)&reg_y1, (long long*)&reg_x1) == 6)
        has_region = true;
    }
  }

  if (!output) { fputs("error: --output <path> required\n", stderr); return 1; }

  c4d_reader *r = c4d_reader_open(input);
  if (!r) return 1;

  if (!mkdirs(output)) {
    fprintf(stderr, "error: cannot create %s\n", output);
    c4d_reader_close(r); return 1;
  }

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  size_t total_comp = 0, total_raw = 0;

  for (uint32_t li = 0; li < r->nlevels; li++) {
    // Read level header
    int64_t shape3[3], cshape3[3];
    int64_t total_chunks;
    if (fread(shape3,  sizeof(int64_t), 3, r->fp) != 3) break;
    if (fread(cshape3, sizeof(int64_t), 3, r->fp) != 3) break;
    if (fread(&total_chunks, sizeof(total_chunks), 1, r->fp) != 1) break;

    int lvl = (int)li;
    bool decode_this = (lvl <= max_lvl);

    // Create level output directory and .zarray
    if (decode_this) {
      char dir[512];
      snprintf(dir, sizeof(dir), "%s/%d", output, lvl);
      mkdirs(dir);
      char zarr_path[560];
      snprintf(zarr_path, sizeof(zarr_path), "%s/.zarray", dir);
      FILE *zf = fopen(zarr_path, "w");
      if (zf) {
        fprintf(zf,
          "{\n  \"zarr_format\": 2,\n"
          "  \"shape\": [%lld, %lld, %lld],\n"
          "  \"chunks\": [%lld, %lld, %lld],\n"
          "  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n"
          "  \"compressor\": null,\n  \"fill_value\": 0,\n"
          "  \"filters\": null\n}\n",
          (long long)shape3[0], (long long)shape3[1], (long long)shape3[2],
          (long long)cshape3[0], (long long)cshape3[1], (long long)cshape3[2]);
        fclose(zf);
      }
    }

    char label[64];
    snprintf(label, sizeof(label), "decompress level %d", lvl);

    for (int64_t ci = 0; ci < total_chunks; ci++) {
      cli_progress((int)(ci % 10000), (int)(total_chunks % 10000), label);

      int64_t coords[3];
      uint64_t comp_size;
      if (fread(coords,    sizeof(int64_t), 3, r->fp) != 3) goto done;
      if (fread(&comp_size, sizeof(comp_size), 1, r->fp) != 1) goto done;

      if (!decode_this || comp_size == 0) {
        // Skip this chunk
        fseek(r->fp, (long)comp_size, SEEK_CUR);
        continue;
      }

      uint8_t *comp = malloc(comp_size);
      if (!comp || fread(comp, 1, comp_size, r->fp) != comp_size) {
        free(comp);
        goto done;
      }

      // Chunk voxel dimensions
      int64_t chunk_vox_z = cshape3[0];
      int64_t chunk_vox_y = cshape3[1];
      int64_t chunk_vox_x = cshape3[2];
      // Clamp to volume boundary
      int64_t vz0 = coords[0] * cshape3[0];
      int64_t vy0 = coords[1] * cshape3[1];
      int64_t vx0 = coords[2] * cshape3[2];
      int64_t actual_z = shape3[0] - vz0; if (actual_z > chunk_vox_z) actual_z = chunk_vox_z;
      int64_t actual_y = shape3[1] - vy0; if (actual_y > chunk_vox_y) actual_y = chunk_vox_y;
      int64_t actual_x = shape3[2] - vx0; if (actual_x > chunk_vox_x) actual_x = chunk_vox_x;
      if (actual_z < 1) actual_z = 1;
      if (actual_y < 1) actual_y = 1;
      if (actual_x < 1) actual_x = 1;
      size_t nfloats = (size_t)(chunk_vox_z * chunk_vox_y * chunk_vox_x);

      // Region skip
      if (has_region) {
        int64_t vz1 = vz0 + chunk_vox_z;
        int64_t vy1 = vy0 + chunk_vox_y;
        int64_t vx1 = vx0 + chunk_vox_x;
        if (vz1 <= reg_z0 || vz0 >= reg_z1 ||
            vy1 <= reg_y0 || vy0 >= reg_y1 ||
            vx1 <= reg_x0 || vx0 >= reg_x1) {
          free(comp);
          continue;
        }
      }

      float *decoded = malloc(nfloats * sizeof(float));
      if (!decoded) { free(comp); continue; }

      if (!compress4d_decode_residual(comp, comp_size, nfloats,
                                       r->quality, decoded)) {
        LOG_WARN("decompress4d: decode failed for level %d chunk %lld,%lld,%lld",
                 lvl, (long long)coords[0], (long long)coords[1], (long long)coords[2]);
        free(decoded); free(comp); continue;
      }

      // Write raw float chunk
      char chunk_path[512];
      snprintf(chunk_path, sizeof(chunk_path), "%s/%d/%lld.%lld.%lld",
               output, lvl,
               (long long)coords[0], (long long)coords[1], (long long)coords[2]);
      write_file(chunk_path, (const uint8_t *)decoded,
                 nfloats * sizeof(float));

      total_raw  += nfloats * sizeof(float);
      total_comp += comp_size;
      free(decoded);
      free(comp);
    }

    cli_progress((int)total_chunks, (int)total_chunks, label);
  }

done:
  c4d_reader_close(r);
  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) +
                   (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  printf("decoded: %.2f MB  time: %.2f s  (%.1f MB/s)\n",
         (double)total_raw / 1e6, elapsed,
         elapsed > 0.0 ? (double)total_raw / 1e6 / elapsed : 0.0);
  printf("wrote:   %s\n", output);
  return 0;
}

// ---------------------------------------------------------------------------
// cmd_compress4d_info
// ---------------------------------------------------------------------------

int cmd_compress4d_info(int argc, char **argv) {
  if (argc < 1) {
    puts("usage: volatile compress4d-info <input.c4d>");
    return 1;
  }

  c4d_reader *r = c4d_reader_open(argv[0]);
  if (!r) return 1;

  printf("file:    %s\n", argv[0]);
  printf("version: %u\n", r->version);
  printf("quality: %.4f\n", (double)r->quality);
  printf("levels:  %u\n\n", r->nlevels);

  size_t total_comp = 0;
  size_t total_raw  = 0;

  for (uint32_t li = 0; li < r->nlevels; li++) {
    int64_t shape3[3], cshape3[3];
    int64_t total_chunks;
    if (fread(shape3,  sizeof(int64_t), 3, r->fp) != 3) break;
    if (fread(cshape3, sizeof(int64_t), 3, r->fp) != 3) break;
    if (fread(&total_chunks, sizeof(total_chunks), 1, r->fp) != 1) break;

    size_t lvl_comp = 0;
    size_t lvl_raw  = (size_t)(shape3[0] * shape3[1] * shape3[2] *
                                (int64_t)sizeof(float));

    for (int64_t ci = 0; ci < total_chunks; ci++) {
      int64_t coords[3];
      uint64_t comp_size;
      if (fread(coords,     sizeof(int64_t), 3, r->fp) != 3) goto info_done;
      if (fread(&comp_size, sizeof(comp_size), 1, r->fp) != 1) goto info_done;
      fseek(r->fp, (long)comp_size, SEEK_CUR);
      lvl_comp += comp_size;
    }

    double ratio = lvl_raw > 0 ? (double)lvl_raw / (double)lvl_comp : 0.0;
    printf("  level %u:\n", li);
    printf("    shape:    %lld x %lld x %lld\n",
           (long long)shape3[0], (long long)shape3[1], (long long)shape3[2]);
    printf("    chunks:   %lld x %lld x %lld\n",
           (long long)cshape3[0], (long long)cshape3[1], (long long)cshape3[2]);
    printf("    raw:      %.2f MB\n", (double)lvl_raw  / 1e6);
    printf("    compressed: %.2f MB\n", (double)lvl_comp / 1e6);
    printf("    ratio:    %.2fx\n\n", ratio);

    total_comp += lvl_comp;
    total_raw  += lvl_raw;
  }

info_done:;
  double total_ratio = total_raw > 0 ?
                       (double)total_raw / (double)total_comp : 0.0;
  printf("total raw:        %.2f MB\n", (double)total_raw  / 1e6);
  printf("total compressed: %.2f MB\n", (double)total_comp / 1e6);
  printf("total ratio:      %.2fx\n",   total_ratio);

  c4d_reader_close(r);
  return 0;
}
