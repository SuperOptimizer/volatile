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
#include <time.h>
#include <errno.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Codec selection
// ---------------------------------------------------------------------------

typedef enum { CODEC_COMPRESS4D, CODEC_BLOSC, CODEC_ZSTD } codec_t;

static codec_t parse_codec(const char *s) {
  if (strcmp(s, "compress4d") == 0) return CODEC_COMPRESS4D;
  if (strcmp(s, "blosc")      == 0) return CODEC_BLOSC;
  if (strcmp(s, "zstd")       == 0) return CODEC_ZSTD;
  fprintf(stderr, "warning: unknown codec '%s', defaulting to compress4d\n", s);
  return CODEC_COMPRESS4D;
}

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

// Create directory (and parents) if needed.  Returns false on failure.
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

// Write raw bytes to a file (creates parent dirs on demand).
static bool write_file(const char *path, const uint8_t *data, size_t len) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  size_t written = fwrite(data, 1, len, f);
  fclose(f);
  return written == len;
}

// Minimal .zarray JSON for OME-Zarr v2 output.
static void write_zarray(const char *out_dir, int level,
                         const zarr_level_meta *m,
                         const int rechunk[3], const char *compressor_id,
                         int clevel) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%d/.zarray", out_dir, level);
  mkdirs(path);  // ensure parent dir

  // Strip filename to get the dir
  char dir[512];
  snprintf(dir, sizeof(dir), "%s/%d", out_dir, level);
  mkdirs(dir);

  FILE *f = fopen(path, "w");
  if (!f) { LOG_WARN("write_zarray: cannot open %s", path); return; }

  // chunk shape: use rechunk if provided, else original
  int64_t cs[8];
  for (int d = 0; d < m->ndim; d++)
    cs[d] = (rechunk && d < 3) ? rechunk[2 - d] : m->chunk_shape[d];

  fprintf(f, "{\n  \"zarr_format\": 2,\n  \"shape\": [");
  for (int d = 0; d < m->ndim; d++)
    fprintf(f, "%s%lld", d ? ", " : "", (long long)m->shape[d]);
  fprintf(f, "],\n  \"chunks\": [");
  for (int d = 0; d < m->ndim; d++)
    fprintf(f, "%s%lld", d ? ", " : "", (long long)cs[d]);
  fprintf(f, "],\n  \"dtype\": \"<f4\",\n  \"order\": \"C\",\n");
  fprintf(f, "  \"compressor\": {\"id\": \"%s\", \"clevel\": %d},\n",
          compressor_id, clevel);
  fprintf(f, "  \"fill_value\": 0,\n  \"filters\": null\n}\n");
  fclose(f);
}

// ---------------------------------------------------------------------------
// Compress one chunk with selected codec.  Returns heap-allocated bytes.
// ---------------------------------------------------------------------------

static uint8_t *compress_chunk(codec_t codec, int level,
                                const uint8_t *raw, size_t raw_size,
                                size_t *out_size) {
  switch (codec) {
    case CODEC_COMPRESS4D: {
      // Treat raw bytes as float32; if not aligned/float just byte-quantise.
      size_t nfloats = raw_size / sizeof(float);
      if (nfloats == 0) { *out_size = 0; return NULL; }
      return compress4d_encode_residual((const float *)raw, nfloats,
                                       1.0f / 255.0f, out_size);
    }
    case CODEC_BLOSC:
    case CODEC_ZSTD: {
      // Fallback: store uncompressed (blosc2 linkage is optional at CLI level).
      // TODO: call blosc2_compress() when linked.
      (void)level;
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
// cmd_compress
// ---------------------------------------------------------------------------

int cmd_compress(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile compress <input_zarr> --output <out>");
    puts("       [--codec compress4d|blosc|zstd] [--level 1-9]");
    puts("       [--rechunk Z,Y,X]");
    puts("");
    puts("  Re-compress all chunks of a Zarr volume with the chosen codec.");
    puts("  Defaults: --codec compress4d  --level 5");
    return argc < 1 ? 1 : 0;
  }

  const char *input    = argv[0];
  const char *output   = NULL;
  codec_t     codec    = CODEC_COMPRESS4D;
  int         clevel   = 5;
  int         rechunk[3] = {0, 0, 0};  // 0 = no rechunking
  bool        do_rechunk = false;

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--output") == 0) {
      output = argv[++i];
    } else if (strcmp(argv[i], "--codec") == 0) {
      codec = parse_codec(argv[++i]);
    } else if (strcmp(argv[i], "--level") == 0) {
      clevel = atoi(argv[++i]);
      if (clevel < 1) clevel = 1;
      if (clevel > 9) clevel = 9;
    } else if (strcmp(argv[i], "--rechunk") == 0) {
      if (sscanf(argv[++i], "%d,%d,%d", &rechunk[0], &rechunk[1], &rechunk[2]) == 3)
        do_rechunk = true;
    }
  }

  if (!output) {
    fputs("error: --output <path> is required\n", stderr);
    return 1;
  }

  volume *v = vol_open(input);
  if (!v) {
    fprintf(stderr, "error: cannot open volume: %s\n", input);
    return 1;
  }

  const char *codec_name =
      codec == CODEC_COMPRESS4D ? "compress4d" :
      codec == CODEC_BLOSC      ? "blosc" : "zstd";

  int nlevels = vol_num_levels(v);
  size_t total_in = 0, total_out = 0;

  struct timespec t0, t1;
  clock_gettime(CLOCK_MONOTONIC, &t0);

  if (!mkdirs(output)) {
    fprintf(stderr, "error: cannot create output dir: %s\n", output);
    vol_free(v);
    return 1;
  }

  for (int lvl = 0; lvl < nlevels; lvl++) {
    const zarr_level_meta *m = vol_level_meta(v, lvl);
    if (!m) continue;

    write_zarray(output, lvl, m, do_rechunk ? rechunk : NULL, codec_name, clevel);

    // Count chunks
    int64_t num_chunks[8], total_chunks = 1;
    for (int d = 0; d < m->ndim; d++) {
      num_chunks[d] = (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
      total_chunks *= num_chunks[d];
    }

    int64_t done = 0;
    int64_t coords[8] = {0};
    char label[64];

    while (true) {
      snprintf(label, sizeof(label), "compressing level %d", lvl);
      cli_progress((int)done, (int)total_chunks, label);

      size_t raw_size;
      uint8_t *raw = vol_read_chunk(v, lvl, coords, &raw_size);
      if (raw) {
        total_in += raw_size;

        size_t comp_size;
        uint8_t *compressed = compress_chunk(codec, clevel, raw, raw_size, &comp_size);
        free(raw);

        if (compressed && comp_size > 0) {
          // Build output chunk path: <out>/<level>/<c0>.<c1>.<c2>...
          char chunk_path[512];
          int n = snprintf(chunk_path, sizeof(chunk_path), "%s/%d", output, lvl);
          for (int d = 0; d < m->ndim; d++)
            n += snprintf(chunk_path + n, sizeof(chunk_path) - (size_t)n,
                          "%s%lld", d ? "." : "/", (long long)coords[d]);

          if (!write_file(chunk_path, compressed, comp_size))
            LOG_WARN("compress: failed to write %s", chunk_path);

          total_out += comp_size;
          free(compressed);
        }
      }

      done++;

      // Advance N-dimensional chunk coordinate
      int carry = m->ndim - 1;
      while (carry >= 0) {
        coords[carry]++;
        if (coords[carry] < num_chunks[carry]) break;
        coords[carry] = 0;
        carry--;
      }
      if (carry < 0) break;
    }

    cli_progress((int)total_chunks, (int)total_chunks, label);
  }

  vol_free(v);

  clock_gettime(CLOCK_MONOTONIC, &t1);
  double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;

  double ratio = total_in > 0 ? (double)total_out / (double)total_in : 0.0;
  printf("input:   %zu bytes (%.1f MB)\n", total_in,  (double)total_in  / 1e6);
  printf("output:  %zu bytes (%.1f MB)\n", total_out, (double)total_out / 1e6);
  printf("ratio:   %.3f (%.1f%% of original)\n", ratio, ratio * 100.0);
  printf("time:    %.2f s\n", elapsed);
  printf("written: %s\n", output);
  return 0;
}
