#define _POSIX_C_SOURCE 200809L

#include "cli_stats.h"
#include "cli_progress.h"
#include "core/vol.h"
#include "core/io.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Running statistics accumulator (Welford's online algorithm)
// ---------------------------------------------------------------------------

typedef struct {
  double  min, max;
  double  mean;
  double  m2;         // sum of squared deviations (for variance)
  int64_t count;
  // 256-bin histogram over the raw float range
  double  hist_min, hist_max;
  int64_t hist[256];
  bool    hist_ready;
} stats_acc;

static void stats_init(stats_acc *s) {
  memset(s, 0, sizeof(*s));
  s->min = 1e300;
  s->max = -1e300;
}

static void stats_push(stats_acc *s, double v) {
  if (v < s->min) s->min = v;
  if (v > s->max) s->max = v;
  s->count++;
  double delta = v - s->mean;
  s->mean += delta / (double)s->count;
  double delta2 = v - s->mean;
  s->m2 += delta * delta2;
}

static double stats_stddev(const stats_acc *s) {
  if (s->count < 2) return 0.0;
  return sqrt(s->m2 / (double)(s->count - 1));
}

// Second pass: bin values into histogram.
static void stats_histogram(stats_acc *s, double v) {
  if (s->hist_max == s->hist_min) return;
  double t = (v - s->hist_min) / (s->hist_max - s->hist_min);
  int bin = (int)(t * 255.0);
  if (bin < 0)   bin = 0;
  if (bin > 255) bin = 255;
  s->hist[bin]++;
}

// Compute percentile from histogram (approximate).
static double hist_percentile(const stats_acc *s, double pct) {
  int64_t target = (int64_t)(pct * (double)s->count / 100.0);
  int64_t accum = 0;
  for (int b = 0; b < 256; b++) {
    accum += s->hist[b];
    if (accum >= target) {
      double t = (b + 0.5) / 256.0;
      return s->hist_min + t * (s->hist_max - s->hist_min);
    }
  }
  return s->hist_max;
}

// ---------------------------------------------------------------------------
// Decode a raw chunk buffer into doubles and push to accumulator.
// ---------------------------------------------------------------------------

static void accumulate_chunk(stats_acc *s, const uint8_t *raw, size_t nbytes,
                              dtype_t dtype, bool do_hist) {
  size_t stride = 1;
  switch (dtype) {
    case DTYPE_U8:  stride = 1; break;
    case DTYPE_U16: stride = 2; break;
    case DTYPE_F32: stride = 4; break;
    case DTYPE_F64: stride = 8; break;
  }
  size_t n = nbytes / stride;
  for (size_t i = 0; i < n; i++) {
    double v;
    switch (dtype) {
      case DTYPE_U8:  v = ((const uint8_t *)raw)[i];  break;
      case DTYPE_U16: { uint16_t u; memcpy(&u, raw + i*2, 2); v = u; break; }
      case DTYPE_F32: { float f;    memcpy(&f, raw + i*4, 4); v = f; break; }
      case DTYPE_F64: { memcpy(&v, raw + i*8, 8); break; }
      default: v = 0;
    }
    if (do_hist)
      stats_histogram(s, v);
    else
      stats_push(s, v);
  }
}

// ---------------------------------------------------------------------------
// cmd_stats
// ---------------------------------------------------------------------------

int cmd_stats(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile stats <path>");
    puts("  Print volume statistics: min, max, mean, std, and histogram percentiles.");
    return argc < 1 ? 1 : 0;
  }

  const char *path = argv[0];
  volume *v = vol_open(path);
  if (!v) {
    fprintf(stderr, "error: cannot open volume: %s\n", path);
    return 1;
  }

  const zarr_level_meta *m = vol_level_meta(v, 0);
  if (!m) {
    fputs("error: no level 0 metadata\n", stderr);
    vol_free(v);
    return 1;
  }

  // Build chunk iteration counts.
  int64_t num_chunks[8];
  int64_t total = 1;
  for (int d = 0; d < m->ndim; d++) {
    num_chunks[d] = (m->shape[d] + m->chunk_shape[d] - 1) / m->chunk_shape[d];
    total *= num_chunks[d];
  }

  dtype_t dtype = (dtype_t)m->dtype;
  stats_acc acc;
  stats_init(&acc);

  // Pass 1: compute min/max/mean/variance.
  fprintf(stderr, "pass 1/2: scanning %lld chunks...\n", (long long)total);
  int64_t done = 0;
  int64_t coords[8] = {0};
  while (true) {
    size_t sz;
    uint8_t *raw = vol_read_chunk(v, 0, coords, &sz);
    if (raw) {
      accumulate_chunk(&acc, raw, sz, dtype, false);
      free(raw);
    }
    done++;
    cli_progress((int)done, (int)total, "scanning");

    int carry = m->ndim - 1;
    while (carry >= 0) {
      coords[carry]++;
      if (coords[carry] < num_chunks[carry]) break;
      coords[carry] = 0;
      carry--;
    }
    if (carry < 0) break;
  }

  if (acc.count == 0) {
    fputs("error: no voxels found\n", stderr);
    vol_free(v); return 1;
  }

  // Pass 2: histogram.
  acc.hist_min   = acc.min;
  acc.hist_max   = acc.max;
  acc.hist_ready = true;
  fprintf(stderr, "pass 2/2: building histogram...\n");
  done = 0;
  memset(coords, 0, sizeof(coords));
  while (true) {
    size_t sz;
    uint8_t *raw = vol_read_chunk(v, 0, coords, &sz);
    if (raw) {
      accumulate_chunk(&acc, raw, sz, dtype, true);
      free(raw);
    }
    done++;
    cli_progress((int)done, (int)total, "histogram");

    int carry = m->ndim - 1;
    while (carry >= 0) {
      coords[carry]++;
      if (coords[carry] < num_chunks[carry]) break;
      coords[carry] = 0;
      carry--;
    }
    if (carry < 0) break;
  }

  vol_free(v);

  // Print results.
  printf("path:    %s\n", path);
  printf("voxels:  %lld\n", (long long)acc.count);
  printf("min:     %.6g\n", acc.min);
  printf("max:     %.6g\n", acc.max);
  printf("mean:    %.6g\n", acc.mean);
  printf("std:     %.6g\n", stats_stddev(&acc));
  printf("p01:     %.6g\n", hist_percentile(&acc, 1.0));
  printf("p05:     %.6g\n", hist_percentile(&acc, 5.0));
  printf("p25:     %.6g\n", hist_percentile(&acc, 25.0));
  printf("p50:     %.6g\n", hist_percentile(&acc, 50.0));
  printf("p75:     %.6g\n", hist_percentile(&acc, 75.0));
  printf("p95:     %.6g\n", hist_percentile(&acc, 95.0));
  printf("p99:     %.6g\n", hist_percentile(&acc, 99.0));
  return 0;
}
