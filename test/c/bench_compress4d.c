#define _POSIX_C_SOURCE 200809L

#include "core/compress4d.h"

#include <blosc2.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec * 1e-6;
}

// ---------------------------------------------------------------------------
// CT-like synthetic data generation
// ---------------------------------------------------------------------------

// Fills buf[n] with uint8 data that mimics CT papyrus:
//   - slow gradient (large background structure)
//   - medium-frequency sine noise
//   - sparse bright features (fibres)
static void gen_ct_data(uint8_t *buf, size_t n, int side) {
  for (int z = 0; z < side; z++) {
    for (int y = 0; y < side; y++) {
      for (int x = 0; x < side; x++) {
        // Gradient base: papyrus sheet is brighter near z-centre
        float fz = (float)z / (float)(side - 1);
        float fy = (float)y / (float)(side - 1);
        float fx = (float)x / (float)(side - 1);
        float grad = 80.0f + 60.0f * expf(-8.0f * ((fz - 0.5f)*(fz - 0.5f)
                                                   + (fy - 0.5f)*(fy - 0.5f)));
        // Medium-frequency noise
        float noise = 15.0f * sinf(fx * 12.0f) * cosf(fy * 10.0f + fz * 8.0f);
        // Occasional bright fibre (1-voxel-wide, linear)
        float fibre = 0.0f;
        if ((x + y * 3) % 31 == 0) fibre = 80.0f;
        float v = grad + noise + fibre;
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        buf[(size_t)z * side * side + (size_t)y * side + x] = (uint8_t)v;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Benchmark helpers
// ---------------------------------------------------------------------------

typedef struct {
  double encode_ms;
  double decode_ms;
  size_t compressed;
  size_t original;
} bench_result;

// blosc2 benchmark
static bench_result bench_blosc(const uint8_t *src, size_t n,
                                int clevel, const char *cname) {
  size_t dst_size = n + BLOSC2_MAX_OVERHEAD;
  uint8_t *dst = malloc(dst_size);
  uint8_t *dec = malloc(n);

  blosc2_init();
  blosc1_set_compressor(cname);

  double t0 = now_ms();
  int csize = blosc1_compress(clevel, BLOSC_SHUFFLE, sizeof(uint8_t),
                              src, (int32_t)n, dst, (int32_t)dst_size);
  double encode_ms = now_ms() - t0;

  double t1 = now_ms();
  blosc1_decompress(dst, dec, (int32_t)n);
  double decode_ms = now_ms() - t1;

  bench_result r = {
    .encode_ms  = encode_ms,
    .decode_ms  = decode_ms,
    .compressed = csize > 0 ? (size_t)csize : 0,
    .original   = n,
  };

  free(dst);
  free(dec);
  blosc2_destroy();
  return r;
}

// compress4d rANS benchmark (encode = build table + ans_encode; decode = ans_decode)
static bench_result bench_compress4d_ans(const uint8_t *src, size_t n) {
  double t0 = now_ms();
  ans_table *t = ans_table_build(src, n);
  size_t enc_len = 0;
  uint8_t *enc = ans_encode(t, src, n, &enc_len);
  double encode_ms = now_ms() - t0;

  double t1 = now_ms();
  uint8_t *dec = ans_decode(t, enc, enc_len, n);
  double decode_ms = now_ms() - t1;

  bench_result r = {
    .encode_ms  = encode_ms,
    .decode_ms  = decode_ms,
    .compressed = enc_len,
    .original   = n,
  };

  free(enc);
  free(dec);
  ans_table_free(t);
  return r;
}

// ---------------------------------------------------------------------------
// Pyramid benchmark: 4-level compress4d (residual per level) vs blosc per-level
// ---------------------------------------------------------------------------

typedef struct {
  size_t c4d_total;   // total bytes for all compress4d levels
  size_t blosc_total; // total bytes for all blosc levels (zstd-1)
  size_t orig_total;  // uncompressed total across levels
  double c4d_encode_ms;
  double blosc_encode_ms;
} pyramid_result;

static pyramid_result bench_pyramid(int base_side, int nlevels) {
  pyramid_result pr = {0};

  // Build float pyramid: level 0 = full, level k = half-resolution
  int side = base_side;
  float *prev = NULL;

  // allocate and fill base level as float
  size_t base_n = (size_t)side * side * side;
  uint8_t *raw = malloc(base_n);
  gen_ct_data(raw, base_n, side);
  float *base_f = malloc(base_n * sizeof(float));
  for (size_t i = 0; i < base_n; i++) base_f[i] = (float)raw[i];
  free(raw);

  prev = base_f;
  int cur_side = side;

  for (int lvl = 0; lvl < nlevels; lvl++) {
    size_t n = (size_t)cur_side * cur_side * cur_side;
    pr.orig_total += n;

    // --- compress4d residual encode ---
    float scale = 0.5f;
    double t0 = now_ms();
    size_t enc_len = 0;
    uint8_t *enc = compress4d_encode_residual(prev, n, scale, &enc_len);
    pr.c4d_encode_ms += now_ms() - t0;
    pr.c4d_total += enc_len;
    free(enc);

    // --- blosc zstd-1 encode (on float bytes) ---
    size_t nbytes = n * sizeof(float);
    size_t dst_size = nbytes + BLOSC2_MAX_OVERHEAD;
    uint8_t *dst = malloc(dst_size);
    blosc2_init();
    blosc1_set_compressor("zstd");
    double t1 = now_ms();
    int csize = blosc1_compress(1, BLOSC_SHUFFLE, sizeof(float),
                                prev, (int32_t)nbytes, dst, (int32_t)dst_size);
    pr.blosc_encode_ms += now_ms() - t1;
    if (csize > 0) pr.blosc_total += (size_t)csize;
    free(dst);
    blosc2_destroy();

    // Downsample 2x (box filter) for next level
    if (lvl < nlevels - 1) {
      int next_side = cur_side / 2;
      if (next_side < 1) break;
      size_t ns = (size_t)next_side;
      size_t cs = (size_t)cur_side;
      size_t next_n = ns * ns * ns;
      float *next = malloc(next_n * sizeof(float));
      for (int z = 0; z < next_side; z++)
        for (int y = 0; y < next_side; y++)
          for (int x = 0; x < next_side; x++) {
            float s = 0.0f;
            for (int dz = 0; dz < 2; dz++)
              for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++)
                  s += prev[((size_t)(z*2+dz)*cs*cs
                             + (size_t)(y*2+dy)*cs + (size_t)(x*2+dx))];
            next[(size_t)z*ns*ns + (size_t)y*ns + (size_t)x] = s * 0.125f;
          }
      if (prev != base_f) free(prev);
      prev = next;
      cur_side = next_side;
    }
  }

  if (prev != base_f) free(prev);
  free(base_f);
  return pr;
}

// ---------------------------------------------------------------------------
// Print row
// ---------------------------------------------------------------------------

static void print_row(const char *size_str, const char *codec,
                      bench_result r) {
  double ratio     = (r.compressed > 0) ? (double)r.original / (double)r.compressed : 0.0;
  double orig_mb   = (double)r.original / (1024.0 * 1024.0);
  double enc_mbs   = (r.encode_ms > 0) ? orig_mb / (r.encode_ms * 1e-3) : 0.0;
  double dec_mbs   = (r.decode_ms > 0) ? orig_mb / (r.decode_ms * 1e-3) : 0.0;

  printf("%-8s | %-18s | %9.2f | %9.2f | %6.2fx | %11.1f | %11.1f\n",
         size_str, codec,
         r.encode_ms, r.decode_ms,
         ratio, enc_mbs, dec_mbs);
}

static void print_sep(void) {
  puts("---------+--------------------+-----------+-----------+--------+-------------+-------------");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(void) {
  printf("bench_compress4d: synthetic CT-like data\n\n");
  printf("%-8s | %-18s | %9s | %9s | %7s | %11s | %11s\n",
         "Size", "Codec", "Encode ms", "Decode ms", "Ratio", "MB/s enc", "MB/s dec");
  print_sep();

  int sides[] = {64, 128, 256};
  int nsides  = (int)(sizeof(sides) / sizeof(sides[0]));

  for (int si = 0; si < nsides; si++) {
    int side    = sides[si];
    size_t n    = (size_t)side * (size_t)side * (size_t)side;
    uint8_t *src = malloc(n);
    gen_ct_data(src, n, side);

    char label[16];
    snprintf(label, sizeof(label), "%d^3", side);

    bench_result br;

    br = bench_blosc(src, n, 1, "zstd");
    print_row(label, "blosc-zstd-1", br);

    br = bench_blosc(src, n, 5, "zstd");
    print_row(label, "blosc-zstd-5", br);

    br = bench_compress4d_ans(src, n);
    print_row(label, "compress4d-ans", br);

    print_sep();
    free(src);
  }

  // Pyramid benchmark
  printf("\nPyramid benchmark (128^3 base, 4 levels):\n");
  pyramid_result pr = bench_pyramid(128, 4);
  double orig_mb   = (double)pr.orig_total / (1024.0 * 1024.0);
  double c4d_mb    = (double)pr.c4d_total  / (1024.0 * 1024.0);
  double blosc_mb  = (double)pr.blosc_total / (1024.0 * 1024.0);
  printf("  Original total:     %.2f MB (all levels, float32)\n", orig_mb);
  printf("  compress4d total:   %.2f MB  ratio=%.2fx  encode=%.1f ms\n",
         c4d_mb,
         pr.orig_total > 0 ? (double)pr.orig_total / (double)pr.c4d_total : 0.0,
         pr.c4d_encode_ms);
  printf("  blosc-zstd-1 total: %.2f MB  ratio=%.2fx  encode=%.1f ms\n",
         blosc_mb,
         pr.blosc_total > 0 ? (double)pr.orig_total * sizeof(float) / (double)pr.blosc_total : 0.0,
         pr.blosc_encode_ms);

  return 0;
}
