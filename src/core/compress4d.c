#include "core/compress4d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// rANS implementation
// NOTE: We use the "Fabian Giesen" style rANS with 32-bit state.
// M = 1<<SCALE_BITS (normalised frequency table size).
// State range: [RANS_BYTE_L, RANS_BYTE_L * 256).
// Encode is done right-to-left (reverse); decode left-to-right (forward).
// Ref: https://arxiv.org/abs/1402.3392 and rans_byte.h by Fabian Giesen.
// ---------------------------------------------------------------------------

#define SCALE_BITS 12u
#define ANS_M      (1u << SCALE_BITS)  // 4096
#define ANS_NSYM   256u
#define RANS_L     (1u << (32u - SCALE_BITS - 8u))  // 1 << 12 = 4096
// State in [RANS_L, RANS_L*256); renorm emits/reads 1 byte at a time.

// ---------------------------------------------------------------------------
// Frequency table
// ---------------------------------------------------------------------------

struct ans_table {
  uint16_t freq[ANS_NSYM];     // normalised frequency for each symbol
  uint16_t cumul[ANS_NSYM+1];  // exclusive prefix-sum; cumul[NSYM] == ANS_M
  uint8_t  sym_of_slot[ANS_M]; // slot index -> symbol  (for decode)
};

static void normalise_counts(const uint32_t raw[ANS_NSYM], uint16_t freq_out[ANS_NSYM]) {
  uint64_t total    = 0;
  uint32_t n_nonzero = 0;
  for (uint32_t i = 0; i < ANS_NSYM; i++) { total += raw[i]; if (raw[i]) n_nonzero++; }
  if (!total || !n_nonzero) {
    for (uint32_t i = 0; i < ANS_NSYM; i++) freq_out[i] = (uint16_t)(ANS_M / ANS_NSYM);
    return;
  }

  uint32_t alloc[ANS_NSYM] = {0};
  uint32_t assigned = 0;
  // Floor-scale each non-zero symbol, guaranteeing >= 1.
  for (uint32_t i = 0; i < ANS_NSYM; i++) {
    if (!raw[i]) continue;
    uint32_t a = (uint32_t)((uint64_t)raw[i] * (ANS_M - n_nonzero) / total) + 1;
    alloc[i]   = a;
    assigned  += a;
  }
  int32_t leftover = (int32_t)ANS_M - (int32_t)assigned;
  for (uint32_t i = 0; leftover > 0 && i < ANS_NSYM; i++)
    if (raw[i]) { alloc[i]++; leftover--; }
  for (uint32_t i = 0; leftover < 0 && i < ANS_NSYM; i++)
    if (alloc[i] > 1) { alloc[i]--; leftover++; }
  for (uint32_t i = 0; i < ANS_NSYM; i++) freq_out[i] = (uint16_t)alloc[i];
}

static void build_sym_of_slot(ans_table *t) {
  // NOTE: rANS requires sym_of_slot[cumul[s] + k] == s for k in [0, freq[s]).
  // Sequential (non-interleaved) layout ensures the decode step
  //   slot = x & (ANS_M-1),  s = sym_of_slot[slot]
  // correctly identifies the symbol when slot is in [cumul[s], cumul[s]+freq[s]).
  for (uint32_t s = 0; s < ANS_NSYM; s++)
    for (uint32_t k = 0; k < t->freq[s]; k++)
      t->sym_of_slot[t->cumul[s] + k] = (uint8_t)s;
}

static ans_table *table_from_freq(const uint16_t freq[ANS_NSYM]) {
  ans_table *t = malloc(sizeof(*t));
  if (!t) return NULL;
  memcpy(t->freq, freq, ANS_NSYM * sizeof(uint16_t));
  t->cumul[0] = 0;
  for (uint32_t i = 0; i < ANS_NSYM; i++) t->cumul[i+1] = t->cumul[i] + t->freq[i];
  build_sym_of_slot(t);
  return t;
}

ans_table *ans_table_build(const uint8_t *data, size_t len) {
  uint32_t raw[ANS_NSYM] = {0};
  for (size_t i = 0; i < len; i++) raw[data[i]]++;
  for (uint32_t i = 0; i < ANS_NSYM; i++) if (!raw[i]) raw[i] = 1;
  uint16_t freq[ANS_NSYM];
  normalise_counts(raw, freq);
  return table_from_freq(freq);
}

ans_table *ans_table_from_counts(const uint32_t counts[256]) {
  uint16_t freq[ANS_NSYM];
  normalise_counts(counts, freq);
  return table_from_freq(freq);
}

ans_table *ans_table_from_freqs(const uint16_t freqs[256]) { return table_from_freq(freqs); }
void       ans_table_free(ans_table *t) { free(t); }
void       ans_table_get_freqs(const ans_table *t, uint16_t freqs_out[256]) {
  memcpy(freqs_out, t->freq, ANS_NSYM * sizeof(uint16_t));
}

// ---------------------------------------------------------------------------
// rANS encode  (writes output in reverse; caller reverses before returning)
//
// Standard rANS encode step for symbol s with freq fs, cumul cs:
//   1. Renorm x down: while x >= ((RANS_L >> SCALE_BITS) << 8) * fs  -> emit low byte
//   2. x_new = (x / fs) * ANS_M + cs + (x % fs)
//
// Decode (inverse):
//   1. slot = x & (ANS_M - 1);  s = sym[slot];
//   2. x_new = fs * (x >> SCALE_BITS) + slot - cs
//   3. Renorm up: while x < RANS_L -> x = (x << 8) | read_byte
// ---------------------------------------------------------------------------

uint8_t *ans_encode(const ans_table *t, const uint8_t *src, size_t len, size_t *out_len) {
  size_t   cap = len + 16;
  uint8_t *buf = malloc(cap);
  if (!buf) return NULL;

  uint32_t x      = RANS_L;
  size_t   nbytes = 0;

  // NOTE: Encode right-to-left; decoder will process left-to-right.
  for (size_t i = len; i-- > 0; ) {
    uint8_t  s  = src[i];
    uint32_t fs = t->freq[s];
    uint32_t cs = t->cumul[s];

    // Renorm: upper bound is (RANS_L/ANS_M * 256) * fs = RANS_L * fs / ANS_M * 256
    // But we work with integer arithmetic: limit = (RANS_L >> SCALE_BITS) * fs * 256
    // NOTE: RANS_L >> SCALE_BITS = 1, so limit = fs * 256 = fs << 8
    uint32_t x_max = ((RANS_L >> SCALE_BITS) * 256u) * fs; // = fs * 256
    while (x >= x_max) {
      if (nbytes >= cap) {
        cap = cap * 2 + 8;
        uint8_t *nb = realloc(buf, cap);
        if (!nb) { free(buf); return NULL; }
        buf = nb;
      }
      buf[nbytes++] = (uint8_t)(x & 0xFF);
      x >>= 8;
    }
    x = (x / fs) * ANS_M + cs + (x % fs);
  }

  // Flush final 4-byte state.
  if (nbytes + 4 > cap) {
    cap = nbytes + 4;
    uint8_t *nb = realloc(buf, cap);
    if (!nb) { free(buf); return NULL; }
    buf = nb;
  }
  buf[nbytes++] = (uint8_t)(x);
  buf[nbytes++] = (uint8_t)(x >> 8);
  buf[nbytes++] = (uint8_t)(x >> 16);
  buf[nbytes++] = (uint8_t)(x >> 24);

  // Reverse so decoder reads left-to-right.
  for (size_t a = 0, b = nbytes - 1; a < b; a++, b--) {
    uint8_t tmp = buf[a]; buf[a] = buf[b]; buf[b] = tmp;
  }

  *out_len = nbytes;
  return buf;
}

// ---------------------------------------------------------------------------
// rANS decode
// ---------------------------------------------------------------------------

uint8_t *ans_decode(const ans_table *t, const uint8_t *src, size_t src_len, size_t orig_len) {
  if (src_len < 4) return NULL;
  uint8_t *out = malloc(orig_len ? orig_len : 1);
  if (!out) return NULL;

  // Read initial state (big-endian, because we reversed the stream on encode).
  uint32_t x   = ((uint32_t)src[0] << 24) | ((uint32_t)src[1] << 16)
               | ((uint32_t)src[2] <<  8) |  (uint32_t)src[3];
  size_t   pos = 4;

  for (size_t i = 0; i < orig_len; i++) {
    uint32_t slot = x & (ANS_M - 1);
    uint8_t  s    = t->sym_of_slot[slot];
    out[i]        = s;

    uint32_t fs   = t->freq[s];
    uint32_t cs   = t->cumul[s];
    x = fs * (x >> SCALE_BITS) + slot - cs;

    // Renorm: refill until x is back in [RANS_L, RANS_L*256).
    while (x < RANS_L && pos < src_len)
      x = (x << 8) | src[pos++];
  }

  return out;
}

// ---------------------------------------------------------------------------
// Lanczos-3 helpers
// ---------------------------------------------------------------------------

static float sinc_f(float x) {
  if (fabsf(x) < 1e-6f) return 1.0f;
  float px = (float)M_PI * x;
  return sinf(px) / px;
}

static float lanczos3_w(float x) {
  if (fabsf(x) >= 3.0f) return 0.0f;
  return sinc_f(x) * sinc_f(x / 3.0f);
}

static void lanczos3_upsample1d(const float *src, size_t n, float *dst) {
  for (size_t i = 0; i < 2 * n; i++) {
    float src_pos = (float)i * 0.5f;
    float sum = 0.0f, wsum = 0.0f;
    int   base = (int)floorf(src_pos);
    for (int k = -2; k <= 3; k++) {
      int j = base + k;
      if (j < 0) j = 0;
      if (j >= (int)n) j = (int)n - 1;
      float w = lanczos3_w(src_pos - (float)j);
      sum  += src[j] * w;
      wsum += w;
    }
    dst[i] = (wsum > 1e-6f) ? sum / wsum : 0.0f;
  }
}

void lanczos3_upsample3d(const float *src, size_t dx, size_t dy, size_t dz, float *dst) {
  size_t dx2 = dx * 2, dy2 = dy * 2, dz2 = dz * 2;
  float *tmp1 = malloc(dz  * dy  * dx2 * sizeof(float));
  float *tmp2 = malloc(dz  * dy2 * dx2 * sizeof(float));
  if (!tmp1 || !tmp2) goto cleanup;

  for (size_t z = 0; z < dz; z++)
    for (size_t y = 0; y < dy; y++)
      lanczos3_upsample1d(src + z*dy*dx + y*dx, dx, tmp1 + z*dy*dx2 + y*dx2);

  {
    float *col = malloc(dy * sizeof(float)), *col2 = malloc(dy2 * sizeof(float));
    if (!col || !col2) { free(col); free(col2); goto cleanup; }
    for (size_t z = 0; z < dz; z++)
      for (size_t x = 0; x < dx2; x++) {
        for (size_t y = 0; y < dy;  y++) col[y]  = tmp1[z*dy*dx2 + y*dx2 + x];
        lanczos3_upsample1d(col, dy, col2);
        for (size_t y = 0; y < dy2; y++) tmp2[z*dy2*dx2 + y*dx2 + x] = col2[y];
      }
    free(col); free(col2);
  }

  {
    float *col = malloc(dz * sizeof(float)), *col2 = malloc(dz2 * sizeof(float));
    if (!col || !col2) { free(col); free(col2); goto cleanup; }
    for (size_t y = 0; y < dy2; y++)
      for (size_t x = 0; x < dx2; x++) {
        for (size_t z = 0; z < dz;  z++) col[z]  = tmp2[z*dy2*dx2 + y*dx2 + x];
        lanczos3_upsample1d(col, dz, col2);
        for (size_t z = 0; z < dz2; z++) dst[z*dy2*dx2 + y*dx2 + x] = col2[z];
      }
    free(col); free(col2);
  }

cleanup:
  free(tmp1);
  free(tmp2);
}

// ---------------------------------------------------------------------------
// Residual encode / decode
// ---------------------------------------------------------------------------

static uint8_t quantise(float v, float scale) {
  float q = v / scale;
  if (q < -127.0f) q = -127.0f;
  if (q >  127.0f) q =  127.0f;
  return (uint8_t)((int)roundf(q) + 128);
}

static float dequantise(uint8_t b, float scale) {
  return (float)((int8_t)(b - 128)) * scale;
}

uint8_t *compress4d_encode_residual(const float *residual, size_t len,
                                    float scale, size_t *out_len) {
  uint8_t *quant = malloc(len);
  if (!quant) return NULL;
  for (size_t i = 0; i < len; i++) quant[i] = quantise(residual[i], scale);

  ans_table *t = ans_table_build(quant, len);
  if (!t) { free(quant); return NULL; }

  size_t   hdr  = ANS_NSYM * sizeof(uint16_t);
  size_t   elen = 0;
  uint8_t *enc  = ans_encode(t, quant, len, &elen);
  free(quant);
  if (!enc) { ans_table_free(t); return NULL; }

  uint8_t *out = malloc(hdr + elen);
  if (!out) { free(enc); ans_table_free(t); return NULL; }

  uint16_t freqs[ANS_NSYM];
  ans_table_get_freqs(t, freqs);
  memcpy(out, freqs, hdr);
  memcpy(out + hdr, enc, elen);
  free(enc);
  ans_table_free(t);
  *out_len = hdr + elen;
  return out;
}

bool compress4d_decode_residual(const uint8_t *src, size_t src_len,
                                size_t len, float scale, float *out) {
  size_t hdr = ANS_NSYM * sizeof(uint16_t);
  if (src_len < hdr) return false;

  uint16_t freqs[ANS_NSYM];
  memcpy(freqs, src, hdr);
  ans_table *t = ans_table_from_freqs(freqs);
  if (!t) return false;

  uint8_t *quant = ans_decode(t, src + hdr, src_len - hdr, len);
  ans_table_free(t);
  if (!quant) return false;

  for (size_t i = 0; i < len; i++) out[i] = dequantise(quant[i], scale);
  free(quant);
  return true;
}
