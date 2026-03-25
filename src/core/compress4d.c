#include "core/compress4d.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

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

// ---------------------------------------------------------------------------
// Little-endian helpers
// ---------------------------------------------------------------------------

static void write_u8(uint8_t **p, uint8_t v)   { **p = v; (*p)++; }
static void write_u16le(uint8_t **p, uint16_t v) {
  (*p)[0] = (uint8_t)(v);
  (*p)[1] = (uint8_t)(v >> 8);
  *p += 2;
}
static void write_u32le(uint8_t **p, uint32_t v) {
  (*p)[0] = (uint8_t)(v);
  (*p)[1] = (uint8_t)(v >> 8);
  (*p)[2] = (uint8_t)(v >> 16);
  (*p)[3] = (uint8_t)(v >> 24);
  *p += 4;
}
static void write_i32le(uint8_t **p, int32_t v)  { write_u32le(p, (uint32_t)v); }
static void write_u64le(uint8_t **p, uint64_t v) {
  write_u32le(p, (uint32_t)(v & 0xFFFFFFFFu));
  write_u32le(p, (uint32_t)(v >> 32));
}
static void write_i64le(uint8_t **p, int64_t v)  { write_u64le(p, (uint64_t)v); }
static void write_f32le(uint8_t **p, float v) {
  uint32_t bits; memcpy(&bits, &v, 4); write_u32le(p, bits);
}

static uint8_t  read_u8(const uint8_t **p)  { return *(*p)++; }
static uint16_t read_u16le(const uint8_t **p) {
  uint16_t v = (uint16_t)((*p)[0]) | ((uint16_t)((*p)[1]) << 8); *p += 2; return v;
}
static uint32_t read_u32le(const uint8_t **p) {
  uint32_t v = (uint32_t)(*p)[0] | ((uint32_t)(*p)[1]<<8) |
               ((uint32_t)(*p)[2]<<16) | ((uint32_t)(*p)[3]<<24);
  *p += 4; return v;
}
static int32_t  read_i32le(const uint8_t **p) { return (int32_t)read_u32le(p); }
static uint64_t read_u64le(const uint8_t **p) {
  uint64_t lo = read_u32le(p), hi = read_u32le(p); return lo | (hi << 32);
}
static int64_t  read_i64le(const uint8_t **p) { return (int64_t)read_u64le(p); }
static float    read_f32le(const uint8_t **p) {
  uint32_t bits = read_u32le(p); float v; memcpy(&v, &bits, 4); return v;
}

// ---------------------------------------------------------------------------
// Growable byte buffer
// ---------------------------------------------------------------------------

typedef struct {
  uint8_t *data;
  size_t   len;
  size_t   cap;
} buf_t;

static bool buf_reserve(buf_t *b, size_t extra) {
  if (b->len + extra <= b->cap) return true;
  size_t nc = (b->cap == 0) ? 4096 : b->cap * 2;
  while (nc < b->len + extra) nc *= 2;
  uint8_t *nd = realloc(b->data, nc);
  if (!nd) return false;
  b->data = nd; b->cap = nc;
  return true;
}

static bool buf_write(buf_t *b, const void *src, size_t n) {
  if (!buf_reserve(b, n)) return false;
  memcpy(b->data + b->len, src, n);
  b->len += n;
  return true;
}

// Write a single byte placeholder, return its index (for patching).
static size_t buf_placeholder_u64(buf_t *b) {
  size_t off = b->len;
  uint8_t zero[8] = {0};
  buf_write(b, zero, 8);
  return off;
}

static void buf_patch_u64le(buf_t *b, size_t off, uint64_t v) {
  uint8_t *p = b->data + off;
  write_u64le(&p, v);
}

// ---------------------------------------------------------------------------
// Spatial chunking helpers
// ---------------------------------------------------------------------------

// Number of chunks along one dimension given volume size and chunk size.
static int64_t n_chunks_dim(int64_t vol, int64_t cs) {
  return (vol + cs - 1) / cs;
}

// Encode a single spatial chunk as residual + ANS.
// prediction: Lanczos upsampled base (or NULL for coarsest level).
// vol: full level float data [depth*height*width] (row-major z,y,x).
// cz,cy,cx: chunk origin. sz,sy,sx: chunk size.
static bool encode_chunk(const float *vol, const float *prediction,
                          int64_t D, int64_t H, int64_t W,
                          int64_t cz, int64_t cy, int64_t cx,
                          int64_t sz, int64_t sy, int64_t sx,
                          float scale, buf_t *out) {
  size_t n = (size_t)(sz * sy * sx);
  float *residual = malloc(n * sizeof(float));
  if (!residual) return false;

  size_t k = 0;
  for (int64_t z = cz; z < cz + sz; z++)
    for (int64_t y = cy; y < cy + sy; y++)
      for (int64_t x = cx; x < cx + sx; x++) {
        float v = vol[((size_t)z * (size_t)H + (size_t)y) * (size_t)W + (size_t)x];
        float p = prediction
          ? prediction[((size_t)z * (size_t)H + (size_t)y) * (size_t)W + (size_t)x]
          : 0.0f;
        residual[k++] = v - p;
      }

  size_t enc_len = 0;
  uint8_t *enc = compress4d_encode_residual(residual, n, scale, &enc_len);
  free(residual);
  if (!enc) return false;

  // Write: compressed_size (u64), freq[256] (u16 each) already embedded in enc
  // (compress4d_encode_residual prepends the freq table), so enc = freqs+data.
  // We need to split them for the format: freq[256] u16, then data.
  // Actually: use the raw enc blob directly — store its length then blob.
  // The decoder knows the freq table is the first 256*2 bytes.
  size_t patch_off = out->len;
  uint8_t *pw = out->data + out->len;
  (void)pw;
  // reserve space for compressed_size
  size_t csz_off = buf_placeholder_u64(out);
  // write the encoded blob (contains freq table + ANS data)
  if (!buf_write(out, enc, enc_len)) { free(enc); return false; }
  free(enc);
  // patch the size
  buf_patch_u64le(out, csz_off, (uint64_t)enc_len);
  (void)patch_off;
  return true;
}

// ---------------------------------------------------------------------------
// compress4d_params_default
// ---------------------------------------------------------------------------

compress4d_params compress4d_params_default(void) {
  return (compress4d_params){
    .num_levels  = 5,
    .quant_step  = 1.0f / 255.0f,
    .chunk_shape = {64, 64, 64},
  };
}

// ---------------------------------------------------------------------------
// compress4d_encode_pyramid
// ---------------------------------------------------------------------------

uint8_t *compress4d_encode_pyramid(const float *const *levels,
                                    const int64_t (*shapes)[3],
                                    int num_levels,
                                    compress4d_params params,
                                    size_t *out_size) {
  if (!levels || !shapes || num_levels < 1 || !out_size) return NULL;

  buf_t buf = {0};

  // --- Header ---
  // magic "C4D\0"
  buf_write(&buf, C4D_MAGIC, 4);
  {
    uint8_t *p = buf.data + buf.len;
    // version, num_levels
    if (!buf_reserve(&buf, 2 + 4 + 12)) goto fail;
    buf.data[buf.len++] = C4D_VERSION;
    buf.data[buf.len++] = (uint8_t)num_levels;
    p = buf.data + buf.len;
    write_f32le(&p, params.quant_step);
    write_i32le(&p, params.chunk_shape[0]);
    write_i32le(&p, params.chunk_shape[1]);
    write_i32le(&p, params.chunk_shape[2]);
    buf.len = (size_t)(p - buf.data);
  }

  // --- Levels: coarsest first ---
  for (int li = num_levels - 1; li >= 0; li--) {
    int64_t D = shapes[li][0], H = shapes[li][1], W = shapes[li][2];

    // Build prediction (upsampled coarser level) for levels > coarsest.
    float *prediction = NULL;
    if (li < num_levels - 1) {
      int64_t Dc = shapes[li+1][0], Hc = shapes[li+1][1], Wc = shapes[li+1][2];
      prediction = malloc((size_t)(D * H * W) * sizeof(float));
      if (!prediction) goto fail;
      lanczos3_upsample3d(levels[li+1], (size_t)Wc, (size_t)Hc, (size_t)Dc, prediction);
      // Clamp prediction to volume size (upsample doubles dimensions; may be larger)
      // lanczos3_upsample3d outputs Dc*2 x Hc*2 x Wc*2 which may be > D,H,W
      // if shapes[li] < 2*shapes[li+1]. We already allocated D*H*W, so the
      // prediction ptr is sized correctly. But upsample writes (Dc*2)*(Hc*2)*(Wc*2).
      // Reallocate to full upsampled size, then trim to (D,H,W) if needed.
      int64_t Du = Dc*2, Hu = Hc*2, Wu = Wc*2;
      if (Du != D || Hu != H || Wu != W) {
        // Re-run into a properly sized buffer
        float *full = malloc((size_t)(Du * Hu * Wu) * sizeof(float));
        if (!full) { free(prediction); goto fail; }
        lanczos3_upsample3d(levels[li+1], (size_t)Wc, (size_t)Hc, (size_t)Dc, full);
        // Copy crop into prediction (already sized D*H*W)
        for (int64_t z = 0; z < D; z++)
          for (int64_t y = 0; y < H; y++)
            for (int64_t x = 0; x < W; x++)
              prediction[((size_t)z*(size_t)H+(size_t)y)*(size_t)W+(size_t)x] =
                full[((size_t)z*(size_t)Hu+(size_t)y)*(size_t)Wu+(size_t)x];
        free(full);
      } else {
        // Re-run directly into prediction
        lanczos3_upsample3d(levels[li+1], (size_t)Wc, (size_t)Hc, (size_t)Dc, prediction);
      }
    }

    // Level block header
    {
      uint8_t tmp[1 + 3*8 + 4];
      uint8_t *p = tmp;
      write_u8(&p, (uint8_t)li);
      write_i64le(&p, D);
      write_i64le(&p, H);
      write_i64le(&p, W);
      // num_chunks placeholder — write after we know it
      int64_t csz = params.chunk_shape[0], csy = params.chunk_shape[1], csx = params.chunk_shape[2];
      if (csz <= 0) csz = D; if (csy <= 0) csy = H; if (csx <= 0) csx = W;
      int64_t ncz = n_chunks_dim(D, csz), ncy = n_chunks_dim(H, csy), ncx = n_chunks_dim(W, csx);
      uint32_t num_chunks = (uint32_t)(ncz * ncy * ncx);
      write_u32le(&p, num_chunks);
      buf_write(&buf, tmp, (size_t)(p - tmp));

      // Encode each chunk
      for (int64_t iz = 0; iz < ncz; iz++) {
        for (int64_t iy = 0; iy < ncy; iy++) {
          for (int64_t ix = 0; ix < ncx; ix++) {
            int64_t oz = iz * csz, oy = iy * csy, ox = ix * csx;
            int64_t esz = (oz + csz <= D) ? csz : D - oz;
            int64_t esy = (oy + csy <= H) ? csy : H - oy;
            int64_t esx = (ox + csx <= W) ? csx : W - ox;
            if (!encode_chunk(levels[li], prediction, D, H, W,
                               oz, oy, ox, esz, esy, esx,
                               params.quant_step, &buf)) {
              free(prediction);
              goto fail;
            }
          }
        }
      }
    }

    free(prediction);
  }

  *out_size = buf.len;
  return buf.data;

fail:
  free(buf.data);
  return NULL;
}

// ---------------------------------------------------------------------------
// Stream parsing helpers
// ---------------------------------------------------------------------------

// Header size: magic(4) + version(1) + num_levels(1) + quant_step(4) + chunk_shape(12) = 22
#define C4D_HEADER_SIZE 22

typedef struct {
  int      level_idx;
  int64_t  shape[3];
  uint32_t num_chunks;
  const uint8_t *chunks_start;  // pointer into the original stream at first chunk
  size_t   total_bytes;         // total bytes for all chunks in this level
} level_index_entry;

// Parse the stream header and build a level index table.
// Returns false on malformed stream.
// entries must be pre-allocated to num_levels size.
static bool parse_stream(const uint8_t *stream, size_t stream_len,
                          int *out_num_levels, float *out_quant_step,
                          int32_t chunk_shape[3],
                          level_index_entry *entries, int max_entries) {
  if (stream_len < C4D_HEADER_SIZE) return false;
  if (memcmp(stream, C4D_MAGIC, 4) != 0) return false;

  const uint8_t *p = stream + 4;
  if (read_u8(&p) != C4D_VERSION) return false;
  int nl = (int)read_u8(&p);
  if (nl < 1 || nl > max_entries) return false;
  *out_num_levels = nl;
  *out_quant_step = read_f32le(&p);
  chunk_shape[0] = read_i32le(&p);
  chunk_shape[1] = read_i32le(&p);
  chunk_shape[2] = read_i32le(&p);

  const uint8_t *end = stream + stream_len;
  for (int i = 0; i < nl; i++) {
    if (p + 1 + 3*8 + 4 > end) return false;
    entries[i].level_idx = (int)read_u8(&p);
    entries[i].shape[0] = read_i64le(&p);
    entries[i].shape[1] = read_i64le(&p);
    entries[i].shape[2] = read_i64le(&p);
    entries[i].num_chunks = read_u32le(&p);
    entries[i].chunks_start = p;

    // Skip over all chunk data
    for (uint32_t c = 0; c < entries[i].num_chunks; c++) {
      if (p + 8 > end) return false;
      uint64_t csz = read_u64le(&p);
      if (p + csz > end) return false;
      p += csz;
    }
    entries[i].total_bytes = (size_t)(p - entries[i].chunks_start);
  }
  return true;
}

// Decode all chunks of a level into a freshly allocated float array.
// base: optional upsampled prediction of same shape (may be NULL for coarsest).
static float *decode_level_data(const level_index_entry *e, float *base,
                                float quant_step,
                                int32_t cs0, int32_t cs1, int32_t cs2) {
  int64_t D = e->shape[0], H = e->shape[1], W = e->shape[2];
  size_t n = (size_t)(D * H * W);
  float *out = malloc(n * sizeof(float));
  if (!out) return NULL;

  int32_t csz = cs0 > 0 ? cs0 : (int32_t)D;
  int32_t csy = cs1 > 0 ? cs1 : (int32_t)H;
  int32_t csx = cs2 > 0 ? cs2 : (int32_t)W;

  int64_t ncz = n_chunks_dim(D, csz), ncy = n_chunks_dim(H, csy), ncx = n_chunks_dim(W, csx);

  const uint8_t *p = e->chunks_start;
  const uint8_t *end = p + e->total_bytes;

  for (int64_t iz = 0; iz < ncz; iz++) {
    for (int64_t iy = 0; iy < ncy; iy++) {
      for (int64_t ix = 0; ix < ncx; ix++) {
        if (p + 8 > end) { free(out); return NULL; }
        uint64_t enc_len = read_u64le(&p);
        if (p + enc_len > end) { free(out); return NULL; }

        int64_t oz = iz * csz, oy = iy * csy, ox = ix * csx;
        int64_t esz = (oz + csz <= D) ? csz : D - oz;
        int64_t esy = (oy + csy <= H) ? csy : H - oy;
        int64_t esx = (ox + csx <= W) ? csx : W - ox;
        size_t cn = (size_t)(esz * esy * esx);

        float *residual = malloc(cn * sizeof(float));
        if (!residual) { free(out); return NULL; }
        if (!compress4d_decode_residual(p, (size_t)enc_len, cn, quant_step, residual)) {
          free(residual); free(out); return NULL;
        }
        p += enc_len;

        // Add prediction and scatter into output
        size_t k = 0;
        for (int64_t z = oz; z < oz + esz; z++)
          for (int64_t y = oy; y < oy + esy; y++)
            for (int64_t x = ox; x < ox + esx; x++) {
              size_t idx = ((size_t)z * (size_t)H + (size_t)y) * (size_t)W + (size_t)x;
              float pred = base ? base[idx] : 0.0f;
              out[idx] = residual[k++] + pred;
            }
        free(residual);
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// compress4d_decode_level
// ---------------------------------------------------------------------------

float *compress4d_decode_level(const uint8_t *stream, size_t stream_len,
                                int target_level, int64_t out_shape[3]) {
  if (!stream || stream_len == 0) return NULL;

  int nl = 0;
  float quant_step = 0.0f;
  int32_t cs[3] = {0};
  level_index_entry entries[64];
  if (!parse_stream(stream, stream_len, &nl, &quant_step, cs, entries, 64))
    return NULL;
  if (target_level < 0 || target_level >= nl) return NULL;

  // Stream is coarsest-first. entries[0] = coarsest (level_idx = nl-1).
  // We need to decode from coarsest up to target_level.
  // Find which stream entry corresponds to each level:
  //   entries[i].level_idx gives the logical level index.

  // Build a map: logical_level -> stream entry index
  int entry_for_level[64];
  for (int i = 0; i < nl; i++) entry_for_level[entries[i].level_idx] = i;

  float *current = NULL;

  // Decode from coarsest (nl-1) down to target_level
  for (int li = nl - 1; li >= target_level; li--) {
    int ei = entry_for_level[li];
    float *base = current; // upsampled from previous (finer) decode? No — base is coarser.
    // Actually we decode coarsest first, so:
    //   li = nl-1: no base (coarsest)
    //   li = nl-2: base = upsample(decode of li=nl-1)
    //   etc.

    float *next_base = NULL;
    if (current != NULL) {
      // Upsample current (which is level li+1) to level li's shape
      int64_t D = entries[ei].shape[0], H = entries[ei].shape[1], W = entries[ei].shape[2];
      int prev_ei = entry_for_level[li + 1];
      int64_t Dc = entries[prev_ei].shape[0], Hc = entries[prev_ei].shape[1], Wc = entries[prev_ei].shape[2];
      int64_t Du = Dc*2, Hu = Hc*2, Wu = Wc*2;

      if (Du == D && Hu == H && Wu == W) {
        next_base = malloc((size_t)(D * H * W) * sizeof(float));
        if (!next_base) { free(current); return NULL; }
        lanczos3_upsample3d(current, (size_t)Wc, (size_t)Hc, (size_t)Dc, next_base);
      } else {
        float *full = malloc((size_t)(Du * Hu * Wu) * sizeof(float));
        if (!full) { free(current); return NULL; }
        lanczos3_upsample3d(current, (size_t)Wc, (size_t)Hc, (size_t)Dc, full);
        next_base = malloc((size_t)(D * H * W) * sizeof(float));
        if (!next_base) { free(full); free(current); return NULL; }
        for (int64_t z = 0; z < D; z++)
          for (int64_t y = 0; y < H; y++)
            for (int64_t x = 0; x < W; x++)
              next_base[((size_t)z*(size_t)H+(size_t)y)*(size_t)W+(size_t)x] =
                full[((size_t)z*(size_t)Hu+(size_t)y)*(size_t)Wu+(size_t)x];
        free(full);
      }
      free(current);
      base = next_base;
    }

    current = decode_level_data(&entries[ei], base, quant_step, cs[0], cs[1], cs[2]);
    free(base);
    if (!current) return NULL;
  }

  if (out_shape) {
    int ei = entry_for_level[target_level];
    out_shape[0] = entries[ei].shape[0];
    out_shape[1] = entries[ei].shape[1];
    out_shape[2] = entries[ei].shape[2];
  }
  return current;
}

// ---------------------------------------------------------------------------
// Streaming decoder
// ---------------------------------------------------------------------------

struct compress4d_decoder {
  const uint8_t   *stream;
  size_t           stream_len;
  int              num_levels;
  float            quant_step;
  int32_t          cs[3];
  level_index_entry entries[64];
  int              next_stream_idx;  // index into entries[] (0 = coarsest)
  float           *last_decoded;     // last decoded data (to build prediction for next)
  int              last_level_idx;   // logical level index of last_decoded
};

compress4d_decoder *compress4d_decoder_new(const uint8_t *stream, size_t len) {
  if (!stream || len == 0) return NULL;
  compress4d_decoder *d = calloc(1, sizeof(*d));
  if (!d) return NULL;
  d->stream = stream;
  d->stream_len = len;
  d->next_stream_idx = 0;
  d->last_decoded = NULL;
  d->last_level_idx = -1;
  if (!parse_stream(stream, len, &d->num_levels, &d->quant_step,
                    d->cs, d->entries, 64)) {
    free(d);
    return NULL;
  }
  return d;
}

bool compress4d_decoder_next(compress4d_decoder *d, float **out_data,
                              int64_t out_shape[3], int *out_level) {
  if (!d || d->next_stream_idx >= d->num_levels) return false;

  int ei = d->next_stream_idx;
  level_index_entry *e = &d->entries[ei];
  int li = e->level_idx;

  // Build prediction from last decoded level if available
  float *base = NULL;
  if (d->last_decoded != NULL) {
    // last_decoded is level (li+1) (one step coarser); upsample it
    int prev_ei = -1;
    for (int i = 0; i < d->num_levels; i++)
      if (d->entries[i].level_idx == d->last_level_idx) { prev_ei = i; break; }

    if (prev_ei >= 0) {
      int64_t D = e->shape[0], H = e->shape[1], W = e->shape[2];
      int64_t Dc = d->entries[prev_ei].shape[0], Hc = d->entries[prev_ei].shape[1], Wc = d->entries[prev_ei].shape[2];
      int64_t Du = Dc*2, Hu = Hc*2, Wu = Wc*2;
      if (Du == D && Hu == H && Wu == W) {
        base = malloc((size_t)(D * H * W) * sizeof(float));
        if (!base) return false;
        lanczos3_upsample3d(d->last_decoded, (size_t)Wc, (size_t)Hc, (size_t)Dc, base);
      } else {
        float *full = malloc((size_t)(Du * Hu * Wu) * sizeof(float));
        if (!full) return false;
        lanczos3_upsample3d(d->last_decoded, (size_t)Wc, (size_t)Hc, (size_t)Dc, full);
        base = malloc((size_t)(D * H * W) * sizeof(float));
        if (!base) { free(full); return false; }
        for (int64_t z = 0; z < D; z++)
          for (int64_t y = 0; y < H; y++)
            for (int64_t x = 0; x < W; x++)
              base[((size_t)z*(size_t)H+(size_t)y)*(size_t)W+(size_t)x] =
                full[((size_t)z*(size_t)Hu+(size_t)y)*(size_t)Wu+(size_t)x];
        free(full);
      }
    }
  }

  float *decoded = decode_level_data(e, base, d->quant_step, d->cs[0], d->cs[1], d->cs[2]);
  free(base);
  if (!decoded) return false;

  free(d->last_decoded);
  d->last_decoded = decoded;
  d->last_level_idx = li;
  d->next_stream_idx++;

  *out_data = decoded;
  if (out_shape) {
    out_shape[0] = e->shape[0];
    out_shape[1] = e->shape[1];
    out_shape[2] = e->shape[2];
  }
  if (out_level) *out_level = li;
  return true;
}

void compress4d_decoder_free(compress4d_decoder *d) {
  if (!d) return;
  free(d->last_decoded);
  free(d);
}

// ---------------------------------------------------------------------------
// Zarr v3 codec registration (stub — actual dispatch lives in vol.c)
// ---------------------------------------------------------------------------

static bool g_c4d_codec_registered = false;

void compress4d_register_zarr_codec(void) {
  if (g_c4d_codec_registered) return;
  // In a full implementation this would register a codec_vtable into the
  // global zarr codec registry in vol.c.  For now we set the flag so callers
  // can check availability and the codec id COMPRESS4D_ZARR_CODEC_ID can be
  // embedded in zarr v3 .zarray metadata by vol_create / vol_write_chunk.
  g_c4d_codec_registered = true;
}
