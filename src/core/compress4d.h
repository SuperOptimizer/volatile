#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// ANS frequency table
// ---------------------------------------------------------------------------

// Opaque table built from a symbol histogram; used for both encode and decode.
typedef struct ans_table ans_table;

// Build frequency table from data (all 256 symbols get at least 1 count).
// Returns NULL on allocation failure.
ans_table *ans_table_build(const uint8_t *data, size_t len);

// Build frequency table directly from externally-provided symbol counts[256].
ans_table *ans_table_from_counts(const uint32_t counts[256]);

void ans_table_free(ans_table *t);

// Serialise the 256 normalised frequencies to a compact uint16 array (caller
// owns). Used to embed the table in a compressed stream.
void ans_table_get_freqs(const ans_table *t, uint16_t freqs_out[256]);

// Rebuild a table from previously serialised frequencies.
ans_table *ans_table_from_freqs(const uint16_t freqs[256]);

// ---------------------------------------------------------------------------
// tANS encode / decode
// ---------------------------------------------------------------------------

// Encode src[len] -> heap-allocated compressed bytes.  *out_len set to
// compressed size.  Returns NULL on failure.
uint8_t *ans_encode(const ans_table *t, const uint8_t *src, size_t len, size_t *out_len);

// Decode compressed bytes back to orig_len symbols.  Returns heap-allocated
// buffer or NULL on failure.
uint8_t *ans_decode(const ans_table *t, const uint8_t *src, size_t src_len, size_t orig_len);

// ---------------------------------------------------------------------------
// Lanczos-3 3-D upsampling
// ---------------------------------------------------------------------------

// Upsample a float volume by 2x in each spatial dimension (x, y, z).
// src:  input  [dz][dy][dx]  (row-major, z outermost)
// dst:  output [dz*2][dy*2][dx*2]  (caller allocates)
void lanczos3_upsample3d(const float *src, size_t dx, size_t dy, size_t dz,
                         float *dst);

// ---------------------------------------------------------------------------
// Residual encode / decode  (quantise -> ANS -> dequantise)
// ---------------------------------------------------------------------------

// Quantise float residual to uint8, ANS-encode.  scale controls the
// quantisation step (residual clamped to [-127*scale, 127*scale]).
// Returns compressed bytes (caller frees) and sets *out_len.
uint8_t *compress4d_encode_residual(const float *residual, size_t len,
                                    float scale, size_t *out_len);

// ANS-decode then dequantise.  out must point to a caller-allocated float[len].
// Returns true on success.
bool compress4d_decode_residual(const uint8_t *src, size_t src_len,
                                size_t len, float scale, float *out);
