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

// ---------------------------------------------------------------------------
// Pyramid encode / decode
//
// Stream format (C4D v1):
//
//   Header (13 bytes):
//     magic[4]       : "C4D\0"
//     version        : uint8  (1)
//     num_levels     : uint8
//     quant_step     : float32 (little-endian)
//     chunk_shape[3] : 3x int32 (little-endian)
//
//   Per-level block (coarsest level first, i.e. levels[num_levels-1] .. levels[0]):
//     level_idx      : uint8
//     shape[3]       : 3x int64 (little-endian)
//     num_chunks     : uint32  (number of spatial chunks for this level)
//     Per-chunk:
//       compressed_size : uint64 (little-endian)
//       freq[256]       : 256x uint16 (ANS frequencies, little-endian)
//       data            : <compressed_size bytes>
//
// levels[0] is finest (full resolution), levels[num_levels-1] is coarsest.
// The stream stores levels coarsest-first so progressive decoders can stop early.
// ---------------------------------------------------------------------------

#define C4D_MAGIC "C4D\0"
#define C4D_VERSION 1

typedef struct {
  int num_levels;          // pyramid depth
  float quant_step;        // quantisation step size
  int32_t chunk_shape[3];  // spatial tile shape for independent chunks
} compress4d_params;

compress4d_params compress4d_params_default(void);

// Encode a complete pyramid.
// levels[i]: float volume for level i (levels[0]=finest, levels[num_levels-1]=coarsest)
// shapes[i]: {depth, height, width} for level i
// Returns the compressed stream (caller frees) or NULL on failure.
uint8_t *compress4d_encode_pyramid(const float *const *levels,
                                    const int64_t (*shapes)[3],
                                    int num_levels,
                                    compress4d_params params,
                                    size_t *out_size);

// Decode a single level from the stream.
// target_level=0 = finest, target_level=num_levels-1 = coarsest.
// out_shape receives {depth, height, width}.  Caller frees returned buffer.
float *compress4d_decode_level(const uint8_t *stream, size_t stream_len,
                                int target_level, int64_t out_shape[3]);

// ---------------------------------------------------------------------------
// Streaming / progressive decoder
// ---------------------------------------------------------------------------

typedef struct compress4d_decoder compress4d_decoder;

// Create a decoder for the given stream (does not copy; stream must outlive decoder).
compress4d_decoder *compress4d_decoder_new(const uint8_t *stream, size_t len);

// Decode the next level (coarsest-first).
// On success: *out_data = malloc'd float array (caller frees), *out_shape and
// *out_level are set.  Returns true while levels remain, false when done.
bool compress4d_decoder_next(compress4d_decoder *d, float **out_data,
                              int64_t out_shape[3], int *out_level);

void compress4d_decoder_free(compress4d_decoder *d);

// ---------------------------------------------------------------------------
// Zarr v3 codec registration
//
// Registers "compress4d" as a named codec so vol.c can look it up by id.
// Must be called once at startup (idempotent).
// ---------------------------------------------------------------------------
void compress4d_register_zarr_codec(void);

// Codec id string used in zarr v3 metadata.
#define COMPRESS4D_ZARR_CODEC_ID "compress4d"
