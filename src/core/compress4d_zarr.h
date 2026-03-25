#pragma once
#include <stddef.h>
#include <stdint.h>
#include "core/json.h"

// Register "compress4d" as a recognized codec in our zarr reader.
// Must be called once before vol_open; safe to call multiple times.
void compress4d_zarr_register(void);

// Decode a compress4d-encoded chunk.  config may be NULL (uses defaults).
// Returns malloc'd buffer of *out_len bytes; caller must free.  NULL on failure.
uint8_t *compress4d_zarr_decode(const uint8_t *data, size_t len, size_t *out_len,
                                const json_value *config);

// Encode raw bytes using compress4d residual coding.  config may be NULL.
// Returns malloc'd buffer of *out_len bytes; caller must free.  NULL on failure.
uint8_t *compress4d_zarr_encode(const uint8_t *data, size_t len, size_t *out_len,
                                const json_value *config);
