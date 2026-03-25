#include "core/compress4d_zarr.h"
#include "core/compress4d.h"
#include "core/json.h"
#include "core/log.h"

#include <assert.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Wire format
//   [4]  magic  "C4DR"
//   [4]  n_floats   (little-endian uint32) — number of float32 elements
//   [4]  scale_bits (little-endian uint32, reinterpreted as float32)
//   [N]  ANS residual payload (compress4d_encode_residual output)
// ---------------------------------------------------------------------------

#define MAGIC "C4DR"
#define MAGIC_LEN 4
#define HEADER_LEN (MAGIC_LEN + 4 + 4)  // 12 bytes

static float default_scale(const json_value *config) {
  if (!config) return 1.0f;
  const json_value *q = json_object_get(config, "quality");
  if (!q) return 1.0f;
  double v = json_get_number(q, 1.0);
  // quality in [0,1]: quality==1 → fine-grain (small scale), quality→0 → coarser
  // scale = 1/quality clamped to [0.01, 100]
  if (v <= 0.0) v = 1.0;
  float s = (float)(1.0 / v);
  if (s < 0.01f) s = 0.01f;
  if (s > 100.0f) s = 100.0f;
  return s;
}

// ---------------------------------------------------------------------------

static _Atomic int g_registered = 0;

void compress4d_zarr_register(void) {
  int was = atomic_exchange(&g_registered, 1);
  if (!was) LOG_INFO("compress4d zarr codec registered");
}

// ---------------------------------------------------------------------------

uint8_t *compress4d_zarr_encode(const uint8_t *data, size_t len, size_t *out_len,
                                const json_value *config) {
  if (!data || len == 0 || len % sizeof(float) != 0) {
    LOG_WARN("compress4d_zarr_encode: len=%zu is not a multiple of 4", len);
    return NULL;
  }

  size_t n_floats = len / sizeof(float);
  float scale = default_scale(config);

  const float *fdata = (const float *)data;
  size_t payload_len = 0;
  uint8_t *payload = compress4d_encode_residual(fdata, n_floats, scale, &payload_len);
  if (!payload) return NULL;

  size_t total = HEADER_LEN + payload_len;
  uint8_t *out = malloc(total);
  if (!out) { free(payload); return NULL; }

  // write header
  memcpy(out, MAGIC, MAGIC_LEN);
  uint32_t nf = (uint32_t)n_floats;
  memcpy(out + MAGIC_LEN, &nf, 4);
  uint32_t scale_bits;
  memcpy(&scale_bits, &scale, 4);
  memcpy(out + MAGIC_LEN + 4, &scale_bits, 4);

  memcpy(out + HEADER_LEN, payload, payload_len);
  free(payload);

  if (out_len) *out_len = total;
  return out;
}

uint8_t *compress4d_zarr_decode(const uint8_t *data, size_t len, size_t *out_len,
                                const json_value *config) {
  (void)config;  // scale is embedded in the stream
  if (!data || len < (size_t)HEADER_LEN) {
    LOG_WARN("compress4d_zarr_decode: too short (%zu bytes)", len);
    return NULL;
  }
  if (memcmp(data, MAGIC, MAGIC_LEN) != 0) {
    LOG_WARN("compress4d_zarr_decode: bad magic");
    return NULL;
  }

  uint32_t n_floats;
  memcpy(&n_floats, data + MAGIC_LEN, 4);
  uint32_t scale_bits;
  memcpy(&scale_bits, data + MAGIC_LEN + 4, 4);
  float scale;
  memcpy(&scale, &scale_bits, 4);

  const uint8_t *payload = data + HEADER_LEN;
  size_t payload_len = len - HEADER_LEN;

  float *out = malloc(n_floats * sizeof(float));
  if (!out) return NULL;

  bool ok = compress4d_decode_residual(payload, payload_len, n_floats, scale, out);
  if (!ok) { free(out); return NULL; }

  if (out_len) *out_len = n_floats * sizeof(float);
  return (uint8_t *)out;
}
