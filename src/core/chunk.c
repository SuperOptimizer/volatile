#include "core/chunk.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

// compute ceil(a / b) for positive integers
static inline int64_t ceildiv(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// flat index into the chunk pointer array from N-dim chunk coordinates (row-major)
static size_t chunk_flat_index(const chunked_array *a, const int64_t *chunk_coords) {
  size_t idx = 0;
  for (int d = 0; d < a->ndim; d++) {
    idx = idx * (size_t)a->nchunks[d] + (size_t)chunk_coords[d];
  }
  return idx;
}

// flat offset of an element within its chunk, given the element's local coords inside the chunk.
// local[d] = indices[d] % chunk_shape[d].
// the actual chunk shape may be smaller at the edge; we use the full chunk_shape stride for layout.
static size_t elem_offset_in_chunk(const chunked_array *a, const int64_t *local) {
  size_t off = 0;
  for (int d = 0; d < a->ndim; d++) {
    off = off * (size_t)a->chunk_shape[d] + (size_t)local[d];
  }
  return off;
}

// ---------------------------------------------------------------------------
// lifecycle
// ---------------------------------------------------------------------------

chunked_array *chunked_array_new(int ndim, const int64_t *shape, const int64_t *chunk_shape, size_t elem_size) {
  REQUIRE(ndim >= 1 && ndim <= CHUNK_MAX_NDIM, "ndim must be 1..%d, got %d", CHUNK_MAX_NDIM, ndim);
  REQUIRE(shape != NULL, "shape must not be NULL");
  REQUIRE(chunk_shape != NULL, "chunk_shape must not be NULL");
  REQUIRE(elem_size > 0, "elem_size must be > 0");

  chunked_array *a = malloc(sizeof(chunked_array));
  REQUIRE(a != NULL, "malloc failed");

  a->ndim = ndim;
  a->elem_size = elem_size;

  size_t total_chunks = 1;
  for (int d = 0; d < ndim; d++) {
    REQUIRE(shape[d] > 0, "shape[%d] must be > 0", d);
    REQUIRE(chunk_shape[d] > 0, "chunk_shape[%d] must be > 0", d);
    a->shape[d] = shape[d];
    a->chunk_shape[d] = chunk_shape[d];
    a->nchunks[d] = ceildiv(shape[d], chunk_shape[d]);
    total_chunks *= (size_t)a->nchunks[d];
  }
  a->total_chunks = total_chunks;

  a->chunks = calloc(total_chunks, sizeof(void *));
  REQUIRE(a->chunks != NULL, "calloc failed");

  return a;
}

void chunked_array_free(chunked_array *a) {
  if (!a) return;
  for (size_t i = 0; i < a->total_chunks; i++) {
    free(a->chunks[i]);
  }
  free(a->chunks);
  free(a);
}

// ---------------------------------------------------------------------------
// chunk management
// ---------------------------------------------------------------------------

size_t chunked_array_chunk_index(const chunked_array *a, const int64_t *chunk_coords) {
  return chunk_flat_index(a, chunk_coords);
}

void *chunked_array_get_chunk(const chunked_array *a, const int64_t *chunk_coords) {
  size_t idx = chunk_flat_index(a, chunk_coords);
  return a->chunks[idx];
}

void chunked_array_set_chunk(chunked_array *a, const int64_t *chunk_coords, void *data) {
  size_t idx = chunk_flat_index(a, chunk_coords);
  free(a->chunks[idx]);  // release previous if present
  a->chunks[idx] = data;
}

size_t chunked_array_chunk_bytes(const chunked_array *a) {
  size_t n = a->elem_size;
  for (int d = 0; d < a->ndim; d++) {
    n *= (size_t)a->chunk_shape[d];
  }
  return n;
}

// ---------------------------------------------------------------------------
// lazy chunk allocation (internal)
// ---------------------------------------------------------------------------

// ensure chunk at chunk_coords is allocated; returns pointer to its data
static void *ensure_chunk(chunked_array *a, const int64_t *chunk_coords) {
  size_t idx = chunk_flat_index(a, chunk_coords);
  if (!a->chunks[idx]) {
    a->chunks[idx] = calloc(1, chunked_array_chunk_bytes(a));
    REQUIRE(a->chunks[idx] != NULL, "calloc failed for chunk");
  }
  return a->chunks[idx];
}

// ---------------------------------------------------------------------------
// element access
// ---------------------------------------------------------------------------

void *chunked_array_get_ptr(const chunked_array *a, const int64_t *indices) {
  int64_t chunk_coords[CHUNK_MAX_NDIM];
  int64_t local[CHUNK_MAX_NDIM];
  for (int d = 0; d < a->ndim; d++) {
    chunk_coords[d] = indices[d] / a->chunk_shape[d];
    local[d] = indices[d] % a->chunk_shape[d];
  }
  void *chunk = chunked_array_get_chunk(a, chunk_coords);
  if (!chunk) return NULL;
  size_t off = elem_offset_in_chunk(a, local);
  return (uint8_t *)chunk + off * a->elem_size;
}

// mutable version that allocates the chunk lazily
static void *get_ptr_rw(chunked_array *a, const int64_t *indices) {
  int64_t chunk_coords[CHUNK_MAX_NDIM];
  int64_t local[CHUNK_MAX_NDIM];
  for (int d = 0; d < a->ndim; d++) {
    chunk_coords[d] = indices[d] / a->chunk_shape[d];
    local[d] = indices[d] % a->chunk_shape[d];
  }
  void *chunk = ensure_chunk(a, chunk_coords);
  size_t off = elem_offset_in_chunk(a, local);
  return (uint8_t *)chunk + off * a->elem_size;
}

float chunked_array_get_f32(const chunked_array *a, const int64_t *indices) {
  void *p = chunked_array_get_ptr(a, indices);
  if (!p) return 0.0f;
  float v;
  memcpy(&v, p, sizeof(float));
  return v;
}

void chunked_array_set_f32(chunked_array *a, const int64_t *indices, float val) {
  void *p = get_ptr_rw(a, indices);
  memcpy(p, &val, sizeof(float));
}

uint8_t chunked_array_get_u8(const chunked_array *a, const int64_t *indices) {
  void *p = chunked_array_get_ptr(a, indices);
  if (!p) return 0;
  return *(uint8_t *)p;
}

void chunked_array_set_u8(chunked_array *a, const int64_t *indices, uint8_t val) {
  void *p = get_ptr_rw(a, indices);
  *(uint8_t *)p = val;
}

// ---------------------------------------------------------------------------
// utilities
// ---------------------------------------------------------------------------

void chunked_array_fill_chunk(chunked_array *a, const int64_t *chunk_coords, const void *data, size_t len) {
  void *chunk = ensure_chunk(a, chunk_coords);
  size_t bytes = chunked_array_chunk_bytes(a);
  size_t copy = len < bytes ? len : bytes;
  memcpy(chunk, data, copy);
}

bool chunked_array_chunk_loaded(const chunked_array *a, const int64_t *chunk_coords) {
  return chunked_array_get_chunk(a, chunk_coords) != NULL;
}
