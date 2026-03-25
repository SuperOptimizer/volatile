#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#define CHUNK_MAX_NDIM 8

typedef struct {
  int ndim;
  int64_t shape[CHUNK_MAX_NDIM];        // total shape
  int64_t chunk_shape[CHUNK_MAX_NDIM];  // per-chunk shape
  int64_t nchunks[CHUNK_MAX_NDIM];      // number of chunks per dim (computed)
  size_t elem_size;                      // bytes per element (1, 2, 4, 8)
  void **chunks;                         // flat array of chunk data pointers (NULL = not loaded)
  size_t total_chunks;                   // product of nchunks
} chunked_array;

// lifecycle
chunked_array *chunked_array_new(int ndim, const int64_t *shape, const int64_t *chunk_shape, size_t elem_size);
void chunked_array_free(chunked_array *a);

// chunk management
size_t chunked_array_chunk_index(const chunked_array *a, const int64_t *chunk_coords);  // chunk coord -> flat index
void *chunked_array_get_chunk(const chunked_array *a, const int64_t *chunk_coords);     // NULL if not loaded
void chunked_array_set_chunk(chunked_array *a, const int64_t *chunk_coords, void *data); // takes ownership
size_t chunked_array_chunk_bytes(const chunked_array *a);                                // bytes per chunk

// element access (goes through chunk lookup)
void *chunked_array_get_ptr(const chunked_array *a, const int64_t *indices);            // pointer to element
float chunked_array_get_f32(const chunked_array *a, const int64_t *indices);            // convenience for float
void chunked_array_set_f32(chunked_array *a, const int64_t *indices, float val);
uint8_t chunked_array_get_u8(const chunked_array *a, const int64_t *indices);
void chunked_array_set_u8(chunked_array *a, const int64_t *indices, uint8_t val);

// utilities
void chunked_array_fill_chunk(chunked_array *a, const int64_t *chunk_coords, const void *data, size_t len);
bool chunked_array_chunk_loaded(const chunked_array *a, const int64_t *chunk_coords);
