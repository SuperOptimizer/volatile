#include "greatest.h"
#include "core/chunk.h"

#include <string.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_create_and_free(void) {
  int64_t shape[3]       = {100, 100, 50};
  int64_t chunk_shape[3] = {32,  32,  16};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  ASSERT(a != NULL);
  ASSERT_EQ(3, a->ndim);
  ASSERT_EQ((size_t)sizeof(float), a->elem_size);

  // nchunks should be ceil(shape/chunk_shape)
  ASSERT_EQ(4, a->nchunks[0]);  // ceil(100/32) = 4
  ASSERT_EQ(4, a->nchunks[1]);  // ceil(100/32) = 4
  ASSERT_EQ(4, a->nchunks[2]);  // ceil(50/16)  = 4 (actually ceil(50/16)=4 since 3*16=48<50)

  ASSERT_EQ((size_t)(4 * 4 * 4), a->total_chunks);

  chunked_array_free(a);
  PASS();
}

TEST test_lazy_chunk_allocation(void) {
  int64_t shape[3]       = {64, 64, 64};
  int64_t chunk_shape[3] = {32, 32, 32};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  int64_t cc[3] = {0, 0, 0};

  // chunk should not be loaded yet
  ASSERT_EQ(false, chunked_array_chunk_loaded(a, cc));
  ASSERT_EQ(NULL, chunked_array_get_chunk(a, cc));

  // writing an element should allocate the chunk
  int64_t idx[3] = {1, 2, 3};
  chunked_array_set_f32(a, idx, 42.0f);

  ASSERT_EQ(true, chunked_array_chunk_loaded(a, cc));
  ASSERT(chunked_array_get_chunk(a, cc) != NULL);

  // reading back should return the value
  float v = chunked_array_get_f32(a, idx);
  ASSERT_IN_RANGE(41.9f, v, 42.1f);

  // other chunks remain unloaded
  int64_t cc2[3] = {1, 0, 0};
  ASSERT_EQ(false, chunked_array_chunk_loaded(a, cc2));

  chunked_array_free(a);
  PASS();
}

TEST test_set_get_elements_3d(void) {
  int64_t shape[3]       = {16, 16, 16};
  int64_t chunk_shape[3] = {8,  8,  8};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  // write several elements across different chunks
  struct { int64_t i[3]; float val; } cases[] = {
    {{0,  0,  0},  1.0f},
    {{7,  7,  7},  2.0f},
    {{8,  0,  0},  3.0f},
    {{15, 15, 15}, 4.0f},
    {{4,  9,  2},  5.5f},
  };
  int n = (int)(sizeof(cases) / sizeof(cases[0]));

  for (int i = 0; i < n; i++) {
    chunked_array_set_f32(a, cases[i].i, cases[i].val);
  }
  for (int i = 0; i < n; i++) {
    float got = chunked_array_get_f32(a, cases[i].i);
    ASSERT_IN_RANGE(cases[i].val - 0.001f, got, cases[i].val + 0.001f);
  }

  chunked_array_free(a);
  PASS();
}

TEST test_u8_set_get(void) {
  int64_t shape[2]       = {256, 256};
  int64_t chunk_shape[2] = {64, 64};
  chunked_array *a = chunked_array_new(2, shape, chunk_shape, sizeof(uint8_t));

  int64_t idx[2] = {100, 200};
  chunked_array_set_u8(a, idx, 255);
  ASSERT_EQ(255, chunked_array_get_u8(a, idx));

  int64_t idx2[2] = {0, 0};
  chunked_array_set_u8(a, idx2, 7);
  ASSERT_EQ(7, chunked_array_get_u8(a, idx2));

  chunked_array_free(a);
  PASS();
}

TEST test_edge_chunks(void) {
  // shape not divisible by chunk_shape -> edge chunks
  int64_t shape[3]       = {10, 10, 10};
  int64_t chunk_shape[3] = {4,  4,  4};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  // nchunks[d] = ceil(10/4) = 3
  ASSERT_EQ(3, a->nchunks[0]);
  ASSERT_EQ(3, a->nchunks[1]);
  ASSERT_EQ(3, a->nchunks[2]);

  // write to a corner element in the last edge chunk
  int64_t last[3] = {9, 9, 9};
  chunked_array_set_f32(a, last, 99.0f);
  float got = chunked_array_get_f32(a, last);
  ASSERT_IN_RANGE(98.9f, got, 99.1f);

  // the edge chunk is allocated; chunk coords = {2,2,2}
  int64_t edge_cc[3] = {2, 2, 2};
  ASSERT_EQ(true, chunked_array_chunk_loaded(a, edge_cc));

  // interior elements of that same chunk
  int64_t interior[3] = {8, 8, 8};
  chunked_array_set_f32(a, interior, 55.0f);
  ASSERT_IN_RANGE(54.9f, chunked_array_get_f32(a, interior), 55.1f);

  chunked_array_free(a);
  PASS();
}

TEST test_chunk_index(void) {
  int64_t shape[3]       = {64, 64, 64};
  int64_t chunk_shape[3] = {32, 32, 32};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  // nchunks = {2, 2, 2}; row-major flat index
  int64_t cc[3];

  cc[0] = 0; cc[1] = 0; cc[2] = 0; ASSERT_EQ((size_t)0, chunked_array_chunk_index(a, cc));
  cc[0] = 0; cc[1] = 0; cc[2] = 1; ASSERT_EQ((size_t)1, chunked_array_chunk_index(a, cc));
  cc[0] = 0; cc[1] = 1; cc[2] = 0; ASSERT_EQ((size_t)2, chunked_array_chunk_index(a, cc));
  cc[0] = 0; cc[1] = 1; cc[2] = 1; ASSERT_EQ((size_t)3, chunked_array_chunk_index(a, cc));
  cc[0] = 1; cc[1] = 0; cc[2] = 0; ASSERT_EQ((size_t)4, chunked_array_chunk_index(a, cc));
  cc[0] = 1; cc[1] = 1; cc[2] = 1; ASSERT_EQ((size_t)7, chunked_array_chunk_index(a, cc));

  chunked_array_free(a);
  PASS();
}

TEST test_set_chunk_takes_ownership(void) {
  int64_t shape[2]       = {8, 8};
  int64_t chunk_shape[2] = {8, 8};
  chunked_array *a = chunked_array_new(2, shape, chunk_shape, sizeof(float));

  size_t nbytes = chunked_array_chunk_bytes(a);
  ASSERT_EQ((size_t)(8 * 8 * sizeof(float)), nbytes);

  float *buf = malloc(nbytes);
  ASSERT(buf != NULL);
  for (size_t i = 0; i < 8 * 8; i++) buf[i] = (float)i;

  int64_t cc[2] = {0, 0};
  chunked_array_set_chunk(a, cc, buf);  // transfers ownership
  ASSERT_EQ(true, chunked_array_chunk_loaded(a, cc));

  int64_t idx[2] = {3, 5};
  float expected = buf[3 * 8 + 5];
  float got = chunked_array_get_f32(a, idx);
  ASSERT_IN_RANGE(expected - 0.001f, got, expected + 0.001f);

  chunked_array_free(a);  // frees buf
  PASS();
}

TEST test_fill_chunk(void) {
  int64_t shape[3]       = {4, 4, 4};
  int64_t chunk_shape[3] = {4, 4, 4};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, sizeof(float));

  size_t nbytes = chunked_array_chunk_bytes(a);
  float *src = malloc(nbytes);
  ASSERT(src != NULL);
  for (size_t i = 0; i < 4 * 4 * 4; i++) src[i] = (float)i * 2.0f;

  int64_t cc[3] = {0, 0, 0};
  chunked_array_fill_chunk(a, cc, src, nbytes);
  free(src);

  int64_t idx[3] = {1, 2, 3};
  // offset = 1*16 + 2*4 + 3 = 27; value = 27 * 2.0 = 54.0
  float got = chunked_array_get_f32(a, idx);
  ASSERT_IN_RANGE(53.9f, got, 54.1f);

  chunked_array_free(a);
  PASS();
}

TEST test_get_ptr_unloaded_returns_null(void) {
  int64_t shape[2]       = {16, 16};
  int64_t chunk_shape[2] = {8,  8};
  chunked_array *a = chunked_array_new(2, shape, chunk_shape, sizeof(float));

  int64_t idx[2] = {0, 0};
  void *p = chunked_array_get_ptr(a, idx);
  ASSERT_EQ(NULL, p);

  // get_f32/get_u8 should return 0 on unloaded chunk
  float f = chunked_array_get_f32(a, idx);
  ASSERT_IN_RANGE(-0.001f, f, 0.001f);

  chunked_array_free(a);
  PASS();
}

TEST test_chunk_bytes(void) {
  int64_t shape[3]       = {64, 64, 64};
  int64_t chunk_shape[3] = {16, 16, 16};
  chunked_array *a = chunked_array_new(3, shape, chunk_shape, 2);
  ASSERT_EQ((size_t)(16 * 16 * 16 * 2), chunked_array_chunk_bytes(a));
  chunked_array_free(a);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(chunk_suite) {
  RUN_TEST(test_create_and_free);
  RUN_TEST(test_lazy_chunk_allocation);
  RUN_TEST(test_set_get_elements_3d);
  RUN_TEST(test_u8_set_get);
  RUN_TEST(test_edge_chunks);
  RUN_TEST(test_chunk_index);
  RUN_TEST(test_set_chunk_takes_ownership);
  RUN_TEST(test_fill_chunk);
  RUN_TEST(test_get_ptr_unloaded_returns_null);
  RUN_TEST(test_chunk_bytes);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(chunk_suite);
  GREATEST_MAIN_END();
}
