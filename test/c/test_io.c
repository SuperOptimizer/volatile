#include "greatest.h"
#include "core/io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// dtype helpers
// ---------------------------------------------------------------------------

TEST test_dtype_size(void) {
  ASSERT_EQ(1u, dtype_size(DTYPE_U8));
  ASSERT_EQ(2u, dtype_size(DTYPE_U16));
  ASSERT_EQ(4u, dtype_size(DTYPE_F32));
  ASSERT_EQ(8u, dtype_size(DTYPE_F64));
  PASS();
}

TEST test_dtype_name(void) {
  ASSERT_STR_EQ("uint8",   dtype_name(DTYPE_U8));
  ASSERT_STR_EQ("uint16",  dtype_name(DTYPE_U16));
  ASSERT_STR_EQ("float32", dtype_name(DTYPE_F32));
  ASSERT_STR_EQ("float64", dtype_name(DTYPE_F64));
  PASS();
}

// ---------------------------------------------------------------------------
// PGM write + ppm_read round-trip
// ---------------------------------------------------------------------------

TEST test_pgm_write_read_roundtrip(void) {
  // 4x4 grayscale image with known pixel values
  uint8_t pixels[16];
  for (int i = 0; i < 16; i++) pixels[i] = (uint8_t)(i * 16);

  const char *path = "/tmp/test_io_pgm.pgm";
  ASSERT(pgm_write(path, pixels, 4, 4));

  image *img = ppm_read(path);
  ASSERT(img != NULL);
  ASSERT_EQ(4, img->width);
  ASSERT_EQ(4, img->height);
  ASSERT_EQ(1, img->depth);
  ASSERT_EQ(1, img->channels);
  ASSERT_EQ(DTYPE_U8, img->dtype);
  ASSERT_EQ(16u, img->data_size);

  uint8_t *d = (uint8_t *)img->data;
  for (int i = 0; i < 16; i++)
    ASSERT_EQ(pixels[i], d[i]);

  image_free(img);
  remove(path);
  PASS();
}

TEST test_ppm_write_read_roundtrip(void) {
  // 3x2 RGB image
  uint8_t pixels[18];
  for (int i = 0; i < 18; i++) pixels[i] = (uint8_t)(i * 14);

  // build an image struct and use ppm_write
  image src = {
    .width     = 3,
    .height    = 2,
    .depth     = 1,
    .channels  = 3,
    .dtype     = DTYPE_U8,
    .data      = pixels,
    .data_size = 18,
  };

  const char *path = "/tmp/test_io_ppm.ppm";
  ASSERT(ppm_write(path, &src));

  image *img = ppm_read(path);
  ASSERT(img != NULL);
  ASSERT_EQ(3, img->width);
  ASSERT_EQ(2, img->height);
  ASSERT_EQ(3, img->channels);
  ASSERT_EQ(DTYPE_U8, img->dtype);

  uint8_t *d = (uint8_t *)img->data;
  for (int i = 0; i < 18; i++)
    ASSERT_EQ(pixels[i], d[i]);

  image_free(img);
  remove(path);
  PASS();
}

TEST test_image_free_null(void) {
  // must not crash
  image_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// OBJ parser
// ---------------------------------------------------------------------------

static obj_mesh *parse_obj_string(const char *src) {
  const char *path = "/tmp/test_io_inline.obj";
  FILE *f = fopen(path, "w");
  if (!f) return NULL;
  fputs(src, f);
  fclose(f);
  return obj_read(path);
}

TEST test_obj_triangle(void) {
  const char *src =
    "v 0.0 0.0 0.0\n"
    "v 1.0 0.0 0.0\n"
    "v 0.0 1.0 0.0\n"
    "f 1 2 3\n";

  obj_mesh *m = parse_obj_string(src);
  ASSERT(m != NULL);
  ASSERT_EQ(3,  m->vertex_count);
  ASSERT_EQ(3,  m->index_count);
  ASSERT_EQ(0,  m->indices[0]);
  ASSERT_EQ(1,  m->indices[1]);
  ASSERT_EQ(2,  m->indices[2]);

  // vertex 0 should be (0,0,0)
  ASSERT_IN_RANGE(0.0f, m->vertices[0], 1e-6f);
  ASSERT_IN_RANGE(0.0f, m->vertices[1], 1e-6f);
  ASSERT_IN_RANGE(0.0f, m->vertices[2], 1e-6f);

  // vertex 1 should be (1,0,0)
  ASSERT_IN_RANGE(1.0f, m->vertices[3], 1e-6f);

  obj_free(m);
  remove("/tmp/test_io_inline.obj");
  PASS();
}

TEST test_obj_quad_triangulated(void) {
  // quad face should produce 2 triangles = 6 indices
  const char *src =
    "v 0.0 0.0 0.0\n"
    "v 1.0 0.0 0.0\n"
    "v 1.0 1.0 0.0\n"
    "v 0.0 1.0 0.0\n"
    "f 1 2 3 4\n";

  obj_mesh *m = parse_obj_string(src);
  ASSERT(m != NULL);
  ASSERT_EQ(4, m->vertex_count);
  ASSERT_EQ(6, m->index_count);

  obj_free(m);
  remove("/tmp/test_io_inline.obj");
  PASS();
}

TEST test_obj_normals(void) {
  const char *src =
    "v  0.0 0.0 0.0\n"
    "v  1.0 0.0 0.0\n"
    "v  0.0 1.0 0.0\n"
    "vn 0.0 0.0 1.0\n"
    "f 1//1 2//1 3//1\n";

  obj_mesh *m = parse_obj_string(src);
  ASSERT(m != NULL);
  ASSERT(m->normals != NULL);
  // normal for vertex 0 should be (0,0,1)
  ASSERT_IN_RANGE(0.0f, m->normals[0], 1e-6f);
  ASSERT_IN_RANGE(0.0f, m->normals[1], 1e-6f);
  ASSERT_IN_RANGE(1.0f, m->normals[2], 1e-6f);

  obj_free(m);
  remove("/tmp/test_io_inline.obj");
  PASS();
}

TEST test_obj_slash_separator(void) {
  // f v/vt/vn format
  const char *src =
    "v  0.0 0.0 0.0\n"
    "v  1.0 0.0 0.0\n"
    "v  0.0 1.0 0.0\n"
    "vt 0.0 0.0\n"
    "vn 0.0 0.0 1.0\n"
    "f 1/1/1 2/1/1 3/1/1\n";

  obj_mesh *m = parse_obj_string(src);
  ASSERT(m != NULL);
  ASSERT_EQ(3, m->vertex_count);
  ASSERT_EQ(3, m->index_count);

  obj_free(m);
  remove("/tmp/test_io_inline.obj");
  PASS();
}

TEST test_obj_free_null(void) {
  obj_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// NRRD (minimal — raw encoding only)
// ---------------------------------------------------------------------------

TEST test_nrrd_raw_roundtrip(void) {
  // write a minimal NRRD file manually and read it back
  const char *path = "/tmp/test_io.nrrd";
  FILE *f = fopen(path, "wb");
  ASSERT(f != NULL);

  uint8_t pixels[6] = {10, 20, 30, 40, 50, 60};
  fprintf(f, "NRRD0001\n");
  fprintf(f, "type: uint8\n");
  fprintf(f, "dimension: 2\n");
  fprintf(f, "sizes: 3 2\n");
  fprintf(f, "encoding: raw\n");
  fprintf(f, "\n");
  fwrite(pixels, 1, 6, f);
  fclose(f);

  nrrd_data *n = nrrd_read(path);
  ASSERT(n != NULL);
  ASSERT_EQ(2, n->ndim);
  ASSERT_EQ(3, n->sizes[0]);
  ASSERT_EQ(2, n->sizes[1]);
  ASSERT_EQ(DTYPE_U8, n->dtype);
  ASSERT_EQ(6u, n->data_size);

  uint8_t *d = (uint8_t *)n->data;
  for (int i = 0; i < 6; i++)
    ASSERT_EQ(pixels[i], d[i]);

  nrrd_free(n);
  remove(path);
  PASS();
}

TEST test_nrrd_free_null(void) {
  nrrd_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(io_suite) {
  RUN_TEST(test_dtype_size);
  RUN_TEST(test_dtype_name);
  RUN_TEST(test_pgm_write_read_roundtrip);
  RUN_TEST(test_ppm_write_read_roundtrip);
  RUN_TEST(test_image_free_null);
  RUN_TEST(test_obj_triangle);
  RUN_TEST(test_obj_quad_triangulated);
  RUN_TEST(test_obj_normals);
  RUN_TEST(test_obj_slash_separator);
  RUN_TEST(test_obj_free_null);
  RUN_TEST(test_nrrd_raw_roundtrip);
  RUN_TEST(test_nrrd_free_null);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(io_suite);
  GREATEST_MAIN_END();
}
