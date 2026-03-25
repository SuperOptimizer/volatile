#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef enum { DTYPE_U8, DTYPE_U16, DTYPE_F32, DTYPE_F64 } dtype_t;

typedef struct {
  int width, height, depth;  // depth=1 for 2D images, >1 for stacks
  int channels;              // 1 for grayscale
  dtype_t dtype;
  void *data;                // row-major, channels interleaved
  size_t data_size;
} image;

void image_free(image *img);

// TIFF reader (supports: uncompressed, multi-page/stack, uint8/uint16/float32)
image *tiff_read(const char *path);

// NRRD reader (supports: raw encoding, gzip encoding, basic header fields)
typedef struct {
  int ndim;
  int sizes[8];
  dtype_t dtype;
  float space_directions[8][3];
  float space_origin[3];
  void *data;
  size_t data_size;
} nrrd_data;

nrrd_data *nrrd_read(const char *path);
void nrrd_free(nrrd_data *n);

// TIFF writer (uncompressed; RGB or grayscale, uint8 or uint16)
bool tiff_write(const char *path, const image *img);

// PPM/PGM reader/writer (simple, for debugging)
image *ppm_read(const char *path);
bool ppm_write(const char *path, const image *img);
bool pgm_write(const char *path, const uint8_t *data, int width, int height);

// OBJ mesh reader (triangles only)
typedef struct {
  float *vertices;   // 3 floats per vertex
  int *indices;      // 3 ints per triangle
  float *normals;    // 3 floats per vertex (or NULL)
  int vertex_count;
  int index_count;   // number of indices (triangle_count * 3)
} obj_mesh;

obj_mesh *obj_read(const char *path);
void obj_free(obj_mesh *m);

// utility
size_t dtype_size(dtype_t dt);
const char *dtype_name(dtype_t dt);
