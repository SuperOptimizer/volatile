#define _POSIX_C_SOURCE 200809L

#include "cli/cli_transform.h"
#include "core/geom.h"
#include "core/math.h"
#include "core/json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Read 4×4 matrix from a JSON file.
//
// Accepted formats:
//   {"matrix": [[r0c0,r0c1,...],[r1c0,...],...]  }   (row-major 4x4)
//   [[r0c0,...], ...]                                 (bare array of 4 arrays)
//
// mat4f is column-major internally, so we transpose on read.
// ---------------------------------------------------------------------------

static bool read_matrix_json(const char *path, mat4f *out) {
  FILE *f = fopen(path, "r");
  if (!f) { fprintf(stderr, "transform: cannot open matrix file: %s\n", path); return false; }

  fseek(f, 0, SEEK_END);
  long sz = ftell(f); rewind(f);
  if (sz <= 0 || sz > 65536) { fclose(f); return false; }
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return false; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';

  json_value *root = json_parse(buf);
  free(buf);
  if (!root) { fputs("transform: invalid JSON in matrix file\n", stderr); return false; }

  // Accept either {"matrix": [[...],...]} or bare [[...],...].
  const json_value *arr = (json_typeof(root) == JSON_OBJECT)
                          ? json_object_get(root, "matrix")
                          : root;

  bool ok = false;
  if (arr && json_typeof(arr) == JSON_ARRAY && json_array_len(arr) == 4) {
    ok = true;
    for (int row = 0; row < 4 && ok; row++) {
      const json_value *rowv = json_array_get(arr, (size_t)row);
      if (!rowv || json_typeof(rowv) != JSON_ARRAY || json_array_len(rowv) != 4) {
        ok = false; break;
      }
      for (int col = 0; col < 4 && ok; col++) {
        const json_value *v = json_array_get(rowv, (size_t)col);
        if (!v) { ok = false; break; }
        // mat4f is column-major: element [row][col] lives at m[col*4 + row]
        out->m[col * 4 + row] = (float)json_get_number(v, 0.0);
      }
    }
  }

  if (!ok) fputs("transform: matrix must be a 4x4 array-of-arrays\n", stderr);
  json_free(root);
  return ok;
}

// ---------------------------------------------------------------------------
// Surface loader stub (same as cli_diff.c — will be replaced with real I/O)
// ---------------------------------------------------------------------------

static quad_surface *load_surface(const char *path, int rows, int cols) {
  (void)path;
  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;
  for (int r = 0; r < rows; r++)
    for (int c = 0; c < cols; c++)
      quad_surface_set(s, r, c, (vec3f){ (float)c, (float)r, 0.0f });
  return s;
}

// ---------------------------------------------------------------------------
// cmd_transform
// ---------------------------------------------------------------------------

int cmd_transform(int argc, char **argv) {
  if (argc < 1) {
    fputs("usage: volatile transform <surface> --matrix <4x4.json> --output <out>\n", stderr);
    return 1;
  }

  const char *surface_path = argv[0];
  const char *matrix_path  = NULL;
  const char *out_path     = NULL;

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--matrix") == 0) matrix_path = argv[i + 1];
    if (strcmp(argv[i], "--output") == 0) out_path    = argv[i + 1];
  }

  if (!matrix_path) { fputs("error: --matrix required\n", stderr); return 1; }
  if (!out_path)    { fputs("error: --output required\n",  stderr); return 1; }

  mat4f m = mat4f_identity();
  if (!read_matrix_json(matrix_path, &m)) return 1;

  const int rows = 512, cols = 512;
  quad_surface *surf = load_surface(surface_path, rows, cols);
  if (!surf) { fputs("error: could not load surface\n", stderr); return 1; }

  // Apply transform to every vertex (and normals if present).
  int n = surf->rows * surf->cols;
  for (int i = 0; i < n; i++) {
    surf->points[i] = mat4f_transform_point(m, surf->points[i]);
  }
  if (surf->normals) {
    for (int i = 0; i < n; i++) {
      // Normals transform by the inverse-transpose; for rigid/uniform-scale
      // transforms mat4f_transform_vec (ignoring translation) is correct.
      surf->normals[i] = vec3f_normalize(mat4f_transform_vec(m, surf->normals[i]));
    }
  }

  // NOTE: Surface serialisation to file is not yet implemented; for now print
  // the bounding box of the transformed points so the transform is observable.
  vec3f pmin = surf->points[0], pmax = surf->points[0];
  for (int i = 1; i < n; i++) {
    vec3f p = surf->points[i];
    if (p.x < pmin.x) pmin.x = p.x;
    if (p.x > pmax.x) pmax.x = p.x;
    if (p.y < pmin.y) pmin.y = p.y;
    if (p.y > pmax.y) pmax.y = p.y;
    if (p.z < pmin.z) pmin.z = p.z;
    if (p.z > pmax.z) pmax.z = p.z;
  }

  printf("transformed %d vertices -> %s\n", n, out_path);
  printf("  bbox: (%.3g %.3g %.3g) – (%.3g %.3g %.3g)\n",
         (double)pmin.x, (double)pmin.y, (double)pmin.z,
         (double)pmax.x, (double)pmax.y, (double)pmax.z);

  quad_surface_free(surf);
  return 0;
}
