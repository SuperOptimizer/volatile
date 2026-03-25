#define _POSIX_C_SOURCE 200809L

#include "cli/cli_grow.h"
#include "core/geom.h"
#include "core/vol.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void grow_usage(void) {
  puts("Usage: volatile grow <volume_path> --seed <z,y,x> [options]\n");
  puts("Grow a quad surface from a seed point by following the local normal.");
  puts("");
  puts("Options:");
  puts("  --seed <z,y,x>         Seed voxel coordinate (required)");
  puts("  --generations <N>      Number of expansion steps (default: 10)");
  puts("  --search-range <R>     Search range along normal in voxels (default: 5)");
  puts("  --output <path>        Output JSON path (default: surface.json)");
  puts("  --help                 Print this message");
}

// ---------------------------------------------------------------------------
// Normal search: sample along the normal from center, return offset that
// maximises intensity gradient magnitude (zero-crossing of derivative).
// Returns displacement in voxels (negative or positive along normal).
// ---------------------------------------------------------------------------

static float find_best_offset(volume *vol, float z, float y, float x,
                               float nz, float ny, float nx,
                               float search_range) {
  float best_offset = 0.0f;
  float best_grad   = -1.0f;
  int   steps       = (int)(search_range * 4.0f); // 4 samples per voxel

  float prev = vol_sample(vol, 0, z, y, x);
  for (int i = 1; i <= steps; i++) {
    float t    = (float)i / 4.0f - search_range / 2.0f;
    float sz   = z + nz * t;
    float sy   = y + ny * t;
    float sx   = x + nx * t;
    float curr = vol_sample(vol, 0, sz, sy, sx);
    float grad = fabsf(curr - prev);
    if (grad > best_grad) { best_grad = grad; best_offset = t; }
    prev = curr;
  }
  return best_offset;
}

// ---------------------------------------------------------------------------
// Seed surface: a (2*half+1) x (2*half+1) flat patch centred at seed,
// oriented in the XY plane (normal = +Z).  This is the starting guess.
// ---------------------------------------------------------------------------

static quad_surface *make_seed_surface(float sz, float sy, float sx, int half) {
  int dim = 2 * half + 1;
  quad_surface *s = quad_surface_new(dim, dim);
  if (!s) return NULL;
  for (int r = 0; r < dim; r++)
    for (int c = 0; c < dim; c++)
      quad_surface_set(s, r, c,
        (vec3f){ sx + (float)(c - half), sy + (float)(r - half), sz });
  return s;
}

// ---------------------------------------------------------------------------
// One growth step: expand the surface by adding one row/column on each edge,
// then snap each new boundary vertex to the best position along its normal.
// Returns a new (larger) quad_surface; caller frees both old and new.
// ---------------------------------------------------------------------------

static quad_surface *grow_one_generation(quad_surface *s, volume *vol, float search_range) {
  // Ensure normals are up-to-date.
  if (!s->normals) quad_surface_compute_normals(s);

  int old_rows = s->rows, old_cols = s->cols;
  int new_rows = old_rows + 2, new_cols = old_cols + 2;
  quad_surface *n = quad_surface_new(new_rows, new_cols);
  if (!n) return NULL;

  // Copy existing interior (offset by 1).
  for (int r = 0; r < old_rows; r++)
    for (int c = 0; c < old_cols; c++)
      quad_surface_set(n, r + 1, c + 1, quad_surface_get(s, r, c));

  // Fill the four new border strips by extrapolating from the edge row/col
  // and then snapping to the volume.

  // Top row (r=0) — extrapolate from old row 0 using the delta to old row 1.
  for (int c = 1; c < new_cols - 1; c++) {
    int oc = c - 1;
    vec3f edge  = quad_surface_get(s, 0, oc);
    vec3f inner = quad_surface_get(s, 1, oc);
    vec3f ext   = vec3f_sub(edge, vec3f_sub(inner, edge)); // edge - (inner-edge)
    quad_surface_set(n, 0, c, ext);
  }
  // Bottom row (r=new_rows-1).
  for (int c = 1; c < new_cols - 1; c++) {
    int oc = c - 1;
    vec3f edge  = quad_surface_get(s, old_rows - 1, oc);
    vec3f inner = quad_surface_get(s, old_rows - 2, oc);
    vec3f ext   = vec3f_sub(edge, vec3f_sub(inner, edge));
    quad_surface_set(n, new_rows - 1, c, ext);
  }
  // Left col (c=0).
  for (int r = 0; r < new_rows; r++) {
    vec3f edge, inner;
    if (r == 0 || r == new_rows - 1) {
      // corner: extrapolate from the extrapolated top/bottom edges
      edge  = quad_surface_get(n, r, 1);
      inner = quad_surface_get(n, r, 2);
    } else {
      edge  = quad_surface_get(s, r - 1, 0);
      inner = quad_surface_get(s, r - 1, 1);
    }
    vec3f ext = vec3f_sub(edge, vec3f_sub(inner, edge));
    quad_surface_set(n, r, 0, ext);
  }
  // Right col (c=new_cols-1).
  for (int r = 0; r < new_rows; r++) {
    vec3f edge, inner;
    if (r == 0 || r == new_rows - 1) {
      edge  = quad_surface_get(n, r, new_cols - 2);
      inner = quad_surface_get(n, r, new_cols - 3);
    } else {
      edge  = quad_surface_get(s, r - 1, old_cols - 1);
      inner = quad_surface_get(s, r - 1, old_cols - 2);
    }
    vec3f ext = vec3f_sub(edge, vec3f_sub(inner, edge));
    quad_surface_set(n, r, new_cols - 1, ext);
  }

  // Snap new boundary vertices to volume.
  quad_surface_compute_normals(n);

  // Top and bottom rows.
  for (int pass = 0; pass < 2; pass++) {
    int r = (pass == 0) ? 0 : new_rows - 1;
    for (int c = 0; c < new_cols; c++) {
      vec3f pt  = quad_surface_get(n, r, c);
      vec3f nm  = n->normals[r * new_cols + c];
      float off = find_best_offset(vol, pt.z, pt.y, pt.x, nm.z, nm.y, nm.x, search_range);
      quad_surface_set(n, r, c, vec3f_add(pt, vec3f_scale(nm, off)));
    }
  }
  // Left and right columns (skip already-processed corners).
  for (int pass = 0; pass < 2; pass++) {
    int c = (pass == 0) ? 0 : new_cols - 1;
    for (int r = 1; r < new_rows - 1; r++) {
      vec3f pt  = quad_surface_get(n, r, c);
      vec3f nm  = n->normals[r * new_cols + c];
      float off = find_best_offset(vol, pt.z, pt.y, pt.x, nm.z, nm.y, nm.x, search_range);
      quad_surface_set(n, r, c, vec3f_add(pt, vec3f_scale(nm, off)));
    }
  }

  // Invalidate normals after modification.
  if (n->normals) { free(n->normals); n->normals = NULL; }
  return n;
}

// ---------------------------------------------------------------------------
// Save quad_surface to a minimal JSON file.
// Format: {"rows":R,"cols":C,"points":[[x,y,z],...]}
// ---------------------------------------------------------------------------

static int save_surface_json(const quad_surface *s, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "grow: cannot write output file: %s\n", path);
    return 1;
  }
  fprintf(f, "{\"rows\":%d,\"cols\":%d,\"points\":[", s->rows, s->cols);
  int n = s->rows * s->cols;
  for (int i = 0; i < n; i++) {
    vec3f p = s->points[i];
    fprintf(f, "%s[%.6g,%.6g,%.6g]", i ? "," : "", (double)p.x, (double)p.y, (double)p.z);
  }
  fputs("]}\n", f);
  fclose(f);
  return 0;
}

// ---------------------------------------------------------------------------
// cmd_grow
// ---------------------------------------------------------------------------

int cmd_grow(int argc, char **argv) {
  if (argc < 1) { grow_usage(); return 1; }

  // Check for --help first (before requiring volume_path).
  for (int i = 0; i < argc; i++) {
    if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      grow_usage();
      return 0;
    }
  }

  const char *volume_path  = NULL;
  float       seed_z = 0, seed_y = 0, seed_x = 0;
  bool        has_seed     = false;
  int         generations  = 10;
  float       search_range = 5.0f;
  const char *output_path  = "surface.json";

  // First positional arg = volume path.
  if (argv[0][0] != '-') {
    volume_path = argv[0];
  } else {
    fputs("grow: volume_path is required\n", stderr);
    grow_usage();
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      if (sscanf(argv[++i], "%f,%f,%f", &seed_z, &seed_y, &seed_x) != 3) {
        fprintf(stderr, "grow: --seed requires z,y,x format\n");
        return 1;
      }
      has_seed = true;
    } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
      generations = atoi(argv[++i]);
      if (generations < 1) generations = 1;
    } else if (strcmp(argv[i], "--search-range") == 0 && i + 1 < argc) {
      search_range = (float)atof(argv[++i]);
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      output_path = argv[++i];
    }
  }

  if (!has_seed) {
    fputs("grow: --seed <z,y,x> is required\n", stderr);
    grow_usage();
    return 1;
  }

  volume *vol = vol_open(volume_path);
  if (!vol) {
    fprintf(stderr, "grow: cannot open volume: %s\n", volume_path);
    return 1;
  }

  printf("grow: seed=(%.1f,%.1f,%.1f) generations=%d search_range=%.1f\n",
         (double)seed_z, (double)seed_y, (double)seed_x, generations, (double)search_range);

  // Initial 5x5 patch.
  quad_surface *s = make_seed_surface(seed_z, seed_y, seed_x, 2);
  if (!s) {
    fputs("grow: out of memory\n", stderr);
    vol_free(vol);
    return 1;
  }

  for (int g = 0; g < generations; g++) {
    quad_surface *next = grow_one_generation(s, vol, search_range);
    quad_surface_free(s);
    if (!next) {
      fputs("grow: out of memory during growth\n", stderr);
      vol_free(vol);
      return 1;
    }
    s = next;
    printf("grow: generation %d/%d — surface %dx%d\n", g + 1, generations, s->rows, s->cols);
    fflush(stdout);
  }

  int rc = save_surface_json(s, output_path);
  if (rc == 0)
    printf("grow: saved surface to %s (%d points)\n", output_path, s->rows * s->cols);

  quad_surface_free(s);
  vol_free(vol);
  return rc;
}
