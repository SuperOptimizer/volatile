#define _POSIX_C_SOURCE 200809L
#include "core/gridstore.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <dirent.h>
#include <errno.h>
#include <inttypes.h>

struct gridstore {
  char     root[4096];
  int64_t  chunk_shape[3];
};

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void _chunk_path(const gridstore *g, const int64_t *coords, char *out, size_t cap) {
  snprintf(out, cap, "%s/%" PRId64 "/%" PRId64 "/%" PRId64 ".bin",
           g->root, coords[0], coords[1], coords[2]);
}

// mkdir -p for a single path segment (no-op if exists)
static bool _mkdir_p(const char *path) {
  char tmp[4096];
  snprintf(tmp, sizeof(tmp), "%s", path);
  for (char *p = tmp + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return false;
      *p = '/';
    }
  }
  if (mkdir(tmp, 0755) != 0 && errno != EEXIST) return false;
  return true;
}

static bool _ensure_dir(const gridstore *g, const int64_t *coords) {
  char dir[4096];
  snprintf(dir, sizeof(dir), "%s/%" PRId64 "/%" PRId64,
           g->root, coords[0], coords[1]);
  return _mkdir_p(dir);
}

// Count .bin files under root (recursive, max 3 levels deep)
static int _count_bins(const char *dir, int depth) {
  if (depth > 3) return 0;
  DIR *d = opendir(dir);
  if (!d) return 0;
  int count = 0;
  struct dirent *e;
  while ((e = readdir(d))) {
    if (e->d_name[0] == '.') continue;
    char sub[4096];
    snprintf(sub, sizeof(sub), "%s/%s", dir, e->d_name);
    struct stat st;
    if (stat(sub, &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      count += _count_bins(sub, depth + 1);
    } else if (S_ISREG(st.st_mode)) {
      size_t n = strlen(e->d_name);
      if (n > 4 && strcmp(e->d_name + n - 4, ".bin") == 0) count++;
    }
  }
  closedir(d);
  return count;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

gridstore *gridstore_new(const char *path, const int64_t chunk_shape[3]) {
  gridstore *g = calloc(1, sizeof(*g));
  if (!g) return NULL;
  snprintf(g->root, sizeof(g->root), "%s", path);
  for (int i = 0; i < 3; i++) g->chunk_shape[i] = chunk_shape[i];
  if (!_mkdir_p(g->root)) { free(g); return NULL; }
  return g;
}

void gridstore_free(gridstore *g) { free(g); }

bool gridstore_write(gridstore *g, const int64_t *coords, const void *data, size_t len) {
  if (!_ensure_dir(g, coords)) return false;
  char path[4096];
  _chunk_path(g, coords, path, sizeof(path));
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  bool ok = (fwrite(data, 1, len, f) == len);
  fclose(f);
  return ok;
}

uint8_t *gridstore_read(const gridstore *g, const int64_t *coords, size_t *out_len) {
  char path[4096];
  _chunk_path(g, coords, path, sizeof(path));
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  rewind(f);
  uint8_t *buf = malloc((size_t)sz);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { free(buf); fclose(f); return NULL; }
  fclose(f);
  *out_len = (size_t)sz;
  return buf;
}

bool gridstore_exists(const gridstore *g, const int64_t *coords) {
  char path[4096];
  _chunk_path(g, coords, path, sizeof(path));
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int gridstore_count(const gridstore *g) {
  return _count_bins(g->root, 0);
}
