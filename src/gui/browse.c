#include "gui/browse.h"
#include "core/log.h"
#include "core/json.h"
#include "core/net.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// vol_browser internals
// ---------------------------------------------------------------------------

struct vol_browser {
  vol_entry **entries;
  int         count;
  int         cap;
};

// ---------------------------------------------------------------------------
// vol_entry
// ---------------------------------------------------------------------------

void vol_entry_free(vol_entry *e) {
  if (!e) return;
  free(e->name);
  free(e->path);
  free(e->dtype_name);
  free(e);
}

static vol_entry *entry_new(const char *name, const char *path, bool is_remote) {
  vol_entry *e = calloc(1, sizeof(vol_entry));
  if (!e) return NULL;
  e->name      = strdup(name);
  e->path      = strdup(path);
  e->is_remote = is_remote;
  e->dtype_name = strdup("unknown");
  if (!e->name || !e->path || !e->dtype_name) { vol_entry_free(e); return NULL; }
  return e;
}

// ---------------------------------------------------------------------------
// vol_browser lifecycle
// ---------------------------------------------------------------------------

vol_browser *vol_browser_new(void) {
  vol_browser *b = calloc(1, sizeof(vol_browser));
  return b;
}

void vol_browser_free(vol_browser *b) {
  if (!b) return;
  for (int i = 0; i < b->count; i++) vol_entry_free(b->entries[i]);
  free(b->entries);
  free(b);
}

// ---------------------------------------------------------------------------
// internal append
// ---------------------------------------------------------------------------

static bool browser_append(vol_browser *b, vol_entry *e) {
  if (b->count >= b->cap) {
    int new_cap = b->cap ? b->cap * 2 : 8;
    vol_entry **p = realloc(b->entries, (size_t)new_cap * sizeof(vol_entry *));
    if (!p) return false;
    b->entries = p;
    b->cap = new_cap;
  }
  b->entries[b->count++] = e;
  return true;
}

// ---------------------------------------------------------------------------
// basename helper (does not modify src)
// ---------------------------------------------------------------------------

static const char *path_basename(const char *path) {
  const char *p = strrchr(path, '/');
  return p ? p + 1 : path;
}

// ---------------------------------------------------------------------------
// zarr detection helpers
// ---------------------------------------------------------------------------

// Check if a directory looks like a zarr store by probing for .zarray or
// .zattrs at the top level or one level down (multi-scale).
static bool dir_is_zarr(const char *dir_path) {
  char probe[4096];
  struct stat st;

  // top-level .zattrs (OME-Zarr root)
  snprintf(probe, sizeof(probe), "%s/.zattrs", dir_path);
  if (stat(probe, &st) == 0) return true;

  // top-level .zarray (single-array zarr)
  snprintf(probe, sizeof(probe), "%s/.zarray", dir_path);
  if (stat(probe, &st) == 0) return true;

  // level 0 sub-array (multiscale: <store>/0/.zarray)
  snprintf(probe, sizeof(probe), "%s/0/.zarray", dir_path);
  if (stat(probe, &st) == 0) return true;

  return false;
}

// Count pyramid levels: look for numeric sub-directories 0, 1, 2, ...
static int count_pyramid_levels(const char *dir_path) {
  char probe[4096];
  struct stat st;
  int level = 0;
  while (level < 16) {
    snprintf(probe, sizeof(probe), "%s/%d", dir_path, level);
    if (stat(probe, &st) != 0 || !S_ISDIR(st.st_mode)) break;
    level++;
  }
  return level > 0 ? level : 1;
}

// Read shape from <store>/<level>/.zarray JSON
static void read_zarray_shape(const char *dir_path, int level, int64_t shape[3]) {
  char path[4096];
  snprintf(path, sizeof(path), "%s/%d/.zarray", dir_path, level);

  FILE *f = fopen(path, "r");
  if (!f) {
    // try top-level .zarray
    snprintf(path, sizeof(path), "%s/.zarray", dir_path);
    f = fopen(path, "r");
  }
  if (!f) return;

  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) { fclose(f); return; }

  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return; }
  buf[fread(buf, 1, (size_t)sz, f)] = '\0';
  fclose(f);

  json_value *root = json_parse(buf);
  free(buf);
  if (!root) return;

  const json_value *chunks_arr = json_object_get(root, "shape");
  if (chunks_arr) {
    size_t n = json_array_len(chunks_arr);
    // fill shape from the last 3 dims (z,y,x)
    int off = (int)n >= 3 ? (int)n - 3 : 0;
    for (int i = 0; i < 3 && (off + i) < (int)n; i++)
      shape[i] = json_get_int(json_array_get(chunks_arr, (size_t)(off + i)), 0);
  }

  const json_value *dtype_v = json_object_get(root, "dtype");
  // dtype_name is set separately; just read it here for future use
  (void)dtype_v;

  json_free(root);
}

// ---------------------------------------------------------------------------
// vol_browser_scan_local
// ---------------------------------------------------------------------------

int vol_browser_scan_local(vol_browser *b, const char *dir) {
  DIR *d = opendir(dir);
  if (!d) {
    LOG_WARN("vol_browser_scan_local: cannot open %s: %s", dir, strerror(errno));
    return -1;
  }

  int added = 0;
  struct dirent *ent;
  while ((ent = readdir(d)) != NULL) {
    if (ent->d_name[0] == '.') continue;

    char full[4096];
    snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);

    struct stat st;
    if (stat(full, &st) != 0 || !S_ISDIR(st.st_mode)) continue;
    if (!dir_is_zarr(full)) continue;

    vol_entry *e = entry_new(ent->d_name, full, false);
    if (!e) continue;

    e->num_levels = count_pyramid_levels(full);
    read_zarray_shape(full, 0, e->shape);

    if (browser_append(b, e)) added++;
    else { vol_entry_free(e); }
  }
  closedir(d);
  return added;
}

// ---------------------------------------------------------------------------
// vol_browser_add_remote  (stub — real impl would HTTP GET the catalog)
// ---------------------------------------------------------------------------

int vol_browser_add_remote(vol_browser *b, const char *server_url) {
  (void)b;
  // NOTE: stub. A real implementation would:
  //   http_response *r = http_get(server_url, 5000);
  //   parse JSON array of vol_entry records from r->data;
  //   append each to b;
  //   http_response_free(r);
  LOG_INFO("vol_browser_add_remote: remote catalog stub (url=%s)", server_url);
  return 0;
}

// ---------------------------------------------------------------------------
// vol_browser_add — single path or URL
// ---------------------------------------------------------------------------

bool vol_browser_add(vol_browser *b, const char *path_or_url) {
  bool is_remote = (strncmp(path_or_url, "http://",  7) == 0 ||
                    strncmp(path_or_url, "https://", 8) == 0 ||
                    strncmp(path_or_url, "s3://",    5) == 0);

  const char *name = path_basename(path_or_url);
  vol_entry *e = entry_new(name, path_or_url, is_remote);
  if (!e) return false;

  if (!is_remote) {
    e->num_levels = count_pyramid_levels(path_or_url);
    read_zarray_shape(path_or_url, 0, e->shape);
  } else {
    e->num_levels = 1;
  }

  if (!browser_append(b, e)) { vol_entry_free(e); return false; }
  return true;
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

int vol_browser_count(const vol_browser *b) {
  return b ? b->count : 0;
}

const vol_entry *vol_browser_get(const vol_browser *b, int index) {
  if (!b || index < 0 || index >= b->count) return NULL;
  return b->entries[index];
}

// ---------------------------------------------------------------------------
// search — substring match on name or path
// ---------------------------------------------------------------------------

int vol_browser_search(const vol_browser *b, const char *query,
                        int *results, int max_results) {
  if (!b || !query || !results || max_results <= 0) return 0;
  int found = 0;
  for (int i = 0; i < b->count && found < max_results; i++) {
    const vol_entry *e = b->entries[i];
    if ((e->name && strstr(e->name, query)) ||
        (e->path && strstr(e->path, query))) {
      results[found++] = i;
    }
  }
  return found;
}
