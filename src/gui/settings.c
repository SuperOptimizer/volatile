#include "gui/settings.h"

#include "core/hash.h"
#include "core/json.h"

#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static char *xstrdup(const char *src) {
  if (!src) return NULL;
  size_t n = strlen(src) + 1;
  char  *d = malloc(n);
  if (d) memcpy(d, src, n);
  return d;
}

static bool mkdir_p(const char *path) {
  struct stat st;
  if (stat(path, &st) == 0) return true;
  return mkdir(path, 0755) == 0;
}

static char *path_join(const char *dir, const char *file) {
  size_t n = strlen(dir) + 1 + strlen(file) + 1;
  char  *p = malloc(n);
  if (p) snprintf(p, n, "%s/%s", dir, file);
  return p;
}

static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  rewind(f);
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';
  return buf;
}

// ---------------------------------------------------------------------------
// JSON -> hash_map loader
// ---------------------------------------------------------------------------

static void json_kv_cb(const char *key, const json_value *val, void *ctx) {
  hash_map *map = ctx;
  const char *sv = NULL;
  char tmp[64];
  switch (json_typeof(val)) {
    case JSON_STRING: sv = json_get_str(val); break;
    case JSON_NUMBER:
      snprintf(tmp, sizeof(tmp), "%.17g", json_get_number(val, 0.0));
      sv = tmp;
      break;
    case JSON_BOOL: sv = json_get_bool(val, false) ? "true" : "false"; break;
    case JSON_NULL:  sv = "null"; break;
    default: return;  // skip nested objects/arrays
  }
  void *old = hash_map_get(map, key);
  if (old) { free(old); hash_map_del(map, key); }
  char *dup = xstrdup(sv);
  if (dup) hash_map_put(map, key, dup);
}

static void load_json_into_map(const char *path, hash_map *map) {
  char *text = read_file(path);
  if (!text) return;
  json_value *root = json_parse(text);
  free(text);
  if (!root || json_typeof(root) != JSON_OBJECT) { json_free(root); return; }
  json_object_iter(root, json_kv_cb, map);
  json_free(root);
}

// ---------------------------------------------------------------------------
// hash_map -> JSON file writer
// ---------------------------------------------------------------------------

static void write_json_str(FILE *f, const char *s) {
  fputc('"', f);
  for (; *s; s++) {
    unsigned char c = (unsigned char)*s;
    if      (c == '"')  fputs("\\\"", f);
    else if (c == '\\') fputs("\\\\", f);
    else if (c == '\n') fputs("\\n",  f);
    else if (c == '\r') fputs("\\r",  f);
    else if (c == '\t') fputs("\\t",  f);
    else if (c < 0x20)  fprintf(f, "\\u%04x", c);
    else                fputc(c, f);
  }
  fputc('"', f);
}

static bool save_map_to_file(hash_map *map, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fputs("{\n", f);
  hash_map_iter *it = hash_map_iter_new(map);
  hash_map_entry e;
  bool first = true;
  while (hash_map_iter_next(it, &e)) {
    if (!first) fputs(",\n", f);
    first = false;
    fputs("  ", f);
    write_json_str(f, e.key);
    fputs(": ", f);
    write_json_str(f, (const char *)e.val);
  }
  hash_map_iter_free(it);
  fputs("\n}\n", f);
  return fclose(f) == 0;
}

// ---------------------------------------------------------------------------
// settings struct
// ---------------------------------------------------------------------------

struct settings {
  hash_map *global;
  hash_map *project;   // NULL when no project_dir given
  char     *global_path;
  char     *project_path;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

settings *settings_open(const char *project_dir) {
  settings *s = calloc(1, sizeof(settings));
  if (!s) return NULL;

  const char *xdg  = getenv("XDG_CONFIG_HOME");
  char        gdir[4096];
  if (xdg && *xdg)
    snprintf(gdir, sizeof(gdir), "%s/volatile", xdg);
  else {
    const char *home = getenv("HOME");
    snprintf(gdir, sizeof(gdir), "%s/.config/volatile", home ? home : "/tmp");
  }
  s->global_path = path_join(gdir, "config.json");
  s->global      = hash_map_new();
  if (!s->global) { settings_close(s); return NULL; }
  load_json_into_map(s->global_path, s->global);

  if (project_dir) {
    char pdir[4096];
    snprintf(pdir, sizeof(pdir), "%s/.volatile", project_dir);
    s->project_path = path_join(pdir, "config.json");
    s->project      = hash_map_new();
    if (!s->project) { settings_close(s); return NULL; }
    load_json_into_map(s->project_path, s->project);
  }
  return s;
}

static void free_map_values(hash_map *m) {
  if (!m) return;
  hash_map_iter *it = hash_map_iter_new(m);
  hash_map_entry e;
  while (hash_map_iter_next(it, &e)) free(e.val);
  hash_map_iter_free(it);
}

void settings_close(settings *s) {
  if (!s) return;
  free_map_values(s->global);  hash_map_free(s->global);
  free_map_values(s->project); hash_map_free(s->project);
  free(s->global_path);
  free(s->project_path);
  free(s);
}

static const char *lookup(settings *s, const char *key) {
  if (s->project) {
    void *v = hash_map_get(s->project, key);
    if (v) return (const char *)v;
  }
  return (const char *)hash_map_get(s->global, key);
}

const char *settings_get_str(settings *s, const char *key, const char *def) {
  const char *v = lookup(s, key);
  return v ? v : def;
}

int settings_get_int(settings *s, const char *key, int def) {
  const char *v = lookup(s, key);
  if (!v) return def;
  char *end; long n = strtol(v, &end, 10);
  return (*end == '\0') ? (int)n : def;
}

float settings_get_float(settings *s, const char *key, float def) {
  const char *v = lookup(s, key);
  if (!v) return def;
  char *end; double d = strtod(v, &end);
  return (*end == '\0') ? (float)d : def;
}

bool settings_get_bool(settings *s, const char *key, bool def) {
  const char *v = lookup(s, key);
  if (!v) return def;
  if (strcmp(v, "true") == 0 || strcmp(v, "1") == 0)  return true;
  if (strcmp(v, "false") == 0 || strcmp(v, "0") == 0) return false;
  return def;
}

static void set_str_in(hash_map *map, const char *key, const char *val) {
  void *old = hash_map_get(map, key);
  if (old) { free(old); hash_map_del(map, key); }
  char *dup = xstrdup(val);
  if (dup) hash_map_put(map, key, dup);
}

void settings_set_str(settings *s, const char *key, const char *val) {
  set_str_in(s->project ? s->project : s->global, key, val);
}

void settings_set_int(settings *s, const char *key, int val) {
  char buf[32]; snprintf(buf, sizeof(buf), "%d", val);
  settings_set_str(s, key, buf);
}

bool settings_save(settings *s) {
  hash_map   *map  = s->project ? s->project      : s->global;
  const char *path = s->project ? s->project_path : s->global_path;
  if (!path) return false;
  char *dir = xstrdup(path);
  if (!dir) return false;
  char *slash = strrchr(dir, '/');
  if (slash) { *slash = '\0'; mkdir_p(dir); }
  free(dir);
  return save_map_to_file(map, path);
}

void settings_dump(const settings *s, FILE *out) {
  if (!out) out = stdout;
  if (s->global) {
    fprintf(out, "# global: %s\n", s->global_path ? s->global_path : "(none)");
    hash_map_iter *it = hash_map_iter_new(s->global);
    hash_map_entry e;
    while (hash_map_iter_next(it, &e)) fprintf(out, "  %s = %s\n", e.key, (char *)e.val);
    hash_map_iter_free(it);
  }
  if (s->project) {
    fprintf(out, "# project: %s\n", s->project_path ? s->project_path : "(none)");
    hash_map_iter *it = hash_map_iter_new(s->project);
    hash_map_entry e;
    while (hash_map_iter_next(it, &e)) fprintf(out, "  %s = %s\n", e.key, (char *)e.val);
    hash_map_iter_free(it);
  }
}
