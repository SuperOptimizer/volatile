#define _POSIX_C_SOURCE 200809L
#include "core/project.h"
#include "core/json.h"
#include "core/log.h"

#include <dirent.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// data_type <-> JSON string
// ---------------------------------------------------------------------------

static const char *const k_type_names[] = {
  "segments", "zarr_volume", "normalgrids",
  "direction_volume", "tiff_stack", "remote_zarr", "other",
};

static data_type type_from_str(const char *s) {
  if (!s) return DATA_OTHER;
  for (int i = 0; i < (int)(sizeof(k_type_names)/sizeof(*k_type_names)); i++)
    if (strcmp(s, k_type_names[i]) == 0) return (data_type)i;
  return DATA_OTHER;
}

// ---------------------------------------------------------------------------
// entry helpers
// ---------------------------------------------------------------------------

static void entry_free_fields(project_entry *e) {
  free(e->name); free(e->path);
  for (int i = 0; i < e->num_tags; i++) free(e->tags[i]);
  free(e->tags);
  memset(e, 0, sizeof(*e));
}

static bool entry_copy(project_entry *dst, const project_entry *src) {
  memset(dst, 0, sizeof(*dst));
  dst->name         = strdup(src->name  ? src->name  : "");
  dst->path         = strdup(src->path  ? src->path  : "");
  dst->type         = src->type;
  dst->is_remote    = src->is_remote;
  dst->track_changes = src->track_changes;
  dst->recursive    = src->recursive;
  if (!dst->name || !dst->path) { entry_free_fields(dst); return false; }
  if (src->num_tags > 0) {
    dst->tags = calloc((size_t)src->num_tags, sizeof(char *));
    if (!dst->tags) { entry_free_fields(dst); return false; }
    for (int i = 0; i < src->num_tags; i++) {
      dst->tags[i] = strdup(src->tags[i] ? src->tags[i] : "");
      if (!dst->tags[i]) { entry_free_fields(dst); return false; }
      dst->num_tags++;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// project lifecycle
// ---------------------------------------------------------------------------

project *project_new(const char *name) {
  project *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->name = strdup(name ? name : "untitled");
  if (!p->name) { free(p); return NULL; }
  return p;
}

void project_free(project *p) {
  if (!p) return;
  free(p->name); free(p->description);
  free(p->output_dir); free(p->sync_dir); free(p->path);
  for (int i = 0; i < p->num_entries; i++)
    entry_free_fields(&p->entries[i]);
  free(p->entries);
  free(p);
}

// ---------------------------------------------------------------------------
// grow backing array
// ---------------------------------------------------------------------------

static bool ensure_cap(project *p) {
  if (p->num_entries < p->capacity) return true;
  int nc = p->capacity ? p->capacity * 2 : 8;
  project_entry *buf = realloc(p->entries, (size_t)nc * sizeof(project_entry));
  if (!buf) return false;
  p->entries = buf;
  p->capacity = nc;
  return true;
}

// ---------------------------------------------------------------------------
// Data management
// ---------------------------------------------------------------------------

int project_add_entry(project *p, const project_entry *entry) {
  if (!p || !entry) return -1;
  if (!ensure_cap(p)) return -1;
  if (!entry_copy(&p->entries[p->num_entries], entry)) return -1;
  p->modified = true;
  return p->num_entries++;
}

int project_add_local(project *p, const char *path, data_type type,
                      bool recursive) {
  if (!p || !path) return -1;
  project_entry e = {
    .name      = (char *)path,
    .path      = (char *)path,
    .type      = type,
    .is_remote = false,
    .recursive = recursive,
  };
  return project_add_entry(p, &e);
}

int project_add_remote(project *p, const char *url, data_type type) {
  if (!p || !url) return -1;
  project_entry e = {
    .name      = (char *)url,
    .path      = (char *)url,
    .type      = type,
    .is_remote = true,
  };
  return project_add_entry(p, &e);
}

bool project_remove_entry(project *p, int index) {
  if (!p || index < 0 || index >= p->num_entries) return false;
  entry_free_fields(&p->entries[index]);
  int tail = p->num_entries - index - 1;
  if (tail > 0)
    memmove(&p->entries[index], &p->entries[index + 1],
            (size_t)tail * sizeof(project_entry));
  p->num_entries--;
  p->modified = true;
  return true;
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int project_count(const project *p)              { return p ? p->num_entries : 0; }
const project_entry *project_get(const project *p, int i) {
  return (p && i >= 0 && i < p->num_entries) ? &p->entries[i] : NULL;
}

int project_count_type(const project *p, data_type type) {
  int n = 0;
  if (!p) return 0;
  for (int i = 0; i < p->num_entries; i++)
    if (p->entries[i].type == type) n++;
  return n;
}

int project_find_by_type(const project *p, data_type type,
                         int *results, int max) {
  if (!p || !results || max <= 0) return 0;
  int n = 0;
  for (int i = 0; i < p->num_entries && n < max; i++)
    if (p->entries[i].type == type) results[n++] = i;
  return n;
}

int project_find_by_tag(const project *p, const char *tag,
                        int *results, int max) {
  if (!p || !tag || !results || max <= 0) return 0;
  int n = 0;
  for (int i = 0; i < p->num_entries && n < max; i++) {
    const project_entry *e = &p->entries[i];
    for (int t = 0; t < e->num_tags; t++) {
      if (e->tags[t] && strcmp(e->tags[t], tag) == 0) {
        results[n++] = i; break;
      }
    }
  }
  return n;
}

// ---------------------------------------------------------------------------
// Tagging
// ---------------------------------------------------------------------------

void project_tag_entry(project *p, int index, const char *tag) {
  if (!p || !tag || index < 0 || index >= p->num_entries) return;
  project_entry *e = &p->entries[index];
  // No-op if already tagged.
  for (int i = 0; i < e->num_tags; i++)
    if (e->tags[i] && strcmp(e->tags[i], tag) == 0) return;
  char **tags = realloc(e->tags, (size_t)(e->num_tags + 1) * sizeof(char *));
  if (!tags) return;
  e->tags = tags;
  e->tags[e->num_tags] = strdup(tag);
  if (e->tags[e->num_tags]) { e->num_tags++; p->modified = true; }
}

void project_untag_entry(project *p, int index, const char *tag) {
  if (!p || !tag || index < 0 || index >= p->num_entries) return;
  project_entry *e = &p->entries[index];
  for (int i = 0; i < e->num_tags; i++) {
    if (e->tags[i] && strcmp(e->tags[i], tag) == 0) {
      free(e->tags[i]);
      int tail = e->num_tags - i - 1;
      if (tail > 0) memmove(&e->tags[i], &e->tags[i+1],
                            (size_t)tail * sizeof(char *));
      e->num_tags--;
      p->modified = true;
      return;
    }
  }
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

static void fprint_escaped(FILE *f, const char *s) {
  fputc('"', f);
  for (; s && *s; s++) {
    if (*s == '"')       fputs("\\\"", f);
    else if (*s == '\\') fputs("\\\\", f);
    else if (*s == '\n') fputs("\\n",  f);
    else                 fputc(*s, f);
  }
  fputc('"', f);
}

static void fprint_str_field(FILE *f, const char *key, const char *val,
                              const char *indent, bool comma) {
  if (!val) return;
  fprintf(f, "%s%s: ", indent, key);
  fprint_escaped(f, val);
  if (comma) fputc(',', f);
  fputc('\n', f);
}

// ---------------------------------------------------------------------------
// project_save
// ---------------------------------------------------------------------------

bool project_save(const project *p, const char *path) {
  if (!p || !path) return false;
  FILE *f = fopen(path, "w");
  if (!f) { LOG_ERROR("project_save: fopen %s: %s", path, strerror(errno)); return false; }

  fprintf(f, "{\n");
  fprint_str_field(f, "  \"name\"",       p->name,       "", true);
  fprint_str_field(f, "  \"description\"",p->description,"", true);
  fprint_str_field(f, "  \"output_dir\"", p->output_dir, "", true);
  fprint_str_field(f, "  \"sync_dir\"",   p->sync_dir,   "", true);
  fprintf(f, "  \"entries\": [\n");

  for (int i = 0; i < p->num_entries; i++) {
    const project_entry *e = &p->entries[i];
    bool last = (i == p->num_entries - 1);
    fprintf(f, "    {\n");
    fprint_str_field(f, "      \"name\"", e->name, "", true);
    fprint_str_field(f, "      \"path\"", e->path, "", true);
    fprintf(f, "      \"type\": \"%s\",\n", k_type_names[(int)e->type]);
    fprintf(f, "      \"remote\": %s,\n",   e->is_remote    ? "true" : "false");
    fprintf(f, "      \"recursive\": %s,\n",e->recursive    ? "true" : "false");
    fprintf(f, "      \"track_changes\": %s,\n", e->track_changes ? "true":"false");
    fprintf(f, "      \"tags\": [");
    for (int t = 0; t < e->num_tags; t++) {
      fprint_escaped(f, e->tags[t]);
      if (t < e->num_tags - 1) fputc(',', f);
    }
    fprintf(f, "]\n    }%s\n", last ? "" : ",");
  }
  fprintf(f, "  ]\n}\n");
  fclose(f);

  // Remove const to set path/modified — caller pattern mirrors volpkg tools.
  ((project *)p)->path     = p->path ? p->path : strdup(path);
  ((project *)p)->modified = false;
  return true;
}

// ---------------------------------------------------------------------------
// project_load
// ---------------------------------------------------------------------------

project *project_load(const char *path) {
  if (!path) return NULL;
  FILE *f = fopen(path, "r");
  if (!f) { LOG_ERROR("project_load: fopen %s: %s", path, strerror(errno)); return NULL; }

  fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
  if (sz <= 0) { fclose(f); return NULL; }
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  buf[fread(buf, 1, (size_t)sz, f)] = '\0';
  fclose(f);

  json_value *root = json_parse(buf);
  free(buf);
  if (!root) { LOG_ERROR("project_load: JSON parse failed: %s", path); return NULL; }

  project *p = project_new(json_get_str(json_object_get(root, "name")));
  if (!p) { json_free(root); return NULL; }

  const json_value *v;
  if ((v = json_object_get(root, "description")))
    p->description = strdup(json_get_str(v) ? json_get_str(v) : "");
  if ((v = json_object_get(root, "output_dir")))
    p->output_dir  = strdup(json_get_str(v) ? json_get_str(v) : "");
  if ((v = json_object_get(root, "sync_dir")))
    p->sync_dir    = strdup(json_get_str(v) ? json_get_str(v) : "");
  p->path = strdup(path);

  const json_value *arr = json_object_get(root, "entries");
  size_t n = arr ? json_array_len(arr) : 0;
  for (size_t i = 0; i < n; i++) {
    const json_value *ej = json_array_get(arr, i);
    if (!ej) continue;
    project_entry e = {0};
    const json_value *nv = json_object_get(ej, "name");
    const json_value *pv = json_object_get(ej, "path");
    const json_value *tv = json_object_get(ej, "type");
    e.name         = (char *)(nv ? json_get_str(nv) : "");
    e.path         = (char *)(pv ? json_get_str(pv) : "");
    e.type         = type_from_str(tv ? json_get_str(tv) : NULL);
    e.is_remote    = json_get_bool(json_object_get(ej, "remote"),       false);
    e.recursive    = json_get_bool(json_object_get(ej, "recursive"),    false);
    e.track_changes= json_get_bool(json_object_get(ej, "track_changes"),false);

    // Temporarily borrow tags from JSON; entry_copy will strdup them.
    const json_value *ta = json_object_get(ej, "tags");
    size_t nt = ta ? json_array_len(ta) : 0;
    char **tmp_tags = nt ? calloc(nt, sizeof(char *)) : NULL;
    for (size_t t = 0; t < nt; t++) {
      const json_value *tv2 = json_array_get(ta, t);
      tmp_tags[t] = (char *)(tv2 ? json_get_str(tv2) : "");
    }
    e.tags = tmp_tags; e.num_tags = (int)nt;
    project_add_entry(p, &e);
    free(tmp_tags);
  }

  json_free(root);
  p->modified = false;
  return p;
}

// ---------------------------------------------------------------------------
// project_from_volpkg
// ---------------------------------------------------------------------------

project *project_from_volpkg(const char *volpkg_path) {
  if (!volpkg_path) return NULL;

  char cfg_path[4096];
  snprintf(cfg_path, sizeof(cfg_path), "%s/config.json", volpkg_path);
  FILE *f = fopen(cfg_path, "r");
  if (!f) { LOG_WARN("project_from_volpkg: no config.json at %s", volpkg_path); }

  project *p = project_new("Imported volpkg");
  if (!p) { if (f) fclose(f); return NULL; }
  p->output_dir = strdup(volpkg_path);

  if (f) {
    fseek(f, 0, SEEK_END); long sz = ftell(f); rewind(f);
    char *buf = sz > 0 ? malloc((size_t)sz + 1) : NULL;
    if (buf) {
      buf[fread(buf, 1, (size_t)sz, f)] = '\0';
      json_value *root = json_parse(buf);
      free(buf);
      if (root) {
        const json_value *nm = json_object_get(root, "name");
        if (nm && json_get_str(nm)) {
          free(p->name);
          p->name = strdup(json_get_str(nm));
        }
        json_free(root);
      }
    }
    fclose(f);
  }

  // Scan volumes/
  char vol_dir[4096];
  snprintf(vol_dir, sizeof(vol_dir), "%s/volumes", volpkg_path);
  DIR *d = opendir(vol_dir);
  if (d) {
    struct dirent *ent;
    while ((ent = readdir(d))) {
      if (ent->d_name[0] == '.') continue;
      char full[4096];
      snprintf(full, sizeof(full), "%s/%s", vol_dir, ent->d_name);
      project_add_local(p, full, DATA_ZARR_VOLUME, false);
    }
    closedir(d);
  }

  // Scan paths/ (segments)
  char seg_dir[4096];
  snprintf(seg_dir, sizeof(seg_dir), "%s/paths", volpkg_path);
  d = opendir(seg_dir);
  if (d) {
    struct dirent *ent;
    while ((ent = readdir(d))) {
      if (ent->d_name[0] == '.') continue;
      char full[4096];
      snprintf(full, sizeof(full), "%s/%s", seg_dir, ent->d_name);
      project_add_local(p, full, DATA_SEGMENTS, false);
    }
    closedir(d);
  }

  return p;
}

// ---------------------------------------------------------------------------
// project_scan_dir — detect data type from directory heuristics
// ---------------------------------------------------------------------------

static data_type detect_type(const char *path) {
  char probe[4096];
  struct stat st;
  snprintf(probe, sizeof(probe), "%s/.zarray",  path);
  if (stat(probe, &st) == 0) return DATA_ZARR_VOLUME;
  snprintf(probe, sizeof(probe), "%s/.zattrs",  path);
  if (stat(probe, &st) == 0) return DATA_ZARR_VOLUME;
  snprintf(probe, sizeof(probe), "%s/meta.json", path);
  if (stat(probe, &st) == 0) return DATA_SEGMENTS;
  // Check extension of last path component for TIFFs
  const char *ext = strrchr(path, '.');
  if (ext && (strcasecmp(ext, ".tif") == 0 || strcasecmp(ext, ".tiff") == 0))
    return DATA_TIFF_STACK;
  return DATA_OTHER;
}

int project_scan_dir(project *p, const char *dir, bool recursive) {
  if (!p || !dir) return 0;
  DIR *d = opendir(dir);
  if (!d) return 0;
  int added = 0;
  struct dirent *ent;
  while ((ent = readdir(d))) {
    if (ent->d_name[0] == '.') continue;
    char full[4096];
    snprintf(full, sizeof(full), "%s/%s", dir, ent->d_name);
    struct stat st;
    if (stat(full, &st) != 0) continue;
    if (S_ISDIR(st.st_mode)) {
      data_type t = detect_type(full);
      if (t != DATA_OTHER) {
        project_add_local(p, full, t, false);
        added++;
      } else if (recursive) {
        added += project_scan_dir(p, full, true);
      }
    } else if (S_ISREG(st.st_mode)) {
      data_type t = detect_type(full);
      if (t != DATA_OTHER) { project_add_local(p, full, t, false); added++; }
    }
  }
  closedir(d);
  return added;
}

// ---------------------------------------------------------------------------
// project_import_from
// ---------------------------------------------------------------------------

int project_import_from(project *p, const project *other) {
  if (!p || !other) return 0;
  int added = 0;
  for (int i = 0; i < other->num_entries; i++) {
    if (project_add_entry(p, &other->entries[i]) >= 0) added++;
  }
  return added;
}
