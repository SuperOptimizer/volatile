#include "greatest.h"
#include "core/project.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Write a file; returns true on success.
static bool write_file(const char *path, const char *contents) {
  FILE *f = fopen(path, "w");
  if (!f) return false;
  fputs(contents, f);
  fclose(f);
  return true;
}

// Return a temp path that doesn't exist yet (caller must free).
static char *tmp_path(const char *suffix) {
  char *p = malloc(256);
  snprintf(p, 256, "/tmp/test_project_%d_%s", (int)getpid(), suffix);
  return p;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  project *p = project_new("Test");
  ASSERT(p != NULL);
  ASSERT(strcmp(p->name, "Test") == 0);
  ASSERT_EQ(0, project_count(p));
  project_free(p);
  PASS();
}

TEST test_free_null(void) {
  project_free(NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Add / remove entries
// ---------------------------------------------------------------------------

TEST test_add_local(void) {
  project *p = project_new("p");
  int idx = project_add_local(p, "/data/vol", DATA_ZARR_VOLUME, false);
  ASSERT(idx >= 0);
  ASSERT_EQ(1, project_count(p));
  const project_entry *e = project_get(p, 0);
  ASSERT(e != NULL);
  ASSERT(strstr(e->path, "/data/vol") != NULL);
  ASSERT_EQ(DATA_ZARR_VOLUME, e->type);
  ASSERT(!e->is_remote);
  project_free(p);
  PASS();
}

TEST test_add_remote(void) {
  project *p = project_new("p");
  int idx = project_add_remote(p, "https://example.com/vol/", DATA_REMOTE_ZARR);
  ASSERT(idx >= 0);
  const project_entry *e = project_get(p, 0);
  ASSERT(e->is_remote);
  ASSERT_EQ(DATA_REMOTE_ZARR, e->type);
  project_free(p);
  PASS();
}

TEST test_remove_entry(void) {
  project *p = project_new("p");
  project_add_local(p, "/a", DATA_SEGMENTS, false);
  project_add_local(p, "/b", DATA_ZARR_VOLUME, false);
  project_add_local(p, "/c", DATA_TIFF_STACK, false);
  ASSERT_EQ(3, project_count(p));

  bool ok = project_remove_entry(p, 1);  // remove /b
  ASSERT(ok);
  ASSERT_EQ(2, project_count(p));
  ASSERT(strstr(project_get(p, 0)->path, "/a") != NULL);
  ASSERT(strstr(project_get(p, 1)->path, "/c") != NULL);

  ASSERT(!project_remove_entry(p, 99));  // out of range
  project_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Count by type
// ---------------------------------------------------------------------------

TEST test_count_type(void) {
  project *p = project_new("p");
  project_add_local(p, "/a", DATA_SEGMENTS, false);
  project_add_local(p, "/b", DATA_SEGMENTS, false);
  project_add_local(p, "/c", DATA_ZARR_VOLUME, false);
  ASSERT_EQ(2, project_count_type(p, DATA_SEGMENTS));
  ASSERT_EQ(1, project_count_type(p, DATA_ZARR_VOLUME));
  ASSERT_EQ(0, project_count_type(p, DATA_TIFF_STACK));
  project_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Tagging
// ---------------------------------------------------------------------------

TEST test_tag_untag(void) {
  project *p = project_new("p");
  project_add_local(p, "/a", DATA_SEGMENTS, false);

  project_tag_entry(p, 0, "active");
  project_tag_entry(p, 0, "alice");
  ASSERT_EQ(2, project_get(p, 0)->num_tags);

  // Duplicate tag is a no-op.
  project_tag_entry(p, 0, "active");
  ASSERT_EQ(2, project_get(p, 0)->num_tags);

  project_untag_entry(p, 0, "active");
  ASSERT_EQ(1, project_get(p, 0)->num_tags);
  ASSERT(strcmp(project_get(p, 0)->tags[0], "alice") == 0);

  project_free(p);
  PASS();
}

TEST test_find_by_tag(void) {
  project *p = project_new("p");
  project_add_local(p, "/a", DATA_SEGMENTS, false);
  project_add_local(p, "/b", DATA_SEGMENTS, false);
  project_add_local(p, "/c", DATA_ZARR_VOLUME, false);
  project_tag_entry(p, 0, "scroll1");
  project_tag_entry(p, 2, "scroll1");

  int results[8];
  int n = project_find_by_tag(p, "scroll1", results, 8);
  ASSERT_EQ(2, n);
  ASSERT_EQ(0, results[0]);
  ASSERT_EQ(2, results[1]);

  n = project_find_by_tag(p, "nope", results, 8);
  ASSERT_EQ(0, n);
  project_free(p);
  PASS();
}

TEST test_find_by_type(void) {
  project *p = project_new("p");
  project_add_local(p, "/a", DATA_SEGMENTS, false);
  project_add_local(p, "/b", DATA_ZARR_VOLUME, false);
  project_add_local(p, "/c", DATA_SEGMENTS, false);

  int results[8];
  int n = project_find_by_type(p, DATA_SEGMENTS, results, 8);
  ASSERT_EQ(2, n);
  ASSERT_EQ(0, results[0]);
  ASSERT_EQ(2, results[1]);
  project_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Save / load roundtrip
// ---------------------------------------------------------------------------

TEST test_save_load_roundtrip(void) {
  char *path = tmp_path("roundtrip.json");

  project *p = project_new("Roundtrip Project");
  p->description = strdup("test description");
  p->output_dir  = strdup("./output");
  p->sync_dir    = strdup("s3://bucket/segs");

  project_add_local(p, "/vol/main", DATA_ZARR_VOLUME, false);
  project_add_remote(p, "https://remote/seg/1", DATA_REMOTE_ZARR);
  project_tag_entry(p, 0, "main");
  project_tag_entry(p, 1, "remote");
  project_tag_entry(p, 1, "active");

  bool saved = project_save(p, path);
  ASSERT(saved);
  project_free(p);

  project *q = project_load(path);
  ASSERT(q != NULL);
  ASSERT(strcmp(q->name, "Roundtrip Project") == 0);
  ASSERT(strcmp(q->description, "test description") == 0);
  ASSERT_EQ(2, project_count(q));

  const project_entry *e0 = project_get(q, 0);
  ASSERT_EQ(DATA_ZARR_VOLUME, e0->type);
  ASSERT(!e0->is_remote);
  ASSERT_EQ(1, e0->num_tags);
  ASSERT(strcmp(e0->tags[0], "main") == 0);

  const project_entry *e1 = project_get(q, 1);
  ASSERT(e1->is_remote);
  ASSERT_EQ(2, e1->num_tags);

  project_free(q);
  unlink(path);
  free(path);
  PASS();
}

// ---------------------------------------------------------------------------
// project_import_from
// ---------------------------------------------------------------------------

TEST test_import_from(void) {
  project *src = project_new("src");
  project_add_local(src, "/x", DATA_SEGMENTS, false);
  project_add_local(src, "/y", DATA_ZARR_VOLUME, false);

  project *dst = project_new("dst");
  project_add_local(dst, "/z", DATA_TIFF_STACK, false);

  int added = project_import_from(dst, src);
  ASSERT_EQ(2, added);
  ASSERT_EQ(3, project_count(dst));

  project_free(src);
  project_free(dst);
  PASS();
}

// ---------------------------------------------------------------------------
// project_from_volpkg — minimal mock: create a temp volpkg-like tree
// ---------------------------------------------------------------------------

TEST test_from_volpkg(void) {
  char base[256];
  snprintf(base, sizeof(base), "/tmp/test_volpkg_%d", (int)getpid());
  mkdir(base, 0755);

  char vol_dir[256], seg_dir[256];
  snprintf(vol_dir, sizeof(vol_dir), "%s/volumes", base);
  snprintf(seg_dir, sizeof(seg_dir), "%s/paths",   base);
  mkdir(vol_dir, 0755);
  mkdir(seg_dir, 0755);

  // Create one volume subdir and one segment subdir.
  char v1[256], s1[256];
  snprintf(v1, sizeof(v1), "%s/20230205180739", vol_dir);
  snprintf(s1, sizeof(s1), "%s/00042",           seg_dir);
  mkdir(v1, 0755); mkdir(s1, 0755);

  char cfg[256];
  snprintf(cfg, sizeof(cfg), "%s/config.json", base);
  write_file(cfg, "{\"name\": \"PHercParis4\"}");

  project *p = project_from_volpkg(base);
  ASSERT(p != NULL);
  ASSERT(strcmp(p->name, "PHercParis4") == 0);

  int vols = project_count_type(p, DATA_ZARR_VOLUME);
  int segs = project_count_type(p, DATA_SEGMENTS);
  ASSERT_EQ(1, vols);
  ASSERT_EQ(1, segs);

  project_free(p);

  // Cleanup.
  rmdir(v1); rmdir(s1);
  rmdir(vol_dir); rmdir(seg_dir);
  unlink(cfg); rmdir(base);
  PASS();
}

// ---------------------------------------------------------------------------
// Null-safety
// ---------------------------------------------------------------------------

TEST test_null_safety(void) {
  ASSERT_EQ(0, project_count(NULL));
  ASSERT(project_get(NULL, 0) == NULL);
  ASSERT_EQ(-1, project_add_entry(NULL, NULL));
  ASSERT(!project_remove_entry(NULL, 0));
  ASSERT(!project_save(NULL, NULL));
  ASSERT(project_load(NULL) == NULL);
  ASSERT(project_from_volpkg(NULL) == NULL);
  ASSERT_EQ(0, project_scan_dir(NULL, NULL, false));
  ASSERT_EQ(0, project_import_from(NULL, NULL));
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(project_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_add_local);
  RUN_TEST(test_add_remote);
  RUN_TEST(test_remove_entry);
  RUN_TEST(test_count_type);
  RUN_TEST(test_tag_untag);
  RUN_TEST(test_find_by_tag);
  RUN_TEST(test_find_by_type);
  RUN_TEST(test_save_load_roundtrip);
  RUN_TEST(test_import_from);
  RUN_TEST(test_from_volpkg);
  RUN_TEST(test_null_safety);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(project_suite);
  GREATEST_MAIN_END();
}
