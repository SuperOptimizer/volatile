#include "greatest.h"
#include "gui/browse.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// helpers — create minimal zarr directory structures in /tmp
// ---------------------------------------------------------------------------

static void mkdir_p(const char *path) {
  mkdir(path, 0755);
}

// write a minimal .zarray JSON for a given path
static void write_zarray(const char *path, int z, int y, int x) {
  FILE *f = fopen(path, "w");
  if (!f) return;
  fprintf(f,
    "{\n"
    "  \"chunks\": [64, 64, 64],\n"
    "  \"compressor\": null,\n"
    "  \"dtype\": \"<u2\",\n"
    "  \"fill_value\": 0,\n"
    "  \"order\": \"C\",\n"
    "  \"shape\": [%d, %d, %d],\n"
    "  \"zarr_format\": 2\n"
    "}\n", z, y, x);
  fclose(f);
}

// create a multiscale .zarr with n levels under /tmp/base/name
// returns allocated path string (caller must free)
static char *make_zarr(const char *base, const char *name, int levels) {
  char *dir = malloc(512);
  snprintf(dir, 512, "%s/%s", base, name);
  mkdir_p(dir);

  // .zattrs at top level marks it as OME-Zarr
  char probe[512];
  snprintf(probe, sizeof(probe), "%s/.zattrs", dir);
  FILE *f = fopen(probe, "w");
  if (f) { fprintf(f, "{}\n"); fclose(f); }

  for (int i = 0; i < levels; i++) {
    char lvl[512];
    snprintf(lvl, sizeof(lvl), "%s/%d", dir, i);
    mkdir_p(lvl);
    snprintf(probe, sizeof(probe), "%s/.zarray", lvl);
    int div = 1 << i;
    write_zarray(probe, 256 / div, 1024 / div, 1024 / div);
  }
  return dir;
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  vol_browser *b = vol_browser_new();
  ASSERT(b != NULL);
  ASSERT_EQ(0, vol_browser_count(b));
  vol_browser_free(b);
  PASS();
}

TEST test_free_null(void) {
  vol_browser_free(NULL);
  PASS();
}

TEST test_vol_entry_free_null(void) {
  vol_entry_free(NULL);
  PASS();
}

TEST test_scan_local_empty_dir(void) {
  mkdir_p("/tmp/test_browse_empty");
  vol_browser *b = vol_browser_new();
  int n = vol_browser_scan_local(b, "/tmp/test_browse_empty");
  ASSERT(n >= 0);
  ASSERT_EQ(0, vol_browser_count(b));
  vol_browser_free(b);
  PASS();
}

TEST test_scan_local_finds_zarr(void) {
  char *dir = make_zarr("/tmp", "test_browse_vol1.zarr", 2);
  vol_browser *b = vol_browser_new();
  int n = vol_browser_scan_local(b, "/tmp");
  // we might find more than one if other zarrs exist; just check >= 1 found
  ASSERT(n >= 1);
  ASSERT(vol_browser_count(b) >= 1);

  // verify our volume is present
  bool found = false;
  for (int i = 0; i < vol_browser_count(b); i++) {
    const vol_entry *e = vol_browser_get(b, i);
    ASSERT(e != NULL);
    if (strstr(e->name, "test_browse_vol1.zarr")) {
      found = true;
      ASSERT(!e->is_remote);
      ASSERT(e->num_levels >= 1);
    }
  }
  ASSERT(found);
  vol_browser_free(b);
  free(dir);
  PASS();
}

TEST test_scan_reads_shape(void) {
  char *dir = make_zarr("/tmp", "test_browse_shape.zarr", 1);
  vol_browser *b = vol_browser_new();
  vol_browser_scan_local(b, "/tmp");

  bool found = false;
  for (int i = 0; i < vol_browser_count(b); i++) {
    const vol_entry *e = vol_browser_get(b, i);
    if (strstr(e->name, "test_browse_shape.zarr")) {
      // shape should be (256, 1024, 1024)
      ASSERT_EQ(256,  e->shape[0]);
      ASSERT_EQ(1024, e->shape[1]);
      ASSERT_EQ(1024, e->shape[2]);
      found = true;
    }
  }
  ASSERT(found);
  vol_browser_free(b);
  free(dir);
  PASS();
}

TEST test_add_single_local(void) {
  char *dir = make_zarr("/tmp", "test_browse_single.zarr", 3);
  vol_browser *b = vol_browser_new();
  bool ok = vol_browser_add(b, dir);
  ASSERT(ok);
  ASSERT_EQ(1, vol_browser_count(b));

  const vol_entry *e = vol_browser_get(b, 0);
  ASSERT(e != NULL);
  ASSERT_STR_EQ("test_browse_single.zarr", e->name);
  ASSERT_STR_EQ(dir, e->path);
  ASSERT(!e->is_remote);
  ASSERT_EQ(3, e->num_levels);

  vol_browser_free(b);
  free(dir);
  PASS();
}

TEST test_add_remote_url(void) {
  vol_browser *b = vol_browser_new();
  bool ok = vol_browser_add(b, "https://example.com/volumes/my_vol.zarr");
  ASSERT(ok);
  ASSERT_EQ(1, vol_browser_count(b));

  const vol_entry *e = vol_browser_get(b, 0);
  ASSERT(e != NULL);
  ASSERT(e->is_remote);
  ASSERT_STR_EQ("my_vol.zarr", e->name);

  vol_browser_free(b);
  PASS();
}

TEST test_get_out_of_bounds(void) {
  vol_browser *b = vol_browser_new();
  ASSERT(vol_browser_get(b, 0)  == NULL);
  ASSERT(vol_browser_get(b, -1) == NULL);
  vol_browser_free(b);
  PASS();
}

TEST test_add_remote_catalog_stub(void) {
  vol_browser *b = vol_browser_new();
  // stub returns 0 entries, should not crash
  int n = vol_browser_add_remote(b, "https://example.com/catalog");
  ASSERT_EQ(0, n);
  ASSERT_EQ(0, vol_browser_count(b));
  vol_browser_free(b);
  PASS();
}

TEST test_search_by_name(void) {
  vol_browser *b = vol_browser_new();
  vol_browser_add(b, "/tmp/alpha.zarr");
  vol_browser_add(b, "/tmp/beta.zarr");
  vol_browser_add(b, "/tmp/alpha_v2.zarr");

  int results[8];
  int n = vol_browser_search(b, "alpha", results, 8);
  ASSERT_EQ(2, n);
  // both results should be alpha entries
  for (int i = 0; i < n; i++) {
    const vol_entry *e = vol_browser_get(b, results[i]);
    ASSERT(strstr(e->name, "alpha") != NULL);
  }
  vol_browser_free(b);
  PASS();
}

TEST test_search_no_match(void) {
  vol_browser *b = vol_browser_new();
  vol_browser_add(b, "/tmp/alpha.zarr");
  int results[4];
  int n = vol_browser_search(b, "zzz_no_match", results, 4);
  ASSERT_EQ(0, n);
  vol_browser_free(b);
  PASS();
}

TEST test_search_max_results(void) {
  vol_browser *b = vol_browser_new();
  for (int i = 0; i < 10; i++) {
    char path[64];
    snprintf(path, sizeof(path), "/tmp/vol%d.zarr", i);
    vol_browser_add(b, path);
  }
  int results[3];
  int n = vol_browser_search(b, "vol", results, 3);
  ASSERT_EQ(3, n);
  vol_browser_free(b);
  PASS();
}

TEST test_search_by_path(void) {
  vol_browser *b = vol_browser_new();
  vol_browser_add(b, "https://server.example.com/data/scroll1.zarr");
  int results[4];
  int n = vol_browser_search(b, "scroll1", results, 4);
  ASSERT_EQ(1, n);
  vol_browser_free(b);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(browse_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_free_null);
  RUN_TEST(test_vol_entry_free_null);
  RUN_TEST(test_scan_local_empty_dir);
  RUN_TEST(test_scan_local_finds_zarr);
  RUN_TEST(test_scan_reads_shape);
  RUN_TEST(test_add_single_local);
  RUN_TEST(test_add_remote_url);
  RUN_TEST(test_get_out_of_bounds);
  RUN_TEST(test_add_remote_catalog_stub);
  RUN_TEST(test_search_by_name);
  RUN_TEST(test_search_no_match);
  RUN_TEST(test_search_max_results);
  RUN_TEST(test_search_by_path);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(browse_suite);
  GREATEST_MAIN_END();
}
