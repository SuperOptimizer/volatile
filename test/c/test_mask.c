#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "cli/cli_mask.h"

#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void rm_f(const char *path) {
  remove(path);
}

// Build a minimal OME-Zarr store so vol_open succeeds and returns level meta.
static bool make_test_zarr(const char *path) {
  char buf[512];
  snprintf(buf, sizeof(buf), "mkdir -p '%s/0'", path);
  if (system(buf) != 0) return false;

  snprintf(buf, sizeof(buf), "%s/.zgroup", path);
  FILE *f = fopen(buf, "w"); if (!f) return false;
  fputs("{\"zarr_format\":2}\n", f); fclose(f);

  snprintf(buf, sizeof(buf), "%s/0/.zarray", path);
  f = fopen(buf, "w"); if (!f) return false;
  fputs("{\n"
        "  \"zarr_format\": 2,\n"
        "  \"shape\": [8, 8, 8],\n"
        "  \"chunks\": [8, 8, 8],\n"
        "  \"dtype\": \"|u1\",\n"
        "  \"order\": \"C\",\n"
        "  \"compressor\": null,\n"
        "  \"fill_value\": 128,\n"
        "  \"filters\": null\n"
        "}\n", f);
  fclose(f);

  snprintf(buf, sizeof(buf), "%s/0/0.0.0", path);
  f = fopen(buf, "wb"); if (!f) return false;
  for (int i = 0; i < 512; i++) { uint8_t v = 128; fwrite(&v, 1, 1, f); }
  fclose(f);
  return true;
}

static void rm_rf(const char *path) {
  char buf[512];
  snprintf(buf, sizeof(buf), "rm -rf '%s'", path);
  system(buf);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_mask_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_mask(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_mask_no_args(void) {
  int rc = cmd_mask(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

TEST test_mask_no_volume(void) {
  char *args[] = { "plane:z=0", "--output", "/tmp/volatile_test_mask.vmsk" };
  int rc = cmd_mask(3, args);
  ASSERT_EQ(1, rc);  // --volume not provided
  PASS();
}

TEST test_mask_no_output(void) {
  char *args[] = { "plane:z=0", "--volume", "/nonexistent" };
  int rc = cmd_mask(3, args);
  ASSERT_EQ(1, rc);  // --output not provided
  PASS();
}

TEST test_mask_bad_volume(void) {
  char *args[] = { "plane:z=0", "--volume", "/nonexistent", "--output", "/tmp/volatile_test_mask_bad.vmsk" };
  int rc = cmd_mask(5, args);
  ASSERT_EQ(1, rc);  // vol_open fails
  PASS();
}

TEST test_mask_plane_produces_output(void) {
  const char *zarr_path = "/tmp/volatile_test_mask_zarr";
  const char *out_path  = "/tmp/volatile_test_mask_out.vmsk";
  rm_rf(zarr_path);
  rm_f(out_path);
  ASSERT(make_test_zarr(zarr_path));

  char *args[] = {
    "plane:z=4",
    "--volume",  (char *)zarr_path,
    "--output",  (char *)out_path,
    "--radius",  "2",
  };
  int rc = cmd_mask(7, args);
  ASSERT_EQ(0, rc);
  ASSERT(file_exists(out_path));

  // Verify VMSK header magic.
  FILE *f = fopen(out_path, "rb");
  ASSERT(f != NULL);
  char magic[5] = {0};
  fread(magic, 1, 4, f);
  fclose(f);
  ASSERT_STR_EQ("VMSK", magic);

  rm_f(out_path);
  rm_rf(zarr_path);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(mask_suite) {
  RUN_TEST(test_mask_help);
  RUN_TEST(test_mask_no_args);
  RUN_TEST(test_mask_no_volume);
  RUN_TEST(test_mask_no_output);
  RUN_TEST(test_mask_bad_volume);
  RUN_TEST(test_mask_plane_produces_output);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(mask_suite);
  GREATEST_MAIN_END();
}
