#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "cli/cli_inpaint.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static void rm_f(const char *path) { remove(path); }

// Write a minimal P6 PPM: WxH RGB uint8.
static bool write_test_ppm(const char *path, int w, int h) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fprintf(f, "P6\n%d %d\n255\n", w, h);
  for (int i = 0; i < w * h * 3; i++) {
    uint8_t v = (uint8_t)(i & 0xFF);
    fwrite(&v, 1, 1, f);
  }
  fclose(f);
  return true;
}

// Write a minimal P5 PGM mask: WxH grayscale uint8.
// Marks a small central region as hole (value 255), rest 0.
static bool write_test_pgm(const char *path, int w, int h) {
  FILE *f = fopen(path, "wb");
  if (!f) return false;
  fprintf(f, "P5\n%d %d\n255\n", w, h);
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      // Mark the 2x2 center as hole.
      int cx = w / 2, cy = h / 2;
      uint8_t v = (x == cx || x == cx-1) && (y == cy || y == cy-1) ? 255 : 0;
      fwrite(&v, 1, 1, f);
    }
  }
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_inpaint_help(void) {
  char *args[] = { "--help" };
  int rc = cmd_inpaint(1, args);
  ASSERT_EQ(0, rc);
  PASS();
}

TEST test_inpaint_no_args(void) {
  int rc = cmd_inpaint(0, NULL);
  ASSERT_EQ(1, rc);
  PASS();
}

TEST test_inpaint_no_mask(void) {
  char *args[] = { "/tmp/volatile_test_img.ppm", "--output", "/tmp/volatile_test_inpaint_out.ppm" };
  int rc = cmd_inpaint(3, args);
  ASSERT_EQ(1, rc);  // --mask not provided
  PASS();
}

TEST test_inpaint_no_output(void) {
  char *args[] = { "/tmp/volatile_test_img.ppm", "--mask", "/tmp/volatile_test_mask.pgm" };
  int rc = cmd_inpaint(3, args);
  ASSERT_EQ(1, rc);  // --output not provided
  PASS();
}

TEST test_inpaint_bad_image(void) {
  char *args[] = {
    "/nonexistent/img.ppm",
    "--mask",   "/tmp/volatile_test_bad_mask.pgm",
    "--output", "/tmp/volatile_test_bad_out.ppm",
  };
  int rc = cmd_inpaint(5, args);
  ASSERT_EQ(1, rc);  // read_ppm fails
  PASS();
}

TEST test_inpaint_roundtrip(void) {
  const char *img_path  = "/tmp/volatile_test_inpaint_img.ppm";
  const char *mask_path = "/tmp/volatile_test_inpaint_mask.pgm";
  const char *out_path  = "/tmp/volatile_test_inpaint_out.ppm";
  rm_f(img_path); rm_f(mask_path); rm_f(out_path);

  ASSERT(write_test_ppm(img_path, 8, 8));
  ASSERT(write_test_pgm(mask_path, 8, 8));

  char *args[] = {
    (char *)img_path,
    "--mask",   (char *)mask_path,
    "--output", (char *)out_path,
    "--radius", "3",
  };
  int rc = cmd_inpaint(7, args);
  ASSERT_EQ(0, rc);
  ASSERT(file_exists(out_path));

  // Output must be a valid P6 PPM.
  FILE *f = fopen(out_path, "rb");
  ASSERT(f != NULL);
  char magic[3] = {0};
  fscanf(f, "%2s", magic);
  fclose(f);
  ASSERT_STR_EQ("P6", magic);

  rm_f(img_path); rm_f(mask_path); rm_f(out_path);
  PASS();
}

TEST test_inpaint_size_mismatch(void) {
  const char *img_path  = "/tmp/volatile_test_inpaint_mismatch_img.ppm";
  const char *mask_path = "/tmp/volatile_test_inpaint_mismatch_mask.pgm";
  const char *out_path  = "/tmp/volatile_test_inpaint_mismatch_out.ppm";
  rm_f(img_path); rm_f(mask_path); rm_f(out_path);

  ASSERT(write_test_ppm(img_path,  8, 8));
  ASSERT(write_test_pgm(mask_path, 4, 4));  // different size

  char *args[] = {
    (char *)img_path,
    "--mask",   (char *)mask_path,
    "--output", (char *)out_path,
  };
  int rc = cmd_inpaint(5, args);
  ASSERT_EQ(1, rc);  // dimension mismatch

  rm_f(img_path); rm_f(mask_path); rm_f(out_path);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(inpaint_suite) {
  RUN_TEST(test_inpaint_help);
  RUN_TEST(test_inpaint_no_args);
  RUN_TEST(test_inpaint_no_mask);
  RUN_TEST(test_inpaint_no_output);
  RUN_TEST(test_inpaint_bad_image);
  RUN_TEST(test_inpaint_roundtrip);
  RUN_TEST(test_inpaint_size_mismatch);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(inpaint_suite);
  GREATEST_MAIN_END();
}
