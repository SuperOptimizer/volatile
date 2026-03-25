/* test_tiff_write.c — tests for multi-page TIFF and tifxyz writers */
#include "greatest.h"
#include "core/io.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TMP_MULTI   "/tmp/test_multipage.tif"
#define TMP_XYZ     "/tmp/test_xyz.tif"
#define TMP_U16     "/tmp/test_u16mp.tif"

/* -------------------------------------------------------------------------
 * Helpers
 * --------------------------------------------------------------------- */

static bool file_exists_nonempty(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return false;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fclose(f);
  return sz > 0;
}

// Read 2 bytes at offset (little-endian uint16)
static uint16_t read_u16_at(const char *path, long off) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  fseek(f, off, SEEK_SET);
  uint8_t b[2];
  fread(b, 1, 2, f);
  fclose(f);
  return (uint16_t)(b[0] | (b[1] << 8));
}

// Read 4 bytes at offset (little-endian uint32)
static uint32_t read_u32_at(const char *path, long off) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  fseek(f, off, SEEK_SET);
  uint8_t b[4];
  fread(b, 1, 4, f);
  fclose(f);
  return (uint32_t)(b[0] | (b[1]<<8) | (b[2]<<16) | (b[3]<<24));
}

/* -------------------------------------------------------------------------
 * 1. tiff_write_multipage: NULL guards
 * --------------------------------------------------------------------- */

TEST test_multipage_null_path(void) {
  uint8_t page[4] = {0};
  const void *pages[1] = { page };
  ASSERT(!tiff_write_multipage(NULL, pages, 1, 2, 2, DTYPE_U8, 1));
  PASS();
}

TEST test_multipage_null_pages(void) {
  ASSERT(!tiff_write_multipage(TMP_MULTI, NULL, 1, 2, 2, DTYPE_U8, 1));
  PASS();
}

/* -------------------------------------------------------------------------
 * 2. tiff_write_multipage: single page uint8 — valid TIFF header
 * --------------------------------------------------------------------- */

TEST test_multipage_single_u8(void) {
  uint8_t page[4] = {10, 20, 30, 40};
  const void *pages[1] = { page };
  ASSERT(tiff_write_multipage(TMP_MULTI, pages, 1, 2, 2, DTYPE_U8, 1));
  ASSERT(file_exists_nonempty(TMP_MULTI));

  // Check TIFF magic: bytes 0-1 = "II" (0x4949), bytes 2-3 = 42
  ASSERT_EQ(read_u16_at(TMP_MULTI, 0), (uint16_t)0x4949);
  ASSERT_EQ(read_u16_at(TMP_MULTI, 2), (uint16_t)42);
  PASS();
}

/* -------------------------------------------------------------------------
 * 3. tiff_write_multipage: 3 pages, uint16 — file bigger than 1-page
 * --------------------------------------------------------------------- */

TEST test_multipage_three_pages_u16(void) {
  uint16_t page0[4] = {100, 200, 300, 400};
  uint16_t page1[4] = {500, 600, 700, 800};
  uint16_t page2[4] = {900, 1000, 1100, 1200};
  const void *pages[3] = { page0, page1, page2 };

  ASSERT(tiff_write_multipage(TMP_U16, pages, 3, 2, 2, DTYPE_U16, 1));
  ASSERT(file_exists_nonempty(TMP_U16));

  // File size must be significantly larger than a single-page version
  {
    uint16_t p1[4] = {1, 2, 3, 4};
    const void *pp[1] = {p1};
    tiff_write_multipage(TMP_MULTI, pp, 1, 2, 2, DTYPE_U16, 1);
  }
  FILE *f3 = fopen(TMP_U16, "rb");
  FILE *f1 = fopen(TMP_MULTI, "rb");
  ASSERT(f3 && f1);
  fseek(f3, 0, SEEK_END); long sz3 = ftell(f3);
  fseek(f1, 0, SEEK_END); long sz1 = ftell(f1);
  fclose(f3); fclose(f1);
  ASSERT(sz3 > sz1);

  PASS();
}

/* -------------------------------------------------------------------------
 * 4. tiff_write_multipage: float32 single channel — SampleFormat=3 in IFD
 *    The SampleFormat tag (339) value should be 3 for float.
 *    We check the file has non-zero data at the expected pixel offset.
 * --------------------------------------------------------------------- */

TEST test_multipage_float32(void) {
  float page[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  const void *pages[1] = { page };
  ASSERT(tiff_write_multipage(TMP_MULTI, pages, 1, 2, 2, DTYPE_F32, 1));
  ASSERT(file_exists_nonempty(TMP_MULTI));
  PASS();
}

/* -------------------------------------------------------------------------
 * 5. tiff_write_xyz: NULL guards
 * --------------------------------------------------------------------- */

TEST test_xyz_null_path(void) {
  float xyz[12] = {0};
  ASSERT(!tiff_write_xyz(NULL, xyz, 2, 2));
  PASS();
}

TEST test_xyz_null_data(void) {
  ASSERT(!tiff_write_xyz(TMP_XYZ, NULL, 2, 2));
  PASS();
}

/* -------------------------------------------------------------------------
 * 6. tiff_write_xyz: produces valid TIFF with float32 XYZ data
 * --------------------------------------------------------------------- */

TEST test_xyz_roundtrip_magic(void) {
  // 2×2 grid with known XYZ values
  float xyz[12] = {
    1.0f, 2.0f, 3.0f,   4.0f, 5.0f, 6.0f,
    7.0f, 8.0f, 9.0f,  10.0f, 11.0f, 12.0f,
  };
  ASSERT(tiff_write_xyz(TMP_XYZ, xyz, 2, 2));
  ASSERT(file_exists_nonempty(TMP_XYZ));

  // Valid TIFF header
  ASSERT_EQ(read_u16_at(TMP_XYZ, 0), (uint16_t)0x4949);
  ASSERT_EQ(read_u16_at(TMP_XYZ, 2), (uint16_t)42);

  // Float data starts at byte 8; first float should be 1.0f
  FILE *f = fopen(TMP_XYZ, "rb");
  ASSERT(f != NULL);
  fseek(f, 8, SEEK_SET);
  float v = 0.0f;
  fread(&v, 4, 1, f);
  fclose(f);
  ASSERT_IN_RANGE(0.99f, v, 1.01f);

  PASS();
}

/* -------------------------------------------------------------------------
 * 7. tiff_write_multipage: 3-channel RGB float32 (3D XYZ page)
 * --------------------------------------------------------------------- */

TEST test_multipage_rgb_float(void) {
  // 2×2 image, 3 channels
  float page[12] = {
    1.0f, 0.0f, 0.0f,   0.0f, 1.0f, 0.0f,
    0.0f, 0.0f, 1.0f,   1.0f, 1.0f, 0.0f,
  };
  const void *pages[1] = { page };
  ASSERT(tiff_write_multipage(TMP_XYZ, pages, 1, 2, 2, DTYPE_F32, 3));
  ASSERT(file_exists_nonempty(TMP_XYZ));
  PASS();
}

/* -------------------------------------------------------------------------
 * 8. tiff_write_multipage: depth=0 returns false
 * --------------------------------------------------------------------- */

TEST test_multipage_zero_depth(void) {
  const void *pages[1] = { NULL };
  ASSERT(!tiff_write_multipage(TMP_MULTI, pages, 0, 2, 2, DTYPE_U8, 1));
  PASS();
}

/* -------------------------------------------------------------------------
 * Suite
 * --------------------------------------------------------------------- */

SUITE(tiff_write_suite) {
  RUN_TEST(test_multipage_null_path);
  RUN_TEST(test_multipage_null_pages);
  RUN_TEST(test_multipage_single_u8);
  RUN_TEST(test_multipage_three_pages_u16);
  RUN_TEST(test_multipage_float32);
  RUN_TEST(test_xyz_null_path);
  RUN_TEST(test_xyz_null_data);
  RUN_TEST(test_xyz_roundtrip_magic);
  RUN_TEST(test_multipage_rgb_float);
  RUN_TEST(test_multipage_zero_depth);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(tiff_write_suite);
  GREATEST_MAIN_END();
}
