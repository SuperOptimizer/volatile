#include "greatest.h"
#include "core/gridstore.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static char g_tmpdir[256];

static void setup_tmpdir(void) {
  snprintf(g_tmpdir, sizeof(g_tmpdir), "/tmp/gs_test_XXXXXX");
  mkdtemp(g_tmpdir);
}

static void remove_tmpdir(void) {
  char cmd[320];
  snprintf(cmd, sizeof(cmd), "rm -rf %s", g_tmpdir);
  (void)system(cmd);
}

static const int64_t SHAPE[3] = {64, 64, 64};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_new_free(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);
  ASSERT(g != NULL);
  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_write_read_roundtrip(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);
  ASSERT(g != NULL);

  int64_t coords[3] = {0, 1, 2};
  uint8_t data[16];
  for (int i = 0; i < 16; i++) data[i] = (uint8_t)i;

  bool ok = gridstore_write(g, coords, data, sizeof(data));
  ASSERT(ok);

  size_t len = 0;
  uint8_t *buf = gridstore_read(g, coords, &len);
  ASSERT(buf != NULL);
  ASSERT_EQ(sizeof(data), len);
  ASSERT_EQ(0, memcmp(data, buf, len));
  free(buf);

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_exists_absent(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);

  int64_t coords[3] = {5, 5, 5};
  ASSERT(!gridstore_exists(g, coords));

  uint8_t b = 42;
  gridstore_write(g, coords, &b, 1);
  ASSERT(gridstore_exists(g, coords));

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_read_absent_returns_null(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);

  int64_t coords[3] = {99, 99, 99};
  size_t len = 0;
  uint8_t *buf = gridstore_read(g, coords, &len);
  ASSERT_EQ(NULL, buf);

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_count(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);

  ASSERT_EQ(0, gridstore_count(g));

  uint8_t b = 1;
  int64_t c0[3] = {0, 0, 0};
  int64_t c1[3] = {1, 0, 0};
  int64_t c2[3] = {0, 2, 0};
  gridstore_write(g, c0, &b, 1);
  gridstore_write(g, c1, &b, 1);
  gridstore_write(g, c2, &b, 1);
  ASSERT_EQ(3, gridstore_count(g));

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_overwrite(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);

  int64_t coords[3] = {0, 0, 0};
  uint8_t orig[4] = {1, 2, 3, 4};
  uint8_t updated[4] = {9, 8, 7, 6};

  gridstore_write(g, coords, orig, sizeof(orig));
  gridstore_write(g, coords, updated, sizeof(updated));

  size_t len = 0;
  uint8_t *buf = gridstore_read(g, coords, &len);
  ASSERT(buf != NULL);
  ASSERT_EQ(sizeof(updated), len);
  ASSERT_EQ(0, memcmp(updated, buf, len));
  free(buf);

  // Count should still be 1
  ASSERT_EQ(1, gridstore_count(g));

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

TEST test_large_chunk(void) {
  setup_tmpdir();
  gridstore *g = gridstore_new(g_tmpdir, SHAPE);

  size_t sz = 64 * 64 * 64;
  uint8_t *data = malloc(sz);
  for (size_t i = 0; i < sz; i++) data[i] = (uint8_t)(i % 251);

  int64_t coords[3] = {0, 0, 0};
  ASSERT(gridstore_write(g, coords, data, sz));

  size_t len = 0;
  uint8_t *buf = gridstore_read(g, coords, &len);
  ASSERT(buf != NULL);
  ASSERT_EQ(sz, len);
  ASSERT_EQ(0, memcmp(data, buf, sz));
  free(buf);
  free(data);

  gridstore_free(g);
  remove_tmpdir();
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(gridstore_suite) {
  RUN_TEST(test_new_free);
  RUN_TEST(test_write_read_roundtrip);
  RUN_TEST(test_exists_absent);
  RUN_TEST(test_read_absent_returns_null);
  RUN_TEST(test_count);
  RUN_TEST(test_overwrite);
  RUN_TEST(test_large_chunk);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(gridstore_suite);
  GREATEST_MAIN_END();
}
