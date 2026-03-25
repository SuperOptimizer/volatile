#include "greatest.h"
#include "core/hash.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// String-keyed map tests
// ---------------------------------------------------------------------------

TEST test_str_insert_get(void) {
  hash_map *m = hash_map_new();
  ASSERT(m != NULL);

  int a = 1, b = 2;
  ASSERT_EQ(true,  hash_map_put(m, "alpha", &a));
  ASSERT_EQ(true,  hash_map_put(m, "beta",  &b));
  ASSERT_EQ(2u,    hash_map_len(m));

  ASSERT_EQ(&a, hash_map_get(m, "alpha"));
  ASSERT_EQ(&b, hash_map_get(m, "beta"));
  ASSERT_EQ(NULL, hash_map_get(m, "gamma"));

  hash_map_free(m);
  PASS();
}

TEST test_str_put_update(void) {
  hash_map *m = hash_map_new();
  int a = 1, b = 99;

  ASSERT_EQ(true,  hash_map_put(m, "key", &a)); // new
  ASSERT_EQ(false, hash_map_put(m, "key", &b)); // update
  ASSERT_EQ(1u,    hash_map_len(m));
  ASSERT_EQ(&b,    hash_map_get(m, "key"));

  hash_map_free(m);
  PASS();
}

TEST test_str_delete(void) {
  hash_map *m = hash_map_new();
  int v = 7;

  hash_map_put(m, "x", &v);
  ASSERT_EQ(true,  hash_map_del(m, "x"));
  ASSERT_EQ(false, hash_map_del(m, "x")); // already gone
  ASSERT_EQ(0u,    hash_map_len(m));
  ASSERT_EQ(NULL,  hash_map_get(m, "x"));

  hash_map_free(m);
  PASS();
}

TEST test_str_delete_missing(void) {
  hash_map *m = hash_map_new();
  ASSERT_EQ(false, hash_map_del(m, "nope"));
  hash_map_free(m);
  PASS();
}

// Force a rehash: insert enough entries to exceed the 75% load factor on the
// initial capacity-16 table (need > 12 entries).
TEST test_str_grow_rehash(void) {
  hash_map *m = hash_map_new();
  char key[32];
  int  vals[32];

  for (int i = 0; i < 32; i++) {
    vals[i] = i;
    snprintf(key, sizeof(key), "key_%d", i);
    hash_map_put(m, key, &vals[i]);
  }

  ASSERT_EQ(32u, hash_map_len(m));

  // Verify all entries survive rehash
  for (int i = 0; i < 32; i++) {
    snprintf(key, sizeof(key), "key_%d", i);
    ASSERT_EQ(&vals[i], hash_map_get(m, key));
  }

  hash_map_free(m);
  PASS();
}

TEST test_str_collision_handling(void) {
  // These two strings have different hashes but exercise dense probe chains
  // by inserting many entries into a small map.
  hash_map *m = hash_map_new();
  int vals[20];
  char key[32];

  for (int i = 0; i < 20; i++) {
    vals[i] = i * 10;
    snprintf(key, sizeof(key), "collision_%d", i);
    hash_map_put(m, key, &vals[i]);
  }

  for (int i = 0; i < 20; i++) {
    snprintf(key, sizeof(key), "collision_%d", i);
    int *got = hash_map_get(m, key);
    ASSERT(got != NULL);
    ASSERT_EQ(i * 10, *got);
  }

  hash_map_free(m);
  PASS();
}

TEST test_str_iteration(void) {
  hash_map *m = hash_map_new();
  int a = 1, b = 2, c = 3;
  hash_map_put(m, "a", &a);
  hash_map_put(m, "b", &b);
  hash_map_put(m, "c", &c);

  hash_map_iter  *it  = hash_map_iter_new(m);
  hash_map_entry  ent = {0};
  size_t count = 0;
  int    sum   = 0;

  while (hash_map_iter_next(it, &ent)) {
    count++;
    sum += *(int *)ent.val;
  }

  hash_map_iter_free(it);
  ASSERT_EQ(3u, count);
  ASSERT_EQ(6,  sum);

  hash_map_free(m);
  PASS();
}

TEST test_str_iteration_empty(void) {
  hash_map *m = hash_map_new();
  hash_map_iter  *it  = hash_map_iter_new(m);
  hash_map_entry  ent = {0};
  ASSERT_EQ(false, hash_map_iter_next(it, &ent));
  hash_map_iter_free(it);
  hash_map_free(m);
  PASS();
}

// Delete-then-reinsert to exercise backward-shift correctness
TEST test_str_delete_reinsert(void) {
  hash_map *m = hash_map_new();
  int v1 = 10, v2 = 20;

  hash_map_put(m, "foo", &v1);
  hash_map_del(m, "foo");
  hash_map_put(m, "foo", &v2);

  ASSERT_EQ(1u,   hash_map_len(m));
  ASSERT_EQ(&v2,  hash_map_get(m, "foo"));

  hash_map_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Integer-keyed map tests
// ---------------------------------------------------------------------------

TEST test_int_insert_get(void) {
  hash_map_int *m = hash_map_int_new();
  ASSERT(m != NULL);

  int a = 100, b = 200;
  ASSERT_EQ(true,  hash_map_int_put(m, 1,   &a));
  ASSERT_EQ(true,  hash_map_int_put(m, 999, &b));
  ASSERT_EQ(2u,    hash_map_int_len(m));

  ASSERT_EQ(&a,   hash_map_int_get(m, 1));
  ASSERT_EQ(&b,   hash_map_int_get(m, 999));
  ASSERT_EQ(NULL, hash_map_int_get(m, 42));

  hash_map_int_free(m);
  PASS();
}

TEST test_int_put_update(void) {
  hash_map_int *m = hash_map_int_new();
  int a = 1, b = 2;

  ASSERT_EQ(true,  hash_map_int_put(m, 7, &a));
  ASSERT_EQ(false, hash_map_int_put(m, 7, &b));
  ASSERT_EQ(1u,    hash_map_int_len(m));
  ASSERT_EQ(&b,    hash_map_int_get(m, 7));

  hash_map_int_free(m);
  PASS();
}

TEST test_int_delete(void) {
  hash_map_int *m = hash_map_int_new();
  int v = 5;

  hash_map_int_put(m, 42, &v);
  ASSERT_EQ(true,  hash_map_int_del(m, 42));
  ASSERT_EQ(false, hash_map_int_del(m, 42));
  ASSERT_EQ(0u,    hash_map_int_len(m));
  ASSERT_EQ(NULL,  hash_map_int_get(m, 42));

  hash_map_int_free(m);
  PASS();
}

TEST test_int_grow_rehash(void) {
  hash_map_int *m = hash_map_int_new();
  int vals[64];

  for (int i = 0; i < 64; i++) {
    vals[i] = i * 3;
    hash_map_int_put(m, (uint64_t)i, &vals[i]);
  }

  ASSERT_EQ(64u, hash_map_int_len(m));

  for (int i = 0; i < 64; i++) {
    int *got = hash_map_int_get(m, (uint64_t)i);
    ASSERT(got != NULL);
    ASSERT_EQ(i * 3, *got);
  }

  hash_map_int_free(m);
  PASS();
}

TEST test_int_large_keys(void) {
  hash_map_int *m = hash_map_int_new();
  int v = 77;

  hash_map_int_put(m, UINT64_C(0xDEADBEEFCAFEBABE), &v);
  ASSERT_EQ(&v, hash_map_int_get(m, UINT64_C(0xDEADBEEFCAFEBABE)));

  hash_map_int_free(m);
  PASS();
}

TEST test_int_zero_key(void) {
  hash_map_int *m = hash_map_int_new();
  int v = 42;

  hash_map_int_put(m, 0, &v);
  ASSERT_EQ(&v, hash_map_int_get(m, 0));

  hash_map_int_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suites + main
// ---------------------------------------------------------------------------

SUITE(str_map_suite) {
  RUN_TEST(test_str_insert_get);
  RUN_TEST(test_str_put_update);
  RUN_TEST(test_str_delete);
  RUN_TEST(test_str_delete_missing);
  RUN_TEST(test_str_grow_rehash);
  RUN_TEST(test_str_collision_handling);
  RUN_TEST(test_str_iteration);
  RUN_TEST(test_str_iteration_empty);
  RUN_TEST(test_str_delete_reinsert);
}

SUITE(int_map_suite) {
  RUN_TEST(test_int_insert_get);
  RUN_TEST(test_int_put_update);
  RUN_TEST(test_int_delete);
  RUN_TEST(test_int_grow_rehash);
  RUN_TEST(test_int_large_keys);
  RUN_TEST(test_int_zero_key);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(str_map_suite);
  RUN_SUITE(int_map_suite);
  GREATEST_MAIN_END();
}
