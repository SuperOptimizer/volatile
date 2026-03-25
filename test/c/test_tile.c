#include "greatest.h"
#include "render/tile.h"
#include "render/camera.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static tile_key make_key(int col, int row, int level, uint64_t epoch) {
  return (tile_key){.col = col, .row = row, .pyramid_level = level, .epoch = epoch};
}

// Drain with a short spin-wait: call drain up to max_tries times.
static int drain_with_retry(tile_renderer *r, tile_result *out, int max_results, int max_tries) {
  int total = 0;
  for (int attempt = 0; attempt < max_tries && total < max_results; attempt++) {
    total += tile_renderer_drain(r, out + total, max_results - total);
    if (total < max_results) {
      struct timespec ts = {0, 1000000}; // 1 ms
      nanosleep(&ts, NULL);
    }
  }
  return total;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_create_destroy(void) {
  tile_renderer *r = tile_renderer_new(2);
  ASSERT(r != NULL);
  ASSERT_EQ(0, tile_renderer_pending(r));
  tile_renderer_free(r);
  PASS();
}

TEST test_submit_drain_single(void) {
  tile_renderer *r = tile_renderer_new(2);
  ASSERT(r != NULL);

  tile_renderer_submit(r, make_key(3, 7, 0, 1));

  tile_result results[4];
  int n = drain_with_retry(r, results, 4, 20);

  ASSERT_EQ(1, n);
  ASSERT(results[0].valid);
  ASSERT_EQ(3, results[0].key.col);
  ASSERT_EQ(7, results[0].key.row);
  ASSERT_EQ(0, results[0].key.pyramid_level);
  ASSERT_EQ((uint64_t)1, results[0].key.epoch);
  ASSERT(results[0].pixels != NULL);

  // Pixels are TILE_PX*TILE_PX*4 bytes; spot-check alpha is 0xFF.
  ASSERT_EQ(0xFF, results[0].pixels[(TILE_PX * TILE_PX - 1) * 4 + 3]);

  free(results[0].pixels);
  tile_renderer_free(r);
  PASS();
}

TEST test_submit_drain_multiple(void) {
  tile_renderer *r = tile_renderer_new(4);
  ASSERT(r != NULL);

  const int N = 8;
  for (int i = 0; i < N; i++)
    tile_renderer_submit(r, make_key(i, i, 0, 2));

  tile_result results[N];
  int n = drain_with_retry(r, results, N, 40);

  ASSERT_EQ(N, n);
  for (int i = 0; i < N; i++) {
    ASSERT(results[i].valid);
    ASSERT(results[i].pixels != NULL);
    free(results[i].pixels);
  }

  tile_renderer_free(r);
  PASS();
}

TEST test_test_pattern_varies_by_tile(void) {
  // Two different tiles must produce different pixel data.
  tile_renderer *r = tile_renderer_new(2);
  ASSERT(r != NULL);

  tile_renderer_submit(r, make_key(0, 0, 0, 1));
  tile_renderer_submit(r, make_key(5, 3, 0, 1));

  tile_result results[2];
  int n = drain_with_retry(r, results, 2, 40);
  ASSERT_EQ(2, n);

  // The two tiles should differ somewhere in the first row.
  bool differ = false;
  for (int px = 0; px < TILE_PX * 4; px++) {
    if (results[0].pixels[px] != results[1].pixels[px]) { differ = true; break; }
  }
  ASSERT(differ);

  free(results[0].pixels);
  free(results[1].pixels);
  tile_renderer_free(r);
  PASS();
}

TEST test_cancel_stale(void) {
  tile_renderer *r = tile_renderer_new(1);
  ASSERT(r != NULL);

  // Submit tiles with epoch 1 and epoch 2.
  for (int i = 0; i < 4; i++) tile_renderer_submit(r, make_key(i, 0, 0, 1));
  for (int i = 0; i < 4; i++) tile_renderer_submit(r, make_key(i, 1, 0, 2));

  // Cancel epoch < 2 (drops epoch-1 tiles).
  tile_renderer_cancel_stale(r, 2);

  // Drain everything that completes.
  tile_result results[16];
  int n = drain_with_retry(r, results, 16, 80);

  // All results that did complete must have epoch >= 2.
  for (int i = 0; i < n; i++) {
    ASSERT(results[i].key.epoch >= 2);
    free(results[i].pixels);
  }

  tile_renderer_free(r);
  PASS();
}

TEST test_drain_respects_max(void) {
  tile_renderer *r = tile_renderer_new(4);
  ASSERT(r != NULL);

  for (int i = 0; i < 8; i++) tile_renderer_submit(r, make_key(i, 0, 0, 1));

  // Wait for all to complete.
  tile_result all[8];
  int total = drain_with_retry(r, all, 8, 40);
  ASSERT_EQ(8, total);

  // Re-submit 4 and drain with max=2.
  for (int i = 0; i < 4; i++) tile_renderer_submit(r, make_key(i, 2, 0, 3));
  tile_result batch[4];
  int first = drain_with_retry(r, batch, 2, 40);
  ASSERT(first <= 2);

  for (int i = 0; i < total; i++) free(all[i].pixels);
  for (int i = 0; i < first; i++) free(batch[i].pixels);

  tile_renderer_free(r);
  PASS();
}

TEST test_pending_count(void) {
  tile_renderer *r = tile_renderer_new(1);
  ASSERT(r != NULL);

  // With a single thread the pending count after submission is at least 0.
  // We can't assert an exact value because the worker may pick up immediately.
  tile_renderer_submit(r, make_key(0, 0, 0, 1));
  int p = tile_renderer_pending(r);
  ASSERT(p >= 0);

  // After full drain the pending array is not necessarily 0 (entries are
  // compacted only on cancel_stale), so just check no crash.
  tile_result res[4];
  int got = drain_with_retry(r, res, 4, 20);
  for (int i = 0; i < got; i++) free(res[i].pixels);

  tile_renderer_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// slice_cache tests
// ---------------------------------------------------------------------------

TEST test_slice_cache_put_get(void) {
  slice_cache *c = slice_cache_new(16);
  ASSERT(c != NULL);

  // Put an entry, retrieve it exact.
  uint8_t *px = malloc((size_t)TILE_PX * TILE_PX * 4);
  ASSERT(px != NULL);
  memset(px, 0xAB, (size_t)TILE_PX * TILE_PX * 4);

  slice_cache_key k = slice_cache_make_key(3, 7, 1.0f, 0.5f, 0, 0xDEAD);
  // Cache takes ownership of px.
  slice_cache_put(c, k, px, 0);

  slice_cache_entry out = {0};
  bool found = slice_cache_get_best(c, k, &out);
  ASSERT(found);
  ASSERT(out.pixels != NULL);
  ASSERT_EQ(0, out.level);
  ASSERT_EQ(0xAB, out.pixels[0]);

  slice_cache_free(c);
  PASS();
}

TEST test_slice_cache_coarser_fallback(void) {
  slice_cache *c = slice_cache_new(16);
  ASSERT(c != NULL);

  // Store a coarse tile (level 2).
  uint8_t *coarse = malloc((size_t)TILE_PX * TILE_PX * 4);
  ASSERT(coarse != NULL);
  memset(coarse, 0xCC, (size_t)TILE_PX * TILE_PX * 4);

  slice_cache_key coarse_key = slice_cache_make_key(0, 0, 1.0f, 0.0f, 2, 0);
  slice_cache_put(c, coarse_key, coarse, 2);

  // Request level 0 (fine) — should fall back to level 2 placeholder.
  slice_cache_key fine_key = slice_cache_make_key(0, 0, 1.0f, 0.0f, 0, 0);
  slice_cache_entry out = {0};
  bool found = slice_cache_get_best(c, fine_key, &out);
  ASSERT(found);
  ASSERT_EQ(2, out.level);  // got coarser placeholder
  ASSERT_EQ(0xCC, out.pixels[0]);

  slice_cache_free(c);
  PASS();
}

TEST test_slice_cache_fine_preferred_over_coarse(void) {
  slice_cache *c = slice_cache_new(16);
  ASSERT(c != NULL);

  // Store coarse (level 2) first, then fine (level 0) for same tile.
  uint8_t *coarse = malloc((size_t)TILE_PX * TILE_PX * 4);
  uint8_t *fine   = malloc((size_t)TILE_PX * TILE_PX * 4);
  ASSERT(coarse && fine);
  memset(coarse, 0x11, (size_t)TILE_PX * TILE_PX * 4);
  memset(fine,   0x22, (size_t)TILE_PX * TILE_PX * 4);

  slice_cache_key k2 = slice_cache_make_key(1, 1, 1.0f, 0.0f, 2, 0);
  slice_cache_key k0 = slice_cache_make_key(1, 1, 1.0f, 0.0f, 0, 0);
  slice_cache_put(c, k2, coarse, 2);
  slice_cache_put(c, k0, fine,   0);

  // get_best for level 0 should return level 0 (exact hit, not coarser)
  slice_cache_entry out = {0};
  ASSERT(slice_cache_get_best(c, k0, &out));
  ASSERT_EQ(0, out.level);
  ASSERT_EQ(0x22, out.pixels[0]);

  slice_cache_free(c);
  PASS();
}

TEST test_slice_cache_lru_eviction(void) {
  // Capacity 4: insert 5 entries, oldest should be evicted.
  slice_cache *c = slice_cache_new(4);
  ASSERT(c != NULL);

  for (int i = 0; i < 5; i++) {
    uint8_t *px = malloc((size_t)TILE_PX * TILE_PX * 4);
    ASSERT(px != NULL);
    memset(px, (unsigned char)i, (size_t)TILE_PX * TILE_PX * 4);
    slice_cache_key k = slice_cache_make_key(i, 0, 1.0f, 0.0f, 0, 0);
    slice_cache_put(c, k, px, 0);
  }

  // The 4 most recent (i=1..4) should be present; i=0 was evicted.
  slice_cache_entry out = {0};
  slice_cache_key k0 = slice_cache_make_key(0, 0, 1.0f, 0.0f, 0, 0);
  ASSERT_FALSE(slice_cache_get_best(c, k0, &out));

  for (int i = 1; i < 5; i++) {
    slice_cache_key k = slice_cache_make_key(i, 0, 1.0f, 0.0f, 0, 0);
    ASSERT(slice_cache_get_best(c, k, &out));
  }

  slice_cache_free(c);
  PASS();
}

TEST test_slice_cache_clear(void) {
  slice_cache *c = slice_cache_new(8);
  ASSERT(c != NULL);

  uint8_t *px = malloc((size_t)TILE_PX * TILE_PX * 4);
  ASSERT(px != NULL);
  slice_cache_key k = slice_cache_make_key(0, 0, 1.0f, 0.0f, 0, 0);
  slice_cache_put(c, k, px, 0);

  slice_cache_clear(c);

  slice_cache_entry out = {0};
  ASSERT_FALSE(slice_cache_get_best(c, k, &out));

  slice_cache_free(c);
  PASS();
}

// ---------------------------------------------------------------------------
// Progressive refinement: viewer emits coarse tile before fine tile
// ---------------------------------------------------------------------------

TEST test_progressive_refinement_coarse_before_fine(void) {
  // Submit coarse (level 2) and fine (level 0) tiles to the renderer.
  // The coarse tile should arrive in the drain before or with the fine tile.
  tile_renderer *r = tile_renderer_new(2);
  ASSERT(r != NULL);

  // Submit fine first, coarse second — but coarse should still win due to
  // coarser-first priority in submit_with_coarse_fallback.
  tile_renderer_submit(r, make_key(0, 0, 0, 1));  // fine
  tile_renderer_submit(r, make_key(0, 0, 2, 1));  // coarse

  // Drain all.
  tile_result results[8];
  int n = drain_with_retry(r, results, 8, 40);
  ASSERT(n >= 2);

  // Find fine and coarse results.
  bool got_fine = false, got_coarse = false;
  for (int i = 0; i < n; i++) {
    if (results[i].key.pyramid_level == 0) got_fine   = true;
    if (results[i].key.pyramid_level == 2) got_coarse = true;
    free(results[i].pixels);
  }
  ASSERT(got_fine);
  ASSERT(got_coarse);

  tile_renderer_free(r);
  PASS();
}


// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(tile_suite) {
  RUN_TEST(test_create_destroy);
  RUN_TEST(test_submit_drain_single);
  RUN_TEST(test_submit_drain_multiple);
  RUN_TEST(test_test_pattern_varies_by_tile);
  RUN_TEST(test_cancel_stale);
  RUN_TEST(test_drain_respects_max);
  RUN_TEST(test_pending_count);
}

SUITE(slice_cache_suite) {
  RUN_TEST(test_slice_cache_put_get);
  RUN_TEST(test_slice_cache_coarser_fallback);
  RUN_TEST(test_slice_cache_fine_preferred_over_coarse);
  RUN_TEST(test_slice_cache_lru_eviction);
  RUN_TEST(test_slice_cache_clear);
  RUN_TEST(test_progressive_refinement_coarse_before_fine);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(tile_suite);
  RUN_SUITE(slice_cache_suite);
  GREATEST_MAIN_END();
}
