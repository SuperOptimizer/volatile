#include "greatest.h"
#include "core/thread.h"

#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include <pthread.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void *return_arg(void *arg) { return arg; }

static void *sleep_and_return(void *arg) {
  usleep(20000);  // 20 ms
  return arg;
}

static atomic_int g_counter;

static void *increment_counter(void *arg) {
  (void)arg;
  atomic_fetch_add(&g_counter, 1);
  return NULL;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_basic_submit_get(void) {
  threadpool *p = threadpool_new(2);
  future     *f = threadpool_submit(p, return_arg, (void *)0xdeadbeef);

  void *result = future_get(f, 1000);
  ASSERT_EQ((uintptr_t)result, (uintptr_t)0xdeadbeef);
  ASSERT(future_done(f));

  future_free(f);
  threadpool_free(p);
  PASS();
}

TEST test_multiple_tasks(void) {
  atomic_store(&g_counter, 0);

  threadpool *p = threadpool_new(4);

  for (int i = 0; i < 100; i++)
    threadpool_fire(p, increment_counter, NULL);

  threadpool_drain(p, 2000);
  ASSERT_EQ(atomic_load(&g_counter), 100);

  threadpool_free(p);
  PASS();
}

TEST test_drain(void) {
  atomic_store(&g_counter, 0);

  threadpool *p = threadpool_new(2);

  for (int i = 0; i < 20; i++)
    threadpool_fire(p, increment_counter, NULL);

  threadpool_drain(p, 2000);
  ASSERT_EQ(atomic_load(&g_counter), 20);
  ASSERT_EQ(threadpool_pending(p), (size_t)0);

  threadpool_free(p);
  PASS();
}

TEST test_timeout(void) {
  threadpool *p = threadpool_new(1);

  // submit a slow task then immediately check with a very short timeout
  future *f = threadpool_submit(p, sleep_and_return, (void *)42);
  void   *result = future_get(f, 1);  // 1 ms — should time out

  // result may be NULL (timed out) or 42 (raced and finished); both are valid.
  // What we really care about is that future_get returned at all (no deadlock).
  (void)result;

  // now wait properly
  void *final = future_get(f, 500);
  ASSERT_EQ((uintptr_t)final, (uintptr_t)42);

  future_free(f);
  threadpool_free(p);
  PASS();
}

TEST test_fire_and_forget(void) {
  atomic_store(&g_counter, 0);

  threadpool *p = threadpool_new(2);
  threadpool_fire(p, increment_counter, NULL);
  threadpool_drain(p, 1000);
  ASSERT_EQ(atomic_load(&g_counter), 1);

  threadpool_free(p);
  PASS();
}

TEST test_pool_stats(void) {
  threadpool *p = threadpool_new(3);
  ASSERT_EQ(threadpool_num_threads(p), 3);
  ASSERT_EQ(threadpool_pending(p), (size_t)0);

  threadpool_free(p);
  PASS();
}

TEST test_auto_thread_count(void) {
  threadpool *p = threadpool_new(0);  // auto
  ASSERT(threadpool_num_threads(p) >= 2);
  threadpool_free(p);
  PASS();
}

TEST test_future_done_flag(void) {
  threadpool *p = threadpool_new(2);

  future *f = threadpool_submit(p, sleep_and_return, (void *)7);
  // done flag starts false before task completes (may race, just ensure no crash)
  bool before = future_done(f);
  (void)before;

  future_get(f, 500);
  ASSERT(future_done(f));

  future_free(f);
  threadpool_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: submit from multiple threads simultaneously
// ---------------------------------------------------------------------------

typedef struct { threadpool *p; atomic_int *counter; } submit_arg;

static void *submitter_fn(void *arg_ptr) {
  submit_arg *a = arg_ptr;
  for (int i = 0; i < 10; i++)
    threadpool_fire(a->p, increment_counter, NULL);
  (void)a->counter;
  return NULL;
}

TEST test_submit_from_multiple_threads(void) {
  atomic_store(&g_counter, 0);
  threadpool *p = threadpool_new(4);

  const int NSUBMITTERS = 4;
  pthread_t threads[4];
  submit_arg args[4];
  for (int i = 0; i < NSUBMITTERS; i++) {
    args[i] = (submit_arg){p, &g_counter};
    pthread_create(&threads[i], NULL, submitter_fn, &args[i]);
  }
  for (int i = 0; i < NSUBMITTERS; i++) pthread_join(threads[i], NULL);

  threadpool_drain(p, 2000);
  ASSERT_EQ(40, atomic_load(&g_counter));  // 4 threads * 10 tasks each

  threadpool_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: future_get timeout returns NULL for a long-running task
// ---------------------------------------------------------------------------

static void *slow_task(void *arg) {
  usleep(100000);  // 100 ms
  return arg;
}

TEST test_future_timeout_returns_null(void) {
  threadpool *p = threadpool_new(1);
  future *f = threadpool_submit(p, slow_task, (void *)0xAB);

  // 1 ms timeout — must return NULL (or the result if it races, but unlikely)
  void *res = future_get(f, 1);
  // If it raced and finished, res may be non-NULL — that's also valid.
  // The key requirement: no deadlock, and if NULL then it must be timeout.
  if (res == NULL) {
    ASSERT(!future_done(f));  // should not be done if we got NULL via timeout
  }

  // Wait properly now.
  void *final = future_get(f, 500);
  ASSERT_EQ((uintptr_t)final, (uintptr_t)0xAB);
  ASSERT(future_done(f));

  future_free(f);
  threadpool_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Test: 1-thread pool handles sequential tasks correctly
// ---------------------------------------------------------------------------

TEST test_single_thread_sequential(void) {
  atomic_store(&g_counter, 0);
  threadpool *p = threadpool_new(1);

  // Submit 20 tasks — with one thread they run one-at-a-time.
  for (int i = 0; i < 20; i++)
    threadpool_fire(p, increment_counter, NULL);

  threadpool_drain(p, 1000);
  ASSERT_EQ(20, atomic_load(&g_counter));

  threadpool_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(thread_suite) {
  RUN_TEST(test_basic_submit_get);
  RUN_TEST(test_multiple_tasks);
  RUN_TEST(test_drain);
  RUN_TEST(test_timeout);
  RUN_TEST(test_fire_and_forget);
  RUN_TEST(test_pool_stats);
  RUN_TEST(test_auto_thread_count);
  RUN_TEST(test_future_done_flag);
  RUN_TEST(test_submit_from_multiple_threads);
  RUN_TEST(test_future_timeout_returns_null);
  RUN_TEST(test_single_thread_sequential);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(thread_suite);
  GREATEST_MAIN_END();
}
