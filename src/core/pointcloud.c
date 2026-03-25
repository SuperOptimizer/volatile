#define _POSIX_C_SOURCE 200809L
#include "core/pointcloud.h"

#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#define PC_INIT_CAP 256

struct pointcloud {
  vec3f          *pts;
  int             count;
  int             cap;
  pthread_mutex_t mu;
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

pointcloud *pointcloud_new(void) {
  pointcloud *pc = calloc(1, sizeof(*pc));
  if (!pc) return NULL;
  pc->pts = malloc(PC_INIT_CAP * sizeof(vec3f));
  if (!pc->pts) { free(pc); return NULL; }
  pc->cap = PC_INIT_CAP;
  pthread_mutex_init(&pc->mu, NULL);
  return pc;
}

void pointcloud_free(pointcloud *pc) {
  if (!pc) return;
  pthread_mutex_destroy(&pc->mu);
  free(pc->pts);
  free(pc);
}

// ---------------------------------------------------------------------------
// Internal grow (caller must hold mu)
// ---------------------------------------------------------------------------

static bool _grow(pointcloud *pc, int need) {
  if (pc->count + need <= pc->cap) return true;
  int new_cap = pc->cap;
  while (new_cap < pc->count + need) new_cap *= 2;
  vec3f *tmp = realloc(pc->pts, (size_t)new_cap * sizeof(vec3f));
  if (!tmp) return false;
  pc->pts = tmp;
  pc->cap = new_cap;
  return true;
}

// ---------------------------------------------------------------------------
// Add
// ---------------------------------------------------------------------------

void pointcloud_add(pointcloud *pc, vec3f point) {
  pthread_mutex_lock(&pc->mu);
  if (_grow(pc, 1)) pc->pts[pc->count++] = point;
  pthread_mutex_unlock(&pc->mu);
}

void pointcloud_add_batch(pointcloud *pc, const vec3f *points, int count) {
  if (count <= 0) return;
  pthread_mutex_lock(&pc->mu);
  if (_grow(pc, count)) {
    memcpy(pc->pts + pc->count, points, (size_t)count * sizeof(vec3f));
    pc->count += count;
  }
  pthread_mutex_unlock(&pc->mu);
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int pointcloud_count(const pointcloud *pc) {
  return pc ? pc->count : 0;
}

vec3f pointcloud_get(const pointcloud *pc, int index) {
  return pc->pts[index];
}

// ---------------------------------------------------------------------------
// Parallel iteration
// ---------------------------------------------------------------------------

typedef struct {
  pointcloud        *pc;
  pointcloud_iter_fn fn;
  void              *ctx;
  int                start;
  int                end;
} _iter_arg;

static void *_iter_worker(void *arg) {
  _iter_arg *a = (_iter_arg *)arg;
  for (int i = a->start; i < a->end; i++) {
    a->fn(a->pc->pts[i], i, a->ctx);
  }
  return NULL;
}

void pointcloud_parallel_for(pointcloud *pc, pointcloud_iter_fn fn, void *ctx, int num_threads) {
  int n = pc->count;
  if (n == 0) return;

  if (num_threads <= 0) {
    long cpus = sysconf(_SC_NPROCESSORS_ONLN);
    num_threads = (cpus > 0) ? (int)cpus : 1;
  }
  if (num_threads > n) num_threads = n;

  pthread_t    *threads = malloc((size_t)num_threads * sizeof(pthread_t));
  _iter_arg    *args    = malloc((size_t)num_threads * sizeof(_iter_arg));
  if (!threads || !args) { free(threads); free(args); return; }

  int chunk = n / num_threads;
  for (int t = 0; t < num_threads; t++) {
    args[t] = (_iter_arg){
      .pc = pc, .fn = fn, .ctx = ctx,
      .start = t * chunk,
      .end   = (t == num_threads - 1) ? n : (t + 1) * chunk,
    };
    pthread_create(&threads[t], NULL, _iter_worker, &args[t]);
  }
  for (int t = 0; t < num_threads; t++) pthread_join(threads[t], NULL);
  free(threads);
  free(args);
}
