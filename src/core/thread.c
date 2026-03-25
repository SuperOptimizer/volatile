#define _POSIX_C_SOURCE 200809L

#include "core/thread.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

typedef struct task {
  task_fn      fn;
  void        *arg;
  future      *fut;    // NULL for fire-and-forget
  struct task *next;
} task;

struct future {
  pthread_mutex_t mu;
  pthread_cond_t  cv;
  void           *result;
  bool            done;
};

struct threadpool {
  pthread_mutex_t mu;
  pthread_cond_t  work_cv;   // signalled when a task is enqueued
  pthread_cond_t  drain_cv;  // signalled when pending count hits zero

  task  *head;
  task  *tail;
  size_t pending;            // tasks in queue + tasks being executed

  int          num_threads;
  pthread_t   *threads;
  bool         shutdown;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void timespec_add_ms(struct timespec *ts, int ms) {
  ts->tv_sec  += ms / 1000;
  ts->tv_nsec += (long)(ms % 1000) * 1000000L;
  if (ts->tv_nsec >= 1000000000L) {
    ts->tv_sec++;
    ts->tv_nsec -= 1000000000L;
  }
}

static struct timespec deadline_from_now_ms(int ms) {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  timespec_add_ms(&ts, ms);
  return ts;
}

// ---------------------------------------------------------------------------
// Worker thread
// ---------------------------------------------------------------------------

static void *worker_run(void *arg) {
  threadpool *p = arg;

  for (;;) {
    pthread_mutex_lock(&p->mu);

    while (!p->shutdown && p->head == NULL)
      pthread_cond_wait(&p->work_cv, &p->mu);

    if (p->shutdown && p->head == NULL) {
      pthread_mutex_unlock(&p->mu);
      break;
    }

    task *t = p->head;
    p->head = t->next;
    if (p->head == NULL) p->tail = NULL;

    pthread_mutex_unlock(&p->mu);

    // execute
    void *result = t->fn(t->arg);

    // resolve future if present
    if (t->fut) {
      pthread_mutex_lock(&t->fut->mu);
      t->fut->result = result;
      t->fut->done   = true;
      pthread_cond_broadcast(&t->fut->cv);
      pthread_mutex_unlock(&t->fut->mu);
    }

    free(t);

    // decrement pending and signal drain waiters if queue is empty
    pthread_mutex_lock(&p->mu);
    p->pending--;
    if (p->pending == 0)
      pthread_cond_broadcast(&p->drain_cv);
    pthread_mutex_unlock(&p->mu);
  }

  return NULL;
}

// ---------------------------------------------------------------------------
// Pool lifecycle
// ---------------------------------------------------------------------------

threadpool *threadpool_new(int num_threads) {
  if (num_threads <= 0) {
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    num_threads = (cores > 0) ? (int)(cores / 2) : 1;
    if (num_threads < 2) num_threads = 2;
  }

  threadpool *p = calloc(1, sizeof(*p));
  REQUIRE(p, "threadpool_new: calloc failed");

  pthread_mutex_init(&p->mu, NULL);
  pthread_cond_init(&p->work_cv, NULL);
  pthread_cond_init(&p->drain_cv, NULL);

  p->num_threads = num_threads;
  p->threads = calloc((size_t)num_threads, sizeof(pthread_t));
  REQUIRE(p->threads, "threadpool_new: calloc threads failed");

  for (int i = 0; i < num_threads; i++) {
    int rc = pthread_create(&p->threads[i], NULL, worker_run, p);
    REQUIRE(rc == 0, "threadpool_new: pthread_create failed: %d", rc);
  }

  return p;
}

void threadpool_free(threadpool *p) {
  if (!p) return;

  // drain first, then shut down
  pthread_mutex_lock(&p->mu);
  while (p->pending > 0)
    pthread_cond_wait(&p->drain_cv, &p->mu);
  p->shutdown = true;
  pthread_cond_broadcast(&p->work_cv);
  pthread_mutex_unlock(&p->mu);

  for (int i = 0; i < p->num_threads; i++)
    pthread_join(p->threads[i], NULL);

  pthread_cond_destroy(&p->drain_cv);
  pthread_cond_destroy(&p->work_cv);
  pthread_mutex_destroy(&p->mu);
  free(p->threads);
  free(p);
}

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

static task *make_task(task_fn fn, void *arg, future *fut) {
  task *t = malloc(sizeof(*t));
  REQUIRE(t, "make_task: malloc failed");
  t->fn   = fn;
  t->arg  = arg;
  t->fut  = fut;
  t->next = NULL;
  return t;
}

static future *make_future(void) {
  future *f = calloc(1, sizeof(*f));
  REQUIRE(f, "make_future: calloc failed");
  pthread_mutex_init(&f->mu, NULL);
  pthread_cond_init(&f->cv, NULL);
  return f;
}

static void enqueue(threadpool *p, task *t) {
  pthread_mutex_lock(&p->mu);
  REQUIRE(!p->shutdown, "threadpool_submit: pool is shut down");
  if (p->tail) {
    p->tail->next = t;
    p->tail = t;
  } else {
    p->head = p->tail = t;
  }
  p->pending++;
  pthread_cond_signal(&p->work_cv);
  pthread_mutex_unlock(&p->mu);
}

future *threadpool_submit(threadpool *p, task_fn fn, void *arg) {
  REQUIRE(p && fn, "threadpool_submit: null argument");
  future *f = make_future();
  task   *t = make_task(fn, arg, f);
  enqueue(p, t);
  return f;
}

void threadpool_fire(threadpool *p, task_fn fn, void *arg) {
  REQUIRE(p && fn, "threadpool_fire: null argument");
  task *t = make_task(fn, arg, NULL);
  enqueue(p, t);
}

// ---------------------------------------------------------------------------
// Future
// ---------------------------------------------------------------------------

void *future_get(future *f, int timeout_ms) {
  REQUIRE(f, "future_get: null future");

  pthread_mutex_lock(&f->mu);

  if (timeout_ms < 0) {
    // block indefinitely
    while (!f->done)
      pthread_cond_wait(&f->cv, &f->mu);
  } else {
    struct timespec deadline = deadline_from_now_ms(timeout_ms);
    while (!f->done) {
      int rc = pthread_cond_timedwait(&f->cv, &f->mu, &deadline);
      if (rc == ETIMEDOUT) break;
    }
  }

  void *result = f->done ? f->result : NULL;
  pthread_mutex_unlock(&f->mu);
  return result;
}

bool future_done(const future *f) {
  REQUIRE(f, "future_done: null future");
  pthread_mutex_lock((pthread_mutex_t *)&f->mu);
  bool done = f->done;
  pthread_mutex_unlock((pthread_mutex_t *)&f->mu);
  return done;
}

void future_free(future *f) {
  if (!f) return;
  pthread_cond_destroy(&f->cv);
  pthread_mutex_destroy(&f->mu);
  free(f);
}

// ---------------------------------------------------------------------------
// Drain
// ---------------------------------------------------------------------------

void threadpool_drain(threadpool *p, int timeout_ms) {
  REQUIRE(p, "threadpool_drain: null pool");

  pthread_mutex_lock(&p->mu);

  if (timeout_ms < 0) {
    while (p->pending > 0)
      pthread_cond_wait(&p->drain_cv, &p->mu);
  } else {
    struct timespec deadline = deadline_from_now_ms(timeout_ms);
    while (p->pending > 0) {
      int rc = pthread_cond_timedwait(&p->drain_cv, &p->mu, &deadline);
      if (rc == ETIMEDOUT) break;
    }
  }

  pthread_mutex_unlock(&p->mu);
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

size_t threadpool_pending(const threadpool *p) {
  REQUIRE(p, "threadpool_pending: null pool");
  pthread_mutex_lock((pthread_mutex_t *)&p->mu);
  size_t n = p->pending;
  pthread_mutex_unlock((pthread_mutex_t *)&p->mu);
  return n;
}

int threadpool_num_threads(const threadpool *p) {
  REQUIRE(p, "threadpool_num_threads: null pool");
  return p->num_threads;
}
