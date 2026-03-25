#pragma once
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

typedef struct threadpool threadpool;
typedef struct future future;
typedef void *(*task_fn)(void *arg);

// pool lifecycle
threadpool *threadpool_new(int num_threads);  // 0 = auto (num_cores / 2, min 2)
void        threadpool_free(threadpool *p);   // waits for pending tasks, then destroys

// submit work
future *threadpool_submit(threadpool *p, task_fn fn, void *arg);

// future
void *future_get(future *f, int timeout_ms);  // blocks until done or timeout. NULL on timeout.
bool  future_done(const future *f);
void  future_free(future *f);

// convenience: submit and forget (no future returned, result discarded)
void threadpool_fire(threadpool *p, task_fn fn, void *arg);

// wait for all pending tasks to complete
void threadpool_drain(threadpool *p, int timeout_ms);

// stats
size_t threadpool_pending(const threadpool *p);
int    threadpool_num_threads(const threadpool *p);
