#pragma once
#include "core/math.h"
#include <stdatomic.h>

// ---------------------------------------------------------------------------
// pointcloud — thread-safe, dynamically-growing collection of vec3f points
// ---------------------------------------------------------------------------

typedef struct pointcloud pointcloud;

pointcloud *pointcloud_new(void);
void        pointcloud_free(pointcloud *pc);

// Thread-safe single-point append.
void pointcloud_add(pointcloud *pc, vec3f point);

// Thread-safe batch append (acquires lock once for the whole batch).
void pointcloud_add_batch(pointcloud *pc, const vec3f *points, int count);

// Total number of points currently stored (approximate under concurrent writes).
int pointcloud_count(const pointcloud *pc);

// Random-access read (index must be < pointcloud_count). Not thread-safe with
// concurrent writers; call after all adds are done or under external lock.
vec3f pointcloud_get(const pointcloud *pc, int index);

// ---------------------------------------------------------------------------
// Parallel iteration
// ---------------------------------------------------------------------------
typedef void (*pointcloud_iter_fn)(vec3f point, int index, void *ctx);

// Divide points evenly across num_threads pthreads and call fn(point, idx, ctx)
// for each point. Blocks until all threads finish.
// num_threads <= 0 defaults to the number of online CPUs.
void pointcloud_parallel_for(pointcloud *pc, pointcloud_iter_fn fn, void *ctx, int num_threads);
