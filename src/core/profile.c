#define _POSIX_C_SOURCE 200809L
#include "profile.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

#define RING_CAP       4096   // events per thread ring buffer (power of 2)
#define RING_MASK      (RING_CAP - 1)
#define STACK_DEPTH    64     // max nested scopes per thread
#define MAX_THREADS    64     // max threads we track
#define MAX_COUNTERS   128    // max distinct counter names
#define MAX_AGGS       256    // max distinct scope names for aggregation

// ---------------------------------------------------------------------------
// Monotonic clock (nanoseconds)
// ---------------------------------------------------------------------------

static inline int64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

static inline double ns_to_ms(int64_t ns) { return (double)ns * 1e-6; }

// ---------------------------------------------------------------------------
// Per-event record (stored in ring buffer)
// ---------------------------------------------------------------------------

typedef struct {
  const char *name;
  int64_t     start_ns;
  int64_t     end_ns;
  int         depth;
} prof_event;

// ---------------------------------------------------------------------------
// Per-thread state (thread-local)
// ---------------------------------------------------------------------------

typedef struct {
  prof_event  ring[RING_CAP];
  atomic_int  head;          // write cursor (monotonic)
  int         stack_top;     // index into scope_stack
  int64_t     scope_stack_start[STACK_DEPTH];
  const char *scope_stack_name[STACK_DEPTH];
  pthread_t   tid;
  bool        active;
} thread_state;

static thread_state  thread_slots[MAX_THREADS];
static pthread_mutex_t slots_mutex = PTHREAD_MUTEX_INITIALIZER;

static _Thread_local thread_state *tls_state = NULL;

// ---------------------------------------------------------------------------
// Global enable flag + frame tracking
// ---------------------------------------------------------------------------

static atomic_bool g_enabled = false;
static int64_t     g_frame_start_ns = 0;
static int64_t     g_frame_end_ns   = 0;

// ---------------------------------------------------------------------------
// Counter table
// ---------------------------------------------------------------------------

typedef struct {
  char         name[64];
  atomic_int_least64_t value;
} counter_slot;

static counter_slot  g_counters[MAX_COUNTERS];
static atomic_int    g_num_counters = 0;
static pthread_mutex_t counters_mutex = PTHREAD_MUTEX_INITIALIZER;

// ---------------------------------------------------------------------------
// Thread-local state allocation
// ---------------------------------------------------------------------------

static thread_state *get_tls(void) {
  if (tls_state) return tls_state;
  pthread_mutex_lock(&slots_mutex);
  for (int i = 0; i < MAX_THREADS; i++) {
    if (!thread_slots[i].active) {
      thread_slots[i].active    = true;
      thread_slots[i].tid       = pthread_self();
      atomic_store(&thread_slots[i].head, 0);
      thread_slots[i].stack_top = 0;
      tls_state = &thread_slots[i];
      break;
    }
  }
  pthread_mutex_unlock(&slots_mutex);
  return tls_state;
}

// ---------------------------------------------------------------------------
// Init / shutdown
// ---------------------------------------------------------------------------

void profile_init(void) {
  // Only initialize on first call; subsequent calls just reset data.
  // Do not zero thread_slots here — TLS pointers remain valid across calls.
  static atomic_bool initialized = false;
  if (!atomic_exchange(&initialized, true)) {
    memset(thread_slots, 0, sizeof(thread_slots));
  }
  memset(g_counters, 0, sizeof(g_counters));
  atomic_store(&g_num_counters, 0);
  atomic_store(&g_enabled, false);
  g_frame_start_ns = g_frame_end_ns = 0;
  // Reset ring buffers without disturbing active/tid fields
  pthread_mutex_lock(&slots_mutex);
  for (int i = 0; i < MAX_THREADS; i++) {
    atomic_store(&thread_slots[i].head, 0);
    thread_slots[i].stack_top = 0;
  }
  pthread_mutex_unlock(&slots_mutex);
}

void profile_shutdown(void) {
  atomic_store(&g_enabled, false);
}

void profile_enable(bool on) {
  atomic_store(&g_enabled, on);
}

bool profile_enabled(void) {
  return atomic_load(&g_enabled);
}

// ---------------------------------------------------------------------------
// Scoped timing
// ---------------------------------------------------------------------------

void profile_begin(const char *name) {
  if (!atomic_load(&g_enabled)) return;
  thread_state *t = get_tls();
  if (!t || t->stack_top >= STACK_DEPTH) return;
  int d = t->stack_top++;
  t->scope_stack_name[d]  = name;
  t->scope_stack_start[d] = now_ns();
}

void profile_end(void) {
  if (!atomic_load(&g_enabled)) return;
  thread_state *t = get_tls();
  if (!t || t->stack_top <= 0) return;
  int64_t end = now_ns();
  int d = --t->stack_top;

  int slot = atomic_fetch_add(&t->head, 1) & RING_MASK;
  t->ring[slot] = (prof_event){
    .name     = t->scope_stack_name[d],
    .start_ns = t->scope_stack_start[d],
    .end_ns   = end,
    .depth    = d,
  };
}

// ---------------------------------------------------------------------------
// Counters
// ---------------------------------------------------------------------------

static counter_slot *counter_find_or_create(const char *name) {
  int n = atomic_load(&g_num_counters);
  for (int i = 0; i < n; i++) {
    if (strncmp(g_counters[i].name, name, 63) == 0) return &g_counters[i];
  }
  pthread_mutex_lock(&counters_mutex);
  // re-check after lock
  n = atomic_load(&g_num_counters);
  for (int i = 0; i < n; i++) {
    if (strncmp(g_counters[i].name, name, 63) == 0) {
      pthread_mutex_unlock(&counters_mutex);
      return &g_counters[i];
    }
  }
  if (n >= MAX_COUNTERS) { pthread_mutex_unlock(&counters_mutex); return NULL; }
  counter_slot *s = &g_counters[n];
  strncpy(s->name, name, 63);
  s->name[63] = '\0';
  atomic_store(&s->value, 0);
  atomic_fetch_add(&g_num_counters, 1);
  pthread_mutex_unlock(&counters_mutex);
  return s;
}

void profile_counter_inc(const char *name) {
  counter_slot *s = counter_find_or_create(name);
  if (s) atomic_fetch_add(&s->value, 1);
}

void profile_counter_add(const char *name, int64_t delta) {
  counter_slot *s = counter_find_or_create(name);
  if (s) atomic_fetch_add(&s->value, (int_least64_t)delta);
}

int64_t profile_counter_get(const char *name) {
  int n = atomic_load(&g_num_counters);
  for (int i = 0; i < n; i++) {
    if (strncmp(g_counters[i].name, name, 63) == 0)
      return (int64_t)atomic_load(&g_counters[i].value);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Frame markers
// ---------------------------------------------------------------------------

void profile_frame_begin(void) {
  g_frame_start_ns = now_ns();
  profile_begin("frame");
}

void profile_frame_end(void) {
  profile_end();
  g_frame_end_ns = now_ns();
}

bool profile_last_frame_stats(profile_frame_stats *out) {
  if (!out) return false;
  if (g_frame_start_ns == 0) return false;
  out->frame_time_ms  = ns_to_ms(g_frame_end_ns - g_frame_start_ns);
  out->render_time_ms = 0.0;  // populated by render subsystem via counters
  out->cache_hits     = profile_counter_get("cache_hits");
  out->cache_misses   = profile_counter_get("cache_misses");
  out->chunks_loaded  = profile_counter_get("chunks_loaded");
  return true;
}

// ---------------------------------------------------------------------------
// Aggregation helpers
// ---------------------------------------------------------------------------

typedef struct {
  const char *name;
  int64_t     total_ns;
  int64_t     max_ns;
  int64_t     call_count;
} agg_entry;

static int agg_find(agg_entry *aggs, int n, const char *name) {
  for (int i = 0; i < n; i++) if (aggs[i].name == name || strcmp(aggs[i].name, name) == 0) return i;
  return -1;
}

static int agg_cmp(const void *a, const void *b) {
  const agg_entry *ea = a, *eb = b;
  if (eb->total_ns > ea->total_ns) return 1;
  if (eb->total_ns < ea->total_ns) return -1;
  return 0;
}

// Collect all ring buffer events into agg_entry array. Returns count.
static int collect_aggs(agg_entry *aggs, int max_aggs) {
  int n = 0;
  pthread_mutex_lock(&slots_mutex);
  for (int t = 0; t < MAX_THREADS; t++) {
    if (!thread_slots[t].active) continue;
    int head = atomic_load(&thread_slots[t].head);
    int start = head > RING_CAP ? head - RING_CAP : 0;
    for (int i = start; i < head; i++) {
      prof_event *e = &thread_slots[t].ring[i & RING_MASK];
      if (!e->name) continue;
      int64_t dur = e->end_ns - e->start_ns;
      int idx = agg_find(aggs, n, e->name);
      if (idx < 0) {
        if (n >= max_aggs) continue;
        idx = n++;
        aggs[idx] = (agg_entry){ .name = e->name };
      }
      aggs[idx].total_ns  += dur;
      aggs[idx].call_count++;
      if (dur > aggs[idx].max_ns) aggs[idx].max_ns = dur;
    }
  }
  pthread_mutex_unlock(&slots_mutex);
  return n;
}

// ---------------------------------------------------------------------------
// Top entries
// ---------------------------------------------------------------------------

int profile_top_entries(profile_entry *out, int max_entries) {
  if (!out || max_entries <= 0) return 0;
  agg_entry aggs[MAX_AGGS];
  memset(aggs, 0, sizeof(aggs));
  int n = collect_aggs(aggs, MAX_AGGS);
  qsort(aggs, (size_t)n, sizeof(agg_entry), agg_cmp);
  int count = n < max_entries ? n : max_entries;
  for (int i = 0; i < count; i++) {
    out[i].name       = aggs[i].name;
    out[i].total_ms   = ns_to_ms(aggs[i].total_ns);
    out[i].max_ms     = ns_to_ms(aggs[i].max_ns);
    out[i].call_count = aggs[i].call_count;
    out[i].avg_ms     = aggs[i].call_count > 0
                        ? out[i].total_ms / (double)aggs[i].call_count : 0.0;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

void profile_reset(void) {
  pthread_mutex_lock(&slots_mutex);
  for (int i = 0; i < MAX_THREADS; i++) {
    atomic_store(&thread_slots[i].head, 0);
    thread_slots[i].stack_top = 0;
    memset(thread_slots[i].ring, 0, sizeof(thread_slots[i].ring));
  }
  pthread_mutex_unlock(&slots_mutex);
  pthread_mutex_lock(&counters_mutex);
  memset(g_counters, 0, sizeof(g_counters));
  atomic_store(&g_num_counters, 0);
  pthread_mutex_unlock(&counters_mutex);
  g_frame_start_ns = g_frame_end_ns = 0;
}

// ---------------------------------------------------------------------------
// JSON export (Chrome Trace Event Format)
// ---------------------------------------------------------------------------

bool profile_export_json(const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) return false;

  fprintf(f, "{\"traceEvents\":[\n");
  bool first = true;

  pthread_mutex_lock(&slots_mutex);
  for (int t = 0; t < MAX_THREADS; t++) {
    if (!thread_slots[t].active) continue;
    unsigned long tid = (unsigned long)thread_slots[t].tid;
    int head = atomic_load(&thread_slots[t].head);
    int start = head > RING_CAP ? head - RING_CAP : 0;
    for (int i = start; i < head; i++) {
      prof_event *e = &thread_slots[t].ring[i & RING_MASK];
      if (!e->name) continue;
      if (!first) fprintf(f, ",\n");
      first = false;
      // Chrome trace: ts in microseconds, dur in microseconds
      double ts  = (double)e->start_ns / 1000.0;
      double dur = (double)(e->end_ns - e->start_ns) / 1000.0;
      fprintf(f, "{\"ph\":\"X\",\"name\":\"%s\",\"ts\":%.3f,\"dur\":%.3f,\"tid\":%lu,\"pid\":1}",
              e->name, ts, dur, tid);
    }
  }
  pthread_mutex_unlock(&slots_mutex);

  // Append counters as metadata
  int n = atomic_load(&g_num_counters);
  for (int i = 0; i < n; i++) {
    if (!first) fprintf(f, ",\n");
    first = false;
    int64_t v = (int64_t)atomic_load(&g_counters[i].value);
    fprintf(f, "{\"ph\":\"C\",\"name\":\"%s\",\"args\":{\"value\":%lld},\"ts\":0,\"pid\":1,\"tid\":0}",
            g_counters[i].name, (long long)v);
  }

  fprintf(f, "\n]}\n");
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// Human-readable summary
// ---------------------------------------------------------------------------

void profile_print_summary(FILE *out) {
  if (!out) out = stdout;

  fprintf(out, "=== Profile Summary ===\n");
  fprintf(out, "%-40s %10s %10s %10s %10s\n", "Scope", "Total ms", "Avg ms", "Max ms", "Calls");
  fprintf(out, "%-40s %10s %10s %10s %10s\n",
          "----------------------------------------",
          "----------", "----------", "----------", "----------");

  profile_entry entries[MAX_AGGS];
  int n = profile_top_entries(entries, MAX_AGGS);
  for (int i = 0; i < n; i++) {
    fprintf(out, "%-40s %10.3f %10.3f %10.3f %10lld\n",
            entries[i].name,
            entries[i].total_ms,
            entries[i].avg_ms,
            entries[i].max_ms,
            (long long)entries[i].call_count);
  }

  fprintf(out, "\n=== Counters ===\n");
  int nc = atomic_load(&g_num_counters);
  for (int i = 0; i < nc; i++) {
    fprintf(out, "  %-38s %lld\n",
            g_counters[i].name,
            (long long)atomic_load(&g_counters[i].value));
  }
}
