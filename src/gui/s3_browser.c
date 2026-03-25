// ---------------------------------------------------------------------------
// s3_browser.c — S3 bucket browser Nuklear popup dialog.
//
// Features:
//   - Background pthread listing (non-blocking UI)
//   - Per-prefix LRU cache (avoids re-fetch on back navigation)
//   - Filter bar (substring match on current listing)
//   - Recursive search job (matches pattern across prefixes)
//   - Bookmarks sidebar
//   - Recent S3 URLs
//   - Zarr volume preview (shape/dtype from .zarray, fetched in background)
// ---------------------------------------------------------------------------

#define _POSIX_C_SOURCE 200809L

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA
#include <nuklear.h>
#pragma GCC diagnostic pop

#include "gui/s3_browser.h"
#include "core/net.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <pthread.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define POPUP_W        700
#define POPUP_H        480
#define ITEM_H         22
#define MAX_PREFIX     1024
#define MAX_BUCKET     256
#define MAX_URL        1536
#define MAX_NAME       256
#define CACHE_CAP      16
#define BOOKMARK_MAX   16
#define RECENT_MAX     12
#define SEARCH_PAT_MAX 128
#define PREVIEW_MAX    512   // chars for .zarray preview text

// ---------------------------------------------------------------------------
// Job state enum (shared by listing + search + preview jobs)
// ---------------------------------------------------------------------------

typedef enum { JOB_IDLE, JOB_RUNNING, JOB_DONE, JOB_ERROR } job_state;

// ---------------------------------------------------------------------------
// Listing / search job
// ---------------------------------------------------------------------------

typedef struct {
  s3_credentials creds;
  char           bucket[MAX_BUCKET];
  char           prefix[MAX_PREFIX];
  char           pattern[SEARCH_PAT_MAX]; // empty = normal list, else recursive search
  s3_entry      *entries;
  int            count;
  char           error[256];
  job_state      state;
  pthread_t      thread;
  pthread_mutex_t mtx;
} list_job;

// ---------------------------------------------------------------------------
// Volume preview job — fetches .zarray from a zarr prefix
// ---------------------------------------------------------------------------

typedef struct {
  s3_credentials creds;
  char           bucket[MAX_BUCKET];
  char           key[MAX_PREFIX];   // prefix + ".zarray"
  char           text[PREVIEW_MAX]; // result
  job_state      state;
  pthread_t      thread;
  pthread_mutex_t mtx;
} preview_job;

// ---------------------------------------------------------------------------
// Cache
// ---------------------------------------------------------------------------

typedef struct {
  char      bucket[MAX_BUCKET];
  char      prefix[MAX_PREFIX];
  s3_entry *entries;
  int       count;
} cache_entry;

// ---------------------------------------------------------------------------
// Bookmark / recent
// ---------------------------------------------------------------------------

typedef struct { char name[MAX_NAME]; char url[MAX_URL]; } bookmark;

// ---------------------------------------------------------------------------
// s3_browser struct
// ---------------------------------------------------------------------------

struct s3_browser {
  bool           visible;
  bool           has_creds;
  s3_credentials creds;

  // Navigation
  char           bucket[MAX_BUCKET];
  char           prefix[MAX_PREFIX];
  char           bucket_input[MAX_BUCKET];
  int            bucket_input_len;

  // Current full listing (all entries for current prefix)
  s3_entry      *entries;
  int            entry_count;

  // Filter
  char           filter[128];
  int            filter_len;

  // Search
  char           search_pat[SEARCH_PAT_MAX];
  int            search_pat_len;
  bool           search_mode;     // showing search results instead of listing
  s3_entry      *search_results;
  int            search_count;

  // Selection
  int            selected;        // index into visible list (-1 = none)

  // Output
  char           selected_url[MAX_URL];
  bool           confirmed;

  // Jobs
  list_job       job;
  preview_job    prev_job;
  int            spinner_tick;

  // Cache
  cache_entry    cache[CACHE_CAP];
  int            cache_count;

  // Bookmarks / recent
  bookmark       bookmarks[BOOKMARK_MAX];
  int            bookmark_count;
  char           recent[RECENT_MAX][MAX_URL];
  int            recent_count;

  // Preview
  int            preview_entry;   // entry index we fetched preview for
  char           preview_text[PREVIEW_MAX];
};

// ---------------------------------------------------------------------------
// XML helpers
// ---------------------------------------------------------------------------

static const char *xml_next_tag(const char *src, const char *tag,
                                char *out, int out_len) {
  char open[64], close[64];
  snprintf(open,  sizeof(open),  "<%s>",  tag);
  snprintf(close, sizeof(close), "</%s>", tag);
  const char *p = strstr(src, open);
  if (!p) return NULL;
  p += strlen(open);
  const char *e = strstr(p, close);
  if (!e) return NULL;
  int n = (int)(e - p);
  if (n >= out_len) n = out_len - 1;
  memcpy(out, p, (size_t)n);
  out[n] = '\0';
  return e + strlen(close);
}

static void entry_set_name(s3_entry *e, const char *key) {
  const char *slash = strrchr(key, '/');
  if (slash && slash[1] != '\0') {
    snprintf(e->name, sizeof(e->name), "%s", slash + 1);
  } else if (slash && slash[1] == '\0') {
    char tmp[MAX_PREFIX];
    int n = (int)(slash - key);
    if (n >= (int)sizeof(tmp)) n = (int)sizeof(tmp) - 1;
    memcpy(tmp, key, (size_t)n);
    tmp[n] = '\0';
    const char *s2 = strrchr(tmp, '/');
    snprintf(e->name, sizeof(e->name), "%s/", s2 ? s2 + 1 : tmp);
  } else {
    snprintf(e->name, sizeof(e->name), "%s", key);
  }
}

static bool entries_push(s3_entry **arr, int *count, int *cap,
                         const char *key, bool is_prefix, int64_t sz) {
  if (*count == *cap) {
    int nc = *cap * 2;
    s3_entry *nb = realloc(*arr, (size_t)nc * sizeof(s3_entry));
    if (!nb) return false;
    *arr = nb; *cap = nc;
  }
  s3_entry *e = &(*arr)[(*count)++];
  memset(e, 0, sizeof(*e));
  snprintf(e->full_key, sizeof(e->full_key), "%s", key);
  e->is_prefix = is_prefix;
  e->size      = sz;
  entry_set_name(e, key);
  return true;
}

static int parse_list_xml(const char *xml, s3_entry **out) {
  if (!xml || !out) return 0;
  int cap = 64, count = 0;
  s3_entry *arr = calloc((size_t)cap, sizeof(s3_entry));
  if (!arr) return 0;

  const char *cur = xml;
  while (cur) {
    const char *cp = strstr(cur, "<CommonPrefixes>");
    if (!cp) break;
    char prefix[MAX_PREFIX] = {0};
    const char *after = xml_next_tag(cp, "Prefix", prefix, sizeof(prefix));
    if (after && prefix[0]) entries_push(&arr, &count, &cap, prefix, true, 0);
    cur = cp + 1;
  }
  cur = xml;
  while (cur) {
    const char *ct = strstr(cur, "<Contents>");
    if (!ct) break;
    char key[MAX_PREFIX] = {0};
    char sz_str[32]      = {0};
    xml_next_tag(ct, "Key",  key,    sizeof(key));
    xml_next_tag(ct, "Size", sz_str, sizeof(sz_str));
    if (key[0] && key[strlen(key)-1] != '/')
      entries_push(&arr, &count, &cap, key, false, (int64_t)atoll(sz_str));
    cur = ct + 1;
  }
  *out = arr;
  return count;
}

// ---------------------------------------------------------------------------
// Background listing thread
// ---------------------------------------------------------------------------

static void *list_thread(void *arg) {
  list_job *j = arg;
  http_response *resp = s3_list_objects(&j->creds, j->bucket, j->prefix, 10000);
  if (!resp || resp->status_code != 200) {
    snprintf(j->error, sizeof(j->error), "HTTP %d",
             resp ? resp->status_code : 0);
    http_response_free(resp);
    pthread_mutex_lock(&j->mtx);
    j->state = JOB_ERROR;
    pthread_mutex_unlock(&j->mtx);
    return NULL;
  }
  s3_entry *entries = NULL;
  int count = parse_list_xml((const char *)resp->data, &entries);
  http_response_free(resp);
  pthread_mutex_lock(&j->mtx);
  j->entries = entries;
  j->count   = count;
  j->state   = JOB_DONE;
  pthread_mutex_unlock(&j->mtx);
  return NULL;
}

// Recursive search thread: list prefix, collect entries matching pattern,
// then recurse into sub-prefixes (depth-limited).
typedef struct {
  list_job      *job;
  int            depth;
} search_ctx;

static void search_recurse(list_job *j, const char *prefix, int depth) {
  if (depth > 4) return;
  http_response *resp = s3_list_objects(&j->creds, j->bucket, prefix, 10000);
  if (!resp || resp->status_code != 200) { http_response_free(resp); return; }

  s3_entry *page = NULL;
  int n = parse_list_xml((const char *)resp->data, &page);
  http_response_free(resp);

  int cap = j->count ? j->count : 64;
  for (int i = 0; i < n; i++) {
    // match pattern against name (case-insensitive substring)
    char lo_name[MAX_NAME], lo_pat[SEARCH_PAT_MAX];
    snprintf(lo_name, sizeof(lo_name), "%s", page[i].name);
    snprintf(lo_pat,  sizeof(lo_pat),  "%s", j->pattern);
    for (char *p = lo_name; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
    for (char *p = lo_pat;  *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (strstr(lo_name, lo_pat) || strstr(page[i].full_key, j->pattern)) {
      entries_push(&j->entries, &j->count, &cap, page[i].full_key,
                   page[i].is_prefix, page[i].size);
    }
    if (page[i].is_prefix)
      search_recurse(j, page[i].full_key, depth + 1);
  }
  free(page);
}

static void *search_thread(void *arg) {
  list_job *j = arg;
  j->entries = NULL;
  j->count   = 0;
  search_recurse(j, j->prefix, 0);
  pthread_mutex_lock(&j->mtx);
  j->state = JOB_DONE;
  pthread_mutex_unlock(&j->mtx);
  return NULL;
}

static void job_start(list_job *j, const s3_credentials *creds,
                      const char *bucket, const char *prefix,
                      const char *pattern) {
  if (j->state == JOB_RUNNING) return;
  free(j->entries); j->entries = NULL; j->count = 0; j->error[0] = '\0';
  j->state  = JOB_RUNNING;
  j->creds  = *creds;
  snprintf(j->bucket,  sizeof(j->bucket),  "%s", bucket);
  snprintf(j->prefix,  sizeof(j->prefix),  "%s", prefix  ? prefix  : "");
  snprintf(j->pattern, sizeof(j->pattern), "%s", pattern ? pattern : "");
  pthread_create(&j->thread, NULL,
                 pattern && pattern[0] ? search_thread : list_thread, j);
}

static void job_join_if_done(list_job *j) {
  if (j->state == JOB_DONE || j->state == JOB_ERROR)
    pthread_join(j->thread, NULL);
}

static void job_cancel_wait(list_job *j) {
  if (j->state == JOB_RUNNING) pthread_join(j->thread, NULL);
  j->state = JOB_IDLE;
  free(j->entries); j->entries = NULL; j->count = 0;
}

// ---------------------------------------------------------------------------
// Preview job
// ---------------------------------------------------------------------------

static void *preview_thread(void *arg) {
  preview_job *p = arg;
  http_response *resp = s3_get_object(&p->creds, p->bucket, p->key, 5000);
  if (!resp || resp->status_code != 200) {
    snprintf(p->text, sizeof(p->text), "(unavailable)");
    http_response_free(resp);
    pthread_mutex_lock(&p->mtx);
    p->state = JOB_DONE;
    pthread_mutex_unlock(&p->mtx);
    return NULL;
  }
  // Copy first PREVIEW_MAX-1 bytes of .zarray JSON
  int n = (int)resp->size < PREVIEW_MAX - 1 ? (int)resp->size : PREVIEW_MAX - 1;
  memcpy(p->text, resp->data, (size_t)n);
  p->text[n] = '\0';
  http_response_free(resp);
  pthread_mutex_lock(&p->mtx);
  p->state = JOB_DONE;
  pthread_mutex_unlock(&p->mtx);
  return NULL;
}

static void preview_start(preview_job *p, const s3_credentials *creds,
                           const char *bucket, const char *zarr_prefix) {
  if (p->state == JOB_RUNNING) return;
  p->state = JOB_RUNNING;
  p->creds = *creds;
  snprintf(p->bucket, sizeof(p->bucket), "%s", bucket);
  // Try OME-Zarr path first: prefix + ".zarray"
  snprintf(p->key, sizeof(p->key), "%s.zarray", zarr_prefix);
  pthread_create(&p->thread, NULL, preview_thread, p);
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

static cache_entry *cache_find(s3_browser *b, const char *bucket,
                                const char *prefix) {
  for (int i = 0; i < b->cache_count; i++)
    if (!strcmp(b->cache[i].bucket, bucket) &&
        !strcmp(b->cache[i].prefix, prefix))
      return &b->cache[i];
  return NULL;
}

static void cache_store(s3_browser *b, const char *bucket, const char *prefix,
                        s3_entry *entries, int count) {
  if (b->cache_count == CACHE_CAP) {
    free(b->cache[0].entries);
    memmove(&b->cache[0], &b->cache[1],
            (size_t)(CACHE_CAP - 1) * sizeof(cache_entry));
    b->cache_count--;
  }
  cache_entry *c = &b->cache[b->cache_count++];
  snprintf(c->bucket, sizeof(c->bucket), "%s", bucket);
  snprintf(c->prefix, sizeof(c->prefix), "%s", prefix);
  c->entries = calloc((size_t)count, sizeof(s3_entry));
  if (c->entries) {
    memcpy(c->entries, entries, (size_t)count * sizeof(s3_entry));
    c->count = count;
  }
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

s3_browser *s3_browser_new(void) {
  s3_browser *b = calloc(1, sizeof(*b));
  if (!b) { LOG_ERROR("s3_browser_new: calloc failed"); return NULL; }
  b->selected      = -1;
  b->preview_entry = -1;
  pthread_mutex_init(&b->job.mtx, NULL);
  pthread_mutex_init(&b->prev_job.mtx, NULL);
  return b;
}

void s3_browser_free(s3_browser *b) {
  if (!b) return;
  job_cancel_wait(&b->job);
  if (b->prev_job.state == JOB_RUNNING) pthread_join(b->prev_job.thread, NULL);
  pthread_mutex_destroy(&b->job.mtx);
  pthread_mutex_destroy(&b->prev_job.mtx);
  free(b->job.entries);
  free(b->entries);
  free(b->search_results);
  for (int i = 0; i < b->cache_count; i++) free(b->cache[i].entries);
  free(b);
}

// ---------------------------------------------------------------------------
// Public navigation API
// ---------------------------------------------------------------------------

void s3_browser_set_creds(s3_browser *b, const s3_credentials *creds) {
  if (!b || !creds) return;
  b->creds = *creds; b->has_creds = true;
}

void s3_browser_show(s3_browser *b) {
  if (b) { b->visible = true; b->search_mode = false; }
}

bool s3_browser_is_visible(const s3_browser *b) { return b && b->visible; }

const char *s3_browser_get_url(const s3_browser *b) {
  return b ? b->selected_url : NULL;
}

void s3_browser_set_bucket(s3_browser *b, const char *bucket) {
  if (!b || !bucket) return;
  snprintf(b->bucket, sizeof(b->bucket), "%s", bucket);
  snprintf(b->bucket_input, sizeof(b->bucket_input), "%s", bucket);
  b->bucket_input_len = (int)strlen(bucket);
  b->prefix[0] = '\0';
  free(b->entries); b->entries = NULL; b->entry_count = 0;
  b->selected = -1; b->search_mode = false;
}

void s3_browser_navigate(s3_browser *b, const char *prefix) {
  if (!b) return;
  snprintf(b->prefix, sizeof(b->prefix), "%s", prefix ? prefix : "");
  b->selected     = -1;
  b->search_mode  = false;
  b->filter[0]    = '\0';
  b->filter_len   = 0;
  free(b->entries); b->entries = NULL; b->entry_count = 0;
}

void s3_browser_go_up(s3_browser *b) {
  if (!b || b->prefix[0] == '\0') return;
  char tmp[MAX_PREFIX];
  snprintf(tmp, sizeof(tmp), "%s", b->prefix);
  int len = (int)strlen(tmp);
  if (len > 0 && tmp[len-1] == '/') tmp[--len] = '\0';
  char *slash = strrchr(tmp, '/');
  if (slash) { slash[1] = '\0'; s3_browser_navigate(b, tmp); }
  else        s3_browser_navigate(b, "");
}

void s3_browser_add_bookmark(s3_browser *b, const char *name, const char *url) {
  if (!b || !name || !url) return;
  // Deduplicate by URL
  for (int i = 0; i < b->bookmark_count; i++)
    if (!strcmp(b->bookmarks[i].url, url)) return;
  if (b->bookmark_count >= BOOKMARK_MAX) return;
  bookmark *bm = &b->bookmarks[b->bookmark_count++];
  snprintf(bm->name, sizeof(bm->name), "%s", name);
  snprintf(bm->url,  sizeof(bm->url),  "%s", url);
}

void s3_browser_add_recent(s3_browser *b, const char *url) {
  if (!b || !url || !url[0]) return;
  // Dedup: move to front
  for (int i = 0; i < b->recent_count; i++) {
    if (!strcmp(b->recent[i], url)) {
      char tmp[MAX_URL];
      snprintf(tmp, sizeof(tmp), "%s", b->recent[i]);
      for (int j = i; j > 0; j--)
        memcpy(b->recent[j], b->recent[j-1], sizeof(b->recent[0]));
      memcpy(b->recent[0], tmp, sizeof(b->recent[0]));
      return;
    }
  }
  int n = b->recent_count < RECENT_MAX ? b->recent_count : RECENT_MAX - 1;
  for (int i = n; i > 0; i--)
    memcpy(b->recent[i], b->recent[i-1], sizeof(b->recent[0]));
  snprintf(b->recent[0], sizeof(b->recent[0]), "%s", url);
  if (b->recent_count < RECENT_MAX) b->recent_count++;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static bool is_zarr_volume(const s3_entry *e) {
  return e->is_prefix &&
    (strstr(e->name, ".zarr") || strstr(e->full_key, ".zarr"));
}

static void fmt_size(char *buf, int len, int64_t sz) {
  if (sz >= 1024*1024)
    snprintf(buf, (size_t)len, "%.1f MB", (double)sz / (1024.0*1024.0));
  else if (sz >= 1024)
    snprintf(buf, (size_t)len, "%.1f KB", (double)sz / 1024.0);
  else
    snprintf(buf, (size_t)len, "%lld B", (long long)sz);
}

// Collect the visible list (active listing filtered, or search results).
// Returns pointer to entries array and writes count. Caller does NOT free.
static const s3_entry *visible_entries(const s3_browser *b, int *out_count) {
  const s3_entry *src;
  int n;
  if (b->search_mode) { src = b->search_results; n = b->search_count; }
  else                { src = b->entries;         n = b->entry_count;  }

  *out_count = 0;
  if (!src) return NULL;

  // Filter: if filter is non-empty, we skip non-matching entries.
  // We return a const pointer; filtering is done inline during render.
  *out_count = n;
  return src;
}

// Check whether entry i passes the current filter.
static bool passes_filter(const s3_browser *b, const s3_entry *e) {
  if (b->filter_len == 0) return true;
  // Case-insensitive substring
  char lo_name[MAX_NAME], lo_filt[128];
  snprintf(lo_name, sizeof(lo_name), "%s", e->name);
  snprintf(lo_filt, sizeof(lo_filt), "%s", b->filter);
  for (char *p = lo_name; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
  for (char *p = lo_filt; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;
  return strstr(lo_name, lo_filt) != NULL;
}

// Kick off a preview fetch for a zarr entry if not already fetched.
static void maybe_start_preview(s3_browser *b, int idx) {
  if (!b->has_creds || idx < 0) return;
  if (b->preview_entry == idx) return;  // already fetched / fetching
  if (b->prev_job.state == JOB_RUNNING) return;
  if (b->prev_job.state == JOB_DONE)
    pthread_join(b->prev_job.thread, NULL);

  const s3_entry *src; int n;
  src = visible_entries(b, &n);
  if (!src || idx >= n) return;
  const s3_entry *e = &src[idx];
  if (!is_zarr_volume(e)) return;

  b->preview_entry  = idx;
  b->preview_text[0] = '\0';
  b->prev_job.state = JOB_IDLE;
  preview_start(&b->prev_job, &b->creds, b->bucket, e->full_key);
}

// Collect preview result if ready.
static void collect_preview(s3_browser *b) {
  if (b->prev_job.state != JOB_DONE) return;
  pthread_mutex_lock(&b->prev_job.mtx);
  if (b->prev_job.state == JOB_DONE) {
    snprintf(b->preview_text, sizeof(b->preview_text), "%s", b->prev_job.text);
    b->prev_job.state = JOB_IDLE;
    pthread_mutex_unlock(&b->prev_job.mtx);
    pthread_join(b->prev_job.thread, NULL);
    return;
  }
  pthread_mutex_unlock(&b->prev_job.mtx);
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

static void render_sidebar(s3_browser *b, struct nk_context *ctx) {
  // Recent
  if (b->recent_count > 0) {
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    nk_label(ctx, "Recent", NK_TEXT_LEFT);
    for (int i = 0; i < b->recent_count; i++) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      // Trim to last two path components for display
      char disp[48];
      snprintf(disp, sizeof(disp), "%.46s", b->recent[i] + (strlen(b->recent[i]) > 46 ? strlen(b->recent[i]) - 46 : 0));
      if (nk_button_label(ctx, disp)) {
        // Parse s3://bucket/prefix and navigate
        char bkt[MAX_BUCKET], pfx[MAX_PREFIX];
        if (s3_parse_url(b->recent[i], bkt, MAX_BUCKET, pfx, MAX_PREFIX)) {
          s3_browser_set_bucket(b, bkt);
          s3_browser_navigate(b, pfx);
        }
      }
    }
    nk_layout_row_dynamic(ctx, 4, 1);
    nk_rule_horizontal(ctx, ctx->style.window.border_color, false);
  }
  // Bookmarks
  if (b->bookmark_count > 0) {
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    nk_label(ctx, "Bookmarks", NK_TEXT_LEFT);
    for (int i = 0; i < b->bookmark_count; i++) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      if (nk_button_label(ctx, b->bookmarks[i].name)) {
        char bkt[MAX_BUCKET], pfx[MAX_PREFIX];
        if (s3_parse_url(b->bookmarks[i].url, bkt, MAX_BUCKET, pfx, MAX_PREFIX)) {
          s3_browser_set_bucket(b, bkt);
          s3_browser_navigate(b, pfx);
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// s3_browser_render
// ---------------------------------------------------------------------------

bool s3_browser_render(s3_browser *b, struct nk_context *ctx) {
  if (!b || !ctx || !b->visible) return false;
  b->confirmed = false;
  collect_preview(b);

  struct nk_rect bounds = nk_rect(40, 40, (float)POPUP_W, (float)POPUP_H);
  if (!nk_begin(ctx, "Open Remote Volume", bounds,
                NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_MOVABLE | NK_WINDOW_CLOSABLE)) {
    b->visible = false;
    nk_end(ctx);
    return false;
  }

  // -------------------------------------------------------------------------
  // Top bar: bucket + filter + search
  // -------------------------------------------------------------------------
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 5);
  nk_layout_row_push(ctx, 0.12f);
  nk_label(ctx, "Bucket:", NK_TEXT_LEFT);
  nk_layout_row_push(ctx, 0.30f);
  nk_edit_string(ctx, NK_EDIT_FIELD, b->bucket_input, &b->bucket_input_len,
                 MAX_BUCKET - 1, nk_filter_default);
  nk_layout_row_push(ctx, 0.10f);
  if (nk_button_label(ctx, "Go")) {
    b->bucket_input[b->bucket_input_len] = '\0';
    s3_browser_set_bucket(b, b->bucket_input);
  }
  nk_layout_row_push(ctx, 0.28f);
  nk_edit_string(ctx, NK_EDIT_FIELD | NK_EDIT_GOTO_END_ON_ACTIVATE,
                 b->filter, &b->filter_len, (int)sizeof(b->filter) - 1,
                 nk_filter_default);
  nk_layout_row_push(ctx, 0.20f);
  nk_label(ctx, "(filter)", NK_TEXT_LEFT);
  nk_layout_row_end(ctx);

  // Search bar
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
  nk_layout_row_push(ctx, 0.52f);
  nk_edit_string(ctx, NK_EDIT_FIELD, b->search_pat, &b->search_pat_len,
                 SEARCH_PAT_MAX - 1, nk_filter_default);
  nk_layout_row_push(ctx, 0.24f);
  if (nk_button_label(ctx, "Search")) {
    b->search_pat[b->search_pat_len] = '\0';
    if (b->search_pat_len > 0 && b->has_creds && b->bucket[0]) {
      job_cancel_wait(&b->job);
      free(b->search_results); b->search_results = NULL; b->search_count = 0;
      b->search_mode = true;
      b->selected    = -1;
      job_start(&b->job, &b->creds, b->bucket, b->prefix, b->search_pat);
    }
  }
  nk_layout_row_push(ctx, 0.24f);
  if (nk_button_label(ctx, "Clear Search")) {
    b->search_mode = false;
    b->search_pat_len = 0; b->search_pat[0] = '\0';
    b->selected = -1;
  }
  nk_layout_row_end(ctx);

  nk_layout_row_dynamic(ctx, 4, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

  // -------------------------------------------------------------------------
  // Breadcrumb + Up button
  // -------------------------------------------------------------------------
  if (b->bucket[0] != '\0' && !b->search_mode) {
    nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
    nk_layout_row_push(ctx, 0.18f);
    nk_label(ctx, b->bucket, NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.67f);
    char path[MAX_PREFIX + 4];
    snprintf(path, sizeof(path), "/ %s", b->prefix);
    nk_label(ctx, path, NK_TEXT_LEFT);
    nk_layout_row_push(ctx, 0.15f);
    if (nk_button_label(ctx, "Up")) s3_browser_go_up(b);
    nk_layout_row_end(ctx);
  }

  // -------------------------------------------------------------------------
  // Collect completed job results
  // -------------------------------------------------------------------------
  if (b->job.state == JOB_RUNNING) {
    b->spinner_tick++;
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    static const char *spin[] = {"|", "/", "-", "\\"};
    char msg[40];
    snprintf(msg, sizeof(msg), "%s %s",
             b->search_mode ? "Searching..." : "Loading...",
             spin[b->spinner_tick & 3]);
    nk_label(ctx, msg, NK_TEXT_LEFT);
  } else if (b->job.state == JOB_ERROR) {
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    char err[280];
    snprintf(err, sizeof(err), "Error: %s", b->job.error);
    nk_label(ctx, err, NK_TEXT_LEFT);
    job_join_if_done(&b->job);
    b->job.state = JOB_IDLE;
  } else {
    if (b->job.state == JOB_DONE) {
      pthread_join(b->job.thread, NULL);
      if (b->search_mode) {
        free(b->search_results);
        b->search_results = b->job.entries;
        b->search_count   = b->job.count;
      } else {
        free(b->entries);
        b->entries     = b->job.entries;
        b->entry_count = b->job.count;
        if (b->has_creds)
          cache_store(b, b->bucket, b->prefix, b->entries, b->entry_count);
      }
      b->job.entries = NULL; b->job.count = 0;
      b->job.state   = JOB_IDLE;
    }

    // Start list if needed
    if (!b->search_mode && !b->entries && b->bucket[0] && b->has_creds
        && b->job.state == JOB_IDLE) {
      cache_entry *cached = cache_find(b, b->bucket, b->prefix);
      if (cached) {
        free(b->entries);
        b->entries = calloc((size_t)cached->count, sizeof(s3_entry));
        if (b->entries) {
          memcpy(b->entries, cached->entries,
                 (size_t)cached->count * sizeof(s3_entry));
          b->entry_count = cached->count;
        }
      } else {
        job_start(&b->job, &b->creds, b->bucket, b->prefix, NULL);
      }
    } else if (!b->has_creds && b->bucket[0]) {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      nk_label(ctx, "No credentials. Use File > Credentials.", NK_TEXT_LEFT);
    }

    // -------------------------------------------------------------------------
    // Two-column layout: sidebar (left) + entry list (right)
    // -------------------------------------------------------------------------
    float content_h = (float)POPUP_H - 140.0f;
    nk_layout_row_begin(ctx, NK_DYNAMIC, content_h, 2);
    nk_layout_row_push(ctx, 0.22f);

    // Sidebar
    if (nk_group_begin(ctx, "##sidebar", NK_WINDOW_BORDER)) {
      render_sidebar(b, ctx);
      nk_group_end(ctx);
    }

    nk_layout_row_push(ctx, 0.78f);

    // Entry list + preview tooltip
    if (nk_group_begin(ctx, "##entries", NK_WINDOW_BORDER)) {
      int total; const s3_entry *src = visible_entries(b, &total);
      int visible_idx = 0;  // tracks filtered index → selection mapping

      for (int i = 0; i < total; i++) {
        const s3_entry *e = &src[i];
        if (!passes_filter(b, e)) continue;

        nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
        char label[340];
        if (e->is_prefix) {
          snprintf(label, sizeof(label), "%s%s",
                   is_zarr_volume(e) ? "[VOL] " : "[DIR] ", e->name);
        } else {
          char sz[24]; fmt_size(sz, sizeof(sz), e->size);
          snprintf(label, sizeof(label), "[OBJ] %s  (%s)", e->name, sz);
        }

        bool sel = (b->selected == i);
        if (nk_select_label(ctx, label, NK_TEXT_LEFT, sel)) {
          if (b->selected == i && e->is_prefix) {
            s3_browser_navigate(b, e->full_key);
          } else {
            b->selected = i;
            if (is_zarr_volume(e)) maybe_start_preview(b, i);
          }
        }
        // Show preview text below selected zarr entry
        if (sel && is_zarr_volume(e) && b->preview_text[0]) {
          nk_layout_row_dynamic(ctx, (float)ITEM_H * 3, 1);
          nk_label_wrap(ctx, b->preview_text);
        }
        visible_idx++;
      }
      (void)visible_idx;
      nk_group_end(ctx);
    }
    nk_layout_row_end(ctx);
  }

  // -------------------------------------------------------------------------
  // Bottom buttons
  // -------------------------------------------------------------------------
  nk_layout_row_dynamic(ctx, 4, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
  nk_layout_row_push(ctx, 0.5f);
  {
    int total; const s3_entry *src = visible_entries(b, &total);
    if (b->selected >= 0 && b->selected < total && src) {
      char preview[64];
      snprintf(preview, sizeof(preview), "%.60s", src[b->selected].name);
      nk_label(ctx, preview, NK_TEXT_LEFT);
    } else {
      nk_label(ctx, "(none selected)", NK_TEXT_LEFT);
    }
  }
  nk_layout_row_push(ctx, 0.25f);
  if (nk_button_label(ctx, "Open")) {
    int total; const s3_entry *src = visible_entries(b, &total);
    if (b->selected >= 0 && b->selected < total && src) {
      snprintf(b->selected_url, sizeof(b->selected_url),
               "s3://%s/%s", b->bucket, src[b->selected].full_key);
      b->confirmed = true;
      b->visible   = false;
      nk_popup_close(ctx);
    }
  }
  nk_layout_row_push(ctx, 0.25f);
  if (nk_button_label(ctx, "Cancel")) {
    b->visible = false;
    nk_popup_close(ctx);
  }
  nk_layout_row_end(ctx);

  nk_end(ctx);
  return b->confirmed;
}
