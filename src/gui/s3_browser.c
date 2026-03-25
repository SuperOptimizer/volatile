// ---------------------------------------------------------------------------
// s3_browser.c — S3 bucket browser Nuklear popup dialog.
//
// Listing is done in a background pthread so the UI never blocks.
// Parsed results are cached per-prefix to avoid redundant requests.
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

#define POPUP_W       520
#define POPUP_H       420
#define ITEM_H        22
#define MAX_ENTRIES   256
#define MAX_PREFIX    1024
#define MAX_BUCKET    256
#define MAX_URL       1536
#define CACHE_CAP     16     // cached prefix listings

// ---------------------------------------------------------------------------
// Listing cache — avoids re-fetching on back navigation
// ---------------------------------------------------------------------------

typedef struct {
  char      bucket[MAX_BUCKET];
  char      prefix[MAX_PREFIX];
  s3_entry *entries;
  int       count;
} cache_entry;

// ---------------------------------------------------------------------------
// Background listing job
// ---------------------------------------------------------------------------

typedef enum {
  LIST_IDLE,
  LIST_RUNNING,
  LIST_DONE,
  LIST_ERROR,
} list_state;

typedef struct {
  // input (set before thread start)
  s3_credentials creds;
  char           bucket[MAX_BUCKET];
  char           prefix[MAX_PREFIX];

  // output (written by thread, read by main after LIST_DONE)
  s3_entry      *entries;
  int            count;
  char           error[256];

  // state (written by thread, read by main)
  list_state     state;  // atomic-enough on x86/ARM for a bool flag
  pthread_t      thread;
  pthread_mutex_t mtx;
} list_job;

// ---------------------------------------------------------------------------
// s3_browser struct
// ---------------------------------------------------------------------------

struct s3_browser {
  bool           visible;
  bool           has_creds;
  s3_credentials creds;

  char           bucket[MAX_BUCKET];
  char           prefix[MAX_PREFIX];   // current "directory"
  char           bucket_input[MAX_BUCKET]; // text input buffer
  int            bucket_input_len;

  s3_entry      *entries;
  int            entry_count;
  int            selected;             // index into entries, -1 = none

  char           selected_url[MAX_URL];  // built on confirm
  bool           confirmed;

  list_job       job;
  int            spinner_tick;         // frame counter for animation

  cache_entry    cache[CACHE_CAP];
  int            cache_count;
};

// ---------------------------------------------------------------------------
// XML parsing — extract S3 ListObjectsV2 prefixes and keys
// ---------------------------------------------------------------------------

// Minimal strstr-based extraction: find tag, copy content.
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

// Parse ListObjectsV2 XML response into entries array (caller frees).
static int parse_list_xml(const char *xml, s3_entry **out) {
  if (!xml || !out) return 0;

  // Count upper bound: number of <Key> + <Prefix> tags
  int cap = 64;
  s3_entry *arr = calloc((size_t)cap, sizeof(s3_entry));
  if (!arr) return 0;
  int count = 0;

  auto void push(const char *key, bool is_prefix, int64_t size);
  void push(const char *key, bool is_prefix, int64_t sz) {
    if (count == cap) {
      int nc = cap * 2;
      s3_entry *nb = realloc(arr, (size_t)nc * sizeof(s3_entry));
      if (!nb) return;
      arr = nb; cap = nc;
    }
    s3_entry *e = &arr[count++];
    memset(e, 0, sizeof(*e));
    snprintf(e->full_key, sizeof(e->full_key), "%s", key);
    e->is_prefix = is_prefix;
    e->size      = sz;
    // display name = last component
    const char *slash = strrchr(key, '/');
    if (slash && slash[1] != '\0')
      snprintf(e->name, sizeof(e->name), "%s", slash + 1);
    else if (slash && slash[1] == '\0') {
      // prefix ending in /: use component before it
      char tmp[1024];
      snprintf(tmp, sizeof(tmp), "%s", key);
      tmp[slash - key] = '\0';
      const char *s2 = strrchr(tmp, '/');
      snprintf(e->name, sizeof(e->name), "%s/", s2 ? s2 + 1 : tmp);
    } else {
      snprintf(e->name, sizeof(e->name), "%s", key);
    }
  }

  // Parse CommonPrefixes (folders)
  const char *cur = xml;
  while (cur) {
    const char *cp = strstr(cur, "<CommonPrefixes>");
    if (!cp) break;
    char prefix[MAX_PREFIX] = {0};
    const char *after = xml_next_tag(cp, "Prefix", prefix, sizeof(prefix));
    if (after && prefix[0]) push(prefix, true, 0);
    cur = cp + 1;
  }

  // Parse Contents (objects), skip if key ends with /
  cur = xml;
  while (cur) {
    const char *ct = strstr(cur, "<Contents>");
    if (!ct) break;
    char key[MAX_PREFIX] = {0};
    char sz_str[32] = {0};
    xml_next_tag(ct, "Key",  key,    sizeof(key));
    xml_next_tag(ct, "Size", sz_str, sizeof(sz_str));
    if (key[0] && key[strlen(key)-1] != '/')
      push(key, false, (int64_t)atoll(sz_str));
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
  if (!resp) {
    snprintf(j->error, sizeof(j->error), "s3_list_objects returned NULL");
    pthread_mutex_lock(&j->mtx);
    j->state = LIST_ERROR;
    pthread_mutex_unlock(&j->mtx);
    return NULL;
  }
  if (resp->status_code != 200) {
    snprintf(j->error, sizeof(j->error), "HTTP %d", resp->status_code);
    http_response_free(resp);
    pthread_mutex_lock(&j->mtx);
    j->state = LIST_ERROR;
    pthread_mutex_unlock(&j->mtx);
    return NULL;
  }

  s3_entry *entries = NULL;
  int count = parse_list_xml((const char *)resp->data, &entries);
  http_response_free(resp);

  pthread_mutex_lock(&j->mtx);
  j->entries = entries;
  j->count   = count;
  j->state   = LIST_DONE;
  pthread_mutex_unlock(&j->mtx);
  return NULL;
}

static void job_start(list_job *j, const s3_credentials *creds,
                      const char *bucket, const char *prefix) {
  // free previous results
  free(j->entries);
  j->entries = NULL;
  j->count   = 0;
  j->error[0] = '\0';
  j->state    = LIST_RUNNING;

  j->creds = *creds;
  snprintf(j->bucket, sizeof(j->bucket), "%s", bucket);
  snprintf(j->prefix, sizeof(j->prefix), "%s", prefix ? prefix : "");

  pthread_create(&j->thread, NULL, list_thread, j);
}

static void job_join(list_job *j) {
  if (j->state == LIST_RUNNING)
    pthread_join(j->thread, NULL);
}

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

static cache_entry *cache_find(s3_browser *b, const char *bucket,
                                const char *prefix) {
  for (int i = 0; i < b->cache_count; i++) {
    if (strcmp(b->cache[i].bucket, bucket) == 0 &&
        strcmp(b->cache[i].prefix, prefix) == 0)
      return &b->cache[i];
  }
  return NULL;
}

static void cache_store(s3_browser *b, const char *bucket, const char *prefix,
                        s3_entry *entries, int count) {
  // evict oldest if full
  if (b->cache_count == CACHE_CAP) {
    free(b->cache[0].entries);
    memmove(&b->cache[0], &b->cache[1],
            (size_t)(CACHE_CAP - 1) * sizeof(cache_entry));
    b->cache_count--;
  }
  cache_entry *c = &b->cache[b->cache_count++];
  snprintf(c->bucket, sizeof(c->bucket), "%s", bucket);
  snprintf(c->prefix, sizeof(c->prefix), "%s", prefix);
  // duplicate entries for cache
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
  b->selected = -1;
  pthread_mutex_init(&b->job.mtx, NULL);
  return b;
}

void s3_browser_free(s3_browser *b) {
  if (!b) return;
  job_join(&b->job);
  free(b->job.entries);
  free(b->entries);
  for (int i = 0; i < b->cache_count; i++)
    free(b->cache[i].entries);
  pthread_mutex_destroy(&b->job.mtx);
  free(b);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void s3_browser_set_creds(s3_browser *b, const s3_credentials *creds) {
  if (!b || !creds) return;
  b->creds     = *creds;
  b->has_creds = true;
}

void s3_browser_show(s3_browser *b) {
  if (b) b->visible = true;
}

bool s3_browser_is_visible(const s3_browser *b) {
  return b && b->visible;
}

const char *s3_browser_get_url(const s3_browser *b) {
  return b ? b->selected_url : NULL;
}

void s3_browser_set_bucket(s3_browser *b, const char *bucket) {
  if (!b || !bucket) return;
  snprintf(b->bucket, sizeof(b->bucket), "%s", bucket);
  snprintf(b->bucket_input, sizeof(b->bucket_input), "%s", bucket);
  b->bucket_input_len = (int)strlen(bucket);
  b->prefix[0] = '\0';
  b->entries    = NULL;
  b->entry_count = 0;
  b->selected   = -1;
}

void s3_browser_navigate(s3_browser *b, const char *prefix) {
  if (!b || !prefix) return;
  snprintf(b->prefix, sizeof(b->prefix), "%s", prefix);
  b->selected = -1;
  // trigger re-list (entries will be refreshed in render)
  b->entries    = NULL;
  b->entry_count = 0;
}

void s3_browser_go_up(s3_browser *b) {
  if (!b || b->prefix[0] == '\0') return;
  // Remove trailing slash, then strip last component
  char tmp[MAX_PREFIX];
  snprintf(tmp, sizeof(tmp), "%s", b->prefix);
  int len = (int)strlen(tmp);
  if (len > 0 && tmp[len-1] == '/') tmp[--len] = '\0';
  char *slash = strrchr(tmp, '/');
  if (slash) {
    slash[1] = '\0';
    s3_browser_navigate(b, tmp);
  } else {
    s3_browser_navigate(b, "");
  }
}

// ---------------------------------------------------------------------------
// Render helpers
// ---------------------------------------------------------------------------

static bool is_zarr_volume(const s3_entry *e) {
  // A prefix is considered a volume if its name suggests .zarr or zarr.json
  return e->is_prefix && (
    strstr(e->name, ".zarr") != NULL ||
    strstr(e->full_key, ".zarr") != NULL
  );
}

static void render_breadcrumb(s3_browser *b, struct nk_context *ctx) {
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
  nk_layout_row_push(ctx, 0.15f);
  nk_label(ctx, b->bucket, NK_TEXT_LEFT);
  nk_layout_row_push(ctx, 0.7f);
  char path[MAX_PREFIX + 4];
  snprintf(path, sizeof(path), "/ %s", b->prefix);
  nk_label(ctx, path, NK_TEXT_LEFT);
  nk_layout_row_push(ctx, 0.15f);
  if (nk_button_label(ctx, "Up") && b->prefix[0] != '\0')
    s3_browser_go_up(b);
  nk_layout_row_end(ctx);
}

// ---------------------------------------------------------------------------
// s3_browser_render
// ---------------------------------------------------------------------------

bool s3_browser_render(s3_browser *b, struct nk_context *ctx) {
  if (!b || !ctx || !b->visible) return false;

  b->confirmed = false;

  // Popup dimensions centred-ish
  struct nk_rect bounds = nk_rect(60, 60, (float)POPUP_W, (float)POPUP_H);
  if (!nk_popup_begin(ctx, NK_POPUP_STATIC, "Open Remote Volume",
                      NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_CLOSABLE,
                      bounds)) {
    b->visible = false;
    return false;
  }

  // --- Bucket input row ---
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
  nk_layout_row_push(ctx, 0.15f);
  nk_label(ctx, "Bucket:", NK_TEXT_LEFT);
  nk_layout_row_push(ctx, 0.65f);
  nk_edit_string(ctx, NK_EDIT_SIMPLE, b->bucket_input, &b->bucket_input_len,
                 MAX_BUCKET - 1, nk_filter_default);
  nk_layout_row_push(ctx, 0.2f);
  if (nk_button_label(ctx, "Go")) {
    b->bucket_input[b->bucket_input_len] = '\0';
    s3_browser_set_bucket(b, b->bucket_input);
  }
  nk_layout_row_end(ctx);

  // Separator
  nk_layout_row_dynamic(ctx, 4, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

  // --- Breadcrumb ---
  if (b->bucket[0] != '\0')
    render_breadcrumb(b, ctx);

  // --- Check job state and collect results ---
  if (b->job.state == LIST_RUNNING) {
    // spinner
    b->spinner_tick++;
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    static const char *spin[] = { "|", "/", "-", "\\" };
    char msg[32];
    snprintf(msg, sizeof(msg), "Loading... %s", spin[b->spinner_tick & 3]);
    nk_label(ctx, msg, NK_TEXT_LEFT);
  } else if (b->job.state == LIST_ERROR) {
    nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
    char err[280];
    snprintf(err, sizeof(err), "Error: %s", b->job.error);
    nk_label(ctx, err, NK_TEXT_LEFT);
    b->job.state = LIST_IDLE;
  } else {
    // Collect results from completed job
    if (b->job.state == LIST_DONE) {
      pthread_join(b->job.thread, NULL);
      // update entries from job result
      free(b->entries);
      b->entries     = b->job.entries;
      b->entry_count = b->job.count;
      b->job.entries = NULL;
      b->job.count   = 0;
      b->job.state   = LIST_IDLE;
      // store in cache
      if (b->has_creds)
        cache_store(b, b->bucket, b->prefix, b->entries, b->entry_count);
    }

    // If no entries yet and bucket is set, check cache or start listing
    if (!b->entries && b->bucket[0] != '\0' && b->has_creds
        && b->job.state == LIST_IDLE) {
      cache_entry *cached = cache_find(b, b->bucket, b->prefix);
      if (cached) {
        // serve from cache
        free(b->entries);
        b->entries = calloc((size_t)cached->count, sizeof(s3_entry));
        if (b->entries) {
          memcpy(b->entries, cached->entries,
                 (size_t)cached->count * sizeof(s3_entry));
          b->entry_count = cached->count;
        }
      } else {
        job_start(&b->job, &b->creds, b->bucket, b->prefix);
      }
    } else if (!b->has_creds && b->bucket[0] != '\0') {
      nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);
      nk_label(ctx, "No credentials set. Configure via Edit > Credentials.",
               NK_TEXT_LEFT);
    }

    // --- Entry list ---
    float list_h = (float)POPUP_H - 120.0f;
    nk_layout_row_dynamic(ctx, list_h, 1);
    if (nk_group_begin(ctx, "##entries", NK_WINDOW_BORDER)) {
      for (int i = 0; i < b->entry_count; i++) {
        const s3_entry *e = &b->entries[i];
        nk_layout_row_dynamic(ctx, (float)ITEM_H, 1);

        // Build label: folder prefix vs object, volume indicator
        char label[320];
        if (e->is_prefix) {
          const char *icon = is_zarr_volume(e) ? "[VOL] " : "[DIR] ";
          snprintf(label, sizeof(label), "%s%s", icon, e->name);
        } else {
          char sz[32];
          if (e->size >= 1024*1024)
            snprintf(sz, sizeof(sz), "%.1f MB", (double)e->size / (1024.0*1024.0));
          else if (e->size >= 1024)
            snprintf(sz, sizeof(sz), "%.1f KB", (double)e->size / 1024.0);
          else
            snprintf(sz, sizeof(sz), "%lld B", (long long)e->size);
          snprintf(label, sizeof(label), "[OBJ] %s  (%s)", e->name, sz);
        }

        bool selected = (b->selected == i);
        if (nk_select_label(ctx, label, NK_TEXT_LEFT, selected)) {
          if (b->selected == i && e->is_prefix) {
            // double-click simulation: second click on already-selected prefix navigates
            s3_browser_navigate(b, e->full_key);
          } else {
            b->selected = i;
          }
        }
      }
      nk_group_end(ctx);
    }
  }

  // --- Bottom buttons ---
  nk_layout_row_dynamic(ctx, 4, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);
  nk_layout_row_begin(ctx, NK_DYNAMIC, (float)ITEM_H, 3);
  nk_layout_row_push(ctx, 0.5f);
  // Show selected path
  if (b->selected >= 0 && b->selected < b->entry_count) {
    const s3_entry *sel = &b->entries[b->selected];
    char preview[64];
    snprintf(preview, sizeof(preview), "%.60s", sel->name);
    nk_label(ctx, preview, NK_TEXT_LEFT);
  } else {
    nk_label(ctx, "(none selected)", NK_TEXT_LEFT);
  }
  nk_layout_row_push(ctx, 0.25f);
  if (nk_button_label(ctx, "Open")) {
    if (b->selected >= 0 && b->selected < b->entry_count) {
      const s3_entry *sel = &b->entries[b->selected];
      if (sel->is_prefix) {
        snprintf(b->selected_url, sizeof(b->selected_url),
                 "s3://%s/%s", b->bucket, sel->full_key);
      } else {
        snprintf(b->selected_url, sizeof(b->selected_url),
                 "s3://%s/%s", b->bucket, sel->full_key);
      }
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

  nk_popup_end(ctx);
  return b->confirmed;
}
