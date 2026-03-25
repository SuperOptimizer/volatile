#pragma once
#include "core/net.h"
#include <stdbool.h>
#include <stdint.h>

struct nk_context;

// ---------------------------------------------------------------------------
// s3_browser — Nuklear popup dialog for browsing S3 buckets
// ---------------------------------------------------------------------------

typedef struct {
  char    name[256];      // display name (last path component)
  char    full_key[1024]; // full S3 key or prefix
  bool    is_prefix;      // true = "folder", false = object
  int64_t size;           // bytes (0 for prefixes)
} s3_entry;

typedef struct s3_browser s3_browser;

s3_browser *s3_browser_new(void);
void        s3_browser_free(s3_browser *b);

// Set credentials (from cred_dialog or env). Copies the struct.
void s3_browser_set_creds(s3_browser *b, const s3_credentials *creds);

// Show the browser dialog.
void s3_browser_show(s3_browser *b);
bool s3_browser_is_visible(const s3_browser *b);

// Render one frame. Returns true exactly once when user confirms a selection.
// ctx may be NULL — returns false immediately.
bool s3_browser_render(s3_browser *b, struct nk_context *ctx);

// Get the selected S3 URL (s3://bucket/prefix/) valid until next show/free.
const char *s3_browser_get_url(const s3_browser *b);

// Navigation
void s3_browser_set_bucket(s3_browser *b, const char *bucket);
void s3_browser_navigate(s3_browser *b, const char *prefix);
void s3_browser_go_up(s3_browser *b);

// Bookmarks — saved bucket+prefix paths shown in sidebar.
// url should be s3://bucket/prefix/ form.
void s3_browser_add_bookmark(s3_browser *b, const char *name, const char *url);

// Recent — recently opened S3 URLs shown at the top of the listing.
void s3_browser_add_recent(s3_browser *b, const char *url);
