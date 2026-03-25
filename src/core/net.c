#include "core/net.h"
#include "core/log.h"
#include <curl/curl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Write callback — accumulates response bytes into a growing buffer
// ---------------------------------------------------------------------------

typedef struct {
  uint8_t *buf;
  size_t   len;
  size_t   cap;
} write_buf;

static size_t write_cb(void *ptr, size_t size, size_t nmemb, void *userdata) {
  write_buf *wb    = userdata;
  size_t     bytes = size * nmemb;
  size_t     need  = wb->len + bytes + 1;

  if (need > wb->cap) {
    size_t   new_cap = (wb->cap == 0) ? 4096 : wb->cap * 2;
    while (new_cap < need) new_cap *= 2;
    uint8_t *nb = realloc(wb->buf, new_cap);
    if (!nb) return 0;  // signals error to curl
    wb->buf = nb;
    wb->cap = new_cap;
  }

  memcpy(wb->buf + wb->len, ptr, bytes);
  wb->len += bytes;
  wb->buf[wb->len] = '\0';  // keep NUL-terminated for text responses
  return bytes;
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static http_response *response_alloc(void) {
  http_response *r = calloc(1, sizeof(*r));
  return r;
}

// Extract Content-Type from curl after the transfer.
static char *extract_content_type(CURL *curl) {
  char *ct = NULL;
  curl_easy_getinfo(curl, CURLINFO_CONTENT_TYPE, &ct);
  if (!ct) return NULL;
  return strdup(ct);
}

// Build and execute a curl easy handle, returning a populated http_response.
// The caller provides an optional range string (e.g. "0-1023").
static http_response *do_request(const char *url, const char *range,
                                 bool head_only, int timeout_ms) {
  http_response *r  = response_alloc();
  if (!r) return NULL;

  CURL *curl = curl_easy_init();
  if (!curl) {
    r->error = strdup("curl_easy_init failed");
    return r;
  }

  write_buf wb = {0};

  curl_easy_setopt(curl, CURLOPT_URL,            url);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,  1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS,       10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,      (long)timeout_ms);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,   write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &wb);
  curl_easy_setopt(curl, CURLOPT_USERAGENT,       "volatile/0.1");
  // NOTE: NOSIGNAL is required for multi-threaded use — curl's DNS timeout
  // uses SIGALRM by default which is unsafe in a threaded process.
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL,        1L);

  if (head_only) {
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  }

  if (range) {
    curl_easy_setopt(curl, CURLOPT_RANGE, range);
  }

  CURLcode rc = curl_easy_perform(curl);

  if (rc != CURLE_OK) {
    r->error = strdup(curl_easy_strerror(rc));
    free(wb.buf);
    curl_easy_cleanup(curl);
    return r;
  }

  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  r->status_code  = (int)status;
  r->data         = wb.buf;
  r->size         = wb.len;
  r->content_type = extract_content_type(curl);

  curl_easy_cleanup(curl);
  return r;
}

// ---------------------------------------------------------------------------
// Public API — http_response_free
// ---------------------------------------------------------------------------

void http_response_free(http_response *r) {
  if (!r) return;
  free(r->data);
  free(r->content_type);
  free(r->error);
  free(r);
}

// ---------------------------------------------------------------------------
// Public API — HTTP operations
// ---------------------------------------------------------------------------

http_response *http_get(const char *url, int timeout_ms) {
  REQUIRE(url != NULL);
  return do_request(url, NULL, false, timeout_ms);
}

http_response *http_get_range(const char *url, int64_t offset, int64_t len, int timeout_ms) {
  REQUIRE(url != NULL);
  REQUIRE(offset >= 0 && len > 0);

  // CURLOPT_RANGE expects "first-last" (inclusive, byte positions).
  char range[64];
  snprintf(range, sizeof(range), "%" PRId64 "-%" PRId64, offset, offset + len - 1);
  return do_request(url, range, false, timeout_ms);
}

http_response *http_head(const char *url, int timeout_ms) {
  REQUIRE(url != NULL);
  return do_request(url, NULL, true, timeout_ms);
}

// ---------------------------------------------------------------------------
// URL parsing
// ---------------------------------------------------------------------------

// NOTE: Simple character-level splitting; no regex, no external libs.
// Handles:  scheme://[host[:port]][/path][?query]
bool url_parse(const char *url, parsed_url *out) {
  if (!url || !out) return false;
  memset(out, 0, sizeof(*out));

  // --- scheme ---
  const char *p = strstr(url, "://");
  if (!p) return false;
  size_t scheme_len = (size_t)(p - url);
  if (scheme_len == 0 || scheme_len >= sizeof(out->scheme)) return false;
  memcpy(out->scheme, url, scheme_len);
  out->scheme[scheme_len] = '\0';

  p += 3;  // skip "://"

  // --- host[:port] ---
  // Ends at the first '/', '?', or '\0'.
  const char *host_start = p;
  while (*p && *p != '/' && *p != '?') p++;
  size_t authority_len = (size_t)(p - host_start);

  if (authority_len >= sizeof(out->host)) return false;

  // Look for ':' inside the authority to split off port.
  const char *colon = memchr(host_start, ':', authority_len);
  if (colon) {
    size_t host_len = (size_t)(colon - host_start);
    if (host_len >= sizeof(out->host)) return false;
    memcpy(out->host, host_start, host_len);
    out->host[host_len] = '\0';
    out->port = atoi(colon + 1);
  } else {
    memcpy(out->host, host_start, authority_len);
    out->host[authority_len] = '\0';
    out->port = 0;
  }

  // --- path ---
  if (*p == '/') {
    const char *path_start = p;
    while (*p && *p != '?') p++;
    size_t path_len = (size_t)(p - path_start);
    if (path_len >= sizeof(out->path)) return false;
    memcpy(out->path, path_start, path_len);
    out->path[path_len] = '\0';
  } else {
    out->path[0] = '\0';
  }

  // --- query ---
  if (*p == '?') {
    p++;  // skip '?'
    size_t query_len = strlen(p);
    if (query_len >= sizeof(out->query)) return false;
    memcpy(out->query, p, query_len);
    out->query[query_len] = '\0';
  }

  return true;
}

// ---------------------------------------------------------------------------
// Process-level init / cleanup
// ---------------------------------------------------------------------------

void http_init(void) {
  CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
  if (rc != CURLE_OK) {
    LOG_FATAL("curl_global_init failed: %s", curl_easy_strerror(rc));
  }
}

void http_cleanup(void) {
  curl_global_cleanup();
}

// ---------------------------------------------------------------------------
// S3 credentials
// ---------------------------------------------------------------------------

s3_credentials *s3_creds_from_env(void) {
  s3_credentials *c = calloc(1, sizeof(*c));
  if (!c) return NULL;

  const char *ak = getenv("AWS_ACCESS_KEY_ID");
  const char *sk = getenv("AWS_SECRET_ACCESS_KEY");
  const char *rg = getenv("AWS_REGION");
  const char *ep = getenv("AWS_ENDPOINT_URL");
  const char *tk = getenv("AWS_SESSION_TOKEN");

  if (!ak || !sk) { free(c); return NULL; }

  snprintf(c->access_key, sizeof(c->access_key), "%s", ak);
  snprintf(c->secret_key, sizeof(c->secret_key), "%s", sk);
  if (rg) snprintf(c->region,   sizeof(c->region),   "%s", rg);
  else    snprintf(c->region,   sizeof(c->region),   "us-east-1");
  if (ep) snprintf(c->endpoint, sizeof(c->endpoint), "%s", ep);
  if (tk) snprintf(c->token,    sizeof(c->token),    "%s", tk);
  return c;
}

// Minimal INI parser for ~/.aws/credentials:
//   [profile]
//   aws_access_key_id     = ...
//   aws_secret_access_key = ...
//   region                = ...   (optional, non-standard but common)
s3_credentials *s3_creds_from_file(const char *profile) {
  if (!profile) profile = "default";

  const char *home = getenv("HOME");
  if (!home) return NULL;

  char path[512];
  snprintf(path, sizeof(path), "%s/.aws/credentials", home);

  FILE *f = fopen(path, "r");
  if (!f) return NULL;

  s3_credentials *c = calloc(1, sizeof(*c));
  if (!c) { fclose(f); return NULL; }
  snprintf(c->region, sizeof(c->region), "us-east-1");  // default

  // Build target section header "[profile]"
  char want[160];
  snprintf(want, sizeof(want), "[%s]", profile);

  char line[512];
  bool in_section = false;

  while (fgets(line, sizeof(line), f)) {
    // strip trailing newline/CR
    size_t n = strlen(line);
    while (n > 0 && (line[n-1] == '\n' || line[n-1] == '\r')) line[--n] = '\0';

    if (line[0] == '[') {
      in_section = (strcmp(line, want) == 0);
      continue;
    }
    if (!in_section) continue;

    // split on '='
    char *eq = strchr(line, '=');
    if (!eq) continue;
    *eq = '\0';
    const char *k = line;
    const char *v = eq + 1;
    // trim leading spaces from value
    while (*v == ' ' || *v == '\t') v++;
    // trim trailing spaces from key
    char *ke = eq - 1;
    while (ke >= k && (*ke == ' ' || *ke == '\t')) *ke-- = '\0';

    if      (strcmp(k, "aws_access_key_id")     == 0) snprintf(c->access_key, sizeof(c->access_key), "%s", v);
    else if (strcmp(k, "aws_secret_access_key") == 0) snprintf(c->secret_key, sizeof(c->secret_key), "%s", v);
    else if (strcmp(k, "region")                == 0) snprintf(c->region,     sizeof(c->region),     "%s", v);
    else if (strcmp(k, "endpoint_url")          == 0) snprintf(c->endpoint,   sizeof(c->endpoint),   "%s", v);
    else if (strcmp(k, "aws_session_token")     == 0) snprintf(c->token,      sizeof(c->token),      "%s", v);
  }

  fclose(f);

  if (c->access_key[0] == '\0' || c->secret_key[0] == '\0') {
    free(c);
    return NULL;
  }
  return c;
}

void s3_creds_free(s3_credentials *c) {
  if (!c) return;
  // zero sensitive fields before freeing
  memset(c->secret_key, 0, sizeof(c->secret_key));
  memset(c->token,      0, sizeof(c->token));
  free(c);
}

// ---------------------------------------------------------------------------
// S3 — internal: build HTTPS URL and perform signed request via CURLOPT_AWS_SIGV4
// ---------------------------------------------------------------------------

// Build the HTTPS URL for an S3 object given credentials and bucket/key.
// If creds->endpoint is set, use that as the base (path-style).
// Otherwise construct the standard AWS virtual-hosted URL.
static void s3_build_url(const s3_credentials *creds, const char *bucket,
                          const char *key, char *buf, size_t bufsz) {
  if (creds->endpoint[0]) {
    // path-style: https://endpoint/bucket/key
    const char *sep = (key && key[0] == '/') ? "" : "/";
    snprintf(buf, bufsz, "%s/%s%s%s", creds->endpoint, bucket, sep, key ? key : "");
  } else {
    // virtual-hosted: https://bucket.s3.region.amazonaws.com/key
    snprintf(buf, bufsz, "https://%s.s3.%s.amazonaws.com/%s",
             bucket, creds->region, key ? key : "");
  }
}

// Configure CURLOPT_AWS_SIGV4 on the handle (available since curl 7.75).
// provider1:provider2:region:service  —  for AWS S3 this is "aws:amz:REGION:s3".
// Token is passed as x-amz-security-token via a custom header list.
static void s3_setup_sigv4(CURL *curl, const s3_credentials *creds,
                            struct curl_slist **hlist) {
  char sigv4[128];
  snprintf(sigv4, sizeof(sigv4), "aws:amz:%s:s3", creds->region);
  curl_easy_setopt(curl, CURLOPT_AWS_SIGV4, sigv4);
  curl_easy_setopt(curl, CURLOPT_USERNAME,  creds->access_key);
  curl_easy_setopt(curl, CURLOPT_PASSWORD,  creds->secret_key);

  if (creds->token[0]) {
    char hdr[1152];
    snprintf(hdr, sizeof(hdr), "x-amz-security-token: %s", creds->token);
    *hlist = curl_slist_append(*hlist, hdr);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, *hlist);
  }
}

static http_response *s3_do_request(const s3_credentials *creds,
                                     const char *url,
                                     const char *range,
                                     bool head_only,
                                     int timeout_ms) {
  http_response *r = response_alloc();
  if (!r) return NULL;

  CURL *curl = curl_easy_init();
  if (!curl) { r->error = strdup("curl_easy_init failed"); return r; }

  struct curl_slist *hlist = NULL;
  write_buf wb = {0};

  curl_easy_setopt(curl, CURLOPT_URL,           url);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,    (long)timeout_ms);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,     &wb);
  curl_easy_setopt(curl, CURLOPT_USERAGENT,     "volatile/0.1");
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL,      1L);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);

  s3_setup_sigv4(curl, creds, &hlist);

  if (head_only) curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
  if (range)     curl_easy_setopt(curl, CURLOPT_RANGE, range);

  CURLcode rc = curl_easy_perform(curl);

  if (hlist) curl_slist_free_all(hlist);

  if (rc != CURLE_OK) {
    r->error = strdup(curl_easy_strerror(rc));
    free(wb.buf);
    curl_easy_cleanup(curl);
    return r;
  }

  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  r->status_code  = (int)status;
  r->data         = wb.buf;
  r->size         = wb.len;
  r->content_type = extract_content_type(curl);
  curl_easy_cleanup(curl);
  return r;
}

// ---------------------------------------------------------------------------
// Public S3 API
// ---------------------------------------------------------------------------

bool s3_parse_url(const char *url, char *bucket, int bucket_len,
                  char *key, int key_len) {
  if (!url || !bucket || !key) return false;
  if (strncmp(url, "s3://", 5) != 0) return false;

  const char *p = url + 5;
  const char *slash = strchr(p, '/');
  if (!slash) {
    // no key — just bucket
    int blen = (int)(strlen(p));
    if (blen == 0 || blen >= bucket_len) return false;
    memcpy(bucket, p, (size_t)blen);
    bucket[blen] = '\0';
    key[0] = '\0';
    return true;
  }

  int blen = (int)(slash - p);
  if (blen == 0 || blen >= bucket_len) return false;
  memcpy(bucket, p, (size_t)blen);
  bucket[blen] = '\0';

  const char *kstart = slash + 1;
  int klen = (int)strlen(kstart);
  if (klen >= key_len) return false;
  memcpy(key, kstart, (size_t)klen);
  key[klen] = '\0';
  return true;
}

http_response *s3_get_object(const s3_credentials *creds, const char *bucket,
                              const char *key, int timeout_ms) {
  if (!creds || !bucket || !key) return NULL;
  char url[2048];
  s3_build_url(creds, bucket, key, url, sizeof(url));
  return s3_do_request(creds, url, NULL, false, timeout_ms);
}

http_response *s3_get_object_range(const s3_credentials *creds,
                                    const char *bucket, const char *key,
                                    int64_t offset, int64_t len, int timeout_ms) {
  if (!creds || !bucket || !key) return NULL;
  char url[2048];
  s3_build_url(creds, bucket, key, url, sizeof(url));
  char range[64];
  snprintf(range, sizeof(range), "%" PRId64 "-%" PRId64, offset, offset + len - 1);
  return s3_do_request(creds, url, range, false, timeout_ms);
}

http_response *s3_head_object(const s3_credentials *creds, const char *bucket,
                               const char *key, int timeout_ms) {
  if (!creds || !bucket || !key) return NULL;
  char url[2048];
  s3_build_url(creds, bucket, key, url, sizeof(url));
  return s3_do_request(creds, url, NULL, true, timeout_ms);
}

http_response *s3_list_objects(const s3_credentials *creds, const char *bucket,
                                const char *prefix, int timeout_ms) {
  if (!creds || !bucket) return NULL;
  char url[2560];
  if (prefix && prefix[0]) {
    char base[2048];
    s3_build_url(creds, bucket, "", base, sizeof(base));
    snprintf(url, sizeof(url), "%s?list-type=2&prefix=%s", base, prefix);
  } else {
    char base[2048];
    s3_build_url(creds, bucket, "", base, sizeof(base));
    snprintf(url, sizeof(url), "%s?list-type=2", base);
  }
  return s3_do_request(creds, url, NULL, false, timeout_ms);
}

// ---------------------------------------------------------------------------
// Connection pool
// ---------------------------------------------------------------------------

#define HTTP_POOL_MAX 32

typedef struct {
  CURL *curl;
  char  host[256];   // hostname this handle last connected to
  bool  in_use;
} pool_slot;

struct http_pool {
  pool_slot slots[HTTP_POOL_MAX];
  int       cap;
};

http_pool *http_pool_new(int max_connections) {
  if (max_connections <= 0 || max_connections > HTTP_POOL_MAX)
    max_connections = HTTP_POOL_MAX;
  http_pool *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->cap = max_connections;
  for (int i = 0; i < p->cap; i++) {
    p->slots[i].curl = curl_easy_init();
    p->slots[i].in_use = false;
  }
  return p;
}

void http_pool_free(http_pool *p) {
  if (!p) return;
  for (int i = 0; i < p->cap; i++) {
    if (p->slots[i].curl) curl_easy_cleanup(p->slots[i].curl);
  }
  free(p);
}

// Borrow a curl handle for `host` (prefers one that has already connected there).
// If all slots are in_use, falls back to slot 0 (simple LRU-approximate).
static CURL *pool_acquire(http_pool *p, const char *host) {
  // prefer a free handle that last talked to this host
  for (int i = 0; i < p->cap; i++) {
    if (!p->slots[i].in_use && strcmp(p->slots[i].host, host) == 0) {
      p->slots[i].in_use = true;
      return p->slots[i].curl;
    }
  }
  // otherwise any free slot
  for (int i = 0; i < p->cap; i++) {
    if (!p->slots[i].in_use) {
      p->slots[i].in_use = true;
      snprintf(p->slots[i].host, sizeof(p->slots[i].host), "%s", host);
      return p->slots[i].curl;
    }
  }
  // all busy — create a transient handle (degenerate case)
  return curl_easy_init();
}

static void pool_release(http_pool *p, CURL *curl, const char *host) {
  for (int i = 0; i < p->cap; i++) {
    if (p->slots[i].curl == curl) {
      snprintf(p->slots[i].host, sizeof(p->slots[i].host), "%s", host);
      p->slots[i].in_use = false;
      curl_easy_reset(curl);  // clear per-request options, keep connection alive
      return;
    }
  }
  // transient handle — clean it up
  curl_easy_cleanup(curl);
}

static http_response *pool_do_request(http_pool *p, const char *url,
                                       const char *range, int timeout_ms) {
  http_response *r = response_alloc();
  if (!r) return NULL;

  // extract host for pool lookup
  parsed_url pu;
  char host[256] = "";
  if (url_parse(url, &pu)) snprintf(host, sizeof(host), "%s", pu.host);

  CURL *curl = pool_acquire(p, host);
  if (!curl) { r->error = strdup("pool exhausted"); return r; }

  write_buf wb = {0};

  curl_easy_setopt(curl, CURLOPT_URL,            url);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION,  1L);
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS,       10L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS,      (long)timeout_ms);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION,   write_cb);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,       &wb);
  curl_easy_setopt(curl, CURLOPT_USERAGENT,       "volatile/0.1");
  curl_easy_setopt(curl, CURLOPT_NOSIGNAL,        1L);
  // keep TCP connection alive for the next request to the same host
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE,   1L);

  if (range) curl_easy_setopt(curl, CURLOPT_RANGE, range);

  CURLcode rc = curl_easy_perform(curl);

  if (rc != CURLE_OK) {
    r->error = strdup(curl_easy_strerror(rc));
    free(wb.buf);
    pool_release(p, curl, host);
    return r;
  }

  long status = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
  r->status_code  = (int)status;
  r->data         = wb.buf;
  r->size         = wb.len;
  r->content_type = extract_content_type(curl);

  pool_release(p, curl, host);
  return r;
}

http_response *http_pool_get(http_pool *p, const char *url, int timeout_ms) {
  if (!p || !url) return NULL;
  return pool_do_request(p, url, NULL, timeout_ms);
}

http_response *http_pool_get_range(http_pool *p, const char *url,
                                    int64_t offset, int64_t len, int timeout_ms) {
  if (!p || !url) return NULL;
  char range[64];
  snprintf(range, sizeof(range), "%" PRId64 "-%" PRId64, offset, offset + len - 1);
  return pool_do_request(p, url, range, timeout_ms);
}
