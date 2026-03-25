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
