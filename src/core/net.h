#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// HTTP response
// ---------------------------------------------------------------------------

typedef struct {
  int      status_code;
  uint8_t *data;
  size_t   size;
  char    *content_type;
  char    *error;  // NULL on success
} http_response;

void http_response_free(http_response *r);

// ---------------------------------------------------------------------------
// HTTP operations (synchronous)
// ---------------------------------------------------------------------------

// Simple GET — fetches the full body.
http_response *http_get(const char *url, int timeout_ms);

// GET with byte range — for fetching zarr/S3 chunks without reading the whole object.
http_response *http_get_range(const char *url, int64_t offset, int64_t len, int timeout_ms);

// HEAD — check existence and retrieve headers (Content-Length etc.) without body.
http_response *http_head(const char *url, int timeout_ms);

// ---------------------------------------------------------------------------
// URL parsing
// ---------------------------------------------------------------------------

typedef struct {
  char scheme[16];   // "http", "https", "s3"
  char host[256];
  int  port;         // 0 = use scheme default
  char path[2048];
  char query[1024];
} parsed_url;

bool url_parse(const char *url, parsed_url *out);

// ---------------------------------------------------------------------------
// Process-level init/cleanup (wraps curl_global_init / curl_global_cleanup)
// ---------------------------------------------------------------------------

void http_init(void);
void http_cleanup(void);
