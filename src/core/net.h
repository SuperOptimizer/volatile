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

// ---------------------------------------------------------------------------
// S3 credentials
// ---------------------------------------------------------------------------

typedef struct {
  char access_key[128];
  char secret_key[128];
  char region[32];
  char endpoint[256];  // custom endpoint (e.g., MinIO, Wasabi); empty = AWS default
  char token[1024];    // optional session token (for STS)
} s3_credentials;

// Load from env: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION,
// AWS_ENDPOINT_URL, AWS_SESSION_TOKEN.
s3_credentials *s3_creds_from_env(void);

// Load from ~/.aws/credentials ini file.  profile="default" if NULL.
s3_credentials *s3_creds_from_file(const char *profile);

void s3_creds_free(s3_credentials *c);

// ---------------------------------------------------------------------------
// S3 operations  (all synchronous, return NULL on alloc failure)
// ---------------------------------------------------------------------------

http_response *s3_get_object(const s3_credentials *creds, const char *bucket,
                              const char *key, int timeout_ms);
http_response *s3_get_object_range(const s3_credentials *creds, const char *bucket,
                                    const char *key,
                                    int64_t offset, int64_t len, int timeout_ms);
http_response *s3_head_object(const s3_credentials *creds, const char *bucket,
                               const char *key, int timeout_ms);
http_response *s3_list_objects(const s3_credentials *creds, const char *bucket,
                                const char *prefix, int timeout_ms);

// Parse s3://bucket/key → bucket + key buffers.  Returns false on bad input.
bool s3_parse_url(const char *url, char *bucket, int bucket_len,
                  char *key, int key_len);

// ---------------------------------------------------------------------------
// Connection pool  (reuse curl handles per host to avoid TLS overhead)
// ---------------------------------------------------------------------------

typedef struct http_pool http_pool;

http_pool      *http_pool_new(int max_connections);
void            http_pool_free(http_pool *p);
http_response  *http_pool_get(http_pool *p, const char *url, int timeout_ms);
http_response  *http_pool_get_range(http_pool *p, const char *url,
                                     int64_t offset, int64_t len, int timeout_ms);
