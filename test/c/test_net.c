#define _POSIX_C_SOURCE 200809L

#include "greatest.h"
#include "core/net.h"

#include <stdlib.h>
#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// url_parse tests
// ---------------------------------------------------------------------------

TEST test_parse_http_simple(void) {
  parsed_url u;
  ASSERT(url_parse("http://example.com/path/to/file", &u));
  ASSERT_STR_EQ("http",            u.scheme);
  ASSERT_STR_EQ("example.com",     u.host);
  ASSERT_EQ(0,                     u.port);
  ASSERT_STR_EQ("/path/to/file",   u.path);
  ASSERT_STR_EQ("",                u.query);
  PASS();
}

TEST test_parse_https_with_port(void) {
  parsed_url u;
  ASSERT(url_parse("https://myhost:8443/api/v1", &u));
  ASSERT_STR_EQ("https",    u.scheme);
  ASSERT_STR_EQ("myhost",   u.host);
  ASSERT_EQ(8443,            u.port);
  ASSERT_STR_EQ("/api/v1",  u.path);
  PASS();
}

TEST test_parse_s3_url(void) {
  parsed_url u;
  ASSERT(url_parse("s3://my-bucket/zarr/store/0/0/0", &u));
  ASSERT_STR_EQ("s3",                    u.scheme);
  ASSERT_STR_EQ("my-bucket",             u.host);
  ASSERT_EQ(0,                           u.port);
  ASSERT_STR_EQ("/zarr/store/0/0/0",     u.path);
  PASS();
}

TEST test_parse_url_with_query(void) {
  parsed_url u;
  ASSERT(url_parse("http://host.example/search?q=foo&page=2", &u));
  ASSERT_STR_EQ("http",           u.scheme);
  ASSERT_STR_EQ("host.example",   u.host);
  ASSERT_STR_EQ("/search",        u.path);
  ASSERT_STR_EQ("q=foo&page=2",   u.query);
  PASS();
}

TEST test_parse_url_no_path(void) {
  parsed_url u;
  ASSERT(url_parse("http://bare.host", &u));
  ASSERT_STR_EQ("http",       u.scheme);
  ASSERT_STR_EQ("bare.host",  u.host);
  ASSERT_EQ(0,                 u.port);
  ASSERT_STR_EQ("",            u.path);
  PASS();
}

TEST test_parse_url_port_and_query(void) {
  parsed_url u;
  ASSERT(url_parse("http://localhost:9000/bucket/key?versionId=abc", &u));
  ASSERT_STR_EQ("http",           u.scheme);
  ASSERT_STR_EQ("localhost",      u.host);
  ASSERT_EQ(9000,                  u.port);
  ASSERT_STR_EQ("/bucket/key",    u.path);
  ASSERT_STR_EQ("versionId=abc",  u.query);
  PASS();
}

TEST test_parse_url_missing_scheme_separator(void) {
  parsed_url u;
  // No "://" — must return false.
  ASSERT(!url_parse("not-a-url", &u));
  PASS();
}

TEST test_parse_url_null(void) {
  parsed_url u;
  ASSERT(!url_parse(NULL, &u));
  ASSERT(!url_parse("http://ok.example/", NULL));
  PASS();
}

TEST test_parse_url_trailing_slash(void) {
  // Trailing slash after hostname — path should be "/" or "".
  parsed_url u;
  ASSERT(url_parse("http://hostname/", &u));
  ASSERT_STR_EQ("http",     u.scheme);
  ASSERT_STR_EQ("hostname", u.host);
  ASSERT_EQ(0,               u.port);
  ASSERT(strcmp(u.path, "/") == 0 || strcmp(u.path, "") == 0);
  PASS();
}

TEST test_parse_url_query_special_chars(void) {
  // Percent-encoded chars in query — parser must preserve them verbatim.
  parsed_url u;
  ASSERT(url_parse("http://host/path?key=hello%20world&x=1%2B2", &u));
  ASSERT_STR_EQ("http",                       u.scheme);
  ASSERT_STR_EQ("host",                       u.host);
  ASSERT_STR_EQ("/path",                      u.path);
  ASSERT_STR_EQ("key=hello%20world&x=1%2B2",  u.query);
  PASS();
}

TEST test_parse_url_ipv4_with_port(void) {
  parsed_url u;
  ASSERT(url_parse("http://127.0.0.1:8080/v1/chunks", &u));
  ASSERT_STR_EQ("http",       u.scheme);
  ASSERT_STR_EQ("127.0.0.1", u.host);
  ASSERT_EQ(8080,              u.port);
  ASSERT_STR_EQ("/v1/chunks", u.path);
  PASS();
}

TEST test_parse_url_deep_path(void) {
  parsed_url u;
  ASSERT(url_parse("https://cdn.example.com/a/b/c/d/e/f.zarr", &u));
  ASSERT_STR_EQ("https",             u.scheme);
  ASSERT_STR_EQ("cdn.example.com",  u.host);
  ASSERT_STR_EQ("/a/b/c/d/e/f.zarr", u.path);
  PASS();
}

// ---------------------------------------------------------------------------
// http_response_free — must not crash on NULL or partial structs
// ---------------------------------------------------------------------------

TEST test_response_free_null(void) {
  http_response_free(NULL);  // must not crash
  PASS();
}

TEST test_response_free_empty_struct(void) {
  // Allocate with all-zero fields — free must handle NULL data/content_type/error.
  http_response *r = calloc(1, sizeof(*r));
  ASSERT(r != NULL);
  http_response_free(r);  // must not crash, must not double-free
  PASS();
}

TEST test_response_free_with_error(void) {
  http_response *r = calloc(1, sizeof(*r));
  ASSERT(r != NULL);
  r->error = strdup("some error message");
  http_response_free(r);
  PASS();
}

TEST test_response_free_with_data(void) {
  http_response *r = calloc(1, sizeof(*r));
  ASSERT(r != NULL);
  r->data         = malloc(16);
  r->size         = 16;
  r->content_type = strdup("application/octet-stream");
  ASSERT(r->data         != NULL);
  ASSERT(r->content_type != NULL);
  http_response_free(r);
  PASS();
}

// ---------------------------------------------------------------------------
// s3_parse_url tests
// ---------------------------------------------------------------------------

TEST test_s3_parse_basic(void) {
  char bucket[64], key[256];
  ASSERT(s3_parse_url("s3://my-bucket/some/prefix/0/0/0", bucket, 64, key, 256));
  ASSERT_STR_EQ("my-bucket",       bucket);
  ASSERT_STR_EQ("some/prefix/0/0/0", key);
  PASS();
}

TEST test_s3_parse_no_key(void) {
  char bucket[64], key[256];
  ASSERT(s3_parse_url("s3://just-bucket", bucket, 64, key, 256));
  ASSERT_STR_EQ("just-bucket", bucket);
  ASSERT_STR_EQ("",            key);
  PASS();
}

TEST test_s3_parse_leading_slash_key(void) {
  char bucket[64], key[256];
  ASSERT(s3_parse_url("s3://bucket/key/with/slashes", bucket, 64, key, 256));
  ASSERT_STR_EQ("bucket",           bucket);
  ASSERT_STR_EQ("key/with/slashes", key);
  PASS();
}

TEST test_s3_parse_wrong_scheme(void) {
  char bucket[64], key[256];
  ASSERT(!s3_parse_url("http://bucket/key",  bucket, 64, key, 256));
  ASSERT(!s3_parse_url("s4://bucket/key",    bucket, 64, key, 256));
  ASSERT(!s3_parse_url(NULL,                 bucket, 64, key, 256));
  PASS();
}

// ---------------------------------------------------------------------------
// s3_creds_from_env tests
// ---------------------------------------------------------------------------

TEST test_s3_creds_env_missing(void) {
  // Unset required vars — must return NULL.
  unsetenv("AWS_ACCESS_KEY_ID");
  unsetenv("AWS_SECRET_ACCESS_KEY");
  s3_credentials *c = s3_creds_from_env();
  ASSERT(c == NULL);
  PASS();
}

TEST test_s3_creds_env_present(void) {
  setenv("AWS_ACCESS_KEY_ID",     "AKIAIOSFODNN7EXAMPLE", 1);
  setenv("AWS_SECRET_ACCESS_KEY", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", 1);
  setenv("AWS_REGION",            "eu-west-1", 1);
  setenv("AWS_ENDPOINT_URL",      "https://minio.local:9000", 1);
  setenv("AWS_SESSION_TOKEN",     "tok123", 1);

  s3_credentials *c = s3_creds_from_env();
  ASSERT(c != NULL);
  ASSERT_STR_EQ("AKIAIOSFODNN7EXAMPLE",                    c->access_key);
  ASSERT_STR_EQ("wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", c->secret_key);
  ASSERT_STR_EQ("eu-west-1",                               c->region);
  ASSERT_STR_EQ("https://minio.local:9000",                c->endpoint);
  ASSERT_STR_EQ("tok123",                                  c->token);
  s3_creds_free(c);

  // cleanup
  unsetenv("AWS_ACCESS_KEY_ID");
  unsetenv("AWS_SECRET_ACCESS_KEY");
  unsetenv("AWS_REGION");
  unsetenv("AWS_ENDPOINT_URL");
  unsetenv("AWS_SESSION_TOKEN");
  PASS();
}

TEST test_s3_creds_env_default_region(void) {
  setenv("AWS_ACCESS_KEY_ID",     "AK", 1);
  setenv("AWS_SECRET_ACCESS_KEY", "SK", 1);
  unsetenv("AWS_REGION");

  s3_credentials *c = s3_creds_from_env();
  ASSERT(c != NULL);
  ASSERT_STR_EQ("us-east-1", c->region);  // default
  s3_creds_free(c);

  unsetenv("AWS_ACCESS_KEY_ID");
  unsetenv("AWS_SECRET_ACCESS_KEY");
  PASS();
}

// ---------------------------------------------------------------------------
// s3_creds_free — must handle NULL
// ---------------------------------------------------------------------------

TEST test_s3_creds_free_null(void) {
  s3_creds_free(NULL);  // must not crash
  PASS();
}

// ---------------------------------------------------------------------------
// http_pool lifecycle tests
// ---------------------------------------------------------------------------

TEST test_pool_create_free(void) {
  http_pool *p = http_pool_new(4);
  ASSERT(p != NULL);
  http_pool_free(p);
  PASS();
}

TEST test_pool_free_null(void) {
  http_pool_free(NULL);  // must not crash
  PASS();
}

TEST test_pool_create_zero_uses_max(void) {
  // 0 max_connections → clamp to internal max, must not crash
  http_pool *p = http_pool_new(0);
  ASSERT(p != NULL);
  http_pool_free(p);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(net_suite) {
  // url_parse
  RUN_TEST(test_parse_http_simple);
  RUN_TEST(test_parse_https_with_port);
  RUN_TEST(test_parse_s3_url);
  RUN_TEST(test_parse_url_with_query);
  RUN_TEST(test_parse_url_no_path);
  RUN_TEST(test_parse_url_port_and_query);
  RUN_TEST(test_parse_url_missing_scheme_separator);
  RUN_TEST(test_parse_url_null);
  RUN_TEST(test_parse_url_trailing_slash);
  RUN_TEST(test_parse_url_query_special_chars);
  RUN_TEST(test_parse_url_ipv4_with_port);
  RUN_TEST(test_parse_url_deep_path);

  // http_response_free
  RUN_TEST(test_response_free_null);
  RUN_TEST(test_response_free_empty_struct);
  RUN_TEST(test_response_free_with_error);
  RUN_TEST(test_response_free_with_data);

  // s3_parse_url
  RUN_TEST(test_s3_parse_basic);
  RUN_TEST(test_s3_parse_no_key);
  RUN_TEST(test_s3_parse_leading_slash_key);
  RUN_TEST(test_s3_parse_wrong_scheme);

  // s3_creds_from_env
  RUN_TEST(test_s3_creds_env_missing);
  RUN_TEST(test_s3_creds_env_present);
  RUN_TEST(test_s3_creds_env_default_region);
  RUN_TEST(test_s3_creds_free_null);

  // connection pool lifecycle
  RUN_TEST(test_pool_create_free);
  RUN_TEST(test_pool_free_null);
  RUN_TEST(test_pool_create_zero_uses_max);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(net_suite);
  GREATEST_MAIN_END();
}
