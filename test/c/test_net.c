#include "greatest.h"
#include "core/net.h"

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

  // http_response_free
  RUN_TEST(test_response_free_null);
  RUN_TEST(test_response_free_empty_struct);
  RUN_TEST(test_response_free_with_error);
  RUN_TEST(test_response_free_with_data);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(net_suite);
  GREATEST_MAIN_END();
}
