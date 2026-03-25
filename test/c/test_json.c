#include "greatest.h"
#include "core/json.h"

#include <math.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Primitives
// ---------------------------------------------------------------------------

TEST test_null(void) {
  json_value *v = json_parse("null");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_NULL, json_typeof(v));
  json_free(v);
  PASS();
}

TEST test_bool_true(void) {
  json_value *v = json_parse("true");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_BOOL, json_typeof(v));
  ASSERT_EQ(true, json_get_bool(v, false));
  json_free(v);
  PASS();
}

TEST test_bool_false(void) {
  json_value *v = json_parse("false");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_BOOL, json_typeof(v));
  ASSERT_EQ(false, json_get_bool(v, true));
  json_free(v);
  PASS();
}

TEST test_number_integer(void) {
  json_value *v = json_parse("42");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_NUMBER, json_typeof(v));
  ASSERT_EQ(42, json_get_int(v, 0));
  json_free(v);
  PASS();
}

TEST test_number_negative(void) {
  json_value *v = json_parse("-7");
  ASSERT(v != NULL);
  ASSERT_EQ(-7, json_get_int(v, 0));
  json_free(v);
  PASS();
}

TEST test_number_float(void) {
  json_value *v = json_parse("3.14");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_NUMBER, json_typeof(v));
  ASSERT(fabs(json_get_number(v, 0.0) - 3.14) < 1e-9);
  json_free(v);
  PASS();
}

TEST test_number_exponent(void) {
  json_value *v = json_parse("1e3");
  ASSERT(v != NULL);
  ASSERT(fabs(json_get_number(v, 0.0) - 1000.0) < 1e-9);
  json_free(v);
  PASS();
}

TEST test_number_negative_exponent(void) {
  json_value *v = json_parse("2.5e-1");
  ASSERT(v != NULL);
  ASSERT(fabs(json_get_number(v, 0.0) - 0.25) < 1e-9);
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// String + escapes
// ---------------------------------------------------------------------------

TEST test_string_plain(void) {
  json_value *v = json_parse("\"hello\"");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_STRING, json_typeof(v));
  ASSERT_STR_EQ("hello", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_escapes(void) {
  json_value *v = json_parse("\"tab:\\there\\nnewline\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("tab:\there\nnewline", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_escape_quote(void) {
  json_value *v = json_parse("\"say \\\"hi\\\"\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("say \"hi\"", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_escape_backslash(void) {
  json_value *v = json_parse("\"a\\\\b\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("a\\b", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_escape_slash(void) {
  json_value *v = json_parse("\"a\\/b\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("a/b", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_unicode_escape(void) {
  // \u0041 == 'A', \u0042 == 'B'
  json_value *v = json_parse("\"\\u0041\\u0042\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("AB", json_get_str(v));
  json_free(v);
  PASS();
}

TEST test_string_empty(void) {
  json_value *v = json_parse("\"\"");
  ASSERT(v != NULL);
  ASSERT_STR_EQ("", json_get_str(v));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Arrays
// ---------------------------------------------------------------------------

TEST test_array_empty(void) {
  json_value *v = json_parse("[]");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_ARRAY, json_typeof(v));
  ASSERT_EQ(0u, json_array_len(v));
  json_free(v);
  PASS();
}

TEST test_array_ints(void) {
  json_value *v = json_parse("[1, 2, 3]");
  ASSERT(v != NULL);
  ASSERT_EQ(3u, json_array_len(v));
  ASSERT_EQ(1, json_get_int(json_array_get(v, 0), 0));
  ASSERT_EQ(2, json_get_int(json_array_get(v, 1), 0));
  ASSERT_EQ(3, json_get_int(json_array_get(v, 2), 0));
  ASSERT(json_array_get(v, 3) == NULL);
  json_free(v);
  PASS();
}

TEST test_array_mixed(void) {
  json_value *v = json_parse("[null, true, 42, \"hi\"]");
  ASSERT(v != NULL);
  ASSERT_EQ(4u, json_array_len(v));
  ASSERT_EQ(JSON_NULL,   json_typeof(json_array_get(v, 0)));
  ASSERT_EQ(JSON_BOOL,   json_typeof(json_array_get(v, 1)));
  ASSERT_EQ(JSON_NUMBER, json_typeof(json_array_get(v, 2)));
  ASSERT_EQ(JSON_STRING, json_typeof(json_array_get(v, 3)));
  json_free(v);
  PASS();
}

TEST test_array_nested(void) {
  json_value *v = json_parse("[[1,2],[3,4]]");
  ASSERT(v != NULL);
  ASSERT_EQ(2u, json_array_len(v));
  const json_value *inner = json_array_get(v, 1);
  ASSERT(inner != NULL);
  ASSERT_EQ(2u, json_array_len(inner));
  ASSERT_EQ(4, json_get_int(json_array_get(inner, 1), 0));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Objects
// ---------------------------------------------------------------------------

TEST test_object_empty(void) {
  json_value *v = json_parse("{}");
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_OBJECT, json_typeof(v));
  ASSERT_EQ(0u, json_object_len(v));
  json_free(v);
  PASS();
}

TEST test_object_simple(void) {
  json_value *v = json_parse("{\"x\":1,\"y\":2}");
  ASSERT(v != NULL);
  ASSERT_EQ(2u, json_object_len(v));
  ASSERT_EQ(1, json_get_int(json_object_get(v, "x"), 0));
  ASSERT_EQ(2, json_get_int(json_object_get(v, "y"), 0));
  ASSERT(json_object_get(v, "z") == NULL);
  json_free(v);
  PASS();
}

static int iter_count;
static void count_keys(const char *key, const json_value *val, void *ctx) {
  (void)key; (void)val; (void)ctx;
  iter_count++;
}

TEST test_object_iter(void) {
  json_value *v = json_parse("{\"a\":1,\"b\":2,\"c\":3}");
  ASSERT(v != NULL);
  iter_count = 0;
  json_object_iter(v, count_keys, NULL);
  ASSERT_EQ(3, iter_count);
  json_free(v);
  PASS();
}

TEST test_object_nested(void) {
  json_value *v = json_parse("{\"outer\":{\"inner\":99}}");
  ASSERT(v != NULL);
  const json_value *outer = json_object_get(v, "outer");
  ASSERT(outer != NULL);
  ASSERT_EQ(99, json_get_int(json_object_get(outer, "inner"), 0));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Accessor defaults on type mismatch
// ---------------------------------------------------------------------------

TEST test_defaults_on_mismatch(void) {
  json_value *v = json_parse("\"not_a_number\"");
  ASSERT(v != NULL);
  ASSERT(fabs(json_get_number(v, -1.0) - (-1.0)) < 1e-9);
  ASSERT_EQ((int64_t)-1, json_get_int(v, -1));
  ASSERT_EQ(true, json_get_bool(v, true));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

TEST test_parse_error_empty(void) {
  ASSERT(json_parse("") == NULL);
  ASSERT(json_parse(NULL) == NULL);
  PASS();
}

TEST test_parse_error_trailing_garbage(void) {
  ASSERT(json_parse("42 garbage") == NULL);
  PASS();
}

TEST test_parse_error_bad_value(void) {
  ASSERT(json_parse("tru") == NULL);
  ASSERT(json_parse("nul") == NULL);
  PASS();
}

TEST test_parse_error_unclosed_array(void) {
  ASSERT(json_parse("[1,2") == NULL);
  PASS();
}

TEST test_parse_error_unclosed_object(void) {
  ASSERT(json_parse("{\"a\":1") == NULL);
  PASS();
}

// ---------------------------------------------------------------------------
// Zarr-like metadata
// ---------------------------------------------------------------------------

TEST test_zarr_zarray(void) {
  const char *json =
    "{\"chunks\":[128,128,128],"
    "\"compressor\":{\"id\":\"blosc\",\"cname\":\"zstd\",\"clevel\":1},"
    "\"dtype\":\"|u1\","
    "\"fill_value\":0,"
    "\"order\":\"C\","
    "\"shape\":[1000,2000,3000],"
    "\"zarr_format\":2}";

  json_value *v = json_parse(json);
  ASSERT(v != NULL);
  ASSERT_EQ(JSON_OBJECT, json_typeof(v));

  // chunks
  const json_value *chunks = json_object_get(v, "chunks");
  ASSERT(chunks != NULL);
  ASSERT_EQ(JSON_ARRAY, json_typeof(chunks));
  ASSERT_EQ(3u, json_array_len(chunks));
  ASSERT_EQ(128, json_get_int(json_array_get(chunks, 0), 0));
  ASSERT_EQ(128, json_get_int(json_array_get(chunks, 1), 0));
  ASSERT_EQ(128, json_get_int(json_array_get(chunks, 2), 0));

  // compressor
  const json_value *comp = json_object_get(v, "compressor");
  ASSERT(comp != NULL);
  ASSERT_EQ(JSON_OBJECT, json_typeof(comp));
  ASSERT_STR_EQ("blosc", json_get_str(json_object_get(comp, "id")));
  ASSERT_STR_EQ("zstd",  json_get_str(json_object_get(comp, "cname")));
  ASSERT_EQ(1, json_get_int(json_object_get(comp, "clevel"), 0));

  // dtype
  ASSERT_STR_EQ("|u1", json_get_str(json_object_get(v, "dtype")));

  // fill_value
  ASSERT_EQ(0, json_get_int(json_object_get(v, "fill_value"), -1));

  // order
  ASSERT_STR_EQ("C", json_get_str(json_object_get(v, "order")));

  // shape
  const json_value *shape = json_object_get(v, "shape");
  ASSERT(shape != NULL);
  ASSERT_EQ(3u, json_array_len(shape));
  ASSERT_EQ(1000, json_get_int(json_array_get(shape, 0), 0));
  ASSERT_EQ(2000, json_get_int(json_array_get(shape, 1), 0));
  ASSERT_EQ(3000, json_get_int(json_array_get(shape, 2), 0));

  // zarr_format
  ASSERT_EQ(2, json_get_int(json_object_get(v, "zarr_format"), 0));

  json_free(v);
  PASS();
}

TEST test_zarr_zattrs(void) {
  const char *json = "{\"multiscales\":[{\"version\":\"0.4\",\"axes\":[{\"name\":\"z\",\"type\":\"space\"}]}]}";
  json_value *v = json_parse(json);
  ASSERT(v != NULL);

  const json_value *ms = json_object_get(v, "multiscales");
  ASSERT(ms != NULL);
  ASSERT_EQ(1u, json_array_len(ms));

  const json_value *entry = json_array_get(ms, 0);
  ASSERT_STR_EQ("0.4", json_get_str(json_object_get(entry, "version")));

  const json_value *axes = json_object_get(entry, "axes");
  ASSERT(axes != NULL);
  ASSERT_EQ(1u, json_array_len(axes));

  const json_value *axis = json_array_get(axes, 0);
  ASSERT_STR_EQ("z",     json_get_str(json_object_get(axis, "name")));
  ASSERT_STR_EQ("space", json_get_str(json_object_get(axis, "type")));

  json_free(v);
  PASS();
}

TEST test_zarr_zgroup(void) {
  const char *json = "{\"zarr_format\":2}";
  json_value *v = json_parse(json);
  ASSERT(v != NULL);
  ASSERT_EQ(2, json_get_int(json_object_get(v, "zarr_format"), 0));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Whitespace tolerance
// ---------------------------------------------------------------------------

TEST test_whitespace(void) {
  json_value *v = json_parse("  {  \"k\"  :  [  1  ,  2  ]  }  ");
  ASSERT(v != NULL);
  const json_value *arr = json_object_get(v, "k");
  ASSERT(arr != NULL);
  ASSERT_EQ(2u, json_array_len(arr));
  json_free(v);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(json_suite) {
  // primitives
  RUN_TEST(test_null);
  RUN_TEST(test_bool_true);
  RUN_TEST(test_bool_false);
  RUN_TEST(test_number_integer);
  RUN_TEST(test_number_negative);
  RUN_TEST(test_number_float);
  RUN_TEST(test_number_exponent);
  RUN_TEST(test_number_negative_exponent);
  // strings
  RUN_TEST(test_string_plain);
  RUN_TEST(test_string_escapes);
  RUN_TEST(test_string_escape_quote);
  RUN_TEST(test_string_escape_backslash);
  RUN_TEST(test_string_escape_slash);
  RUN_TEST(test_string_unicode_escape);
  RUN_TEST(test_string_empty);
  // arrays
  RUN_TEST(test_array_empty);
  RUN_TEST(test_array_ints);
  RUN_TEST(test_array_mixed);
  RUN_TEST(test_array_nested);
  // objects
  RUN_TEST(test_object_empty);
  RUN_TEST(test_object_simple);
  RUN_TEST(test_object_iter);
  RUN_TEST(test_object_nested);
  // defaults
  RUN_TEST(test_defaults_on_mismatch);
  // errors
  RUN_TEST(test_parse_error_empty);
  RUN_TEST(test_parse_error_trailing_garbage);
  RUN_TEST(test_parse_error_bad_value);
  RUN_TEST(test_parse_error_unclosed_array);
  RUN_TEST(test_parse_error_unclosed_object);
  // zarr metadata
  RUN_TEST(test_zarr_zarray);
  RUN_TEST(test_zarr_zattrs);
  RUN_TEST(test_zarr_zgroup);
  // whitespace
  RUN_TEST(test_whitespace);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(json_suite);
  GREATEST_MAIN_END();
}
