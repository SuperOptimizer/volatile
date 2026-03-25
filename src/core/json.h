#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef enum {
  JSON_NULL,
  JSON_BOOL,
  JSON_NUMBER,
  JSON_STRING,
  JSON_ARRAY,
  JSON_OBJECT,
} json_type;

typedef struct json_value json_value;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Parse a null-terminated JSON string. Returns NULL on error.
json_value *json_parse(const char *str);
void        json_free(json_value *v);

// ---------------------------------------------------------------------------
// Type checking
// ---------------------------------------------------------------------------

json_type json_typeof(const json_value *v);

// ---------------------------------------------------------------------------
// Scalar accessors (return defaults on type mismatch)
// ---------------------------------------------------------------------------

bool        json_get_bool(const json_value *v, bool def);
double      json_get_number(const json_value *v, double def);
int64_t     json_get_int(const json_value *v, int64_t def);
const char *json_get_str(const json_value *v);  // NULL if not string

// ---------------------------------------------------------------------------
// Array access
// ---------------------------------------------------------------------------

size_t           json_array_len(const json_value *v);
const json_value *json_array_get(const json_value *v, size_t idx);

// ---------------------------------------------------------------------------
// Object access
// ---------------------------------------------------------------------------

const json_value *json_object_get(const json_value *v, const char *key);
size_t            json_object_len(const json_value *v);

typedef void (*json_object_iter_fn)(const char *key, const json_value *val, void *ctx);
void json_object_iter(const json_value *v, json_object_iter_fn fn, void *ctx);
