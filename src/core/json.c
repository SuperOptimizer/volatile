#include "core/json.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal structures
// ---------------------------------------------------------------------------

typedef struct {
  char       *key;
  json_value *val;
} json_kv;

struct json_value {
  json_type type;
  union {
    bool    b;
    double  n;
    char   *s;
    struct {
      json_value **items;
      size_t       len;
      size_t       cap;
    } arr;
    struct {
      json_kv *pairs;
      size_t   len;
      size_t   cap;
    } obj;
  };
};

// ---------------------------------------------------------------------------
// Parser state
// ---------------------------------------------------------------------------

typedef struct {
  const char *cur;
  const char *end;
  bool        error;
} parser;

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------

static json_value *parse_value(parser *p);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void skip_ws(parser *p) {
  while (p->cur < p->end && isspace((unsigned char)*p->cur))
    p->cur++;
}

static bool peek(parser *p, char c) {
  skip_ws(p);
  return p->cur < p->end && *p->cur == c;
}

static bool consume(parser *p, char c) {
  skip_ws(p);
  if (p->cur < p->end && *p->cur == c) {
    p->cur++;
    return true;
  }
  p->error = true;
  return false;
}

static json_value *alloc_value(json_type type) {
  json_value *v = calloc(1, sizeof(json_value));
  if (v) v->type = type;
  return v;
}

// ---------------------------------------------------------------------------
// \uXXXX helper: encode a codepoint as UTF-8 into buf, return bytes written
// ---------------------------------------------------------------------------

static int encode_utf8(uint32_t cp, char *buf) {
  if (cp <= 0x7F) {
    buf[0] = (char)cp;
    return 1;
  } else if (cp <= 0x7FF) {
    buf[0] = (char)(0xC0 | (cp >> 6));
    buf[1] = (char)(0x80 | (cp & 0x3F));
    return 2;
  } else if (cp <= 0xFFFF) {
    buf[0] = (char)(0xE0 | (cp >> 12));
    buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
  } else {
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
  }
}

// ---------------------------------------------------------------------------
// String parser
// ---------------------------------------------------------------------------

static char *parse_string_raw(parser *p) {
  if (!consume(p, '"')) return NULL;

  // First pass: measure output length
  const char *scan = p->cur;
  size_t      out_len = 0;
  while (scan < p->end && *scan != '"') {
    if (*scan == '\\') {
      scan++;
      if (scan >= p->end) { p->error = true; return NULL; }
      if (*scan == 'u') {
        if (scan + 4 >= p->end) { p->error = true; return NULL; }
        uint32_t cp = 0;
        for (int i = 1; i <= 4; i++) {
          char hc = scan[i];
          if      (hc >= '0' && hc <= '9') cp = cp * 16 + (uint32_t)(hc - '0');
          else if (hc >= 'a' && hc <= 'f') cp = cp * 16 + (uint32_t)(hc - 'a' + 10);
          else if (hc >= 'A' && hc <= 'F') cp = cp * 16 + (uint32_t)(hc - 'A' + 10);
          else { p->error = true; return NULL; }
        }
        // surrogate pair check
        if (cp >= 0xD800 && cp <= 0xDBFF) {
          // high surrogate — expect \uXXXX low surrogate
          if (scan + 10 < p->end && scan[5] == '\\' && scan[6] == 'u') {
            uint32_t lo = 0;
            for (int i = 7; i <= 10; i++) {
              char hc = scan[i];
              if      (hc >= '0' && hc <= '9') lo = lo * 16 + (uint32_t)(hc - '0');
              else if (hc >= 'a' && hc <= 'f') lo = lo * 16 + (uint32_t)(hc - 'a' + 10);
              else if (hc >= 'A' && hc <= 'F') lo = lo * 16 + (uint32_t)(hc - 'A' + 10);
              else { p->error = true; return NULL; }
            }
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            scan += 6;  // skip second \uXXXX (6 extra chars before the +4 below)
          }
        }
        char tmp[4];
        out_len += (size_t)encode_utf8(cp, tmp);
        scan += 5;  // skip uXXXX (we advanced past '\' already)
      } else {
        out_len++;
        scan++;
      }
    } else {
      out_len++;
      scan++;
    }
  }
  if (scan >= p->end) { p->error = true; return NULL; }

  // Second pass: fill the output buffer
  char *out = malloc(out_len + 1);
  if (!out) { p->error = true; return NULL; }
  char *w = out;

  while (p->cur < p->end && *p->cur != '"') {
    if (*p->cur == '\\') {
      p->cur++;
      switch (*p->cur) {
        case '"':  *w++ = '"';  p->cur++; break;
        case '\\': *w++ = '\\'; p->cur++; break;
        case '/':  *w++ = '/';  p->cur++; break;
        case 'n':  *w++ = '\n'; p->cur++; break;
        case 'r':  *w++ = '\r'; p->cur++; break;
        case 't':  *w++ = '\t'; p->cur++; break;
        case 'b':  *w++ = '\b'; p->cur++; break;
        case 'f':  *w++ = '\f'; p->cur++; break;
        case 'u': {
          p->cur++;  // skip 'u'
          uint32_t cp = 0;
          for (int i = 0; i < 4; i++) {
            char hc = *p->cur++;
            if      (hc >= '0' && hc <= '9') cp = cp * 16 + (uint32_t)(hc - '0');
            else if (hc >= 'a' && hc <= 'f') cp = cp * 16 + (uint32_t)(hc - 'a' + 10);
            else if (hc >= 'A' && hc <= 'F') cp = cp * 16 + (uint32_t)(hc - 'A' + 10);
          }
          if (cp >= 0xD800 && cp <= 0xDBFF && p->cur + 5 < p->end &&
              p->cur[0] == '\\' && p->cur[1] == 'u') {
            p->cur += 2;
            uint32_t lo = 0;
            for (int i = 0; i < 4; i++) {
              char hc = *p->cur++;
              if      (hc >= '0' && hc <= '9') lo = lo * 16 + (uint32_t)(hc - '0');
              else if (hc >= 'a' && hc <= 'f') lo = lo * 16 + (uint32_t)(hc - 'a' + 10);
              else if (hc >= 'A' && hc <= 'F') lo = lo * 16 + (uint32_t)(hc - 'A' + 10);
            }
            cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
          }
          w += encode_utf8(cp, w);
          break;
        }
        default:
          *w++ = *p->cur++;
          break;
      }
    } else {
      *w++ = *p->cur++;
    }
  }
  *w = '\0';
  p->cur++;  // skip closing '"'
  return out;
}

// ---------------------------------------------------------------------------
// Value parsers
// ---------------------------------------------------------------------------

static json_value *parse_string(parser *p) {
  char *s = parse_string_raw(p);
  if (!s) return NULL;
  json_value *v = alloc_value(JSON_STRING);
  if (!v) { free(s); return NULL; }
  v->s = s;
  return v;
}

static json_value *parse_number(parser *p) {
  skip_ws(p);
  char *end_ptr;
  double n = strtod(p->cur, &end_ptr);
  if (end_ptr == p->cur) { p->error = true; return NULL; }
  p->cur = end_ptr;
  json_value *v = alloc_value(JSON_NUMBER);
  if (!v) return NULL;
  v->n = n;
  return v;
}

static json_value *parse_array(parser *p) {
  if (!consume(p, '[')) return NULL;
  json_value *v = alloc_value(JSON_ARRAY);
  if (!v) { p->error = true; return NULL; }

  if (peek(p, ']')) { p->cur++; return v; }

  do {
    skip_ws(p);
    json_value *item = parse_value(p);
    if (!item || p->error) { json_free(v); return NULL; }

    if (v->arr.len == v->arr.cap) {
      size_t new_cap = v->arr.cap ? v->arr.cap * 2 : 4;
      json_value **tmp = realloc(v->arr.items, new_cap * sizeof(json_value *));
      if (!tmp) { json_free(item); json_free(v); p->error = true; return NULL; }
      v->arr.items = tmp;
      v->arr.cap   = new_cap;
    }
    v->arr.items[v->arr.len++] = item;

    skip_ws(p);
  } while (p->cur < p->end && *p->cur == ',' && p->cur++);

  if (!consume(p, ']')) { json_free(v); return NULL; }
  return v;
}

static json_value *parse_object(parser *p) {
  if (!consume(p, '{')) return NULL;
  json_value *v = alloc_value(JSON_OBJECT);
  if (!v) { p->error = true; return NULL; }

  skip_ws(p);
  if (peek(p, '}')) { p->cur++; return v; }

  do {
    skip_ws(p);
    char *key = parse_string_raw(p);
    if (!key || p->error) { json_free(v); return NULL; }

    if (!consume(p, ':')) { free(key); json_free(v); return NULL; }

    skip_ws(p);
    json_value *val = parse_value(p);
    if (!val || p->error) { free(key); json_free(v); return NULL; }

    if (v->obj.len == v->obj.cap) {
      size_t   new_cap = v->obj.cap ? v->obj.cap * 2 : 4;
      json_kv *tmp = realloc(v->obj.pairs, new_cap * sizeof(json_kv));
      if (!tmp) { free(key); json_free(val); json_free(v); p->error = true; return NULL; }
      v->obj.pairs = tmp;
      v->obj.cap   = new_cap;
    }
    v->obj.pairs[v->obj.len].key = key;
    v->obj.pairs[v->obj.len].val = val;
    v->obj.len++;

    skip_ws(p);
  } while (p->cur < p->end && *p->cur == ',' && p->cur++);

  if (!consume(p, '}')) { json_free(v); return NULL; }
  return v;
}

static json_value *parse_value(parser *p) {
  skip_ws(p);
  if (p->cur >= p->end) { p->error = true; return NULL; }

  char c = *p->cur;

  if (c == '"')  return parse_string(p);
  if (c == '[')  return parse_array(p);
  if (c == '{')  return parse_object(p);

  if (c == 't') {
    if (p->cur + 3 < p->end && memcmp(p->cur, "true", 4) == 0) {
      p->cur += 4;
      json_value *v = alloc_value(JSON_BOOL);
      if (v) v->b = true;
      return v;
    }
    p->error = true; return NULL;
  }
  if (c == 'f') {
    if (p->cur + 4 < p->end && memcmp(p->cur, "false", 5) == 0) {
      p->cur += 5;
      json_value *v = alloc_value(JSON_BOOL);
      if (v) v->b = false;
      return v;
    }
    p->error = true; return NULL;
  }
  if (c == 'n') {
    if (p->cur + 3 < p->end && memcmp(p->cur, "null", 4) == 0) {
      p->cur += 4;
      return alloc_value(JSON_NULL);
    }
    p->error = true; return NULL;
  }
  if (c == '-' || isdigit((unsigned char)c)) return parse_number(p);

  p->error = true;
  return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

json_value *json_parse(const char *str) {
  if (!str) return NULL;
  parser p = {.cur = str, .end = str + strlen(str), .error = false};
  json_value *v = parse_value(&p);
  if (p.error || !v) {
    json_free(v);
    return NULL;
  }
  // reject trailing non-whitespace
  skip_ws(&p);
  if (p.cur != p.end) {
    json_free(v);
    return NULL;
  }
  return v;
}

void json_free(json_value *v) {
  if (!v) return;
  switch (v->type) {
    case JSON_STRING:
      free(v->s);
      break;
    case JSON_ARRAY:
      for (size_t i = 0; i < v->arr.len; i++)
        json_free(v->arr.items[i]);
      free(v->arr.items);
      break;
    case JSON_OBJECT:
      for (size_t i = 0; i < v->obj.len; i++) {
        free(v->obj.pairs[i].key);
        json_free(v->obj.pairs[i].val);
      }
      free(v->obj.pairs);
      break;
    default:
      break;
  }
  free(v);
}

json_type json_typeof(const json_value *v) {
  return v ? v->type : JSON_NULL;
}

bool json_get_bool(const json_value *v, bool def) {
  return (v && v->type == JSON_BOOL) ? v->b : def;
}

double json_get_number(const json_value *v, double def) {
  return (v && v->type == JSON_NUMBER) ? v->n : def;
}

int64_t json_get_int(const json_value *v, int64_t def) {
  return (v && v->type == JSON_NUMBER) ? (int64_t)v->n : def;
}

const char *json_get_str(const json_value *v) {
  return (v && v->type == JSON_STRING) ? v->s : NULL;
}

size_t json_array_len(const json_value *v) {
  return (v && v->type == JSON_ARRAY) ? v->arr.len : 0;
}

const json_value *json_array_get(const json_value *v, size_t idx) {
  if (!v || v->type != JSON_ARRAY || idx >= v->arr.len) return NULL;
  return v->arr.items[idx];
}

const json_value *json_object_get(const json_value *v, const char *key) {
  if (!v || v->type != JSON_OBJECT || !key) return NULL;
  for (size_t i = 0; i < v->obj.len; i++) {
    if (strcmp(v->obj.pairs[i].key, key) == 0)
      return v->obj.pairs[i].val;
  }
  return NULL;
}

size_t json_object_len(const json_value *v) {
  return (v && v->type == JSON_OBJECT) ? v->obj.len : 0;
}

void json_object_iter(const json_value *v, json_object_iter_fn fn, void *ctx) {
  if (!v || v->type != JSON_OBJECT || !fn) return;
  for (size_t i = 0; i < v->obj.len; i++)
    fn(v->obj.pairs[i].key, v->obj.pairs[i].val, ctx);
}
