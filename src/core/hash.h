#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// String-keyed hash map
// ---------------------------------------------------------------------------

typedef struct hash_map hash_map;

hash_map *hash_map_new(void);
void      hash_map_free(hash_map *m);
void     *hash_map_get(hash_map *m, const char *key);
bool      hash_map_put(hash_map *m, const char *key, void *val); // true = new key
bool      hash_map_del(hash_map *m, const char *key);
size_t    hash_map_len(hash_map *m);

// Iteration
typedef struct { const char *key; void *val; } hash_map_entry;
typedef struct hash_map_iter hash_map_iter;

hash_map_iter *hash_map_iter_new(hash_map *m);
bool           hash_map_iter_next(hash_map_iter *it, hash_map_entry *out);
void           hash_map_iter_free(hash_map_iter *it);

// ---------------------------------------------------------------------------
// Integer-keyed hash map
// ---------------------------------------------------------------------------

typedef struct hash_map_int hash_map_int;

hash_map_int *hash_map_int_new(void);
void          hash_map_int_free(hash_map_int *m);
void         *hash_map_int_get(hash_map_int *m, uint64_t key);
bool          hash_map_int_put(hash_map_int *m, uint64_t key, void *val); // true = new key
bool          hash_map_int_del(hash_map_int *m, uint64_t key);
size_t        hash_map_int_len(hash_map_int *m);
