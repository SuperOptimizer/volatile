#pragma once
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

typedef struct seg_db seg_db;

// Opens (or creates) a SQLite database at `path`. Use ":memory:" for tests.
// Creates the segments and annotations tables if they don't exist.
// Returns NULL on error.
seg_db *seg_db_open(const char *path);
void    seg_db_close(seg_db *db);

// ---------------------------------------------------------------------------
// Segment record
// ---------------------------------------------------------------------------

typedef struct {
  int64_t     id;
  char        volume_id[128];
  char        name[256];
  char        surface_path[1024];
  int64_t     created_at;   // Unix timestamp
  int64_t     updated_at;
} segment_row;

// Returns new row id on success, -1 on error.
int64_t seg_db_insert_segment(seg_db *db, const char *volume_id,
                               const char *name, const char *surface_path);

// Fills `out`; returns false if not found or on error.
bool seg_db_get_segment(seg_db *db, int64_t id, segment_row *out);

// Callback invoked once per matching row. Return false to stop iteration.
typedef bool (*segment_cb)(const segment_row *row, void *userdata);

// Iterates segments for a volume_id, calling cb for each.
bool seg_db_list_segments(seg_db *db, const char *volume_id,
                           segment_cb cb, void *userdata);

// Iterates ALL segments in the database regardless of volume_id.
bool seg_db_list_all_segments(seg_db *db, segment_cb cb, void *userdata);

bool seg_db_delete_segment(seg_db *db, int64_t id);

// ---------------------------------------------------------------------------
// Annotation record
// ---------------------------------------------------------------------------

typedef struct {
  int64_t id;
  int64_t segment_id;
  char    type[64];
  char   *data_json;   // heap-allocated; caller must free
  int64_t created_at;
} annotation_row;

void annotation_row_free(annotation_row *a);

// Returns new row id on success, -1 on error.
int64_t seg_db_insert_annotation(seg_db *db, int64_t segment_id,
                                  const char *type, const char *data_json);

// Callback invoked once per matching annotation. Return false to stop.
typedef bool (*annotation_cb)(const annotation_row *row, void *userdata);

bool seg_db_list_annotations(seg_db *db, int64_t segment_id,
                               annotation_cb cb, void *userdata);

bool seg_db_delete_annotation(seg_db *db, int64_t id);
