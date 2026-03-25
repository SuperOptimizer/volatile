#define _POSIX_C_SOURCE 200809L

#include "server/db.h"
#include "core/log.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

static const char *k_schema =
  "CREATE TABLE IF NOT EXISTS segments ("
  "  id           INTEGER PRIMARY KEY AUTOINCREMENT,"
  "  volume_id    TEXT    NOT NULL,"
  "  name         TEXT    NOT NULL,"
  "  surface_path TEXT    NOT NULL DEFAULT '',"
  "  created_at   INTEGER NOT NULL,"
  "  updated_at   INTEGER NOT NULL"
  ");"
  "CREATE INDEX IF NOT EXISTS idx_segments_volume ON segments(volume_id);"
  "CREATE TABLE IF NOT EXISTS annotations ("
  "  id           INTEGER PRIMARY KEY AUTOINCREMENT,"
  "  segment_id   INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,"
  "  type         TEXT    NOT NULL,"
  "  data_json    TEXT    NOT NULL DEFAULT '{}',"
  "  created_at   INTEGER NOT NULL"
  ");"
  "CREATE INDEX IF NOT EXISTS idx_annotations_segment ON annotations(segment_id);"
  "PRAGMA foreign_keys = ON;";

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

struct seg_db {
  sqlite3 *sql;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t now_unix(void) {
  return (int64_t)time(NULL);
}

// Prepare, step, and finalize a statement that returns no rows.
// Returns SQLITE_DONE on success.
static int exec_stmt(sqlite3_stmt *stmt) {
  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc;
}

// ---------------------------------------------------------------------------
// Open / close
// ---------------------------------------------------------------------------

seg_db *seg_db_open(const char *path) {
  if (!path) return NULL;

  sqlite3 *sql = NULL;
  int rc = sqlite3_open(path, &sql);
  if (rc != SQLITE_OK) {
    LOG_ERROR("seg_db_open: sqlite3_open(%s): %s", path, sqlite3_errmsg(sql));
    sqlite3_close(sql);
    return NULL;
  }

  // Enable WAL for better concurrent read performance.
  sqlite3_exec(sql, "PRAGMA journal_mode=WAL;", NULL, NULL, NULL);
  sqlite3_exec(sql, "PRAGMA foreign_keys=ON;",  NULL, NULL, NULL);

  rc = sqlite3_exec(sql, k_schema, NULL, NULL, NULL);
  if (rc != SQLITE_OK) {
    LOG_ERROR("seg_db_open: schema init: %s", sqlite3_errmsg(sql));
    sqlite3_close(sql);
    return NULL;
  }

  seg_db *db = malloc(sizeof(*db));
  if (!db) { sqlite3_close(sql); return NULL; }
  db->sql = sql;
  return db;
}

void seg_db_close(seg_db *db) {
  if (!db) return;
  sqlite3_close(db->sql);
  free(db);
}

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

int64_t seg_db_insert_segment(seg_db *db, const char *volume_id,
                               const char *name, const char *surface_path) {
  if (!db || !volume_id || !name) return -1;
  if (!surface_path) surface_path = "";

  const char *sql =
    "INSERT INTO segments(volume_id, name, surface_path, created_at, updated_at)"
    " VALUES(?,?,?,?,?)";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) {
    LOG_ERROR("seg_db_insert_segment: prepare: %s", sqlite3_errmsg(db->sql));
    return -1;
  }

  int64_t ts = now_unix();
  sqlite3_bind_text(stmt, 1, volume_id,    -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 2, name,         -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt, 3, surface_path, -1, SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 4, ts);
  sqlite3_bind_int64(stmt, 5, ts);

  if (exec_stmt(stmt) != SQLITE_DONE) {
    LOG_ERROR("seg_db_insert_segment: step: %s", sqlite3_errmsg(db->sql));
    return -1;
  }
  return (int64_t)sqlite3_last_insert_rowid(db->sql);
}

bool seg_db_get_segment(seg_db *db, int64_t id, segment_row *out) {
  if (!db || !out) return false;

  const char *sql =
    "SELECT id, volume_id, name, surface_path, created_at, updated_at"
    " FROM segments WHERE id=?";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) return false;
  sqlite3_bind_int64(stmt, 1, id);

  bool found = false;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    out->id = sqlite3_column_int64(stmt, 0);

    const char *vol = (const char *)sqlite3_column_text(stmt, 1);
    const char *nm  = (const char *)sqlite3_column_text(stmt, 2);
    const char *sp  = (const char *)sqlite3_column_text(stmt, 3);

    // NOTE: strncpy + explicit NUL-terminate to guard against oversized DB values.
    strncpy(out->volume_id,    vol ? vol : "", sizeof(out->volume_id)    - 1);
    strncpy(out->name,         nm  ? nm  : "", sizeof(out->name)         - 1);
    strncpy(out->surface_path, sp  ? sp  : "", sizeof(out->surface_path) - 1);
    out->volume_id[sizeof(out->volume_id)       - 1] = '\0';
    out->name[sizeof(out->name)                 - 1] = '\0';
    out->surface_path[sizeof(out->surface_path) - 1] = '\0';

    out->created_at = sqlite3_column_int64(stmt, 4);
    out->updated_at = sqlite3_column_int64(stmt, 5);
    found = true;
  }

  sqlite3_finalize(stmt);
  return found;
}

bool seg_db_list_segments(seg_db *db, const char *volume_id,
                           segment_cb cb, void *userdata) {
  if (!db || !volume_id || !cb) return false;

  const char *sql =
    "SELECT id, volume_id, name, surface_path, created_at, updated_at"
    " FROM segments WHERE volume_id=? ORDER BY id";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) return false;
  sqlite3_bind_text(stmt, 1, volume_id, -1, SQLITE_STATIC);

  bool ok = true;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    segment_row row = {0};
    row.id = sqlite3_column_int64(stmt, 0);

    const char *vol = (const char *)sqlite3_column_text(stmt, 1);
    const char *nm  = (const char *)sqlite3_column_text(stmt, 2);
    const char *sp  = (const char *)sqlite3_column_text(stmt, 3);

    strncpy(row.volume_id,    vol ? vol : "", sizeof(row.volume_id)    - 1);
    strncpy(row.name,         nm  ? nm  : "", sizeof(row.name)         - 1);
    strncpy(row.surface_path, sp  ? sp  : "", sizeof(row.surface_path) - 1);
    row.volume_id[sizeof(row.volume_id)       - 1] = '\0';
    row.name[sizeof(row.name)                 - 1] = '\0';
    row.surface_path[sizeof(row.surface_path) - 1] = '\0';

    row.created_at = sqlite3_column_int64(stmt, 4);
    row.updated_at = sqlite3_column_int64(stmt, 5);

    if (!cb(&row, userdata)) break;
  }

  sqlite3_finalize(stmt);
  return ok;
}

bool seg_db_list_all_segments(seg_db *db, segment_cb cb, void *userdata) {
  if (!db || !cb) return false;

  const char *sql =
    "SELECT id, volume_id, name, surface_path, created_at, updated_at"
    " FROM segments ORDER BY id";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) return false;

  bool ok = true;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    segment_row row = {0};
    row.id = sqlite3_column_int64(stmt, 0);

    const char *vol = (const char *)sqlite3_column_text(stmt, 1);
    const char *nm  = (const char *)sqlite3_column_text(stmt, 2);
    const char *sp  = (const char *)sqlite3_column_text(stmt, 3);

    strncpy(row.volume_id,    vol ? vol : "", sizeof(row.volume_id)    - 1);
    strncpy(row.name,         nm  ? nm  : "", sizeof(row.name)         - 1);
    strncpy(row.surface_path, sp  ? sp  : "", sizeof(row.surface_path) - 1);

    row.volume_id[sizeof(row.volume_id)       - 1] = '\0';
    row.name[sizeof(row.name)                 - 1] = '\0';
    row.surface_path[sizeof(row.surface_path) - 1] = '\0';

    row.created_at = sqlite3_column_int64(stmt, 4);
    row.updated_at = sqlite3_column_int64(stmt, 5);

    if (!cb(&row, userdata)) break;
  }

  sqlite3_finalize(stmt);
  return ok;
}

bool seg_db_delete_segment(seg_db *db, int64_t id) {
  if (!db) return false;

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, "DELETE FROM segments WHERE id=?",
                         -1, &stmt, NULL) != SQLITE_OK) return false;
  sqlite3_bind_int64(stmt, 1, id);

  bool ok = exec_stmt(stmt) == SQLITE_DONE && sqlite3_changes(db->sql) > 0;
  return ok;
}

// ---------------------------------------------------------------------------
// Annotations
// ---------------------------------------------------------------------------

void annotation_row_free(annotation_row *a) {
  if (!a) return;
  free(a->data_json);
  a->data_json = NULL;
}

int64_t seg_db_insert_annotation(seg_db *db, int64_t segment_id,
                                  const char *type, const char *data_json) {
  if (!db || !type) return -1;
  if (!data_json) data_json = "{}";

  const char *sql =
    "INSERT INTO annotations(segment_id, type, data_json, created_at)"
    " VALUES(?,?,?,?)";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) {
    LOG_ERROR("seg_db_insert_annotation: prepare: %s", sqlite3_errmsg(db->sql));
    return -1;
  }

  sqlite3_bind_int64(stmt, 1, segment_id);
  sqlite3_bind_text(stmt,  2, type,      -1, SQLITE_STATIC);
  sqlite3_bind_text(stmt,  3, data_json, -1, SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 4, now_unix());

  if (exec_stmt(stmt) != SQLITE_DONE) {
    LOG_ERROR("seg_db_insert_annotation: step: %s", sqlite3_errmsg(db->sql));
    return -1;
  }
  return (int64_t)sqlite3_last_insert_rowid(db->sql);
}

bool seg_db_list_annotations(seg_db *db, int64_t segment_id,
                               annotation_cb cb, void *userdata) {
  if (!db || !cb) return false;

  const char *sql =
    "SELECT id, segment_id, type, data_json, created_at"
    " FROM annotations WHERE segment_id=? ORDER BY id";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) return false;
  sqlite3_bind_int64(stmt, 1, segment_id);

  bool ok = true;
  while (sqlite3_step(stmt) == SQLITE_ROW) {
    annotation_row row = {0};
    row.id         = sqlite3_column_int64(stmt, 0);
    row.segment_id = sqlite3_column_int64(stmt, 1);

    const char *tp  = (const char *)sqlite3_column_text(stmt, 2);
    const char *dj  = (const char *)sqlite3_column_text(stmt, 3);

    strncpy(row.type, tp ? tp : "", sizeof(row.type) - 1);
    row.type[sizeof(row.type) - 1] = '\0';
    row.data_json  = strdup(dj ? dj : "{}");
    row.created_at = sqlite3_column_int64(stmt, 4);

    bool cont = cb(&row, userdata);
    annotation_row_free(&row);
    if (!cont) break;
  }

  sqlite3_finalize(stmt);
  return ok;
}

bool seg_db_delete_annotation(seg_db *db, int64_t id) {
  if (!db) return false;

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(db->sql, "DELETE FROM annotations WHERE id=?",
                         -1, &stmt, NULL) != SQLITE_OK) return false;
  sqlite3_bind_int64(stmt, 1, id);

  bool ok = exec_stmt(stmt) == SQLITE_DONE && sqlite3_changes(db->sql) > 0;
  return ok;
}
