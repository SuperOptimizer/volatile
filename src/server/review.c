#define _POSIX_C_SOURCE 200809L
#include "server/review.h"
#include "core/log.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// The reviews table is created alongside the existing segments/annotations
// schema. review_new() runs CREATE TABLE IF NOT EXISTS so it is safe to call
// on a database that already has the table.
static const char *k_reviews_schema =
  "CREATE TABLE IF NOT EXISTS reviews ("
  "  id          INTEGER PRIMARY KEY AUTOINCREMENT,"
  "  surface_id  INTEGER NOT NULL,"
  "  reviewer_id INTEGER NOT NULL,"
  "  status      INTEGER NOT NULL DEFAULT 0,"  // review_status enum
  "  comment     TEXT    NOT NULL DEFAULT '',"
  "  timestamp   INTEGER NOT NULL"
  ");"
  "CREATE INDEX IF NOT EXISTS idx_reviews_surface ON reviews(surface_id);"
  "CREATE INDEX IF NOT EXISTS idx_reviews_status  ON reviews(status);";

// seg_db exposes its sqlite3* through the opaque handle only via SQL calls;
// we reach the raw handle by declaring the struct here — it matches db.c.
struct seg_db { sqlite3 *sql; };

struct review_system { seg_db *db; };

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static int64_t now_ts(void) { return (int64_t)time(NULL); }

// Prepare → bind status+comment+timestamp+reviewer_id+review_id → step.
static bool set_status(review_system *r, int64_t review_id, int reviewer_id,
                       review_status status, const char *comment) {
  if (!r || !comment) return false;
  const char *sql =
    "UPDATE reviews SET status=?, comment=?, timestamp=?, reviewer_id=?"
    " WHERE id=?";
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(r->db->sql, sql, -1, &stmt, NULL) != SQLITE_OK)
    return false;
  sqlite3_bind_int  (stmt, 1, (int)status);
  sqlite3_bind_text (stmt, 2, comment, -1, SQLITE_STATIC);
  sqlite3_bind_int64(stmt, 3, now_ts());
  sqlite3_bind_int  (stmt, 4, reviewer_id);
  sqlite3_bind_int64(stmt, 5, review_id);
  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  return rc == SQLITE_DONE && sqlite3_changes(r->db->sql) > 0;
}

// Fill a review_entry from the current row of stmt (columns: id, surface_id,
// reviewer_id, status, comment, timestamp).
static void row_to_entry(sqlite3_stmt *stmt, review_entry *e) {
  e->review_id   = sqlite3_column_int64(stmt, 0);
  e->surface_id  = sqlite3_column_int64(stmt, 1);
  e->reviewer_id = sqlite3_column_int  (stmt, 2);
  e->status      = (review_status)sqlite3_column_int(stmt, 3);
  const char *c  = (const char *)sqlite3_column_text(stmt, 4);
  strncpy(e->comment, c ? c : "", sizeof(e->comment) - 1);
  e->comment[sizeof(e->comment) - 1] = '\0';
  e->timestamp = sqlite3_column_int64(stmt, 5);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

review_system *review_new(seg_db *db) {
  if (!db) return NULL;
  if (sqlite3_exec(db->sql, k_reviews_schema, NULL, NULL, NULL) != SQLITE_OK) {
    LOG_ERROR("review_new: schema: %s", sqlite3_errmsg(db->sql));
    return NULL;
  }
  review_system *r = calloc(1, sizeof(*r));
  if (!r) return NULL;
  r->db = db;
  return r;
}

void review_free(review_system *r) { free(r); }

// ---------------------------------------------------------------------------
// Submit
// ---------------------------------------------------------------------------

int64_t review_submit(review_system *r, int64_t surface_id, int submitter_id) {
  if (!r) return -1;
  const char *sql =
    "INSERT INTO reviews(surface_id, reviewer_id, status, comment, timestamp)"
    " VALUES(?,?,0,'',?)";
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(r->db->sql, sql, -1, &stmt, NULL) != SQLITE_OK) {
    LOG_ERROR("review_submit: prepare: %s", sqlite3_errmsg(r->db->sql));
    return -1;
  }
  sqlite3_bind_int64(stmt, 1, surface_id);
  sqlite3_bind_int  (stmt, 2, submitter_id);
  sqlite3_bind_int64(stmt, 3, now_ts());
  int rc = sqlite3_step(stmt);
  sqlite3_finalize(stmt);
  if (rc != SQLITE_DONE) {
    LOG_ERROR("review_submit: step: %s", sqlite3_errmsg(r->db->sql));
    return -1;
  }
  return (int64_t)sqlite3_last_insert_rowid(r->db->sql);
}

// ---------------------------------------------------------------------------
// Review actions
// ---------------------------------------------------------------------------

bool review_approve(review_system *r, int64_t review_id, int reviewer_id,
                    const char *comment) {
  return set_status(r, review_id, reviewer_id, REVIEW_APPROVED,
                    comment ? comment : "");
}

bool review_reject(review_system *r, int64_t review_id, int reviewer_id,
                   const char *comment) {
  return set_status(r, review_id, reviewer_id, REVIEW_REJECTED,
                    comment ? comment : "");
}

bool review_request_changes(review_system *r, int64_t review_id,
                             int reviewer_id, const char *comment) {
  return set_status(r, review_id, reviewer_id, REVIEW_NEEDS_CHANGES,
                    comment ? comment : "");
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

static int query_entries(review_system *r, const char *sql,
                         int bind_int, int64_t bind_val,
                         review_entry *out, int max) {
  if (!r || !out || max <= 0) return 0;
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(r->db->sql, sql, -1, &stmt, NULL) != SQLITE_OK)
    return 0;
  if (bind_int)
    sqlite3_bind_int64(stmt, 1, bind_val);
  int n = 0;
  while (n < max && sqlite3_step(stmt) == SQLITE_ROW)
    row_to_entry(stmt, &out[n++]);
  sqlite3_finalize(stmt);
  return n;
}

int review_list_pending(review_system *r, review_entry *out, int max) {
  return query_entries(r,
    "SELECT id,surface_id,reviewer_id,status,comment,timestamp"
    " FROM reviews WHERE status=0 ORDER BY timestamp",
    1, (int64_t)REVIEW_PENDING, out, max);
}

int review_list_for_surface(review_system *r, int64_t surface_id,
                             review_entry *out, int max) {
  return query_entries(r,
    "SELECT id,surface_id,reviewer_id,status,comment,timestamp"
    " FROM reviews WHERE surface_id=? ORDER BY timestamp",
    1, surface_id, out, max);
}

review_status review_get_status(review_system *r, int64_t surface_id) {
  if (!r) return (review_status)-1;
  const char *sql =
    "SELECT status FROM reviews WHERE surface_id=?"
    " ORDER BY timestamp DESC LIMIT 1";
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(r->db->sql, sql, -1, &stmt, NULL) != SQLITE_OK)
    return (review_status)-1;
  sqlite3_bind_int64(stmt, 1, surface_id);
  review_status st = (review_status)-1;
  if (sqlite3_step(stmt) == SQLITE_ROW)
    st = (review_status)sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return st;
}

int review_count_by_status(review_system *r, review_status status) {
  if (!r) return 0;
  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(r->db->sql,
      "SELECT COUNT(*) FROM reviews WHERE status=?",
      -1, &stmt, NULL) != SQLITE_OK) return 0;
  sqlite3_bind_int(stmt, 1, (int)status);
  int count = 0;
  if (sqlite3_step(stmt) == SQLITE_ROW)
    count = sqlite3_column_int(stmt, 0);
  sqlite3_finalize(stmt);
  return count;
}
