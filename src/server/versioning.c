#define _POSIX_C_SOURCE 200809L

#include "server/versioning.h"
#include "core/geom.h"
#include "core/log.h"

#include <sqlite3.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

static const char *k_schema =
  "CREATE TABLE IF NOT EXISTS surface_versions ("
  "  version_id  INTEGER PRIMARY KEY AUTOINCREMENT,"
  "  surface_id  INTEGER NOT NULL,"
  "  user_id     INTEGER NOT NULL,"
  "  timestamp   INTEGER NOT NULL,"
  "  message     TEXT    NOT NULL DEFAULT '',"
  "  rows        INTEGER NOT NULL,"
  "  cols        INTEGER NOT NULL,"
  "  points_blob BLOB    NOT NULL"
  ");"
  "CREATE INDEX IF NOT EXISTS idx_sv_surface ON surface_versions(surface_id, timestamp DESC);";

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

struct surface_history {
  sqlite3  *db;
  int64_t   surface_id;
  int       autosave_interval;   // 0 = disabled
  time_t    last_autosave;
  int64_t   last_commit_id;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool exec_sql(sqlite3 *db, const char *sql) {
  char *err = NULL;
  int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (rc != SQLITE_OK) {
    LOG_ERROR("versioning: SQL error: %s", err ? err : "unknown");
    sqlite3_free(err);
    return false;
  }
  return true;
}

// Serialise quad_surface points to a flat float blob (rows*cols*3 floats).
static uint8_t *serialise_points(const quad_surface *s, int *out_bytes) {
  int n = s->rows * s->cols;
  *out_bytes = n * 3 * (int)sizeof(float);
  float *buf = malloc((size_t)*out_bytes);
  if (!buf) return NULL;
  for (int i = 0; i < n; i++) {
    buf[i * 3 + 0] = s->points[i].x;
    buf[i * 3 + 1] = s->points[i].y;
    buf[i * 3 + 2] = s->points[i].z;
  }
  return (uint8_t *)buf;
}

// Deserialise blob back into a fresh quad_surface.
static quad_surface *deserialise_points(int rows, int cols,
                                        const void *blob, int blob_bytes) {
  int n = rows * cols;
  if (blob_bytes < n * 3 * (int)sizeof(float)) {
    LOG_ERROR("versioning: blob too small (%d < %d)", blob_bytes,
              n * 3 * (int)sizeof(float));
    return NULL;
  }
  quad_surface *s = quad_surface_new(rows, cols);
  if (!s) return NULL;
  const float *src = blob;
  for (int i = 0; i < n; i++) {
    s->points[i] = (vec3f){ src[i*3], src[i*3+1], src[i*3+2] };
  }
  return s;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

surface_history *surface_history_new(int64_t surface_id, const char *db_path) {
  REQUIRE(db_path, "surface_history_new: null db_path");

  sqlite3 *db = NULL;
  int rc = sqlite3_open(db_path, &db);
  if (rc != SQLITE_OK) {
    LOG_ERROR("surface_history_new: cannot open db '%s': %s",
              db_path, sqlite3_errmsg(db));
    sqlite3_close(db);
    return NULL;
  }

  if (!exec_sql(db, k_schema)) {
    sqlite3_close(db);
    return NULL;
  }

  surface_history *h = calloc(1, sizeof(*h));
  REQUIRE(h, "surface_history_new: calloc failed");
  h->db         = db;
  h->surface_id = surface_id;
  h->last_autosave = time(NULL);
  return h;
}

void surface_history_free(surface_history *h) {
  if (!h) return;
  sqlite3_close(h->db);
  free(h);
}

// ---------------------------------------------------------------------------
// Commit
// ---------------------------------------------------------------------------

int64_t surface_history_commit(surface_history *h, int user_id,
                               const char *message,
                               const quad_surface *surface) {
  REQUIRE(h && surface && surface->points, "surface_history_commit: null arg");

  int blob_bytes;
  uint8_t *blob = serialise_points(surface, &blob_bytes);
  if (!blob) {
    LOG_ERROR("surface_history_commit: serialise failed");
    return -1;
  }

  static const char *sql =
    "INSERT INTO surface_versions"
    " (surface_id, user_id, timestamp, message, rows, cols, points_blob)"
    " VALUES (?, ?, ?, ?, ?, ?, ?);";

  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(h->db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    LOG_ERROR("surface_history_commit: prepare: %s", sqlite3_errmsg(h->db));
    free(blob);
    return -1;
  }

  sqlite3_bind_int64(stmt, 1, h->surface_id);
  sqlite3_bind_int  (stmt, 2, user_id);
  sqlite3_bind_int64(stmt, 3, (int64_t)time(NULL));
  sqlite3_bind_text (stmt, 4, message ? message : "", -1, SQLITE_STATIC);
  sqlite3_bind_int  (stmt, 5, surface->rows);
  sqlite3_bind_int  (stmt, 6, surface->cols);
  sqlite3_bind_blob (stmt, 7, blob, blob_bytes, SQLITE_STATIC);

  rc = sqlite3_step(stmt);
  int64_t version_id = -1;
  if (rc == SQLITE_DONE) {
    version_id = sqlite3_last_insert_rowid(h->db);
    h->last_commit_id = version_id;
    LOG_DEBUG("surface_history_commit: version %lld for surface %lld",
              (long long)version_id, (long long)h->surface_id);
  } else {
    LOG_ERROR("surface_history_commit: step: %s", sqlite3_errmsg(h->db));
  }

  sqlite3_finalize(stmt);
  free(blob);
  return version_id;
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

int surface_history_list(surface_history *h, version_info *out,
                         int max_versions) {
  REQUIRE(h && out && max_versions > 0, "surface_history_list: bad args");

  static const char *sql =
    "SELECT version_id, user_id, timestamp, message"
    " FROM surface_versions"
    " WHERE surface_id = ?"
    " ORDER BY timestamp DESC, version_id DESC"
    " LIMIT ?;";

  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(h->db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    LOG_ERROR("surface_history_list: prepare: %s", sqlite3_errmsg(h->db));
    return -1;
  }

  sqlite3_bind_int64(stmt, 1, h->surface_id);
  sqlite3_bind_int  (stmt, 2, max_versions);

  int count = 0;
  while (sqlite3_step(stmt) == SQLITE_ROW && count < max_versions) {
    version_info *v = &out[count++];
    v->version_id = sqlite3_column_int64(stmt, 0);
    v->user_id    = sqlite3_column_int  (stmt, 1);
    v->timestamp  = sqlite3_column_int64(stmt, 2);
    const char *msg = (const char *)sqlite3_column_text(stmt, 3);
    snprintf(v->message, sizeof(v->message), "%s", msg ? msg : "");
  }

  sqlite3_finalize(stmt);
  return count;
}

// ---------------------------------------------------------------------------
// Checkout
// ---------------------------------------------------------------------------

quad_surface *surface_history_checkout(surface_history *h, int64_t version_id) {
  REQUIRE(h, "surface_history_checkout: null history");

  static const char *sql =
    "SELECT rows, cols, points_blob FROM surface_versions"
    " WHERE version_id = ? AND surface_id = ?;";

  sqlite3_stmt *stmt = NULL;
  int rc = sqlite3_prepare_v2(h->db, sql, -1, &stmt, NULL);
  if (rc != SQLITE_OK) {
    LOG_ERROR("surface_history_checkout: prepare: %s", sqlite3_errmsg(h->db));
    return NULL;
  }

  sqlite3_bind_int64(stmt, 1, version_id);
  sqlite3_bind_int64(stmt, 2, h->surface_id);

  quad_surface *result = NULL;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    int rows       = sqlite3_column_int  (stmt, 0);
    int cols       = sqlite3_column_int  (stmt, 1);
    const void *blob = sqlite3_column_blob(stmt, 2);
    int blob_bytes   = sqlite3_column_bytes(stmt, 2);
    result = deserialise_points(rows, cols, blob, blob_bytes);
  } else {
    LOG_WARN("surface_history_checkout: version %lld not found",
             (long long)version_id);
  }

  sqlite3_finalize(stmt);
  return result;
}

// ---------------------------------------------------------------------------
// Diff
// ---------------------------------------------------------------------------

// Fetch just the points blob for a version.
static float *fetch_points(surface_history *h, int64_t version_id,
                            int *out_n) {
  static const char *sql =
    "SELECT rows, cols, points_blob FROM surface_versions"
    " WHERE version_id = ? AND surface_id = ?;";

  sqlite3_stmt *stmt = NULL;
  if (sqlite3_prepare_v2(h->db, sql, -1, &stmt, NULL) != SQLITE_OK)
    return NULL;

  sqlite3_bind_int64(stmt, 1, version_id);
  sqlite3_bind_int64(stmt, 2, h->surface_id);

  float *pts = NULL;
  if (sqlite3_step(stmt) == SQLITE_ROW) {
    int rows = sqlite3_column_int(stmt, 0);
    int cols = sqlite3_column_int(stmt, 1);
    *out_n   = rows * cols;
    int blob_bytes = sqlite3_column_bytes(stmt, 2);
    const float *src = sqlite3_column_blob(stmt, 2);
    if (blob_bytes >= *out_n * 3 * (int)sizeof(float)) {
      pts = malloc((size_t)(*out_n) * 3 * sizeof(float));
      if (pts) memcpy(pts, src, (size_t)(*out_n) * 3 * sizeof(float));
    }
  }
  sqlite3_finalize(stmt);
  return pts;
}

float *surface_history_diff(surface_history *h, int64_t v1, int64_t v2,
                            int *out_count) {
  REQUIRE(h && out_count, "surface_history_diff: null arg");
  *out_count = 0;

  int n1, n2;
  float *pts1 = fetch_points(h, v1, &n1);
  float *pts2 = fetch_points(h, v2, &n2);

  if (!pts1 || !pts2 || n1 != n2) {
    LOG_WARN("surface_history_diff: version mismatch or fetch failed");
    free(pts1); free(pts2);
    return NULL;
  }

  int n = n1;
  float *diff = malloc((size_t)n * 3 * sizeof(float));
  if (!diff) { free(pts1); free(pts2); return NULL; }

  for (int i = 0; i < n * 3; i++)
    diff[i] = pts2[i] - pts1[i];

  free(pts1);
  free(pts2);
  *out_count = n;
  return diff;
}

// ---------------------------------------------------------------------------
// Autosave
// ---------------------------------------------------------------------------

void surface_history_enable_autosave(surface_history *h, int interval_seconds) {
  REQUIRE(h, "surface_history_enable_autosave: null history");
  h->autosave_interval = interval_seconds > 0 ? interval_seconds : 60;
  h->last_autosave     = time(NULL);
  LOG_INFO("surface_history: autosave enabled, interval=%ds", h->autosave_interval);
}

void surface_history_autosave_tick(surface_history *h,
                                   const quad_surface *surface) {
  if (!h || !surface || h->autosave_interval <= 0) return;
  time_t now = time(NULL);
  if ((now - h->last_autosave) < h->autosave_interval) return;

  int64_t vid = surface_history_commit(h, 0, "[autosave]", surface);
  if (vid >= 0) {
    h->last_autosave = now;
    LOG_DEBUG("surface_history: autosaved as version %lld", (long long)vid);
  }
}
