#pragma once
#include "server/db.h"
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Review workflow for segmentation peer review.
// Backed by a `reviews` table added to the existing seg_db SQLite database.
// ---------------------------------------------------------------------------

typedef enum {
  REVIEW_PENDING        = 0,
  REVIEW_APPROVED       = 1,
  REVIEW_REJECTED       = 2,
  REVIEW_NEEDS_CHANGES  = 3,
} review_status;

typedef struct {
  int64_t       review_id;
  int64_t       surface_id;
  int           reviewer_id;
  review_status status;
  char          comment[512];
  int64_t       timestamp;
} review_entry;

typedef struct review_system review_system;

// Lifecycle — review_new runs CREATE TABLE IF NOT EXISTS on db.
review_system *review_new(seg_db *db);
void           review_free(review_system *r);

// Submit a surface for review; returns new review_id or -1 on error.
int64_t review_submit(review_system *r, int64_t surface_id, int submitter_id);

// Review actions — reviewer_id must differ from submitter where enforced by app.
bool review_approve(review_system *r, int64_t review_id, int reviewer_id,
                    const char *comment);
bool review_reject(review_system *r, int64_t review_id, int reviewer_id,
                   const char *comment);
bool review_request_changes(review_system *r, int64_t review_id, int reviewer_id,
                             const char *comment);

// Query — fills out[0..return-1]; returns count written.
int review_list_pending(review_system *r, review_entry *out, int max);
int review_list_for_surface(review_system *r, int64_t surface_id,
                             review_entry *out, int max);

// Most-recent status for a surface (-1 cast to review_status if not found).
review_status review_get_status(review_system *r, int64_t surface_id);

// Count reviews matching status across all surfaces.
int review_count_by_status(review_system *r, review_status status);
