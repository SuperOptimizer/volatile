#pragma once
#include <stdbool.h>

// Forward declarations
typedef struct seg_db seg_db;
typedef struct review_system review_system;

// ---------------------------------------------------------------------------
// project — lightweight container for a loaded project's databases.
// ---------------------------------------------------------------------------

typedef struct {
  seg_db        *db;      // segment + annotation DB
  review_system *reviews; // review workflow system (may be NULL)
  const char    *name;    // project display name (not owned)
  const char    *thumb_dir; // directory containing thumbnail images (may be NULL)
} project;

// ---------------------------------------------------------------------------
// review_report_generate
//
// Write a summary report of all segments in `proj` to `output_path`.
// Each segment gets: ID, name, area, approval status, author, last modified.
// When html=true the output is an HTML file; otherwise plain text.
// Thumbnail images are included in HTML output when thumb_dir is set.
//
// Returns true on success.
// ---------------------------------------------------------------------------

bool review_report_generate(const project *proj, const char *output_path, bool html);
