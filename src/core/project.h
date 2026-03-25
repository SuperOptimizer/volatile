#pragma once
#include <stdbool.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Project — replaces the rigid volpkg structure.
// A project is a JSON file describing a collection of data sources (volumes,
// segments, annotation grids, remote Zarr stores, etc.).
// ---------------------------------------------------------------------------

typedef enum {
  DATA_SEGMENTS,
  DATA_ZARR_VOLUME,
  DATA_NORMALGRIDS,
  DATA_DIRECTION_VOLUME,
  DATA_TIFF_STACK,
  DATA_REMOTE_ZARR,
  DATA_OTHER,
} data_type;

typedef struct {
  char     *name;           // human-readable name (owned)
  char     *path;           // local path or remote URL (owned)
  data_type type;
  bool      is_remote;
  bool      track_changes;  // re-scan on inotify events
  bool      recursive;      // include subdirectory contents
  char    **tags;           // flexible tagging (owned array of owned strings)
  int       num_tags;
} project_entry;

typedef struct {
  char          *name;
  char          *description;
  char          *output_dir;
  char          *sync_dir;
  project_entry *entries;
  int            num_entries;
  int            capacity;
  char          *path;       // where this project.json lives
  bool           modified;
} project;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

project *project_new(const char *name);
void     project_free(project *p);

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

bool     project_save(const project *p, const char *path);
project *project_load(const char *path);

// ---------------------------------------------------------------------------
// volpkg import (backwards compatible)
// Maps volumes/ -> DATA_ZARR_VOLUME, paths/ -> DATA_SEGMENTS
// ---------------------------------------------------------------------------

project *project_from_volpkg(const char *volpkg_path);

// ---------------------------------------------------------------------------
// Data management
// ---------------------------------------------------------------------------

// Returns the new entry index, or -1 on failure.
int  project_add_entry(project *p, const project_entry *entry);
int  project_add_local(project *p, const char *path, data_type type,
                       bool recursive);
int  project_add_remote(project *p, const char *url, data_type type);
bool project_remove_entry(project *p, int index);

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int                   project_count(const project *p);
int                   project_count_type(const project *p, data_type type);
const project_entry  *project_get(const project *p, int index);

// Fill results[] with matching indices; returns count written.
int project_find_by_tag(const project *p, const char *tag,
                        int *results, int max);
int project_find_by_type(const project *p, data_type type,
                         int *results, int max);

// ---------------------------------------------------------------------------
// Auto-discovery — scan a directory, add all recognized data sources.
// Returns number of entries added.
// ---------------------------------------------------------------------------

int project_scan_dir(project *p, const char *dir, bool recursive);

// ---------------------------------------------------------------------------
// Project composition
// ---------------------------------------------------------------------------

int project_import_from(project *p, const project *other);

// ---------------------------------------------------------------------------
// Tagging
// ---------------------------------------------------------------------------

void project_tag_entry(project *p, int index, const char *tag);
void project_untag_entry(project *p, int index, const char *tag);
