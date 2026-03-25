#pragma once
#include "core/geom.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// git_store — segmentation version control backed by a real git repository.
// All operations shell out to the `git` binary via popen/system.
//
// Repo layout:
//   segments/<id>/surface.json      quad_surface points (JSON)
//   segments/<id>/approval_mask.bin approval mask bytes
//   segments/<id>/metadata.json     name, author, timestamps
//   annotations/<id>.json           annotation data
//   volumes/<vol_id>.json           volume metadata
//   .gitattributes                  LFS tracking rules
//
// Branch convention for review workflow:
//   main                            approved segmentations
//   user/<name>/segment-<id>        per-user work branches
// ---------------------------------------------------------------------------

typedef struct git_store git_store;

// Open or init a git repo at repo_path (runs `git init` if not a repo).
git_store *git_store_open(const char *repo_path);

// Clone a remote repo to local_path.
git_store *git_store_clone(const char *remote_url, const char *local_path);

void git_store_free(git_store *g);

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

// Write data to path (relative to repo root) and stage it with `git add`.
bool git_store_write_file(git_store *g, const char *path,
                          const void *data, size_t len);

// Serialise a quad_surface to segments/<name>/surface.json and stage it.
bool git_store_write_surface(git_store *g, const char *name,
                             const quad_surface *s);

// Write annotation JSON to annotations/<name>.json and stage it.
bool git_store_write_annotation(git_store *g, const char *name,
                                const char *json);

// Read a file from the working tree. Caller must free returned buffer.
uint8_t *git_store_read_file(git_store *g, const char *path, size_t *out_len);

// ---------------------------------------------------------------------------
// Commits
// ---------------------------------------------------------------------------

bool git_store_commit(git_store *g, const char *author, const char *message);

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

bool git_store_create_branch(git_store *g, const char *branch_name);
bool git_store_checkout(git_store *g, const char *branch_or_commit);
bool git_store_merge(git_store *g, const char *branch);

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

typedef struct {
  char    hash[41];
  char    author[64];
  char    message[256];
  int64_t timestamp;
} git_log_entry;

// Fill out[] newest-first. Returns count written, or -1 on error.
int git_store_log(git_store *g, git_log_entry *out, int max_entries);

// Return unified diff text between two refs. Caller must free.
char *git_store_diff(git_store *g, const char *from, const char *to);

// ---------------------------------------------------------------------------
// Remote sync
// ---------------------------------------------------------------------------

bool git_store_push(git_store *g, const char *remote);
bool git_store_pull(git_store *g, const char *remote);

// ---------------------------------------------------------------------------
// LFS
// ---------------------------------------------------------------------------

// Register a glob pattern in .gitattributes for Git LFS tracking.
bool git_store_lfs_track(git_store *g, const char *pattern);

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

int  git_store_modified_count(git_store *g);
bool git_store_is_clean(git_store *g);
