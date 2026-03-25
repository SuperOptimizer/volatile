#pragma once
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// vol_entry — metadata for a single discovered volume
// ---------------------------------------------------------------------------

typedef struct {
  char    *name;       // display name (basename of path)
  char    *path;       // local path or URL
  bool     is_remote;
  int      num_levels;
  int64_t  shape[3];   // level-0 shape (z, y, x); 0 if unknown
  char    *dtype_name;
} vol_entry;

void vol_entry_free(vol_entry *e);

// ---------------------------------------------------------------------------
// vol_browser — growable catalog of volumes
// ---------------------------------------------------------------------------

typedef struct vol_browser vol_browser;

vol_browser *vol_browser_new(void);
void         vol_browser_free(vol_browser *b);

// scan local directory for .zarr volumes (directories containing .zarray)
int  vol_browser_scan_local(vol_browser *b, const char *dir);

// add a remote catalog URL; queries server for available volumes (stub)
int  vol_browser_add_remote(vol_browser *b, const char *server_url);

// add a single volume by path or URL
bool vol_browser_add(vol_browser *b, const char *path_or_url);

// query
int              vol_browser_count(const vol_browser *b);
const vol_entry *vol_browser_get(const vol_browser *b, int index);

// substring search on name/path; fills results[] with matching indices
// returns number of matches written (capped at max_results)
int  vol_browser_search(const vol_browser *b, const char *query,
                         int *results, int max_results);
