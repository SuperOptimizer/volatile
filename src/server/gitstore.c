#define _POSIX_C_SOURCE 200809L

#include "server/gitstore.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Struct
// ---------------------------------------------------------------------------

struct git_store {
  char root[512];   // absolute path to repo working tree
};

// ---------------------------------------------------------------------------
// Shell helpers
// ---------------------------------------------------------------------------

// Run a command string with the repo as cwd. Returns exit code.
static int git_run(const git_store *g, const char *cmd) {
  char buf[1024];
  snprintf(buf, sizeof(buf), "git -C '%s' %s", g->root, cmd);
  LOG_DEBUG("gitstore: %s", buf);
  int rc = system(buf);
  return rc;
}

// Run a command and capture stdout. Returns heap string or NULL. Caller frees.
static char *git_capture(const git_store *g, const char *cmd) {
  char buf[1024];
  snprintf(buf, sizeof(buf), "git -C '%s' %s 2>/dev/null", g->root, cmd);
  FILE *f = popen(buf, "r");
  if (!f) return NULL;

  size_t cap = 4096, len = 0;
  char *out = malloc(cap);
  if (!out) { pclose(f); return NULL; }

  int c;
  while ((c = fgetc(f)) != EOF) {
    if (len + 1 >= cap) {
      cap *= 2;
      char *tmp = realloc(out, cap);
      if (!tmp) { free(out); pclose(f); return NULL; }
      out = tmp;
    }
    out[len++] = (char)c;
  }
  out[len] = '\0';
  pclose(f);
  return out;
}

// Create parent directories for a file path.
static void mkdirs_for(const char *filepath) {
  char buf[512];
  snprintf(buf, sizeof(buf), "%s", filepath);
  for (char *p = buf + 1; *p; p++) {
    if (*p != '/') continue;
    *p = '\0';
    mkdir(buf, 0755);
    *p = '/';
  }
}

// Build an absolute path inside the repo.
static void repo_path(const git_store *g, const char *rel, char *out, size_t outsz) {
  snprintf(out, outsz, "%s/%s", g->root, rel);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

git_store *git_store_open(const char *repo_path) {
  REQUIRE(repo_path, "git_store_open: null path");

  git_store *g = calloc(1, sizeof(*g));
  REQUIRE(g, "git_store_open: calloc failed");
  snprintf(g->root, sizeof(g->root), "%s", repo_path);

  mkdir(repo_path, 0755);

  // Check if already a git repo
  char check[600];
  snprintf(check, sizeof(check), "git -C '%s' rev-parse --git-dir >/dev/null 2>&1",
           repo_path);
  if (system(check) != 0) {
    if (git_run(g, "init -q -b main") != 0) {
      LOG_ERROR("git_store_open: git init failed in '%s'", repo_path);
      free(g);
      return NULL;
    }
    // Initial empty commit so branches can be created from HEAD
    git_run(g, "commit --allow-empty -m 'init' -q "
               "--author='volatile <volatile@localhost>'");
    LOG_INFO("git_store_open: initialised new repo at '%s'", repo_path);
  }

  return g;
}

git_store *git_store_clone(const char *remote_url, const char *local_path) {
  REQUIRE(remote_url && local_path, "git_store_clone: null arg");

  char cmd[1024];
  snprintf(cmd, sizeof(cmd), "git clone -q '%s' '%s'", remote_url, local_path);
  if (system(cmd) != 0) {
    LOG_ERROR("git_store_clone: failed to clone '%s'", remote_url);
    return NULL;
  }
  return git_store_open(local_path);
}

void git_store_free(git_store *g) {
  free(g);
}

// ---------------------------------------------------------------------------
// File I/O
// ---------------------------------------------------------------------------

bool git_store_write_file(git_store *g, const char *path,
                          const void *data, size_t len) {
  REQUIRE(g && path && data, "git_store_write_file: null arg");

  char abs[600];
  repo_path(g, path, abs, sizeof(abs));
  mkdirs_for(abs);

  FILE *f = fopen(abs, "wb");
  if (!f) {
    LOG_ERROR("git_store_write_file: cannot open '%s': %s", abs, strerror(errno));
    return false;
  }
  size_t written = fwrite(data, 1, len, f);
  fclose(f);
  if (written != len) {
    LOG_ERROR("git_store_write_file: short write on '%s'", abs);
    return false;
  }

  // Stage the file
  char add_cmd[600];
  snprintf(add_cmd, sizeof(add_cmd), "add '%s'", path);
  return git_run(g, add_cmd) == 0;
}

bool git_store_write_surface(git_store *g, const char *name,
                             const quad_surface *s) {
  REQUIRE(g && name && s && s->points, "git_store_write_surface: null arg");

  // Serialise to a minimal JSON array of {x,y,z} objects
  int n = s->rows * s->cols;
  size_t cap = (size_t)n * 64 + 128;
  char *json = malloc(cap);
  if (!json) return false;

  int pos = snprintf(json, cap,
    "{\"rows\":%d,\"cols\":%d,\"points\":[", s->rows, s->cols);
  for (int i = 0; i < n; i++) {
    pos += snprintf(json + pos, cap - (size_t)pos,
      "%s{\"x\":%.6g,\"y\":%.6g,\"z\":%.6g}",
      i ? "," : "",
      (double)s->points[i].x,
      (double)s->points[i].y,
      (double)s->points[i].z);
  }
  pos += snprintf(json + pos, cap - (size_t)pos, "]}");

  char rel[256];
  snprintf(rel, sizeof(rel), "segments/%s/surface.json", name);
  bool ok = git_store_write_file(g, rel, json, (size_t)pos);
  free(json);
  return ok;
}

bool git_store_write_annotation(git_store *g, const char *name,
                                const char *json) {
  REQUIRE(g && name && json, "git_store_write_annotation: null arg");
  char rel[256];
  snprintf(rel, sizeof(rel), "annotations/%s.json", name);
  return git_store_write_file(g, rel, json, strlen(json));
}

uint8_t *git_store_read_file(git_store *g, const char *path, size_t *out_len) {
  REQUIRE(g && path && out_len, "git_store_read_file: null arg");

  char abs[600];
  repo_path(g, path, abs, sizeof(abs));

  FILE *f = fopen(abs, "rb");
  if (!f) {
    LOG_WARN("git_store_read_file: cannot open '%s': %s", abs, strerror(errno));
    *out_len = 0;
    return NULL;
  }

  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) { fclose(f); *out_len = 0; return NULL; }

  uint8_t *buf = malloc((size_t)sz);
  if (!buf) { fclose(f); *out_len = 0; return NULL; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  *out_len = n;
  return buf;
}

// ---------------------------------------------------------------------------
// Commits
// ---------------------------------------------------------------------------

bool git_store_commit(git_store *g, const char *author, const char *message) {
  REQUIRE(g && message, "git_store_commit: null arg");
  const char *safe_author = author ? author : "volatile <volatile@localhost>";

  char cmd[768];
  snprintf(cmd, sizeof(cmd),
    "commit -q -m '%s' --author='%s' --allow-empty",
    message, safe_author);
  return git_run(g, cmd) == 0;
}

// ---------------------------------------------------------------------------
// Branches
// ---------------------------------------------------------------------------

bool git_store_create_branch(git_store *g, const char *branch_name) {
  REQUIRE(g && branch_name, "git_store_create_branch: null arg");
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "checkout -q -b '%s'", branch_name);
  return git_run(g, cmd) == 0;
}

bool git_store_checkout(git_store *g, const char *branch_or_commit) {
  REQUIRE(g && branch_or_commit, "git_store_checkout: null arg");
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "checkout -q '%s'", branch_or_commit);
  return git_run(g, cmd) == 0;
}

bool git_store_merge(git_store *g, const char *branch) {
  REQUIRE(g && branch, "git_store_merge: null arg");
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "merge -q --no-edit '%s'", branch);
  return git_run(g, cmd) == 0;
}

// ---------------------------------------------------------------------------
// History
// ---------------------------------------------------------------------------

int git_store_log(git_store *g, git_log_entry *out, int max_entries) {
  REQUIRE(g && out && max_entries > 0, "git_store_log: bad args");

  // Format: hash|author|timestamp|message
  char cmd[128];
  snprintf(cmd, sizeof(cmd),
    "log --pretty=format:'%%H|%%an|%%ct|%%s' -n %d", max_entries);

  char *raw = git_capture(g, cmd);
  if (!raw) return -1;

  int count = 0;
  char *line = raw;
  while (*line && count < max_entries) {
    char *nl = strchr(line, '\n');
    if (nl) *nl = '\0';

    git_log_entry *e = &out[count];
    memset(e, 0, sizeof(*e));

    // Parse hash|author|timestamp|message
    char *p = line;
    char *sep;

    // hash (40 hex + '\0')
    sep = strchr(p, '|');
    if (!sep) goto next;
    size_t hlen = (size_t)(sep - p);
    if (hlen > 40) hlen = 40;
    memcpy(e->hash, p, hlen);
    e->hash[hlen] = '\0';
    p = sep + 1;

    // author
    sep = strchr(p, '|');
    if (!sep) goto next;
    size_t alen = (size_t)(sep - p);
    if (alen >= sizeof(e->author)) alen = sizeof(e->author) - 1;
    memcpy(e->author, p, alen);
    p = sep + 1;

    // timestamp
    sep = strchr(p, '|');
    if (!sep) goto next;
    e->timestamp = atoll(p);
    p = sep + 1;

    // message (remainder of line)
    snprintf(e->message, sizeof(e->message), "%s", p);
    count++;

  next:
    if (!nl) break;
    line = nl + 1;
  }

  free(raw);
  return count;
}

char *git_store_diff(git_store *g, const char *from, const char *to) {
  REQUIRE(g && from && to, "git_store_diff: null arg");
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "diff '%s' '%s'", from, to);
  return git_capture(g, cmd);
}

// ---------------------------------------------------------------------------
// Remote sync
// ---------------------------------------------------------------------------

bool git_store_push(git_store *g, const char *remote) {
  REQUIRE(g, "git_store_push: null store");
  char cmd[256];
  if (remote)
    snprintf(cmd, sizeof(cmd), "push -q '%s'", remote);
  else
    snprintf(cmd, sizeof(cmd), "push -q");
  return git_run(g, cmd) == 0;
}

bool git_store_pull(git_store *g, const char *remote) {
  REQUIRE(g, "git_store_pull: null store");
  char cmd[256];
  if (remote)
    snprintf(cmd, sizeof(cmd), "pull -q '%s'", remote);
  else
    snprintf(cmd, sizeof(cmd), "pull -q");
  return git_run(g, cmd) == 0;
}

// ---------------------------------------------------------------------------
// LFS
// ---------------------------------------------------------------------------

bool git_store_lfs_track(git_store *g, const char *pattern) {
  REQUIRE(g && pattern, "git_store_lfs_track: null arg");
  char cmd[256];
  snprintf(cmd, sizeof(cmd), "lfs track '%s'", pattern);
  if (git_run(g, cmd) != 0) {
    LOG_WARN("git_store_lfs_track: lfs track failed (git-lfs installed?)");
    return false;
  }
  // Stage the updated .gitattributes
  git_run(g, "add .gitattributes");
  return true;
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

int git_store_modified_count(git_store *g) {
  REQUIRE(g, "git_store_modified_count: null store");
  char *out = git_capture(g, "status --porcelain");
  if (!out) return -1;
  int count = 0;
  for (const char *p = out; *p; p++)
    if (*p == '\n') count++;
  // non-empty last line without newline
  if (*out && out[strlen(out) - 1] != '\n') count++;
  free(out);
  return count;
}

bool git_store_is_clean(git_store *g) {
  return git_store_modified_count(g) == 0;
}
