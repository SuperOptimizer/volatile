#define _POSIX_C_SOURCE 200809L

#include "gui/plugin.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <dirent.h>
#include <limits.h>
#include <sys/stat.h>
#include <sys/inotify.h>
#include <unistd.h>
#include <errno.h>

#include "gui/inotify_util.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define MAX_PLUGINS     64
#define PLUGIN_SYMBOL   "volatile_plugin_interface"

// ---------------------------------------------------------------------------
// Internal slot
// ---------------------------------------------------------------------------

typedef struct {
  void            *handle;     // dlopen handle
  plugin_interface iface;       // copy of the loaded interface
  char             path[512];   // path to the .so file
  char             name_buf[64];// owned copy of iface.name (survives dlclose)
  time_t           mtime;       // last-seen modification time
  bool             active;
} plugin_slot;

struct plugin_manager {
  plugin_slot slots[MAX_PLUGINS];
  int         count;
  char        plugin_dir[512];  // directory being watched (may be empty)
  int         inotify_fd;       // -1 if not enabled
  int         inotify_wd;       // watch descriptor
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static time_t file_mtime(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return st.st_mtime;
}

static bool ends_with_so(const char *name) {
  size_t n = strlen(name);
  return n > 3 && strcmp(name + n - 3, ".so") == 0;
}

// Find slot index by plugin name; returns -1 if not found.
static int find_by_name(const plugin_manager *m, const char *name) {
  for (int i = 0; i < m->count; i++) {
    if (m->slots[i].active && strcmp(m->slots[i].iface.name, name) == 0)
      return i;
  }
  return -1;
}

// Find slot index by path; returns -1 if not found.
static int find_by_path(const plugin_manager *m, const char *path) {
  for (int i = 0; i < m->count; i++) {
    if (m->slots[i].active && strcmp(m->slots[i].path, path) == 0)
      return i;
  }
  return -1;
}

// Allocate an empty slot index; returns -1 if full.
static int alloc_slot(plugin_manager *m) {
  if (m->count < MAX_PLUGINS) return m->count++;
  // look for an inactive slot
  for (int i = 0; i < MAX_PLUGINS; i++) {
    if (!m->slots[i].active) return i;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Low-level load into an existing slot (path must already be set).
// Calls plugin->init(NULL) on success.
// ---------------------------------------------------------------------------

static bool slot_load(plugin_slot *s, void *app_ctx) {
  // RTLD_LOCAL: don't pollute the global symbol table.
  // RTLD_NOW:   resolve all symbols immediately so we get errors early.
  s->handle = dlopen(s->path, RTLD_NOW | RTLD_LOCAL);
  if (!s->handle) {
    LOG_ERROR("plugin_mgr: dlopen(%s) failed: %s", s->path, dlerror());
    return false;
  }

  plugin_interface *iface = dlsym(s->handle, PLUGIN_SYMBOL);
  if (!iface) {
    LOG_ERROR("plugin_mgr: %s has no symbol '%s': %s",
              s->path, PLUGIN_SYMBOL, dlerror());
    dlclose(s->handle);
    s->handle = NULL;
    return false;
  }

  s->iface  = *iface;
  // Copy name into owned buffer — iface.name points into .so rodata which
  // becomes invalid after dlclose. name_buf survives the unload for logging.
  snprintf(s->name_buf, sizeof(s->name_buf), "%s",
           iface->name ? iface->name : "(unnamed)");
  s->iface.name = s->name_buf;
  s->mtime  = file_mtime(s->path);
  s->active = true;

  if (s->iface.init) {
    int rc = s->iface.init(app_ctx);
    if (rc != 0) {
      LOG_WARN("plugin_mgr: %s init() returned %d", s->name_buf, rc);
    }
  }

  LOG_INFO("plugin_mgr: loaded '%s' v%s from %s",
           s->name_buf, iface->version, s->path);
  return true;
}

// Low-level unload: calls shutdown, dlclose, marks slot inactive.
static void slot_unload(plugin_slot *s) {
  if (!s->active) return;
  if (s->iface.shutdown) s->iface.shutdown();
  // Log before dlclose — name_buf is in our own memory, safe after unmap.
  LOG_INFO("plugin_mgr: unloaded '%s'", s->name_buf);
  dlclose(s->handle);
  s->handle = NULL;
  s->active = false;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

plugin_manager *plugin_mgr_new(const char *plugin_dir) {
  plugin_manager *m = calloc(1, sizeof(*m));
  REQUIRE(m, "plugin_mgr_new: calloc failed");
  m->inotify_fd = -1;
  m->inotify_wd = -1;

  if (plugin_dir) {
    snprintf(m->plugin_dir, sizeof(m->plugin_dir), "%s", plugin_dir);
    // Scan directory for .so files
    DIR *d = opendir(plugin_dir);
    if (!d) {
      LOG_WARN("plugin_mgr: cannot open plugin dir '%s': %s",
               plugin_dir, strerror(errno));
    } else {
      struct dirent *ent;
      char path[512];
      while ((ent = readdir(d)) != NULL) {
        if (!ends_with_so(ent->d_name)) continue;
        snprintf(path, sizeof(path), "%s/%s", plugin_dir, ent->d_name);
        plugin_mgr_load(m, path);
      }
      closedir(d);
    }
  }

  return m;
}

void plugin_mgr_free(plugin_manager *m) {
  if (!m) return;
  for (int i = 0; i < m->count; i++)
    slot_unload(&m->slots[i]);
  if (m->inotify_fd >= 0) {
    if (m->inotify_wd >= 0) inotify_rm_watch(m->inotify_fd, m->inotify_wd);
    close(m->inotify_fd);
  }
  free(m);
}

// ---------------------------------------------------------------------------
// Load / Unload / Reload
// ---------------------------------------------------------------------------

bool plugin_mgr_load(plugin_manager *m, const char *path) {
  REQUIRE(m && path, "plugin_mgr_load: null argument");

  if (find_by_path(m, path) >= 0) {
    LOG_WARN("plugin_mgr: '%s' already loaded", path);
    return true;
  }

  int idx = alloc_slot(m);
  if (idx < 0) {
    LOG_ERROR("plugin_mgr: MAX_PLUGINS (%d) reached", MAX_PLUGINS);
    return false;
  }

  plugin_slot *s = &m->slots[idx];
  memset(s, 0, sizeof(*s));
  snprintf(s->path, sizeof(s->path), "%s", path);

  return slot_load(s, NULL);
}

bool plugin_mgr_unload(plugin_manager *m, const char *name) {
  REQUIRE(m && name, "plugin_mgr_unload: null argument");
  int idx = find_by_name(m, name);
  if (idx < 0) {
    LOG_WARN("plugin_mgr: '%s' not found for unload", name);
    return false;
  }
  slot_unload(&m->slots[idx]);
  return true;
}

bool plugin_mgr_reload(plugin_manager *m, const char *name) {
  REQUIRE(m && name, "plugin_mgr_reload: null argument");
  int idx = find_by_name(m, name);
  if (idx < 0) {
    LOG_WARN("plugin_mgr: '%s' not found for reload", name);
    return false;
  }
  plugin_slot *s = &m->slots[idx];
  char path[512];
  snprintf(path, sizeof(path), "%s", s->path);

  slot_unload(s);
  snprintf(s->path, sizeof(s->path), "%s", path);
  return slot_load(s, NULL);
}

// ---------------------------------------------------------------------------
// Hot reload via inotify
// ---------------------------------------------------------------------------

void plugin_mgr_enable_hot_reload(plugin_manager *m) {
  REQUIRE(m, "plugin_mgr_enable_hot_reload: null manager");
  if (m->plugin_dir[0] == '\0') {
    LOG_WARN("plugin_mgr: hot reload requested but no plugin_dir set");
    return;
  }
  if (m->inotify_fd >= 0) return;  // already enabled

  m->inotify_fd = inotify_init1(IN_NONBLOCK);
  if (m->inotify_fd < 0) {
    LOG_ERROR("plugin_mgr: inotify_init1 failed: %s", strerror(errno));
    return;
  }

  m->inotify_wd = inotify_add_watch(m->inotify_fd, m->plugin_dir,
                                    IN_CLOSE_WRITE | IN_MOVED_TO);
  if (m->inotify_wd < 0) {
    LOG_ERROR("plugin_mgr: inotify_add_watch('%s') failed: %s",
              m->plugin_dir, strerror(errno));
    close(m->inotify_fd);
    m->inotify_fd = -1;
    return;
  }

  LOG_INFO("plugin_mgr: hot reload enabled for '%s'", m->plugin_dir);
}

void plugin_mgr_check_reload(plugin_manager *m) {
  REQUIRE(m, "plugin_mgr_check_reload: null manager");

  // 1. Drain inotify events — collect changed filenames.
  if (m->inotify_fd >= 0) {
    char buf[INOTIFY_BUF * 8] __attribute__((aligned(alignof(struct inotify_event))));
    ssize_t n = read(m->inotify_fd, buf, sizeof(buf));
    if (n > 0) {
      for (ssize_t off = 0; off < n; ) {
        struct inotify_event *ev = (struct inotify_event *)(buf + off);
        if (ev->len > 0 && ends_with_so(ev->name)) {
          char path[512];
          snprintf(path, sizeof(path), "%s/%s", m->plugin_dir, ev->name);
          int idx = find_by_path(m, path);
          if (idx >= 0) {
            LOG_INFO("plugin_mgr: inotify: reloading '%s'", m->slots[idx].iface.name);
            plugin_mgr_reload(m, m->slots[idx].iface.name);
          } else {
            // New .so appeared — load it
            plugin_mgr_load(m, path);
          }
        }
        off += (ssize_t)(sizeof(struct inotify_event) + ev->len);
      }
    }
  }

  // 2. Fallback: stat-based mtime check for each loaded plugin.
  for (int i = 0; i < m->count; i++) {
    plugin_slot *s = &m->slots[i];
    if (!s->active) continue;
    time_t current = file_mtime(s->path);
    if (current > s->mtime) {
      LOG_INFO("plugin_mgr: mtime changed, reloading '%s'", s->iface.name);
      char name[256];
      snprintf(name, sizeof(name), "%s", s->iface.name);
      plugin_mgr_reload(m, name);
    }
  }
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

void plugin_mgr_render_all(plugin_manager *m, struct nk_context *ctx) {
  REQUIRE(m, "plugin_mgr_render_all: null manager");
  for (int i = 0; i < m->count; i++) {
    plugin_slot *s = &m->slots[i];
    if (s->active && s->iface.render)
      s->iface.render(ctx);
  }
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

int plugin_mgr_count(const plugin_manager *m) {
  REQUIRE(m, "plugin_mgr_count: null manager");
  int n = 0;
  for (int i = 0; i < m->count; i++)
    if (m->slots[i].active) n++;
  return n;
}

const plugin_interface *plugin_mgr_get(const plugin_manager *m, int index) {
  REQUIRE(m, "plugin_mgr_get: null manager");
  int seen = 0;
  for (int i = 0; i < m->count; i++) {
    if (!m->slots[i].active) continue;
    if (seen == index) return &m->slots[i].iface;
    seen++;
  }
  return NULL;
}
