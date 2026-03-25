#pragma once
#include <stdbool.h>

// Forward declaration — callers need not include nuklear.h unless
// they call nk_* functions directly.
struct nk_context;

// ---------------------------------------------------------------------------
// Plugin interface — exported by each .so as "volatile_plugin_interface"
// ---------------------------------------------------------------------------

typedef struct {
  const char *name;
  const char *version;
  const char *description;
  int  (*init)(void *app_ctx);                          // called on load
  void (*shutdown)(void);                               // called on unload
  void (*render)(struct nk_context *ctx);               // called each frame
  void (*on_event)(int event_type, void *event_data);   // optional, may be NULL
} plugin_interface;

// ---------------------------------------------------------------------------
// Plugin manager
// ---------------------------------------------------------------------------

typedef struct plugin_manager plugin_manager;

// Create manager; scans plugin_dir for *.so files and loads them all.
// plugin_dir may be NULL (no initial scan).
plugin_manager *plugin_mgr_new(const char *plugin_dir);
void            plugin_mgr_free(plugin_manager *m);

// Load a single .so by path. Returns false on failure.
bool plugin_mgr_load(plugin_manager *m, const char *path);

// Unload by plugin name (plugin_interface.name). Returns false if not found.
bool plugin_mgr_unload(plugin_manager *m, const char *name);

// Hot-reload by name: dlclose then dlopen. Returns false on failure.
bool plugin_mgr_reload(plugin_manager *m, const char *name);

// Enable inotify-based watch on the plugin directory.
void plugin_mgr_enable_hot_reload(plugin_manager *m);

// Call each frame: checks inotify events and stat mtimes; reloads changed .so files.
void plugin_mgr_check_reload(plugin_manager *m);

// Render all loaded plugins by calling plugin->render(ctx).
void plugin_mgr_render_all(plugin_manager *m, struct nk_context *ctx);

// Query
int                    plugin_mgr_count(const plugin_manager *m);
const plugin_interface *plugin_mgr_get(const plugin_manager *m, int index);
