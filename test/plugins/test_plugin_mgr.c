#include "greatest.h"
#include "gui/plugin.h"

#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <utime.h>

// TEST_PLUGIN_PATH and TEST_PLUGIN_DIR are injected by CMake.
#ifndef TEST_PLUGIN_PATH
#  error "TEST_PLUGIN_PATH must be defined by CMake"
#endif
#ifndef TEST_PLUGIN_DIR
#  error "TEST_PLUGIN_DIR must be defined by CMake"
#endif

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Keep a persistent handle open across tests so dlsym into the .so is safe
// even after a plugin_mgr_free (which decrements the refcount but doesn't
// unmap the library while our handle keeps refcount >= 1).
static void  *g_plugin_handle = NULL;

static void open_persistent_handle(void) {
  if (!g_plugin_handle)
    g_plugin_handle = dlopen(TEST_PLUGIN_PATH, RTLD_NOW | RTLD_LOCAL);
}

typedef int *(*counter_fn)(void);

static int get_counter(const char *sym) {
  open_persistent_handle();
  if (!g_plugin_handle) return -1;
  counter_fn fn = (counter_fn)(uintptr_t)dlsym(g_plugin_handle, sym);
  return fn ? *fn() : -1;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_load_and_get(void) {
  plugin_manager *m = plugin_mgr_new(NULL);
  ASSERT(m != NULL);
  ASSERT_EQ(0, plugin_mgr_count(m));

  bool ok = plugin_mgr_load(m, TEST_PLUGIN_PATH);
  ASSERT(ok);
  ASSERT_EQ(1, plugin_mgr_count(m));

  const plugin_interface *iface = plugin_mgr_get(m, 0);
  ASSERT(iface != NULL);
  ASSERT_STR_EQ("test_plugin", iface->name);
  ASSERT_STR_EQ("1.0.0",       iface->version);
  ASSERT(iface->render   != NULL);
  ASSERT(iface->on_event != NULL);

  plugin_mgr_free(m);
  PASS();
}

TEST test_init_called_on_load(void) {
  plugin_manager *m = plugin_mgr_new(NULL);

  plugin_mgr_load(m, TEST_PLUGIN_PATH);
  int inits = get_counter("test_plugin_init_count");
  ASSERT(inits >= 1);  // init must have been called at least once

  plugin_mgr_free(m);
  PASS();
}

TEST test_shutdown_called_on_unload(void) {
  plugin_manager *m = plugin_mgr_new(NULL);

  plugin_mgr_load(m, TEST_PLUGIN_PATH);

  bool ok = plugin_mgr_unload(m, "test_plugin");
  ASSERT(ok);
  ASSERT_EQ(0, plugin_mgr_count(m));

  plugin_mgr_free(m);
  PASS();
}

TEST test_double_load_is_noop(void) {
  plugin_manager *m = plugin_mgr_new(NULL);

  plugin_mgr_load(m, TEST_PLUGIN_PATH);
  plugin_mgr_load(m, TEST_PLUGIN_PATH);  // second load should be a no-op
  ASSERT_EQ(1, plugin_mgr_count(m));

  plugin_mgr_free(m);
  PASS();
}

TEST test_unload_unknown_returns_false(void) {
  plugin_manager *m = plugin_mgr_new(NULL);
  bool ok = plugin_mgr_unload(m, "no_such_plugin");
  ASSERT(!ok);
  plugin_mgr_free(m);
  PASS();
}

TEST test_reload_cycle(void) {
  plugin_manager *m = plugin_mgr_new(NULL);

  plugin_mgr_load(m, TEST_PLUGIN_PATH);
  ASSERT_EQ(1, plugin_mgr_count(m));

  bool ok = plugin_mgr_reload(m, "test_plugin");
  ASSERT(ok);
  ASSERT_EQ(1, plugin_mgr_count(m));

  // Interface must still be valid after reload
  const plugin_interface *iface = plugin_mgr_get(m, 0);
  ASSERT(iface != NULL);
  ASSERT_STR_EQ("test_plugin", iface->name);

  plugin_mgr_free(m);
  PASS();
}

TEST test_scan_dir_on_new(void) {
  // plugin_mgr_new with the plugin dir should auto-load test_plugin.so
  plugin_manager *m = plugin_mgr_new(TEST_PLUGIN_DIR);

  // The test plugin .so is in TEST_PLUGIN_DIR; it should have been loaded.
  ASSERT(plugin_mgr_count(m) >= 1);

  bool found = false;
  for (int i = 0; i < plugin_mgr_count(m); i++) {
    const plugin_interface *iface = plugin_mgr_get(m, i);
    if (iface && strcmp(iface->name, "test_plugin") == 0) { found = true; break; }
  }
  ASSERT(found);

  plugin_mgr_free(m);
  PASS();
}

TEST test_render_all_calls_render(void) {
  plugin_manager *m = plugin_mgr_new(NULL);
  plugin_mgr_load(m, TEST_PLUGIN_PATH);

  // Call render_all with a NULL ctx — our test plugin tolerates it.
  plugin_mgr_render_all(m, NULL);
  plugin_mgr_render_all(m, NULL);

  // render should have been called twice
  int renders = get_counter("test_plugin_render_count");
  ASSERT(renders >= 2);

  plugin_mgr_free(m);
  PASS();
}

TEST test_mtime_hot_reload(void) {
  plugin_manager *m = plugin_mgr_new(NULL);
  plugin_mgr_load(m, TEST_PLUGIN_PATH);

  // Advance the mtime of the .so artificially so check_reload triggers.
  struct stat st;
  stat(TEST_PLUGIN_PATH, &st);
  struct utimbuf times = {
    .actime  = st.st_atime,
    .modtime = st.st_mtime + 2,
  };
  int rc = utime(TEST_PLUGIN_PATH, &times);
  if (rc != 0) {
    // Skip if we can't modify the file (e.g. read-only build dir)
    plugin_mgr_free(m);
    SKIPm("cannot utime plugin .so, skipping mtime test");
  }

  plugin_mgr_check_reload(m);

  // Still one plugin loaded after reload
  ASSERT_EQ(1, plugin_mgr_count(m));
  const plugin_interface *iface = plugin_mgr_get(m, 0);
  ASSERT(iface != NULL);
  ASSERT_STR_EQ("test_plugin", iface->name);

  // Restore original mtime
  times.modtime = st.st_mtime;
  utime(TEST_PLUGIN_PATH, &times);

  plugin_mgr_free(m);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(plugin_suite) {
  RUN_TEST(test_load_and_get);
  RUN_TEST(test_init_called_on_load);
  RUN_TEST(test_shutdown_called_on_unload);
  RUN_TEST(test_double_load_is_noop);
  RUN_TEST(test_unload_unknown_returns_false);
  RUN_TEST(test_reload_cycle);
  RUN_TEST(test_scan_dir_on_new);
  RUN_TEST(test_render_all_calls_render);
  RUN_TEST(test_mtime_hot_reload);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(plugin_suite);
  GREATEST_MAIN_END();
}
