// Test plugin — compiled as a shared library (.so) for plugin system tests.
// Exported symbol: volatile_plugin_interface

#include "gui/plugin.h"

#include <stdio.h>

// Counters checked by the test harness via the interface's init/shutdown.
static int g_init_count     = 0;
static int g_shutdown_count = 0;
static int g_render_count   = 0;
static int g_event_count    = 0;

static int plugin_init(void *app_ctx) {
  (void)app_ctx;
  g_init_count++;
  return 0;
}

static void plugin_shutdown(void) {
  g_shutdown_count++;
}

static void plugin_render(struct nk_context *ctx) {
  (void)ctx;
  g_render_count++;
}

static void plugin_on_event(int event_type, void *event_data) {
  (void)event_type; (void)event_data;
  g_event_count++;
}

// These counters are accessible via dlsym by the test to verify call counts.
int *test_plugin_init_count(void)     { return &g_init_count; }
int *test_plugin_shutdown_count(void) { return &g_shutdown_count; }
int *test_plugin_render_count(void)   { return &g_render_count; }
int *test_plugin_event_count(void)    { return &g_event_count; }

// The symbol the plugin manager looks up.
plugin_interface volatile_plugin_interface = {
  .name        = "test_plugin",
  .version     = "1.0.0",
  .description = "Test plugin for plugin manager unit tests",
  .init        = plugin_init,
  .shutdown    = plugin_shutdown,
  .render      = plugin_render,
  .on_event    = plugin_on_event,
};
