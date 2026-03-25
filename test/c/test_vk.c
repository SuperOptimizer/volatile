#include "greatest.h"
#include "gpu/vk.h"

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// NOTE: These tests skip gracefully when no Vulkan-capable device is present.

TEST test_init_non_null(void) {
  vk_config cfg = { .validation = false, .headless = true };
  vk_context *ctx = vk_init(cfg);

  if (!ctx) {
    SKIPm("no Vulkan device available");
  }

  vk_shutdown(ctx);
  PASS();
}

TEST test_device_name_non_empty(void) {
  vk_config cfg = { .validation = false, .headless = true };
  vk_context *ctx = vk_init(cfg);

  if (!ctx) {
    SKIPm("no Vulkan device available");
  }

  const char *name = vk_device_name(ctx);
  ASSERT(name != NULL);
  ASSERT(name[0] != '\0');

  vk_shutdown(ctx);
  PASS();
}

TEST test_has_bda_is_bool(void) {
  vk_config cfg = { .validation = false, .headless = true };
  vk_context *ctx = vk_init(cfg);

  if (!ctx) {
    SKIPm("no Vulkan device available");
  }

  // Just verifies the call doesn't crash; BDA may or may not be supported.
  bool bda = vk_has_buffer_device_address(ctx);
  (void)bda;

  vk_shutdown(ctx);
  PASS();
}

TEST test_shutdown_null_safe(void) {
  vk_shutdown(NULL);  // must not crash
  PASS();
}

TEST test_validation_layer(void) {
  vk_config cfg = { .validation = true, .headless = true };
  vk_context *ctx = vk_init(cfg);

  if (!ctx) {
    SKIPm("no Vulkan device available");
  }

  // If we got a context with validation requested, device name is still valid.
  const char *name = vk_device_name(ctx);
  ASSERT(name != NULL && name[0] != '\0');

  vk_shutdown(ctx);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(vk_suite) {
  RUN_TEST(test_init_non_null);
  RUN_TEST(test_device_name_non_empty);
  RUN_TEST(test_has_bda_is_bool);
  RUN_TEST(test_shutdown_null_safe);
  RUN_TEST(test_validation_layer);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vk_suite);
  GREATEST_MAIN_END();
}
