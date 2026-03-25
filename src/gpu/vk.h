#pragma once
#include <stdbool.h>

// Opaque Vulkan context. Holds instance, physical device, logical device,
// and compute queue. Created by vk_init, destroyed by vk_shutdown.
typedef struct vk_context vk_context;

typedef struct {
  bool validation;      // enable VK_LAYER_KHRONOS_validation (debug builds)
  bool headless;        // no surface/swapchain needed (compute-only)
} vk_config;

// Initialise Vulkan. Returns NULL if no suitable device found or Vulkan
// is unavailable. Caller owns the returned context.
vk_context *vk_init(vk_config cfg);

// Destroy context and release all Vulkan resources.
void vk_shutdown(vk_context *ctx);

// Human-readable device name (e.g. "NVIDIA GeForce RTX 4090").
// Pointer valid until vk_shutdown.
const char *vk_device_name(const vk_context *ctx);

// True if VK_KHR_buffer_device_address was successfully enabled.
bool vk_has_buffer_device_address(const vk_context *ctx);

// ---------------------------------------------------------------------------
// Internal accessors used by the gpu abstraction layer (gpu.c)
// ---------------------------------------------------------------------------
#include <vulkan/vulkan.h>

VkDevice         vk_device(const vk_context *ctx);
VkPhysicalDevice vk_physical_device(const vk_context *ctx);
VkQueue          vk_compute_queue(const vk_context *ctx);
uint32_t         vk_compute_queue_family(const vk_context *ctx);
