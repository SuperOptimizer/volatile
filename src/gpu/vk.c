#define _POSIX_C_SOURCE 200809L

#include "gpu/vk.h"
#include "core/log.h"

#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct vk_context {
  VkInstance               instance;
  VkPhysicalDevice         physical_device;
  VkDevice                 device;
  VkQueue                  compute_queue;
  uint32_t                 compute_queue_family;
  VkPhysicalDeviceProperties props;
  bool                     has_bda;   // buffer_device_address
  bool                     validation_enabled;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool layer_available(const char *name) {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, NULL);
  VkLayerProperties *layers = malloc(sizeof(*layers) * count);
  if (!layers) return false;
  vkEnumerateInstanceLayerProperties(&count, layers);
  bool found = false;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(layers[i].layerName, name) == 0) { found = true; break; }
  }
  free(layers);
  return found;
}

static bool device_ext_available(VkPhysicalDevice pd, const char *name) {
  uint32_t count = 0;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &count, NULL);
  VkExtensionProperties *exts = malloc(sizeof(*exts) * count);
  if (!exts) return false;
  vkEnumerateDeviceExtensionProperties(pd, NULL, &count, exts);
  bool found = false;
  for (uint32_t i = 0; i < count; i++) {
    if (strcmp(exts[i].extensionName, name) == 0) { found = true; break; }
  }
  free(exts);
  return found;
}

// Score a physical device. Higher is better. Returns -1 if unusable.
static int device_score(VkPhysicalDevice pd) {
  VkPhysicalDeviceProperties props;
  vkGetPhysicalDeviceProperties(pd, &props);

  // Must support Vulkan 1.3
  if (props.apiVersion < VK_MAKE_API_VERSION(0, 1, 3, 0)) return -1;

  // Must have at least one compute queue family
  uint32_t qfc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, NULL);
  VkQueueFamilyProperties *qfp = malloc(sizeof(*qfp) * qfc);
  if (!qfp) return -1;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, qfp);
  bool has_compute = false;
  for (uint32_t i = 0; i < qfc; i++) {
    if (qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { has_compute = true; break; }
  }
  free(qfp);
  if (!has_compute) return -1;

  int score = 0;
  if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)   score += 1000;
  if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 500;
  if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU)    score += 100;

  return score;
}

static uint32_t find_compute_queue(VkPhysicalDevice pd) {
  uint32_t qfc = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, NULL);
  VkQueueFamilyProperties *qfp = malloc(sizeof(*qfp) * qfc);
  REQUIRE(qfp, "find_compute_queue: malloc");
  vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfc, qfp);
  // Prefer a dedicated compute-only queue (no graphics bit)
  uint32_t best = UINT32_MAX;
  for (uint32_t i = 0; i < qfc; i++) {
    if (!(qfp[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
    if (best == UINT32_MAX) best = i;
    if (!(qfp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { best = i; break; }
  }
  free(qfp);
  return best;
}

// ---------------------------------------------------------------------------
// vk_init
// ---------------------------------------------------------------------------

vk_context *vk_init(vk_config cfg) {
  vk_context *ctx = calloc(1, sizeof(*ctx));
  if (!ctx) return NULL;

  // --- Instance ---
  VkApplicationInfo app_info = {
    .sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO,
    .pApplicationName   = "volatile",
    .applicationVersion = VK_MAKE_VERSION(0, 1, 0),
    .pEngineName        = "volatile",
    .engineVersion      = VK_MAKE_VERSION(0, 1, 0),
    .apiVersion         = VK_API_VERSION_1_3,
  };

  const char *validation_layer = "VK_LAYER_KHRONOS_validation";
  bool use_validation = cfg.validation && layer_available(validation_layer);
  ctx->validation_enabled = use_validation;

  VkInstanceCreateInfo inst_ci = {
    .sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
    .pApplicationInfo        = &app_info,
    .enabledLayerCount       = use_validation ? 1 : 0,
    .ppEnabledLayerNames     = use_validation ? &validation_layer : NULL,
    .enabledExtensionCount   = 0,
    .ppEnabledExtensionNames = NULL,
  };

  VkResult r = vkCreateInstance(&inst_ci, NULL, &ctx->instance);
  if (r != VK_SUCCESS) {
    LOG_WARN("vk_init: vkCreateInstance failed (%d)", r);
    free(ctx);
    return NULL;
  }

  // --- Physical device selection ---
  uint32_t pd_count = 0;
  vkEnumeratePhysicalDevices(ctx->instance, &pd_count, NULL);
  if (pd_count == 0) {
    LOG_WARN("vk_init: no Vulkan physical devices found");
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
    return NULL;
  }

  VkPhysicalDevice *pds = malloc(sizeof(*pds) * pd_count);
  if (!pds) { vkDestroyInstance(ctx->instance, NULL); free(ctx); return NULL; }
  vkEnumeratePhysicalDevices(ctx->instance, &pd_count, pds);

  int best_score = -1;
  for (uint32_t i = 0; i < pd_count; i++) {
    int s = device_score(pds[i]);
    if (s > best_score) { best_score = s; ctx->physical_device = pds[i]; }
  }
  free(pds);

  if (best_score < 0) {
    LOG_WARN("vk_init: no suitable physical device (need Vulkan 1.3 + compute queue)");
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
    return NULL;
  }

  vkGetPhysicalDeviceProperties(ctx->physical_device, &ctx->props);
  LOG_INFO("vk_init: selected device: %s", ctx->props.deviceName);

  // --- Queue family ---
  ctx->compute_queue_family = find_compute_queue(ctx->physical_device);
  if (ctx->compute_queue_family == UINT32_MAX) {
    LOG_WARN("vk_init: failed to find compute queue family");
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
    return NULL;
  }

  // --- Device extensions ---
  bool bda_supported = device_ext_available(
    ctx->physical_device, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);

  const char *dev_exts[1];
  uint32_t dev_ext_count = 0;
  if (bda_supported)
    dev_exts[dev_ext_count++] = VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME;

  // --- Enable buffer_device_address feature if supported ---
  VkPhysicalDeviceBufferDeviceAddressFeatures bda_features = {
    .sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
    .bufferDeviceAddress = bda_supported ? VK_TRUE : VK_FALSE,
  };

  VkPhysicalDeviceVulkan13Features vk13_features = {
    .sType            = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
    .pNext            = bda_supported ? &bda_features : NULL,
    .synchronization2 = VK_TRUE,
    .dynamicRendering = VK_TRUE,
  };

  float queue_priority = 1.0f;
  VkDeviceQueueCreateInfo queue_ci = {
    .sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = ctx->compute_queue_family,
    .queueCount       = 1,
    .pQueuePriorities = &queue_priority,
  };

  VkDeviceCreateInfo dev_ci = {
    .sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
    .pNext                   = &vk13_features,
    .queueCreateInfoCount    = 1,
    .pQueueCreateInfos       = &queue_ci,
    .enabledExtensionCount   = dev_ext_count,
    .ppEnabledExtensionNames = dev_ext_count ? dev_exts : NULL,
  };

  r = vkCreateDevice(ctx->physical_device, &dev_ci, NULL, &ctx->device);
  if (r != VK_SUCCESS) {
    LOG_WARN("vk_init: vkCreateDevice failed (%d)", r);
    vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
    return NULL;
  }

  vkGetDeviceQueue(ctx->device, ctx->compute_queue_family, 0, &ctx->compute_queue);
  ctx->has_bda = bda_supported;

  LOG_INFO("vk_init: logical device ready (compute queue family %u, bda=%s)",
           ctx->compute_queue_family, bda_supported ? "yes" : "no");
  return ctx;
}

// ---------------------------------------------------------------------------
// vk_shutdown
// ---------------------------------------------------------------------------

void vk_shutdown(vk_context *ctx) {
  if (!ctx) return;
  if (ctx->device)   vkDestroyDevice(ctx->device, NULL);
  if (ctx->instance) vkDestroyInstance(ctx->instance, NULL);
  free(ctx);
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const char *vk_device_name(const vk_context *ctx) {
  REQUIRE(ctx, "vk_device_name: null context");
  return ctx->props.deviceName;
}

bool vk_has_buffer_device_address(const vk_context *ctx) {
  REQUIRE(ctx, "vk_has_buffer_device_address: null context");
  return ctx->has_bda;
}

VkDevice vk_device(const vk_context *ctx) {
  REQUIRE(ctx, "vk_device: null context");
  return ctx->device;
}

VkPhysicalDevice vk_physical_device(const vk_context *ctx) {
  REQUIRE(ctx, "vk_physical_device: null context");
  return ctx->physical_device;
}

VkQueue vk_compute_queue(const vk_context *ctx) {
  REQUIRE(ctx, "vk_compute_queue: null context");
  return ctx->compute_queue;
}

uint32_t vk_compute_queue_family(const vk_context *ctx) {
  REQUIRE(ctx, "vk_compute_queue_family: null context");
  return ctx->compute_queue_family;
}
