#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// GPU abstraction layer — wraps Vulkan (now), Metal/DX12 (future)
// ---------------------------------------------------------------------------

typedef enum {
  GPU_BACKEND_NONE   = 0,  // auto-detect
  GPU_BACKEND_VULKAN,
  GPU_BACKEND_METAL,
  GPU_BACKEND_DX12,
} gpu_backend_t;

typedef struct gpu_device   gpu_device;
typedef struct gpu_buffer   gpu_buffer;
typedef struct gpu_pipeline gpu_pipeline;

// Initialise the GPU. GPU_BACKEND_NONE = auto-detect (prefers Vulkan).
// Returns NULL if no suitable backend could be initialised.
gpu_device    *gpu_init(gpu_backend_t preferred);
void           gpu_shutdown(gpu_device *dev);

gpu_backend_t  gpu_active_backend(const gpu_device *dev);
const char    *gpu_device_name(const gpu_device *dev);

// Buffer management
gpu_buffer *gpu_buffer_create(gpu_device *dev, size_t size, bool host_visible);
void        gpu_buffer_destroy(gpu_buffer *buf);
void       *gpu_buffer_map(gpu_buffer *buf);
void        gpu_buffer_unmap(gpu_buffer *buf);
void        gpu_buffer_upload(gpu_buffer *buf, const void *data, size_t size);
void        gpu_buffer_download(gpu_buffer *buf, void *data, size_t size);

// Compute pipeline (SPIR-V)
gpu_pipeline *gpu_pipeline_create(gpu_device *dev, const uint8_t *spirv, size_t spirv_size);
void          gpu_pipeline_destroy(gpu_pipeline *p);

// Dispatch a compute workgroup and optionally wait for completion.
void gpu_dispatch(gpu_device *dev, gpu_pipeline *p,
                  gpu_buffer **buffers, int num_buffers,
                  int groups_x, int groups_y, int groups_z);
void gpu_wait(gpu_device *dev);
