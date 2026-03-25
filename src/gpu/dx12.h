#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// DirectX 12 backend stub — Windows only.
// Full implementation requires the Windows SDK (d3d12.h, dxgi.h) and DXIL
// shader compilation.  This header exposes a C-callable interface so gpu.c
// can call through it on Windows builds.  On non-Windows builds all functions
// return NULL/false.
// ---------------------------------------------------------------------------

typedef struct dx12_context dx12_context;

typedef struct {
  bool debug_layer;       // enable D3D12 debug layer (slows down, dev only)
  bool prefer_warp;       // use WARP software renderer (testing/CI)
  int  adapter_index;     // -1 = pick highest-performance adapter
} dx12_config;

// Initialise Direct3D 12. Returns NULL on non-Windows platforms or if no
// suitable adapter was found.
dx12_context *dx12_init(dx12_config cfg);

// Release device, command queues, and descriptor heaps.
void dx12_shutdown(dx12_context *ctx);

// Human-readable adapter description (e.g. "NVIDIA GeForce RTX 4090").
// Pointer valid until dx12_shutdown.
const char *dx12_device_name(const dx12_context *ctx);

// True if the device supports DirectX Raytracing (DXR) tier 1.0+.
bool dx12_has_raytracing(const dx12_context *ctx);

// True if shader model 6.6+ is supported (required for mesh/amplification shaders).
bool dx12_has_shader_model_6_6(const dx12_context *ctx);

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

typedef struct dx12_buffer dx12_buffer;

typedef enum {
  DX12_HEAP_DEFAULT,   // GPU-local (fast, no CPU access)
  DX12_HEAP_UPLOAD,    // CPU write-once, GPU read
  DX12_HEAP_READBACK,  // GPU write, CPU read
} dx12_heap_type;

dx12_buffer *dx12_buffer_create(dx12_context *ctx, size_t size, dx12_heap_type heap);
void         dx12_buffer_destroy(dx12_buffer *buf);
void        *dx12_buffer_map(dx12_buffer *buf);      // only valid for UPLOAD/READBACK
void         dx12_buffer_unmap(dx12_buffer *buf);
void         dx12_buffer_upload(dx12_buffer *buf, const void *data, size_t size);
void         dx12_buffer_download(dx12_buffer *buf, void *data, size_t size);

// ---------------------------------------------------------------------------
// Compute pipeline (DXIL bytecode)
// ---------------------------------------------------------------------------

typedef struct dx12_pipeline dx12_pipeline;

// Create pipeline from pre-compiled DXIL bytecode.
dx12_pipeline *dx12_pipeline_create(dx12_context *ctx,
                                    const uint8_t *dxil, size_t dxil_size,
                                    const char *entry_point);
void           dx12_pipeline_destroy(dx12_pipeline *p);

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

void dx12_dispatch(dx12_context *ctx, dx12_pipeline *p,
                   dx12_buffer **buffers, int num_buffers,
                   int groups_x, int groups_y, int groups_z);

// Block until all GPU work submitted so far is complete.
void dx12_wait(dx12_context *ctx);
