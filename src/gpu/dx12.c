#include "gpu/dx12.h"
#include "core/log.h"

// ---------------------------------------------------------------------------
// DirectX 12 backend stub.
//
// On Windows this file should be replaced (or augmented) with code that
// includes <d3d12.h>, <dxgi1_6.h>, and calls the D3D12 API.  Until then
// every function logs a warning and returns a safe no-op value so the rest
// of the codebase can compile and link on all platforms.
//
// TODO: implement with D3D12/DXGI on Windows.
// ---------------------------------------------------------------------------

dx12_context *dx12_init(dx12_config cfg) {
  (void)cfg;
  LOG_WARN("dx12_init: DirectX 12 backend is not yet implemented (stub)");
  return NULL;
}

void dx12_shutdown(dx12_context *ctx) {
  (void)ctx;
}

const char *dx12_device_name(const dx12_context *ctx) {
  (void)ctx;
  return "(DX12 stub)";
}

bool dx12_has_raytracing(const dx12_context *ctx) {
  (void)ctx;
  return false;
}

bool dx12_has_shader_model_6_6(const dx12_context *ctx) {
  (void)ctx;
  return false;
}

dx12_buffer *dx12_buffer_create(dx12_context *ctx, size_t size, dx12_heap_type heap) {
  (void)ctx; (void)size; (void)heap;
  LOG_WARN("dx12_buffer_create: DirectX 12 backend not implemented");
  return NULL;
}

void dx12_buffer_destroy(dx12_buffer *buf) {
  (void)buf;
}

void *dx12_buffer_map(dx12_buffer *buf) {
  (void)buf;
  return NULL;
}

void dx12_buffer_unmap(dx12_buffer *buf) {
  (void)buf;
}

void dx12_buffer_upload(dx12_buffer *buf, const void *data, size_t size) {
  (void)buf; (void)data; (void)size;
}

void dx12_buffer_download(dx12_buffer *buf, void *data, size_t size) {
  (void)buf; (void)data; (void)size;
}

dx12_pipeline *dx12_pipeline_create(dx12_context *ctx,
                                    const uint8_t *dxil, size_t dxil_size,
                                    const char *entry_point) {
  (void)ctx; (void)dxil; (void)dxil_size; (void)entry_point;
  LOG_WARN("dx12_pipeline_create: DirectX 12 backend not implemented");
  return NULL;
}

void dx12_pipeline_destroy(dx12_pipeline *p) {
  (void)p;
}

void dx12_dispatch(dx12_context *ctx, dx12_pipeline *p,
                   dx12_buffer **buffers, int num_buffers,
                   int groups_x, int groups_y, int groups_z) {
  (void)ctx; (void)p; (void)buffers; (void)num_buffers;
  (void)groups_x; (void)groups_y; (void)groups_z;
}

void dx12_wait(dx12_context *ctx) {
  (void)ctx;
}
