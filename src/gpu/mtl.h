#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Metal backend stub — macOS/iOS only.
// Full implementation requires Objective-C and the Metal framework;
// this header exposes a C-callable interface so gpu.c can call through it
// on Apple platforms. On non-Apple builds all functions return NULL/false.
// ---------------------------------------------------------------------------

typedef struct mtl_context mtl_context;

typedef struct {
  bool prefer_low_power;  // prefer integrated GPU over discrete
} mtl_config;

// Initialise Metal. Returns NULL on non-Apple platforms or if no Metal
// device is found.
mtl_context *mtl_init(mtl_config cfg);

// Release all Metal resources.
void mtl_shutdown(mtl_context *ctx);

// Human-readable GPU name (e.g. "Apple M3 Pro").
// Pointer valid until mtl_shutdown.
const char *mtl_device_name(const mtl_context *ctx);

// True if the device supports non-uniform threadgroup sizes (Metal 2+).
bool mtl_has_nonuniform_threadgroups(const mtl_context *ctx);

// ---------------------------------------------------------------------------
// Buffer management
// ---------------------------------------------------------------------------

typedef struct mtl_buffer mtl_buffer;

// Allocate a Metal buffer. shared=true maps it to CPU-visible (managed) memory.
mtl_buffer *mtl_buffer_create(mtl_context *ctx, size_t size, bool shared);
void        mtl_buffer_destroy(mtl_buffer *buf);
void       *mtl_buffer_map(mtl_buffer *buf);     // returns CPU pointer
void        mtl_buffer_unmap(mtl_buffer *buf);   // sync GPU-side (no-op on unified memory)
void        mtl_buffer_upload(mtl_buffer *buf, const void *data, size_t size);
void        mtl_buffer_download(mtl_buffer *buf, void *data, size_t size);

// ---------------------------------------------------------------------------
// Compute pipeline (MSL source or pre-compiled metallib)
// ---------------------------------------------------------------------------

typedef struct mtl_pipeline mtl_pipeline;

// Compile MSL source into a pipeline. Returns NULL on failure.
mtl_pipeline *mtl_pipeline_create_msl(mtl_context *ctx,
                                      const char *msl_source,
                                      const char *entry_point);

// Load a pre-compiled .metallib blob.
mtl_pipeline *mtl_pipeline_create_metallib(mtl_context *ctx,
                                           const uint8_t *blob, size_t size,
                                           const char *entry_point);

void mtl_pipeline_destroy(mtl_pipeline *p);

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

// Submit a compute dispatch and (optionally) block until done.
void mtl_dispatch(mtl_context *ctx, mtl_pipeline *p,
                  mtl_buffer **buffers, int num_buffers,
                  int groups_x, int groups_y, int groups_z);
void mtl_wait(mtl_context *ctx);
