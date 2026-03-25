#include "gpu/mtl.h"
#include "core/log.h"

// ---------------------------------------------------------------------------
// Metal backend stub.
//
// On Apple platforms this file should be replaced (or augmented) with
// Objective-C that calls into the Metal framework.  Until then every
// function logs a warning and returns a safe no-op value so the rest of
// the codebase can compile and link on all platforms.
//
// TODO: implement with Objective-C / Metal framework on macOS/iOS.
// ---------------------------------------------------------------------------

mtl_context *mtl_init(mtl_config cfg) {
  (void)cfg;
  LOG_WARN("mtl_init: Metal backend is not yet implemented (stub)");
  return NULL;
}

void mtl_shutdown(mtl_context *ctx) {
  (void)ctx;
}

const char *mtl_device_name(const mtl_context *ctx) {
  (void)ctx;
  return "(Metal stub)";
}

bool mtl_has_nonuniform_threadgroups(const mtl_context *ctx) {
  (void)ctx;
  return false;
}

mtl_buffer *mtl_buffer_create(mtl_context *ctx, size_t size, bool shared) {
  (void)ctx; (void)size; (void)shared;
  LOG_WARN("mtl_buffer_create: Metal backend not implemented");
  return NULL;
}

void mtl_buffer_destroy(mtl_buffer *buf) {
  (void)buf;
}

void *mtl_buffer_map(mtl_buffer *buf) {
  (void)buf;
  return NULL;
}

void mtl_buffer_unmap(mtl_buffer *buf) {
  (void)buf;
}

void mtl_buffer_upload(mtl_buffer *buf, const void *data, size_t size) {
  (void)buf; (void)data; (void)size;
}

void mtl_buffer_download(mtl_buffer *buf, void *data, size_t size) {
  (void)buf; (void)data; (void)size;
}

mtl_pipeline *mtl_pipeline_create_msl(mtl_context *ctx,
                                      const char *msl_source,
                                      const char *entry_point) {
  (void)ctx; (void)msl_source; (void)entry_point;
  LOG_WARN("mtl_pipeline_create_msl: Metal backend not implemented");
  return NULL;
}

mtl_pipeline *mtl_pipeline_create_metallib(mtl_context *ctx,
                                           const uint8_t *blob, size_t size,
                                           const char *entry_point) {
  (void)ctx; (void)blob; (void)size; (void)entry_point;
  LOG_WARN("mtl_pipeline_create_metallib: Metal backend not implemented");
  return NULL;
}

void mtl_pipeline_destroy(mtl_pipeline *p) {
  (void)p;
}

void mtl_dispatch(mtl_context *ctx, mtl_pipeline *p,
                  mtl_buffer **buffers, int num_buffers,
                  int groups_x, int groups_y, int groups_z) {
  (void)ctx; (void)p; (void)buffers; (void)num_buffers;
  (void)groups_x; (void)groups_y; (void)groups_z;
}

void mtl_wait(mtl_context *ctx) {
  (void)ctx;
}
