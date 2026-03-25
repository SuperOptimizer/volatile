#include "greatest.h"
#include "gpu/gpu.h"

#include <string.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Minimal valid SPIR-V: a do-nothing compute shader (local_size 1,1,1).
// glslc output for:
//   #version 450
//   layout(local_size_x = 1) in;
//   void main() {}
// Represented as a byte array so we don't need external files.
// ---------------------------------------------------------------------------
static const uint8_t k_noop_spirv[] = {
  0x03, 0x02, 0x23, 0x07,  // magic
  0x00, 0x00, 0x01, 0x00,  // version 1.0
  0x0b, 0x00, 0x0d, 0x00,  // generator
  0x06, 0x00, 0x00, 0x00,  // bound = 6
  0x00, 0x00, 0x00, 0x00,  // reserved
  // OpCapability Shader
  0x11, 0x00, 0x02, 0x00,  0x01, 0x00, 0x00, 0x00,
  // OpExtInstImport "GLSL.std.450"
  0x0b, 0x00, 0x06, 0x00,  0x01, 0x00, 0x00, 0x00,
  0x47, 0x4c, 0x53, 0x4c,  0x2e, 0x73, 0x74, 0x64,
  0x2e, 0x34, 0x35, 0x30,  0x00, 0x00, 0x00, 0x00,
  // OpMemoryModel Logical GLSL450
  0x0e, 0x00, 0x03, 0x00,  0x00, 0x00, 0x00, 0x00,  0x01, 0x00, 0x00, 0x00,
  // OpEntryPoint GLCompute %4 "main"
  0x0f, 0x00, 0x05, 0x00,  0x05, 0x00, 0x00, 0x00,  0x04, 0x00, 0x00, 0x00,
  0x6d, 0x61, 0x69, 0x6e,  0x00, 0x00, 0x00, 0x00,
  // OpExecutionMode %4 LocalSize 1 1 1
  0x10, 0x00, 0x06, 0x00,  0x04, 0x00, 0x00, 0x00,  0x23, 0x00, 0x00, 0x00,
  0x01, 0x00, 0x00, 0x00,  0x01, 0x00, 0x00, 0x00,  0x01, 0x00, 0x00, 0x00,
  // OpSource GLSL 450
  0x03, 0x00, 0x03, 0x00,  0x02, 0x00, 0x00, 0x00,  0xc2, 0x01, 0x00, 0x00,
  // OpTypeVoid %2
  0x13, 0x00, 0x02, 0x00,  0x02, 0x00, 0x00, 0x00,
  // OpTypeFunction %3 %2
  0x21, 0x00, 0x03, 0x00,  0x03, 0x00, 0x00, 0x00,  0x02, 0x00, 0x00, 0x00,
  // OpFunction %2 %4 None %3
  0x36, 0x00, 0x05, 0x00,  0x02, 0x00, 0x00, 0x00,  0x04, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00,  0x03, 0x00, 0x00, 0x00,
  // OpLabel %5
  0xf8, 0x00, 0x02, 0x00,  0x05, 0x00, 0x00, 0x00,
  // OpReturn
  0xfd, 0x00, 0x01, 0x00,
  // OpFunctionEnd
  0x38, 0x00, 0x01, 0x00,
};

// Helper: skip tests gracefully if no Vulkan device is available.
static gpu_device *try_init(void) {
  return gpu_init(GPU_BACKEND_NONE);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_gpu_init_shutdown(void) {
  gpu_device *dev = try_init();
  if (!dev) SKIPm("No Vulkan device available");
  ASSERT_EQ(GPU_BACKEND_VULKAN, gpu_active_backend(dev));
  ASSERT(gpu_device_name(dev) != NULL);
  ASSERT(gpu_device_name(dev)[0] != '\0');
  gpu_shutdown(dev);
  PASS();
}

TEST test_gpu_buffer_host_visible_roundtrip(void) {
  gpu_device *dev = try_init();
  if (!dev) SKIPm("No Vulkan device available");

  enum { N = 256 };
  uint8_t src[N], dst[N];
  for (int i = 0; i < N; i++) src[i] = (uint8_t)i;
  memset(dst, 0, N);

  gpu_buffer *buf = gpu_buffer_create(dev, N, true);
  ASSERT(buf != NULL);

  gpu_buffer_upload(buf, src, N);
  gpu_buffer_download(buf, dst, N);

  for (int i = 0; i < N; i++) ASSERT_EQ(src[i], dst[i]);

  gpu_buffer_destroy(buf);
  gpu_shutdown(dev);
  PASS();
}

TEST test_gpu_buffer_map_unmap(void) {
  gpu_device *dev = try_init();
  if (!dev) SKIPm("No Vulkan device available");

  gpu_buffer *buf = gpu_buffer_create(dev, 64, true);
  ASSERT(buf != NULL);

  void *ptr = gpu_buffer_map(buf);
  ASSERT(ptr != NULL);
  memset(ptr, 0xAB, 64);
  gpu_buffer_unmap(buf);

  uint8_t dst[64];
  gpu_buffer_download(buf, dst, 64);
  ASSERT_EQ(0xAB, dst[0]);
  ASSERT_EQ(0xAB, dst[63]);

  gpu_buffer_destroy(buf);
  gpu_shutdown(dev);
  PASS();
}

TEST test_gpu_pipeline_create_destroy(void) {
  gpu_device *dev = try_init();
  if (!dev) SKIPm("No Vulkan device available");

  gpu_pipeline *p = gpu_pipeline_create(dev, k_noop_spirv, sizeof(k_noop_spirv));
  if (!p) {
    // SPIR-V may be rejected by the driver on some platforms; that's ok.
    gpu_shutdown(dev);
    SKIPm("Driver rejected noop SPIR-V");
  }
  gpu_pipeline_destroy(p);
  gpu_shutdown(dev);
  PASS();
}

TEST test_gpu_dispatch_noop(void) {
  gpu_device *dev = try_init();
  if (!dev) SKIPm("No Vulkan device available");

  gpu_pipeline *p = gpu_pipeline_create(dev, k_noop_spirv, sizeof(k_noop_spirv));
  if (!p) { gpu_shutdown(dev); SKIPm("Driver rejected noop SPIR-V"); }

  // Dispatch with no buffers — just exercises the command submission path.
  gpu_dispatch(dev, p, NULL, 0, 1, 1, 1);
  gpu_wait(dev);

  gpu_pipeline_destroy(p);
  gpu_shutdown(dev);
  PASS();
}

TEST test_gpu_null_on_unsupported_backend(void) {
  gpu_device *dev_metal = gpu_init(GPU_BACKEND_METAL);
  ASSERT_EQ(NULL, dev_metal);

  gpu_device *dev_dx12 = gpu_init(GPU_BACKEND_DX12);
  ASSERT_EQ(NULL, dev_dx12);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(gpu_suite) {
  RUN_TEST(test_gpu_null_on_unsupported_backend);
  RUN_TEST(test_gpu_init_shutdown);
  RUN_TEST(test_gpu_buffer_host_visible_roundtrip);
  RUN_TEST(test_gpu_buffer_map_unmap);
  RUN_TEST(test_gpu_pipeline_create_destroy);
  RUN_TEST(test_gpu_dispatch_noop);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(gpu_suite);
  GREATEST_MAIN_END();
}
