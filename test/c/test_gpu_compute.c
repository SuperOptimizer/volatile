#define _GNU_SOURCE
#include "greatest.h"
#include "gpu/gpu.h"
#include "gpu/shader.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ---------------------------------------------------------------------------
// Test shader: window/level without push constants
//
// Identical logic to window_level.comp but params are in a storage buffer
// (binding 2) so it works with the gpu_dispatch abstraction (no push consts).
// Output: grayscale RGBA8 packed as uint.
// ---------------------------------------------------------------------------

static const char *WL_GLSL =
  "#version 450\n"
  "layout(local_size_x = 64) in;\n"
  "layout(set=0, binding=0) readonly  buffer In  { float data[]; } in_buf;\n"
  "layout(set=0, binding=1) writeonly buffer Out { uint  data[]; } out_buf;\n"
  "layout(set=0, binding=2) readonly  buffer Params {\n"
  "  float window;\n"
  "  float level;\n"
  "  uint  num_pixels;\n"
  "} params;\n"
  "void main() {\n"
  "  uint idx = gl_GlobalInvocationID.x;\n"
  "  if (idx >= params.num_pixels) return;\n"
  "  float t = clamp((in_buf.data[idx] - (params.level - params.window*0.5))\n"
  "                  / params.window, 0.0, 1.0);\n"
  "  uint g = uint(t * 255.0 + 0.5);\n"
  "  out_buf.data[idx] = (0xFFu << 24) | (g << 16) | (g << 8) | g;\n"
  "}\n";

// ---------------------------------------------------------------------------
// CPU reference for the same window/level grayscale mapping
// ---------------------------------------------------------------------------

static uint32_t cpu_wl(float val, float window, float level) {
  float half = window * 0.5f;
  float t = (val - (level - half)) / window;
  if (t < 0.0f) t = 0.0f;
  if (t > 1.0f) t = 1.0f;
  uint32_t g = (uint32_t)(t * 255.0f + 0.5f);
  if (g > 255) g = 255;
  return (0xFFu << 24) | (g << 16) | (g << 8) | g;
}

// ---------------------------------------------------------------------------
// Shared GPU device (initialised once for all tests)
// ---------------------------------------------------------------------------

static gpu_device *g_dev = NULL;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_gpu_init(void) {
  g_dev = gpu_init(GPU_BACKEND_NONE);
  if (!g_dev) SKIPm("no Vulkan device available");
  ASSERT(gpu_device_name(g_dev) != NULL);
  ASSERT(gpu_device_name(g_dev)[0] != '\0');
  PASS();
}

TEST test_window_level_compute(void) {
  if (!g_dev) SKIPm("no GPU");

  // Compile shader at runtime
  compiled_shader *cs = shader_compile_glsl(WL_GLSL, "main", SHADER_SPIRV, NULL, 0);
  if (!cs) SKIPm("glslc/shaderc unavailable — cannot compile test shader");

  gpu_pipeline *pipe = gpu_pipeline_create(g_dev, cs->code, cs->size);
  compiled_shader_free(cs);
  ASSERT(pipe != NULL);

  // Prepare 256 input floats spanning [0, 1]
  enum { N = 256 };
  float input[N];
  for (int i = 0; i < N; i++) input[i] = (float)i / (float)(N - 1);

  // Params buffer: window=1.0, level=0.5, num_pixels=N
  struct { float window; float level; uint32_t num_pixels; } params = {
    .window = 1.0f, .level = 0.5f, .num_pixels = N
  };

  gpu_buffer *in_buf     = gpu_buffer_create(g_dev, sizeof(input),  true);
  gpu_buffer *out_buf    = gpu_buffer_create(g_dev, N * sizeof(uint32_t), true);
  gpu_buffer *param_buf  = gpu_buffer_create(g_dev, sizeof(params), true);
  ASSERT(in_buf && out_buf && param_buf);

  gpu_buffer_upload(in_buf,    input,   sizeof(input));
  gpu_buffer_upload(param_buf, &params, sizeof(params));

  // Clear output
  uint32_t zeros[N];
  memset(zeros, 0, sizeof(zeros));
  gpu_buffer_upload(out_buf, zeros, sizeof(zeros));

  gpu_buffer *bufs[3] = { in_buf, out_buf, param_buf };
  // N=256 elements, local_size_x=64 → 4 groups
  gpu_dispatch(g_dev, pipe, bufs, 3, (N + 63) / 64, 1, 1);
  gpu_wait(g_dev);

  // Download and verify
  uint32_t result[N];
  gpu_buffer_download(out_buf, result, sizeof(result));

  for (int i = 0; i < N; i++) {
    uint32_t expected = cpu_wl(input[i], params.window, params.level);
    // Allow ±1 in each channel for rounding differences
    for (int ch = 0; ch < 3; ch++) {
      int got = (int)((result[i]   >> (ch * 8)) & 0xFF);
      int exp = (int)((expected >> (ch * 8)) & 0xFF);
      ASSERT(abs(got - exp) <= 1);
    }
    // Alpha must be 0xFF
    ASSERT_EQ((result[i] >> 24) & 0xFF, (uint32_t)0xFF);
  }

  gpu_buffer_destroy(in_buf);
  gpu_buffer_destroy(out_buf);
  gpu_buffer_destroy(param_buf);
  gpu_pipeline_destroy(pipe);
  PASS();
}

TEST test_buffer_roundtrip(void) {
  if (!g_dev) SKIPm("no GPU");

  // Simple identity compute: copy input → output via trivial shader
  static const char *COPY_GLSL =
    "#version 450\n"
    "layout(local_size_x = 64) in;\n"
    "layout(set=0, binding=0) readonly  buffer In  { uint d[]; } a;\n"
    "layout(set=0, binding=1) writeonly buffer Out { uint d[]; } b;\n"
    "void main() { b.d[gl_GlobalInvocationID.x] = a.d[gl_GlobalInvocationID.x]; }\n";

  compiled_shader *cs = shader_compile_glsl(COPY_GLSL, "main", SHADER_SPIRV, NULL, 0);
  if (!cs) SKIPm("glslc/shaderc unavailable");

  gpu_pipeline *pipe = gpu_pipeline_create(g_dev, cs->code, cs->size);
  compiled_shader_free(cs);
  ASSERT(pipe != NULL);

  enum { M = 128 };
  uint32_t input[M], output[M];
  for (int i = 0; i < M; i++) input[i] = (uint32_t)(i * 7919);
  memset(output, 0, sizeof(output));

  gpu_buffer *in_b  = gpu_buffer_create(g_dev, sizeof(input),  true);
  gpu_buffer *out_b = gpu_buffer_create(g_dev, sizeof(output), true);
  ASSERT(in_b && out_b);
  gpu_buffer_upload(in_b, input, sizeof(input));

  gpu_buffer *bufs[2] = { in_b, out_b };
  gpu_dispatch(g_dev, pipe, bufs, 2, (M + 63) / 64, 1, 1);
  gpu_wait(g_dev);
  gpu_buffer_download(out_b, output, sizeof(output));

  ASSERT_EQ(memcmp(input, output, sizeof(input)), 0);

  gpu_buffer_destroy(in_b);
  gpu_buffer_destroy(out_b);
  gpu_pipeline_destroy(pipe);
  PASS();
}

TEST test_marching_cubes_spv_loads(void) {
  if (!g_dev) SKIPm("no GPU");

  // Just verify the pre-compiled marching_cubes.spv can be loaded as a shader
  // module.  Dispatch is skipped because the shader uses push constants which
  // gpu_dispatch does not yet support.
  const char *spv_path =
    "/home/forrest/CLionProjects/volatile/src/gpu/shaders/marching_cubes.spv";

  compiled_shader *cs = shader_load_spirv(spv_path);
  if (!cs) SKIPm("marching_cubes.spv not found — run compile_shaders.sh first");

  ASSERT(cs->size >= 20);
  // SPIR-V magic 0x07230203 (little-endian)
  ASSERT_EQ(cs->code[0], 0x03);
  ASSERT_EQ(cs->code[1], 0x02);
  ASSERT_EQ(cs->code[2], 0x23);
  ASSERT_EQ(cs->code[3], 0x07);

  gpu_pipeline *pipe = gpu_pipeline_create(g_dev, cs->code, cs->size);
  compiled_shader_free(cs);

  // Pipeline creation may fail on some drivers due to missing features; skip.
  if (!pipe) SKIPm("gpu_pipeline_create failed for marching_cubes (driver limitation)");

  gpu_pipeline_destroy(pipe);
  PASS();
}

TEST test_gpu_shutdown(void) {
  if (!g_dev) SKIPm("no GPU");
  gpu_shutdown(g_dev);
  g_dev = NULL;
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(gpu_compute_suite) {
  RUN_TEST(test_gpu_init);
  RUN_TEST(test_window_level_compute);
  RUN_TEST(test_buffer_roundtrip);
  RUN_TEST(test_marching_cubes_spv_loads);
  RUN_TEST(test_gpu_shutdown);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(gpu_compute_suite);
  GREATEST_MAIN_END();
}
