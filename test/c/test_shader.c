#define _DEFAULT_SOURCE   /* mkstemps, mkdtemp */
#include "greatest.h"
#include "gpu/shader.h"

#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Minimal valid GLSL compute shader for testing
// ---------------------------------------------------------------------------

static const char *SIMPLE_COMP =
  "#version 450\n"
  "layout(local_size_x = 1) in;\n"
  "layout(set = 0, binding = 0) writeonly buffer Out { uint data[]; } out_buf;\n"
  "void main() { out_buf.data[gl_GlobalInvocationID.x] = 42u; }\n";

// Fake SPIR-V: valid magic word + minimal header (8 uint32s)
// Used to test shader_load_spirv and shader_from_embedded without running glslc.
static const uint8_t FAKE_SPIRV[] = {
  // SPIR-V magic: 0x07230203 (little-endian)
  0x03, 0x02, 0x23, 0x07,
  // version 1.0
  0x00, 0x00, 0x01, 0x00,
  // generator magic
  0x00, 0x00, 0x00, 0x00,
  // bound
  0x01, 0x00, 0x00, 0x00,
  // reserved
  0x00, 0x00, 0x00, 0x00,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns true if glslc is available on PATH.
static bool glslc_available(void) {
  return system("glslc --version > /dev/null 2>&1") == 0;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST test_from_embedded_basic(void) {
  compiled_shader *s = shader_from_embedded(FAKE_SPIRV, sizeof(FAKE_SPIRV), SHADER_SPIRV);
  ASSERT(s != NULL);
  ASSERT_EQ(s->size, sizeof(FAKE_SPIRV));
  ASSERT_EQ(s->target, SHADER_SPIRV);
  ASSERT(s->entry_point != NULL);
  ASSERT_STR_EQ(s->entry_point, "main");
  compiled_shader_free(s);
  PASS();
}

TEST test_from_embedded_null_data(void) {
  compiled_shader *s = shader_from_embedded(NULL, 16, SHADER_SPIRV);
  ASSERT(s == NULL);
  PASS();
}

TEST test_from_embedded_zero_size(void) {
  compiled_shader *s = shader_from_embedded(FAKE_SPIRV, 0, SHADER_SPIRV);
  ASSERT(s == NULL);
  PASS();
}

TEST test_free_null_safe(void) {
  compiled_shader_free(NULL);  // must not crash
  PASS();
}

TEST test_load_spirv_roundtrip(void) {
  // Write fake SPIR-V to a temp file and load it back
  char tmp[] = "/tmp/volatile_test_XXXXXX.spv";
  int fd = mkstemps(tmp, 4);
  ASSERT(fd >= 0);
  write(fd, FAKE_SPIRV, sizeof(FAKE_SPIRV));
  close(fd);

  compiled_shader *s = shader_load_spirv(tmp);
  unlink(tmp);

  ASSERT(s != NULL);
  ASSERT_EQ(s->size, sizeof(FAKE_SPIRV));
  ASSERT_EQ(memcmp(s->code, FAKE_SPIRV, sizeof(FAKE_SPIRV)), 0);
  compiled_shader_free(s);
  PASS();
}

TEST test_load_spirv_missing_file(void) {
  compiled_shader *s = shader_load_spirv("/tmp/volatile_no_such_file_xyz.spv");
  ASSERT(s == NULL);
  PASS();
}

TEST test_compile_glsl_to_spirv(void) {
  if (!glslc_available()) {
    SKIPm("glslc not available");
  }

  compiled_shader *s = shader_compile_glsl(SIMPLE_COMP, "main", SHADER_SPIRV, NULL, 0);
  ASSERT(s != NULL);
  ASSERT(s->size >= 20);         // any real SPIR-V is at least 5 words
  ASSERT_EQ(s->target, SHADER_SPIRV);
  // Check SPIR-V magic: 0x07230203 in little-endian
  ASSERT(s->size >= 4);
  ASSERT_EQ(s->code[0], 0x03);
  ASSERT_EQ(s->code[1], 0x02);
  ASSERT_EQ(s->code[2], 0x23);
  ASSERT_EQ(s->code[3], 0x07);
  compiled_shader_free(s);
  PASS();
}

TEST test_compile_glsl_with_defines(void) {
  if (!glslc_available()) {
    SKIPm("glslc not available");
  }
  const char *src =
    "#version 450\n"
    "layout(local_size_x = 1) in;\n"
    "#ifndef MY_DEFINE\n"
    "  #error MY_DEFINE not set\n"
    "#endif\n"
    "layout(set=0, binding=0) writeonly buffer B { uint d[]; } b;\n"
    "void main() { b.d[0] = MY_DEFINE; }\n";

  const char *defines[] = {"MY_DEFINE=7"};
  compiled_shader *s = shader_compile_glsl(src, "main", SHADER_SPIRV, defines, 1);
  ASSERT(s != NULL);
  compiled_shader_free(s);
  PASS();
}

TEST test_compile_glsl_msl_stub(void) {
  compiled_shader *s = shader_compile_glsl(SIMPLE_COMP, "main", SHADER_MSL, NULL, 0);
  ASSERT(s == NULL);  // MSL not implemented yet
  PASS();
}

TEST test_cache_hit(void) {
  if (!glslc_available()) {
    SKIPm("glslc not available");
  }

  char cache_dir[] = "/tmp/volatile_shader_cache_XXXXXX";
  ASSERT(mkdtemp(cache_dir) != NULL);

  shader_cache *c = shader_cache_new(cache_dir);
  ASSERT(c != NULL);

  // First call: cache miss, compiles
  compiled_shader *s1 = shader_cache_get_or_compile(c, SIMPLE_COMP, "main",
                                                     SHADER_SPIRV, NULL, 0);
  ASSERT(s1 != NULL);
  size_t size1 = s1->size;
  compiled_shader_free(s1);

  // Second call: cache hit, loads from disk
  compiled_shader *s2 = shader_cache_get_or_compile(c, SIMPLE_COMP, "main",
                                                     SHADER_SPIRV, NULL, 0);
  ASSERT(s2 != NULL);
  ASSERT_EQ(s2->size, size1);
  compiled_shader_free(s2);

  shader_cache_free(c);

  // Cleanup temp dir (non-recursive — just the .spv files inside)
  char clean_cmd[640];
  snprintf(clean_cmd, sizeof(clean_cmd), "rm -rf %s", cache_dir);
  system(clean_cmd);

  PASS();
}

TEST test_cache_new_free(void) {
  shader_cache *c = shader_cache_new("/tmp");
  ASSERT(c != NULL);
  shader_cache_free(c);
  PASS();
}

// Reload callback for test_hot_reload_registration.
static int g_reload_called = 0;
static void test_reload_cb(const char *path, compiled_shader *s, void *ctx) {
  (void)path; (void)ctx;
  g_reload_called++;
  compiled_shader_free(s);
}

TEST test_hot_reload_registration(void) {
  // Register a watch on a real file; verify poll doesn't crash when mtime unchanged.
  char tmp[] = "/tmp/volatile_watch_XXXXXX.comp";
  int fd = mkstemps(tmp, 5);
  ASSERT(fd >= 0);
  write(fd, SIMPLE_COMP, strlen(SIMPLE_COMP));
  close(fd);

  shader_cache *c = shader_cache_new("/tmp");
  ASSERT(c != NULL);

  g_reload_called = 0;
  shader_watch(c, tmp, test_reload_cb, NULL);

  shader_poll_reloads(c);  // mtime unchanged — no recompile
  ASSERT_EQ(0, g_reload_called);

  shader_cache_free(c);
  unlink(tmp);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(shader_suite) {
  RUN_TEST(test_from_embedded_basic);
  RUN_TEST(test_from_embedded_null_data);
  RUN_TEST(test_from_embedded_zero_size);
  RUN_TEST(test_free_null_safe);
  RUN_TEST(test_load_spirv_roundtrip);
  RUN_TEST(test_load_spirv_missing_file);
  RUN_TEST(test_compile_glsl_to_spirv);
  RUN_TEST(test_compile_glsl_with_defines);
  RUN_TEST(test_compile_glsl_msl_stub);
  RUN_TEST(test_cache_hit);
  RUN_TEST(test_cache_new_free);
  RUN_TEST(test_hot_reload_registration);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(shader_suite);
  GREATEST_MAIN_END();
}
