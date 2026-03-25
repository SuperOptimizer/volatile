#pragma once
#include <stddef.h>
#include <stdint.h>

typedef enum {
  SHADER_SPIRV,
  SHADER_MSL,   // stub — requires SPIRV-Cross (future)
  SHADER_DXIL,  // stub — requires dxc (future)
} shader_target_t;

typedef struct {
  uint8_t        *code;
  size_t          size;
  shader_target_t target;
  char           *entry_point;
} compiled_shader;

typedef struct shader_cache shader_cache;

// Compile GLSL source to the requested target.
// For SPIR-V: tries shaderc library first, then falls back to glslc subprocess.
// MSL/DXIL: returns NULL (TODO).
// Returns NULL on failure. Caller owns the result; free with compiled_shader_free.
compiled_shader *shader_compile_glsl(const char *source, const char *entry,
                                     shader_target_t target,
                                     const char **defines, int num_defines);
void compiled_shader_free(compiled_shader *s);

// Load pre-compiled SPIR-V from a .spv file.
compiled_shader *shader_load_spirv(const char *path);

// Embed a shader compiled at build-time (from a static C array).
// Does NOT take ownership of data.
compiled_shader *shader_from_embedded(const uint8_t *data, size_t size,
                                      shader_target_t target);

// ---------------------------------------------------------------------------
// Disk cache
// ---------------------------------------------------------------------------

// Create a shader cache that stores compiled binaries in cache_dir.
// cache_dir is created if it does not exist.
shader_cache *shader_cache_new(const char *cache_dir);
void          shader_cache_free(shader_cache *c);

// Look up (source + entry + defines) in the cache; compile and store on miss.
compiled_shader *shader_cache_get_or_compile(shader_cache *c,
                                             const char *source, const char *entry,
                                             shader_target_t target,
                                             const char **defines, int num_defines);

// ---------------------------------------------------------------------------
// Hot reload
// ---------------------------------------------------------------------------

typedef void (*shader_reload_fn)(const char *path, compiled_shader *new_shader, void *ctx);

// Watch a GLSL source file for changes. On change, recompile and call callback.
void shader_watch(shader_cache *c, const char *glsl_path,
                  shader_reload_fn callback, void *ctx);

// Poll all watched files (call each frame in debug builds).
void shader_poll_reloads(shader_cache *c);
