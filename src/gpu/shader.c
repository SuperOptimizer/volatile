#define _DEFAULT_SOURCE   /* mkstemps, mkdtemp */

#include "gpu/shader.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

// Optional: shaderc C API (compile-time feature flag)
#ifdef HAVE_SHADERC
#  include <shaderc/shaderc.h>
#endif

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

#define MAX_WATCHED       64
#define CACHE_KEY_LEN     17   // 16 hex chars + NUL
#define DEFINE_BUF_MAX    4096

// ---------------------------------------------------------------------------
// FNV-1a 64-bit hash
// ---------------------------------------------------------------------------

static uint64_t fnv1a(const uint8_t *data, size_t len) {
  uint64_t h = 14695981039346656037ULL;
  for (size_t i = 0; i < len; i++) {
    h ^= data[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static void hash_to_hex(uint64_t h, char out[CACHE_KEY_LEN]) {
  snprintf(out, CACHE_KEY_LEN, "%016llx", (unsigned long long)h);
}

// ---------------------------------------------------------------------------
// compiled_shader helpers
// ---------------------------------------------------------------------------

static compiled_shader *shader_alloc(uint8_t *code, size_t size,
                                     shader_target_t target, const char *entry) {
  compiled_shader *s = malloc(sizeof(*s));
  if (!s) return NULL;
  s->code        = code;
  s->size        = size;
  s->target      = target;
  s->entry_point = entry ? strdup(entry) : NULL;
  return s;
}

void compiled_shader_free(compiled_shader *s) {
  if (!s) return;
  free(s->code);
  free(s->entry_point);
  free(s);
}

compiled_shader *shader_from_embedded(const uint8_t *data, size_t size,
                                      shader_target_t target) {
  if (!data || size == 0) return NULL;
  uint8_t *copy = malloc(size);
  if (!copy) return NULL;
  memcpy(copy, data, size);
  return shader_alloc(copy, size, target, "main");
}

// ---------------------------------------------------------------------------
// SPIR-V file I/O
// ---------------------------------------------------------------------------

compiled_shader *shader_load_spirv(const char *path) {
  if (!path) return NULL;
  FILE *f = fopen(path, "rb");
  if (!f) {
    LOG_WARN("shader_load_spirv: cannot open %s: %s", path, strerror(errno));
    return NULL;
  }
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0 || sz % 4 != 0) {
    LOG_WARN("shader_load_spirv: invalid SPIR-V size %ld in %s", sz, path);
    fclose(f);
    return NULL;
  }
  uint8_t *buf = malloc((size_t)sz);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    LOG_WARN("shader_load_spirv: read error in %s", path);
    free(buf); fclose(f); return NULL;
  }
  fclose(f);
  return shader_alloc(buf, (size_t)sz, SHADER_SPIRV, "main");
}

// ---------------------------------------------------------------------------
// GLSL compilation — shaderc library path
// ---------------------------------------------------------------------------

#ifdef HAVE_SHADERC
static compiled_shader *compile_via_shaderc(const char *source, const char *entry,
                                             const char **defines, int num_defines) {
  shaderc_compiler_t compiler = shaderc_compiler_initialize();
  shaderc_compile_options_t opts = shaderc_compile_options_initialize();

  for (int i = 0; i < num_defines; i++) {
    // Each define is "NAME" or "NAME=VALUE"
    const char *eq = strchr(defines[i], '=');
    if (eq) {
      shaderc_compile_options_add_macro_definition(opts,
        defines[i], (size_t)(eq - defines[i]), eq + 1, strlen(eq + 1));
    } else {
      shaderc_compile_options_add_macro_definition(opts,
        defines[i], strlen(defines[i]), "1", 1);
    }
  }
  shaderc_compile_options_set_target_env(opts, shaderc_target_env_vulkan,
                                         shaderc_env_version_vulkan_1_3);

  shaderc_compilation_result_t result = shaderc_compile_into_spv(
    compiler, source, strlen(source),
    shaderc_compute_shader, "<source>", entry, opts);

  compiled_shader *out = NULL;
  if (shaderc_result_get_compilation_status(result) == shaderc_compilation_status_success) {
    size_t sz = shaderc_result_get_length(result);
    uint8_t *buf = malloc(sz);
    if (buf) {
      memcpy(buf, shaderc_result_get_bytes(result), sz);
      out = shader_alloc(buf, sz, SHADER_SPIRV, entry);
    }
  } else {
    LOG_WARN("shader shaderc error: %s", shaderc_result_get_error_message(result));
  }

  shaderc_result_release(result);
  shaderc_compile_options_release(opts);
  shaderc_compiler_release(compiler);
  return out;
}
#endif // HAVE_SHADERC

// ---------------------------------------------------------------------------
// GLSL compilation — glslc subprocess fallback
// ---------------------------------------------------------------------------

static compiled_shader *compile_via_glslc(const char *source, const char *entry,
                                           const char **defines, int num_defines) {
  // Write source to a temp file
  char src_path[] = "/tmp/volatile_shader_XXXXXX.comp";
  // mkstemp doesn't accept suffix; use tmpnam-style with open
  int src_fd = mkstemps(src_path, 5);  // 5 = strlen(".comp")
  if (src_fd < 0) {
    LOG_WARN("compile_via_glslc: mkstemps failed: %s", strerror(errno));
    return NULL;
  }
  FILE *sf = fdopen(src_fd, "w");
  if (!sf) { close(src_fd); return NULL; }
  fputs(source, sf);
  fclose(sf);

  // Output SPIR-V path
  char spv_path[256];
  snprintf(spv_path, sizeof(spv_path), "%s.spv", src_path);

  // Build glslc command with -D defines
  char cmd[4096];
  int pos = snprintf(cmd, sizeof(cmd), "glslc --target-env=vulkan1.3 -fshader-stage=compute");
  for (int i = 0; i < num_defines && pos < (int)sizeof(cmd) - 64; i++) {
    pos += snprintf(cmd + pos, sizeof(cmd) - (size_t)pos, " -D%s", defines[i]);
  }
  // entry point (glslc uses -fentry-point)
  if (entry && strcmp(entry, "main") != 0) {
    pos += snprintf(cmd + pos, sizeof(cmd) - (size_t)pos, " -fentry-point=%s", entry);
  }
  snprintf(cmd + pos, sizeof(cmd) - (size_t)pos, " -o %s %s 2>&1", spv_path, src_path);

  int rc = system(cmd);
  unlink(src_path);

  if (rc != 0) {
    LOG_WARN("compile_via_glslc: glslc exited %d", rc);
    unlink(spv_path);
    return NULL;
  }

  compiled_shader *out = shader_load_spirv(spv_path);
  unlink(spv_path);
  if (out) free(out->entry_point);
  if (out) out->entry_point = entry ? strdup(entry) : strdup("main");
  return out;
}

// ---------------------------------------------------------------------------
// Public compile entry-point
// ---------------------------------------------------------------------------

compiled_shader *shader_compile_glsl(const char *source, const char *entry,
                                     shader_target_t target,
                                     const char **defines, int num_defines) {
  if (!source) return NULL;
  if (!entry)  entry = "main";

  if (target == SHADER_MSL) {
    LOG_WARN("shader_compile_glsl: MSL target not yet supported (needs SPIRV-Cross)");
    return NULL;
  }
  if (target == SHADER_DXIL) {
    LOG_WARN("shader_compile_glsl: DXIL target not yet supported (needs dxc)");
    return NULL;
  }

#ifdef HAVE_SHADERC
  compiled_shader *s = compile_via_shaderc(source, entry, defines, num_defines);
  if (s) return s;
  LOG_WARN("shader_compile_glsl: shaderc failed, falling back to glslc");
#endif

  return compile_via_glslc(source, entry, defines, num_defines);
}

// ---------------------------------------------------------------------------
// Cache internals
// ---------------------------------------------------------------------------

typedef struct {
  char              glsl_path[512];
  time_t            last_mtime;
  shader_reload_fn  callback;
  void             *ctx;
  shader_cache     *owner;
} watched_file;

struct shader_cache {
  char          cache_dir[512];
  watched_file  watched[MAX_WATCHED];
  int           num_watched;
};

// Build the on-disk path for a cached SPIR-V binary.
static void cache_path(const shader_cache *c, const char key[CACHE_KEY_LEN], char *out, size_t cap) {
  snprintf(out, cap, "%s/%s.spv", c->cache_dir, key);
}

static uint64_t source_hash(const char *source, const char *entry,
                             const char **defines, int num_defines) {
  uint64_t h = fnv1a((const uint8_t *)source, strlen(source));
  if (entry) h ^= fnv1a((const uint8_t *)entry, strlen(entry));
  for (int i = 0; i < num_defines; i++)
    h ^= fnv1a((const uint8_t *)defines[i], strlen(defines[i]));
  return h;
}

// ---------------------------------------------------------------------------
// Cache public API
// ---------------------------------------------------------------------------

shader_cache *shader_cache_new(const char *cache_dir) {
  if (!cache_dir) return NULL;
  shader_cache *c = calloc(1, sizeof(*c));
  if (!c) return NULL;
  snprintf(c->cache_dir, sizeof(c->cache_dir), "%s", cache_dir);
  // Create directory if absent (best-effort, non-recursive)
  mkdir(cache_dir, 0755);
  return c;
}

void shader_cache_free(shader_cache *c) {
  free(c);
}

compiled_shader *shader_cache_get_or_compile(shader_cache *c,
                                              const char *source, const char *entry,
                                              shader_target_t target,
                                              const char **defines, int num_defines) {
  if (!c || !source) return NULL;

  uint64_t h = source_hash(source, entry, defines, num_defines);
  char key[CACHE_KEY_LEN];
  hash_to_hex(h, key);

  char path[640];
  cache_path(c, key, path, sizeof(path));

  // Cache hit?
  compiled_shader *cached = shader_load_spirv(path);
  if (cached) {
    LOG_DEBUG("shader_cache: hit %s", key);
    return cached;
  }

  // Cache miss — compile
  LOG_DEBUG("shader_cache: miss %s — compiling", key);
  compiled_shader *s = shader_compile_glsl(source, entry, target, defines, num_defines);
  if (!s) return NULL;

  // Write to cache
  FILE *f = fopen(path, "wb");
  if (f) {
    fwrite(s->code, 1, s->size, f);
    fclose(f);
  }
  return s;
}

// ---------------------------------------------------------------------------
// Hot reload
// ---------------------------------------------------------------------------

static time_t file_mtime(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return 0;
  return st.st_mtime;
}

static char *read_file(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f); rewind(f);
  if (sz <= 0) { fclose(f); return NULL; }
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';
  return buf;
}

void shader_watch(shader_cache *c, const char *glsl_path,
                  shader_reload_fn callback, void *ctx) {
  if (!c || !glsl_path || !callback) return;
  if (c->num_watched >= MAX_WATCHED) {
    LOG_WARN("shader_watch: max watched files reached");
    return;
  }
  watched_file *w = &c->watched[c->num_watched++];
  snprintf(w->glsl_path, sizeof(w->glsl_path), "%s", glsl_path);
  w->last_mtime = file_mtime(glsl_path);
  w->callback   = callback;
  w->ctx        = ctx;
  w->owner      = c;
}

void shader_poll_reloads(shader_cache *c) {
  if (!c) return;
  for (int i = 0; i < c->num_watched; i++) {
    watched_file *w = &c->watched[i];
    time_t mtime = file_mtime(w->glsl_path);
    if (mtime == 0 || mtime == w->last_mtime) continue;

    LOG_INFO("shader_poll_reloads: %s changed, recompiling", w->glsl_path);
    w->last_mtime = mtime;

    char *src = read_file(w->glsl_path);
    if (!src) { LOG_WARN("shader_poll_reloads: cannot read %s", w->glsl_path); continue; }

    compiled_shader *s = shader_compile_glsl(src, "main", SHADER_SPIRV, NULL, 0);
    free(src);
    if (!s) { LOG_WARN("shader_poll_reloads: compile failed for %s", w->glsl_path); continue; }

    w->callback(w->glsl_path, s, w->ctx);
    // NOTE: callback takes ownership of s
  }
}
