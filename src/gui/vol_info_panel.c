#include "gui/vol_info_panel.h"
#include "core/vol.h"
#include "core/cache.h"
#include "core/log.h"
#include "core/io.h"

#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#include <nuklear.h>

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct vol_info_panel {
  bool expanded_levels;  // whether the pyramid level list is expanded
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void fmt_bytes(char *buf, int bufsz, size_t bytes) {
  if (bytes >= (size_t)1 << 30)
    snprintf(buf, (size_t)bufsz, "%.1f GB", (double)bytes / (1 << 30));
  else if (bytes >= (size_t)1 << 20)
    snprintf(buf, (size_t)bufsz, "%.1f MB", (double)bytes / (1 << 20));
  else if (bytes >= (size_t)1 << 10)
    snprintf(buf, (size_t)bufsz, "%.1f KB", (double)bytes / (1 << 10));
  else
    snprintf(buf, (size_t)bufsz, "%zu B", bytes);
}

static const char *dtype_label(int dtype) {
  switch (dtype) {
    case DTYPE_U8:  return "uint8";
    case DTYPE_U16: return "uint16";
    case DTYPE_F32: return "float32";
    case DTYPE_F64: return "float64";
    default:        return "unknown";
  }
}

// Render a two-column label/value row.
static void row_kv(struct nk_context *ctx, const char *key, const char *val) {
  nk_layout_row_dynamic(ctx, 18, 2);
  nk_label(ctx, key, NK_TEXT_LEFT);
  nk_label(ctx, val, NK_TEXT_LEFT);
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

vol_info_panel *vol_info_panel_new(void) {
  vol_info_panel *p = calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->expanded_levels = true;
  return p;
}

void vol_info_panel_free(vol_info_panel *p) {
  free(p);
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

bool vol_info_panel_render(vol_info_panel *p, struct nk_context *ctx,
                           volume *vol, chunk_cache *cache) {
  bool refreshed = false;
  char buf[128];

  // --- Header ---
  nk_layout_row_dynamic(ctx, 22, 1);
  nk_label(ctx, "Volume Info", NK_TEXT_LEFT);

  nk_layout_row_dynamic(ctx, 1, 1);
  nk_rule_horizontal(ctx, ctx->style.window.border_color, false);

  if (!vol) {
    nk_layout_row_dynamic(ctx, 20, 1);
    nk_label(ctx, "No volume loaded.", NK_TEXT_LEFT);
    nk_layout_row_dynamic(ctx, 24, 1);
    refreshed = nk_button_label(ctx, "Refresh");
    return refreshed;
  }

  // --- Volume identity ---
  const char *path = vol_path(vol);
  // Show only the last path component as the name
  const char *name = path ? strrchr(path, '/') : NULL;
  name = name ? name + 1 : (path ? path : "(unknown)");

  row_kv(ctx, "Name:",   name);
  row_kv(ctx, "Source:", vol_is_remote(vol) ? "remote" : "local");
  if (path) row_kv(ctx, "Path:",   path);

  nk_layout_row_dynamic(ctx, 6, 1);
  nk_spacing(ctx, 1);

  // --- Shape of level 0 ---
  int nlevels = vol_num_levels(vol);
  snprintf(buf, sizeof(buf), "%d", nlevels);
  row_kv(ctx, "Pyramid levels:", buf);

  const zarr_level_meta *m0 = vol_level_meta(vol, 0);
  if (m0) {
    if (m0->ndim == 3)
      snprintf(buf, sizeof(buf), "%lld x %lld x %lld",
               (long long)m0->shape[0], (long long)m0->shape[1], (long long)m0->shape[2]);
    else if (m0->ndim == 2)
      snprintf(buf, sizeof(buf), "%lld x %lld",
               (long long)m0->shape[0], (long long)m0->shape[1]);
    else
      snprintf(buf, sizeof(buf), "ndim=%d", m0->ndim);
    row_kv(ctx, "Dimensions:", buf);
    row_kv(ctx, "Dtype:", dtype_label(m0->dtype));
    snprintf(buf, sizeof(buf), "v%d%s", m0->zarr_version, m0->sharded ? " (sharded)" : "");
    row_kv(ctx, "Zarr version:", buf);

    if (m0->compressor_id[0]) {
      snprintf(buf, sizeof(buf), "%s/%s (l%d)",
               m0->compressor_id, m0->compressor_cname, m0->compressor_clevel);
      row_kv(ctx, "Codec:", buf);
    } else {
      row_kv(ctx, "Codec:", "none");
    }
  }

  nk_layout_row_dynamic(ctx, 6, 1);
  nk_spacing(ctx, 1);

  // --- Pyramid levels (collapsible) ---
  nk_layout_row_dynamic(ctx, 20, 1);
  if (nk_tree_push(ctx, NK_TREE_TAB, "Pyramid Levels", p->expanded_levels ? NK_MAXIMIZED : NK_MINIMIZED)) {
    p->expanded_levels = true;
    for (int lvl = 0; lvl < nlevels; lvl++) {
      const zarr_level_meta *m = vol_level_meta(vol, lvl);
      if (!m) continue;
      char key[32];
      snprintf(key, sizeof(key), "  Level %d:", lvl);
      if (m->ndim == 3)
        snprintf(buf, sizeof(buf), "%lld x %lld x %lld  [%lld x %lld x %lld chunks]",
                 (long long)m->shape[0], (long long)m->shape[1], (long long)m->shape[2],
                 (long long)m->chunk_shape[0], (long long)m->chunk_shape[1], (long long)m->chunk_shape[2]);
      else
        snprintf(buf, sizeof(buf), "ndim=%d", m->ndim);
      row_kv(ctx, key, buf);
    }
    nk_tree_pop(ctx);
  } else {
    p->expanded_levels = false;
  }

  nk_layout_row_dynamic(ctx, 6, 1);
  nk_spacing(ctx, 1);

  // --- Cache stats ---
  if (cache) {
    nk_layout_row_dynamic(ctx, 20, 1);
    nk_label(ctx, "Cache:", NK_TEXT_LEFT);

    char hot_s[32], warm_s[32];
    fmt_bytes(hot_s,  sizeof(hot_s),  cache_hot_bytes(cache));
    fmt_bytes(warm_s, sizeof(warm_s), cache_warm_bytes(cache));
    row_kv(ctx, "  Hot (RAM):",  hot_s);
    row_kv(ctx, "  Warm (RAM):", warm_s);

    size_t hits   = cache_hits(cache);
    size_t misses = cache_misses(cache);
    size_t total  = hits + misses;
    if (total > 0)
      snprintf(buf, sizeof(buf), "%zu / %zu  (%.0f%%)", hits, total,
               (double)hits * 100.0 / (double)total);
    else
      snprintf(buf, sizeof(buf), "no requests");
    row_kv(ctx, "  Hit rate:", buf);
  }

  nk_layout_row_dynamic(ctx, 6, 1);
  nk_spacing(ctx, 1);

  // --- Refresh button ---
  nk_layout_row_dynamic(ctx, 24, 1);
  refreshed = nk_button_label(ctx, "Refresh");
  return refreshed;
}
