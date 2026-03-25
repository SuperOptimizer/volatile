// ---------------------------------------------------------------------------
// layout.c — configurable panel layout system (VC3D-style)
//
// Default layout:
//   +-------------------+-------------------+-------------+
//   |    XY Viewer      |    XZ Viewer      |  Surface    |
//   |                   |                   |  Tree       |
//   +-------------------+-------------------+             |
//   |    YZ Viewer      |    3D Viewer      |  Segment.   |
//   |                   |                   |  Tools      |
//   +-------------------+-------------------+-------------+
//   |                 Console                             |
//   +----------------------------------------------------+
// ---------------------------------------------------------------------------

#include "gui/layout.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Nuklear — already implemented by app.c; only declarations needed here.
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#include <nuklear.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define SIDE_W    0.20f   // side panel width fraction
#define CONSOLE_H 0.15f   // console height fraction

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
struct app_layout {
  panel_rect panels[PANEL_COUNT];
};

// ---------------------------------------------------------------------------
// Static panel title table
// ---------------------------------------------------------------------------
static const char *const k_titles[PANEL_COUNT] = {
  [PANEL_VIEWER_XY]      = "XY",
  [PANEL_VIEWER_XZ]      = "XZ",
  [PANEL_VIEWER_YZ]      = "YZ",
  [PANEL_VIEWER_3D]      = "3D",
  [PANEL_SURFACE_TREE]   = "Surface Tree",
  [PANEL_SEGMENTATION]   = "Segmentation",
  [PANEL_SETTINGS]       = "Settings",
  [PANEL_CONSOLE]        = "Console",
  [PANEL_ANNOTATIONS]    = "Annotations",
  [PANEL_VOLUME_BROWSER] = "Volume Browser",
};

// ---------------------------------------------------------------------------
// layout_preset_vc3d
// ---------------------------------------------------------------------------
void layout_preset_vc3d(app_layout *l) {
  if (!l) return;

  // Derived fractions
  const float view_w  = (1.0f - SIDE_W) * 0.5f;
  const float view_h  = (1.0f - CONSOLE_H) * 0.5f;
  const float side_x  = 1.0f - SIDE_W;
  const float cons_y  = 1.0f - CONSOLE_H;

  l->panels[PANEL_VIEWER_XY] = (panel_rect){
    PANEL_VIEWER_XY, 0.0f, 0.0f, view_w, view_h, true,
    k_titles[PANEL_VIEWER_XY]
  };
  l->panels[PANEL_VIEWER_XZ] = (panel_rect){
    PANEL_VIEWER_XZ, view_w, 0.0f, view_w, view_h, true,
    k_titles[PANEL_VIEWER_XZ]
  };
  l->panels[PANEL_VIEWER_YZ] = (panel_rect){
    PANEL_VIEWER_YZ, 0.0f, view_h, view_w, view_h, true,
    k_titles[PANEL_VIEWER_YZ]
  };
  l->panels[PANEL_VIEWER_3D] = (panel_rect){
    PANEL_VIEWER_3D, view_w, view_h, view_w, view_h, true,
    k_titles[PANEL_VIEWER_3D]
  };

  // Side panel: Surface Tree occupies top half, Segmentation the bottom
  const float side_half = (1.0f - CONSOLE_H) * 0.5f;
  l->panels[PANEL_SURFACE_TREE] = (panel_rect){
    PANEL_SURFACE_TREE, side_x, 0.0f, SIDE_W, side_half, true,
    k_titles[PANEL_SURFACE_TREE]
  };
  l->panels[PANEL_SEGMENTATION] = (panel_rect){
    PANEL_SEGMENTATION, side_x, side_half, SIDE_W, side_half, true,
    k_titles[PANEL_SEGMENTATION]
  };

  // Console spans full width at the bottom
  l->panels[PANEL_CONSOLE] = (panel_rect){
    PANEL_CONSOLE, 0.0f, cons_y, 1.0f, CONSOLE_H, true,
    k_titles[PANEL_CONSOLE]
  };

  // Hidden panels — zero geometry, invisible
  const panel_id hidden[] = {
    PANEL_SETTINGS, PANEL_ANNOTATIONS, PANEL_VOLUME_BROWSER
  };
  for (int i = 0; i < (int)(sizeof(hidden)/sizeof(hidden[0])); i++) {
    panel_id pid = hidden[i];
    l->panels[pid] = (panel_rect){
      pid, 0.0f, 0.0f, 0.0f, 0.0f, false, k_titles[pid]
    };
  }
}

// ---------------------------------------------------------------------------
// layout_preset_single
// ---------------------------------------------------------------------------
void layout_preset_single(app_layout *l) {
  if (!l) return;
  const float view_w = 1.0f - SIDE_W;
  const float view_h = 1.0f - CONSOLE_H;

  l->panels[PANEL_VIEWER_XY] = (panel_rect){
    PANEL_VIEWER_XY, 0.0f, 0.0f, view_w, view_h, true,
    k_titles[PANEL_VIEWER_XY]
  };
  l->panels[PANEL_SURFACE_TREE] = (panel_rect){
    PANEL_SURFACE_TREE, view_w, 0.0f, SIDE_W, view_h * 0.5f, true,
    k_titles[PANEL_SURFACE_TREE]
  };
  l->panels[PANEL_SEGMENTATION] = (panel_rect){
    PANEL_SEGMENTATION, view_w, view_h * 0.5f, SIDE_W, view_h * 0.5f, true,
    k_titles[PANEL_SEGMENTATION]
  };
  l->panels[PANEL_CONSOLE] = (panel_rect){
    PANEL_CONSOLE, 0.0f, view_h, 1.0f, CONSOLE_H, true,
    k_titles[PANEL_CONSOLE]
  };

  const panel_id hidden[] = {
    PANEL_VIEWER_XZ, PANEL_VIEWER_YZ, PANEL_VIEWER_3D,
    PANEL_SETTINGS, PANEL_ANNOTATIONS, PANEL_VOLUME_BROWSER
  };
  for (int i = 0; i < (int)(sizeof(hidden)/sizeof(hidden[0])); i++) {
    panel_id pid = hidden[i];
    l->panels[pid] = (panel_rect){
      pid, 0.0f, 0.0f, 0.0f, 0.0f, false, k_titles[pid]
    };
  }
}

// ---------------------------------------------------------------------------
// layout_preset_quad
// ---------------------------------------------------------------------------
void layout_preset_quad(app_layout *l) {
  if (!l) return;
  const float hw = 0.5f;
  const float hh = 1.0f - CONSOLE_H;

  l->panels[PANEL_VIEWER_XY] = (panel_rect){
    PANEL_VIEWER_XY, 0.0f, 0.0f, hw, hh * 0.5f, true, k_titles[PANEL_VIEWER_XY]
  };
  l->panels[PANEL_VIEWER_XZ] = (panel_rect){
    PANEL_VIEWER_XZ, hw, 0.0f, hw, hh * 0.5f, true, k_titles[PANEL_VIEWER_XZ]
  };
  l->panels[PANEL_VIEWER_YZ] = (panel_rect){
    PANEL_VIEWER_YZ, 0.0f, hh * 0.5f, hw, hh * 0.5f, true, k_titles[PANEL_VIEWER_YZ]
  };
  l->panels[PANEL_VIEWER_3D] = (panel_rect){
    PANEL_VIEWER_3D, hw, hh * 0.5f, hw, hh * 0.5f, true, k_titles[PANEL_VIEWER_3D]
  };
  l->panels[PANEL_CONSOLE] = (panel_rect){
    PANEL_CONSOLE, 0.0f, hh, 1.0f, CONSOLE_H, true, k_titles[PANEL_CONSOLE]
  };

  const panel_id hidden[] = {
    PANEL_SURFACE_TREE, PANEL_SEGMENTATION,
    PANEL_SETTINGS, PANEL_ANNOTATIONS, PANEL_VOLUME_BROWSER
  };
  for (int i = 0; i < (int)(sizeof(hidden)/sizeof(hidden[0])); i++) {
    panel_id pid = hidden[i];
    l->panels[pid] = (panel_rect){
      pid, 0.0f, 0.0f, 0.0f, 0.0f, false, k_titles[pid]
    };
  }
}

// ---------------------------------------------------------------------------
// layout_new_default / layout_free
// ---------------------------------------------------------------------------
app_layout *layout_new_default(void) {
  app_layout *l = calloc(1, sizeof(*l));
  if (!l) return NULL;
  layout_preset_vc3d(l);
  return l;
}

void layout_free(app_layout *l) {
  free(l);
}

// ---------------------------------------------------------------------------
// Panel access
// ---------------------------------------------------------------------------
panel_rect layout_get_panel(const app_layout *l, panel_id id) {
  if (!l || id < 0 || id >= PANEL_COUNT) {
    return (panel_rect){0};
  }
  return l->panels[id];
}

void layout_set_panel(app_layout *l, panel_id id, panel_rect rect) {
  if (!l || id < 0 || id >= PANEL_COUNT) return;
  rect.id    = id;
  rect.title = k_titles[id];  // keep canonical title
  l->panels[id] = rect;
}

void layout_toggle_panel(app_layout *l, panel_id id) {
  if (!l || id < 0 || id >= PANEL_COUNT) return;
  l->panels[id].visible = !l->panels[id].visible;
}

// ---------------------------------------------------------------------------
// layout_render
// ---------------------------------------------------------------------------
void layout_render(app_layout *l, struct nk_context *ctx,
                   int window_w, int window_h) {
  if (!l || !ctx || window_w <= 0 || window_h <= 0) return;

  const float fw = (float)window_w;
  const float fh = (float)window_h;

  for (int i = 0; i < PANEL_COUNT; i++) {
    const panel_rect *p = &l->panels[i];
    if (!p->visible || p->w <= 0.0f || p->h <= 0.0f) continue;

    const float px = p->x * fw;
    const float py = p->y * fh;
    const float pw = p->w * fw;
    const float ph = p->h * fh;

    if (nk_begin(ctx, p->title,
                 nk_rect(px, py, pw, ph),
                 NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_NO_SCROLLBAR)) {
      // Panel content is filled by the caller; layout only sets the frame.
    }
    nk_end(ctx);
  }
}

// ---------------------------------------------------------------------------
// layout_hit_test
// ---------------------------------------------------------------------------
panel_id layout_hit_test(const app_layout *l, float sx, float sy,
                          int window_w, int window_h) {
  if (!l || window_w <= 0 || window_h <= 0) return (panel_id)-1;

  const float nx = sx / (float)window_w;
  const float ny = sy / (float)window_h;

  for (int i = 0; i < PANEL_COUNT; i++) {
    const panel_rect *p = &l->panels[i];
    if (!p->visible || p->w <= 0.0f || p->h <= 0.0f) continue;
    if (nx >= p->x && nx < p->x + p->w &&
        ny >= p->y && ny < p->y + p->h) {
      return p->id;
    }
  }
  return (panel_id)-1;
}

// ---------------------------------------------------------------------------
// layout_save
// ---------------------------------------------------------------------------
bool layout_save(const app_layout *l, const char *path) {
  if (!l || !path) return false;

  FILE *f = fopen(path, "w");
  if (!f) {
    LOG_ERROR("layout_save: cannot open %s", path);
    return false;
  }

  fprintf(f, "{\n  \"panels\": [\n");
  for (int i = 0; i < PANEL_COUNT; i++) {
    const panel_rect *p = &l->panels[i];
    fprintf(f,
      "    {\"id\":%d,\"x\":%.6f,\"y\":%.6f,"
      "\"w\":%.6f,\"h\":%.6f,\"visible\":%s}%s\n",
      p->id, p->x, p->y, p->w, p->h,
      p->visible ? "true" : "false",
      (i < PANEL_COUNT - 1) ? "," : "");
  }
  fprintf(f, "  ]\n}\n");
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// layout_load — minimal hand-rolled parser to avoid pulling in json.c dep
// ---------------------------------------------------------------------------
app_layout *layout_load(const char *path) {
  if (!path) return NULL;

  FILE *f = fopen(path, "r");
  if (!f) {
    LOG_ERROR("layout_load: cannot open %s", path);
    return NULL;
  }

  // Slurp file
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0 || sz > 1024 * 64) { fclose(f); return NULL; }

  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[sz] = '\0';

  app_layout *l = calloc(1, sizeof(*l));
  if (!l) { free(buf); return NULL; }

  // Defaults first so any missing panels stay sane
  layout_preset_vc3d(l);

  // Parse each panel object: {"id":N,"x":F,"y":F,"w":F,"h":F,"visible":B}
  const char *cur = buf;
  while ((cur = strstr(cur, "\"id\":")) != NULL) {
    int   id;
    float x, y, w, h;
    char  vis[8];
    if (sscanf(cur, "\"id\":%d,\"x\":%f,\"y\":%f,\"w\":%f,\"h\":%f,\"visible\":%7s",
               &id, &x, &y, &w, &h, vis) == 6) {
      if (id >= 0 && id < PANEL_COUNT) {
        l->panels[id].id      = (panel_id)id;
        l->panels[id].x       = x;
        l->panels[id].y       = y;
        l->panels[id].w       = w;
        l->panels[id].h       = h;
        l->panels[id].visible = (strncmp(vis, "true", 4) == 0);
        l->panels[id].title   = k_titles[id];
      }
    }
    cur++;
  }

  free(buf);
  return l;
}
