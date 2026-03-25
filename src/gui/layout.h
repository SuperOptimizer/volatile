#pragma once
#include <stdbool.h>

struct nk_context;

// ---------------------------------------------------------------------------
// Panel identifiers — order matches default layout slots
// ---------------------------------------------------------------------------
typedef enum {
  PANEL_VIEWER_XY,
  PANEL_VIEWER_XZ,
  PANEL_VIEWER_YZ,
  PANEL_VIEWER_3D,
  PANEL_SURFACE_TREE,
  PANEL_SEGMENTATION,
  PANEL_SETTINGS,
  PANEL_CONSOLE,
  PANEL_ANNOTATIONS,
  PANEL_VOLUME_BROWSER,
  PANEL_COUNT,
} panel_id;

// ---------------------------------------------------------------------------
// Panel geometry — normalized coordinates [0,1] within the window
// ---------------------------------------------------------------------------
typedef struct {
  panel_id    id;
  float       x, y, w, h;  // normalized 0-1 within window
  bool        visible;
  const char *title;
} panel_rect;

// ---------------------------------------------------------------------------
// Opaque layout state
// ---------------------------------------------------------------------------
typedef struct app_layout app_layout;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------
app_layout *layout_new_default(void);
void        layout_free(app_layout *l);

// ---------------------------------------------------------------------------
// Panel access
// ---------------------------------------------------------------------------
panel_rect layout_get_panel(const app_layout *l, panel_id id);
void       layout_set_panel(app_layout *l, panel_id id, panel_rect rect);
void       layout_toggle_panel(app_layout *l, panel_id id);

// ---------------------------------------------------------------------------
// Preset layouts
// ---------------------------------------------------------------------------
void layout_preset_vc3d(app_layout *l);    // 4 viewers + side panel
void layout_preset_single(app_layout *l);  // 1 viewer + side panel
void layout_preset_quad(app_layout *l);    // 4 viewers, no side panel

// ---------------------------------------------------------------------------
// Rendering and hit-testing
// ---------------------------------------------------------------------------
void     layout_render(app_layout *l, struct nk_context *ctx,
                       int window_w, int window_h);
panel_id layout_hit_test(const app_layout *l, float sx, float sy,
                          int window_w, int window_h);

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------
bool        layout_save(const app_layout *l, const char *path);
app_layout *layout_load(const char *path);
