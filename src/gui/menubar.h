// WIDGET TYPE: CONTENT — call inside an nk_begin/nk_end block.
#pragma once
#include <stdbool.h>

// Forward-declare nk_context to avoid pulling nuklear.h into consumers.
struct nk_context;

// ---------------------------------------------------------------------------
// menubar — VC3D-style menu bar (File / Edit / View / Selection / Help)
// ---------------------------------------------------------------------------

typedef struct menubar menubar;
typedef void (*menu_action_fn)(void *ctx);

menubar *menubar_new(void);
void     menubar_free(menubar *m);

// Render the menu bar using Nuklear. Call once at the top of each frame,
// inside an nk_begin / nk_end block that spans the full window width.
void menubar_render(menubar *m, struct nk_context *ctx);

// ---------------------------------------------------------------------------
// Action callbacks — register before the first render call.
// fn may be NULL (item renders but does nothing when clicked).
// ---------------------------------------------------------------------------

// File
void menubar_on_open_volpkg(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_open_zarr(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_open_remote(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_attach_remote_zarr(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_generate_report(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_settings(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_import_obj(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_exit(menubar *m, menu_action_fn fn, void *ctx);

// View
void menubar_on_toggle_volumes(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_toggle_segmentation(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_toggle_distance_transform(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_toggle_drawing(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_toggle_viewer_controls(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_toggle_point_collection(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_sync_cursor(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_reset_seg_views(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_show_console(menubar *m, menu_action_fn fn, void *ctx);

// Selection
void menubar_on_draw_bbox(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_surface_from_selection(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_clear_selection(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_inpaint_rebuild(menubar *m, menu_action_fn fn, void *ctx);

// Help
void menubar_on_keybinds(menubar *m, menu_action_fn fn, void *ctx);
void menubar_on_about(menubar *m, menu_action_fn fn, void *ctx);

// Recent files (shown in File → Open recent submenus)
void menubar_add_recent(menubar *m, const char *path);
