#pragma once
#include <stdbool.h>
#include <stdint.h>

// Forward-declare nk_context to avoid pulling nuklear.h into consumers.
struct nk_context;

// ---------------------------------------------------------------------------
// Range slider: two handles for window/level control (e.g. contrast range).
// Returns true if either handle value changed.
// ---------------------------------------------------------------------------
bool nk_widget_range_slider(struct nk_context *ctx,
                            float *min_val, float *max_val,
                            float lo, float hi, float step);

// ---------------------------------------------------------------------------
// Collapsible section header (like Qt's CollapsibleSettingsGroup).
// Returns true while the section is expanded; caller must call _end when open.
// ---------------------------------------------------------------------------
bool nk_widget_collapsible_begin(struct nk_context *ctx, const char *title, bool *expanded);
void nk_widget_collapsible_end(struct nk_context *ctx);

// ---------------------------------------------------------------------------
// Labeled value display: left-aligned label, right-aligned value.
// ---------------------------------------------------------------------------
void nk_widget_labeled_float(struct nk_context *ctx, const char *label, float value, const char *fmt);
void nk_widget_labeled_int(struct nk_context *ctx, const char *label, int value);
void nk_widget_labeled_str(struct nk_context *ctx, const char *label, const char *value);

// ---------------------------------------------------------------------------
// Color swatch: small filled rectangle for colormap preview.
// ---------------------------------------------------------------------------
void nk_widget_color_swatch(struct nk_context *ctx,
                             uint8_t r, uint8_t g, uint8_t b,
                             float width, float height);

// ---------------------------------------------------------------------------
// Dropdown with incremental search/filter.
// filter_buf/filter_len: caller-managed string buffer for the search field.
// Returns true if selection changed.
// ---------------------------------------------------------------------------
bool nk_widget_searchable_combo(struct nk_context *ctx,
                                const char **items, int count, int *selected,
                                char *filter_buf, int filter_len);

// ---------------------------------------------------------------------------
// Progress bar with an overlaid text label.
// progress in [0,1].
// ---------------------------------------------------------------------------
void nk_widget_progress_labeled(struct nk_context *ctx, const char *label, float progress);

// ---------------------------------------------------------------------------
// Coordinate display: shows (x, y, z) formatted with fmt (e.g. "%.1f").
// ---------------------------------------------------------------------------
void nk_widget_coord_display(struct nk_context *ctx, float x, float y, float z, const char *fmt);
