#pragma once
#include <stdbool.h>

// Forward declaration — callers need not include nuklear.h directly unless
// they call nk_* functions themselves.
struct nk_context;

// ---------------------------------------------------------------------------
// Toolbar
// ---------------------------------------------------------------------------

typedef struct toolbar toolbar;
typedef void (*toolbar_action_fn)(void *ctx);

toolbar *toolbar_new(void);
void     toolbar_free(toolbar *t);

// Add a clickable button.  icon_utf8 may be NULL.
void toolbar_add_button(toolbar *t, const char *label, const char *icon_utf8,
                        toolbar_action_fn fn, void *ctx);

// Add a visual separator (vertical gap between buttons).
void toolbar_add_separator(toolbar *t);

// Add a checkbox/toggle bound to an external bool.
void toolbar_add_toggle(toolbar *t, const char *label, bool *value);

// Render toolbar as a horizontal row inside the current Nuklear panel.
void toolbar_render(toolbar *t, struct nk_context *nk);

// Query
int toolbar_button_count(const toolbar *t);   // excludes separators

// ---------------------------------------------------------------------------
// Context menu
// ---------------------------------------------------------------------------

typedef struct context_menu context_menu;

context_menu *context_menu_new(void);
void          context_menu_free(context_menu *m);

// Add a labelled menu item.
void context_menu_add(context_menu *m, const char *label,
                      toolbar_action_fn fn, void *ctx);

// Add a visual separator.
void context_menu_add_separator(context_menu *m);

// Show the context menu popup anchored at (x, y) inside the current Nuklear
// window.  Returns true while the popup is open.  Call each frame from the
// render loop after detecting a right-click.
bool context_menu_show(context_menu *m, struct nk_context *nk,
                       float x, float y);

// Query
int context_menu_item_count(const context_menu *m);  // excludes separators
