// NK_IMPLEMENTATION is defined in exactly one translation unit (app.c).
// Here we only need the declarations.
#include "nuklear.h"

#include "gui/toolbar.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Item types
// ---------------------------------------------------------------------------

typedef enum {
  ITEM_BUTTON,
  ITEM_TOGGLE,
  ITEM_SEPARATOR,
} item_type;

typedef struct {
  item_type         type;
  char              label[64];
  char              icon[8];    // UTF-8 icon glyph, may be empty
  toolbar_action_fn fn;
  void             *fn_ctx;
  bool             *toggle_value;  // ITEM_TOGGLE only
} toolbar_item;

// ---------------------------------------------------------------------------
// toolbar
// ---------------------------------------------------------------------------

#define TOOLBAR_INIT_CAP  8
#define BUTTON_WIDTH      80
#define SEPARATOR_WIDTH   8

struct toolbar {
  toolbar_item *items;
  int           count;
  int           cap;
};

toolbar *toolbar_new(void) {
  toolbar *t = calloc(1, sizeof(*t));
  REQUIRE(t, "toolbar_new: calloc failed");
  t->items = calloc(TOOLBAR_INIT_CAP, sizeof(toolbar_item));
  REQUIRE(t->items, "toolbar_new: calloc items failed");
  t->cap = TOOLBAR_INIT_CAP;
  return t;
}

void toolbar_free(toolbar *t) {
  if (!t) return;
  free(t->items);
  free(t);
}

static toolbar_item *toolbar_push(toolbar *t) {
  if (t->count == t->cap) {
    int new_cap = t->cap * 2;
    toolbar_item *newbuf = realloc(t->items, (size_t)new_cap * sizeof(toolbar_item));
    REQUIRE(newbuf, "toolbar_push: realloc failed");
    t->items = newbuf;
    t->cap   = new_cap;
  }
  toolbar_item *item = &t->items[t->count++];
  memset(item, 0, sizeof(*item));
  return item;
}

void toolbar_add_button(toolbar *t, const char *label, const char *icon_utf8,
                        toolbar_action_fn fn, void *ctx) {
  REQUIRE(t && label, "toolbar_add_button: null argument");
  toolbar_item *item = toolbar_push(t);
  item->type   = ITEM_BUTTON;
  item->fn     = fn;
  item->fn_ctx = ctx;
  snprintf(item->label, sizeof(item->label), "%s", label);
  if (icon_utf8)
    snprintf(item->icon, sizeof(item->icon), "%s", icon_utf8);
}

void toolbar_add_separator(toolbar *t) {
  REQUIRE(t, "toolbar_add_separator: null toolbar");
  toolbar_push(t)->type = ITEM_SEPARATOR;
}

void toolbar_add_toggle(toolbar *t, const char *label, bool *value) {
  REQUIRE(t && label && value, "toolbar_add_toggle: null argument");
  toolbar_item *item = toolbar_push(t);
  item->type         = ITEM_TOGGLE;
  item->toggle_value = value;
  snprintf(item->label, sizeof(item->label), "%s", label);
}

int toolbar_button_count(const toolbar *t) {
  REQUIRE(t, "toolbar_button_count: null toolbar");
  int n = 0;
  for (int i = 0; i < t->count; i++)
    if (t->items[i].type != ITEM_SEPARATOR) n++;
  return n;
}

void toolbar_render(toolbar *t, struct nk_context *nk) {
  if (!t || !nk || t->count == 0) return;

  // One row with fixed-width cells; separator = thin spacer.
  nk_layout_row_begin(nk, NK_STATIC, 28, t->count);
  for (int i = 0; i < t->count; i++) {
    toolbar_item *item = &t->items[i];
    switch (item->type) {
      case ITEM_SEPARATOR:
        nk_layout_row_push(nk, SEPARATOR_WIDTH);
        nk_spacing(nk, 1);
        break;
      case ITEM_BUTTON: {
        nk_layout_row_push(nk, BUTTON_WIDTH);
        char buf[72];
        if (item->icon[0])
          snprintf(buf, sizeof(buf), "%s %s", item->icon, item->label);
        else
          snprintf(buf, sizeof(buf), "%s", item->label);
        if (nk_button_label(nk, buf) && item->fn)
          item->fn(item->fn_ctx);
        break;
      }
      case ITEM_TOGGLE: {
        nk_layout_row_push(nk, BUTTON_WIDTH);
        int val = item->toggle_value ? (int)*item->toggle_value : 0;
        if (nk_checkbox_label(nk, item->label, &val) && item->toggle_value)
          *item->toggle_value = (bool)val;
        break;
      }
    }
  }
  nk_layout_row_end(nk);
}

// ---------------------------------------------------------------------------
// context_menu
// ---------------------------------------------------------------------------

#define MENU_INIT_CAP 8
#define MENU_ITEM_H   22
#define MENU_WIDTH    180

typedef struct {
  item_type         type;
  char              label[64];
  toolbar_action_fn fn;
  void             *fn_ctx;
} menu_item;

struct context_menu {
  menu_item *items;
  int        count;
  int        cap;
};

context_menu *context_menu_new(void) {
  context_menu *m = calloc(1, sizeof(*m));
  REQUIRE(m, "context_menu_new: calloc failed");
  m->items = calloc(MENU_INIT_CAP, sizeof(menu_item));
  REQUIRE(m->items, "context_menu_new: calloc items failed");
  m->cap = MENU_INIT_CAP;
  return m;
}

void context_menu_free(context_menu *m) {
  if (!m) return;
  free(m->items);
  free(m);
}

static menu_item *menu_push(context_menu *m) {
  if (m->count == m->cap) {
    int new_cap = m->cap * 2;
    menu_item *nb = realloc(m->items, (size_t)new_cap * sizeof(menu_item));
    REQUIRE(nb, "menu_push: realloc failed");
    m->items = nb;
    m->cap   = new_cap;
  }
  menu_item *item = &m->items[m->count++];
  memset(item, 0, sizeof(*item));
  return item;
}

void context_menu_add(context_menu *m, const char *label,
                      toolbar_action_fn fn, void *ctx) {
  REQUIRE(m && label, "context_menu_add: null argument");
  menu_item *item = menu_push(m);
  item->type   = ITEM_BUTTON;
  item->fn     = fn;
  item->fn_ctx = ctx;
  snprintf(item->label, sizeof(item->label), "%s", label);
}

void context_menu_add_separator(context_menu *m) {
  REQUIRE(m, "context_menu_add_separator: null menu");
  menu_push(m)->type = ITEM_SEPARATOR;
}

int context_menu_item_count(const context_menu *m) {
  REQUIRE(m, "context_menu_item_count: null menu");
  int n = 0;
  for (int i = 0; i < m->count; i++)
    if (m->items[i].type != ITEM_SEPARATOR) n++;
  return n;
}

bool context_menu_show(context_menu *m, struct nk_context *nk,
                       float x, float y) {
  if (!m || !nk || m->count == 0) return false;

  // Compute total height: each item is MENU_ITEM_H px, separators are 4 px.
  float height = 0;
  for (int i = 0; i < m->count; i++)
    height += (m->items[i].type == ITEM_SEPARATOR) ? 4.0f : (float)MENU_ITEM_H;

  struct nk_rect bounds = nk_rect(x, y, MENU_WIDTH, height);
  if (!nk_contextual_begin(nk, 0, nk_vec2(MENU_WIDTH, height), bounds))
    return false;

  for (int i = 0; i < m->count; i++) {
    menu_item *item = &m->items[i];
    if (item->type == ITEM_SEPARATOR) {
      nk_layout_row_dynamic(nk, 4, 1);
      nk_spacing(nk, 1);
    } else {
      nk_layout_row_dynamic(nk, MENU_ITEM_H, 1);
      if (nk_contextual_item_label(nk, item->label, NK_TEXT_LEFT) && item->fn) {
        item->fn(item->fn_ctx);
        nk_contextual_end(nk);
        return false;  // closed after selection
      }
    }
  }

  nk_contextual_end(nk);
  return true;
}
