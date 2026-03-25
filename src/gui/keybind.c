#include "keybind.h"
#include "core/json.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// SDL3 scancode constants (numeric to avoid SDL3 header dependency in tests)
// ---------------------------------------------------------------------------
#define SC_A       4
#define SC_B       5
#define SC_D       7
#define SC_L      15
#define SC_S      22
#define SC_W      26
#define SC_Y      28
#define SC_Z      29
#define SC_MINUS  45
#define SC_EQUALS 46
#define SC_COMMA  54
#define SC_PERIOD 55
#define SC_F11    68
#define SC_RIGHT  79
#define SC_LEFT   80
#define SC_DOWN   81
#define SC_UP     82
#define SC_G      10
#define SC_P      19

// SDL3 modifier masks (SDL_KMOD_LCTRL | SDL_KMOD_RCTRL combined = 0x00C0)
#define MOD_NONE  0x0000
#define MOD_CTRL  0x00C0  // either Ctrl
#define MOD_SHIFT 0x0003  // either Shift

// ---------------------------------------------------------------------------
// Internal binding entry (one scancode+mod per action)
// ---------------------------------------------------------------------------

typedef struct {
  int scancode;
  int modifiers;
} binding;

struct keybind_map {
  binding bindings[ACTION_COUNT];
};

// ---------------------------------------------------------------------------
// Defaults (VC3D-compatible)
// ---------------------------------------------------------------------------

static void set_defaults(keybind_map *m) {
  // Pan: WASD and arrow keys — store primary binding per action
  m->bindings[ACTION_PAN_LEFT]  = (binding){ SC_A,      MOD_NONE };
  m->bindings[ACTION_PAN_RIGHT] = (binding){ SC_D,      MOD_NONE };
  m->bindings[ACTION_PAN_UP]    = (binding){ SC_W,      MOD_NONE };
  m->bindings[ACTION_PAN_DOWN]  = (binding){ SC_S,      MOD_NONE };

  m->bindings[ACTION_ZOOM_IN]   = (binding){ SC_EQUALS, MOD_NONE };  // + / =
  m->bindings[ACTION_ZOOM_OUT]  = (binding){ SC_MINUS,  MOD_NONE };

  m->bindings[ACTION_SLICE_NEXT] = (binding){ SC_PERIOD, MOD_NONE };  // .
  m->bindings[ACTION_SLICE_PREV] = (binding){ SC_COMMA,  MOD_NONE };  // ,

  m->bindings[ACTION_TOOL_BRUSH]    = (binding){ SC_B, MOD_NONE };
  m->bindings[ACTION_TOOL_LINE]     = (binding){ SC_L, MOD_NONE };
  m->bindings[ACTION_TOOL_PUSHPULL] = (binding){ SC_P, MOD_NONE };

  m->bindings[ACTION_UNDO] = (binding){ SC_Z, MOD_CTRL };
  m->bindings[ACTION_REDO] = (binding){ SC_Y, MOD_CTRL };

  m->bindings[ACTION_FOCUS_NEXT] = (binding){ SC_RIGHT, MOD_NONE };
  m->bindings[ACTION_FOCUS_PREV] = (binding){ SC_LEFT,  MOD_NONE };

  m->bindings[ACTION_TOGGLE_OVERLAY]      = (binding){ SC_F11, MOD_NONE };
  m->bindings[ACTION_TOGGLE_SEGMENTATION] = (binding){ SC_G,   MOD_CTRL };  // Ctrl+G

  m->bindings[ACTION_SAVE]       = (binding){ SC_S,   MOD_CTRL };
  m->bindings[ACTION_FULLSCREEN] = (binding){ SC_F11, MOD_NONE };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

keybind_map *keybind_new(void) {
  keybind_map *m = malloc(sizeof(keybind_map));
  if (!m) return NULL;
  set_defaults(m);
  return m;
}

void keybind_free(keybind_map *m) {
  free(m);
}

// ---------------------------------------------------------------------------
// Set / lookup
// ---------------------------------------------------------------------------

void keybind_set(keybind_map *m, action_id action, int sdl_scancode, int modifiers) {
  if (action < 0 || action >= ACTION_COUNT) return;
  m->bindings[action].scancode  = sdl_scancode;
  m->bindings[action].modifiers = modifiers;
}

void keybind_get(const keybind_map *m, action_id action, int *sc_out, int *mod_out) {
  if (!m || action < 0 || action >= ACTION_COUNT) return;
  if (sc_out)  *sc_out  = m->bindings[action].scancode;
  if (mod_out) *mod_out = m->bindings[action].modifiers;
}

int keybind_lookup(const keybind_map *m, int sdl_scancode, int modifiers) {
  for (int a = 0; a < ACTION_COUNT; a++) {
    if (m->bindings[a].scancode == sdl_scancode && m->bindings[a].modifiers == modifiers) return a;
  }
  return -1;
}

// ---------------------------------------------------------------------------
// Action names
// ---------------------------------------------------------------------------

const char *keybind_action_name(action_id a) {
  switch (a) {
    case ACTION_PAN_LEFT:             return "pan_left";
    case ACTION_PAN_RIGHT:            return "pan_right";
    case ACTION_PAN_UP:               return "pan_up";
    case ACTION_PAN_DOWN:             return "pan_down";
    case ACTION_ZOOM_IN:              return "zoom_in";
    case ACTION_ZOOM_OUT:             return "zoom_out";
    case ACTION_SLICE_NEXT:           return "slice_next";
    case ACTION_SLICE_PREV:           return "slice_prev";
    case ACTION_TOOL_BRUSH:           return "tool_brush";
    case ACTION_TOOL_LINE:            return "tool_line";
    case ACTION_TOOL_PUSHPULL:        return "tool_pushpull";
    case ACTION_UNDO:                 return "undo";
    case ACTION_REDO:                 return "redo";
    case ACTION_FOCUS_NEXT:           return "focus_next";
    case ACTION_FOCUS_PREV:           return "focus_prev";
    case ACTION_TOGGLE_OVERLAY:       return "toggle_overlay";
    case ACTION_TOGGLE_SEGMENTATION:  return "toggle_segmentation";
    case ACTION_SAVE:                 return "save";
    case ACTION_FULLSCREEN:           return "fullscreen";
    default:                          return "unknown";
  }
}

// ---------------------------------------------------------------------------
// JSON save
// ---------------------------------------------------------------------------

bool keybind_save_json(const keybind_map *m, const char *path) {
  FILE *f = fopen(path, "w");
  if (!f) return false;

  fprintf(f, "{\n");
  for (int a = 0; a < ACTION_COUNT; a++) {
    const char *name = keybind_action_name((action_id)a);
    fprintf(f, "  \"%s\": { \"scancode\": %d, \"modifiers\": %d }%s\n",
            name, m->bindings[a].scancode, m->bindings[a].modifiers,
            (a < ACTION_COUNT - 1) ? "," : "");
  }
  fprintf(f, "}\n");
  fclose(f);
  return true;
}

// ---------------------------------------------------------------------------
// JSON load
// ---------------------------------------------------------------------------

// Map action name string -> action_id
static action_id action_from_name(const char *name) {
  for (int a = 0; a < ACTION_COUNT; a++) {
    if (strcmp(keybind_action_name((action_id)a), name) == 0) return (action_id)a;
  }
  return (action_id)-1;
}

// Read entire file into a heap-allocated buffer (caller frees)
static char *read_file(const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0) { fclose(f); return NULL; }
  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }
  size_t nr = fread(buf, 1, (size_t)sz, f);
  buf[nr] = '\0';
  fclose(f);
  return buf;
}

typedef struct { keybind_map *m; } load_ctx;

static void load_entry(const char *key, const json_value *val, void *ctx) {
  keybind_map *m = ((load_ctx *)ctx)->m;
  action_id a = action_from_name(key);
  if (a < 0 || a >= ACTION_COUNT) return;
  const json_value *sc  = json_object_get(val, "scancode");
  const json_value *mod = json_object_get(val, "modifiers");
  if (!sc || !mod) return;
  m->bindings[a].scancode  = (int)json_get_int(sc,  m->bindings[a].scancode);
  m->bindings[a].modifiers = (int)json_get_int(mod, m->bindings[a].modifiers);
}

keybind_map *keybind_load_json(const char *path) {
  keybind_map *m = keybind_new();
  if (!m) return NULL;

  char *buf = read_file(path);
  if (!buf) return m;  // return defaults on failure

  json_value *root = json_parse(buf);
  free(buf);

  if (!root) return m;  // return defaults on parse failure

  load_ctx ctx = { m };
  json_object_iter(root, load_entry, &ctx);
  json_free(root);

  return m;
}
