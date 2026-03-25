#pragma once

#include <stdbool.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Forward declarations — callers need not include SDL3 or Nuklear headers
// directly unless they want to call nk_* functions.
// ---------------------------------------------------------------------------
struct nk_context;

// ---------------------------------------------------------------------------
// Configuration passed to app_init
// ---------------------------------------------------------------------------
typedef struct {
  const char *title;       // window title (defaults to "Volatile" if NULL)
  int         width;       // initial window width  (default 1280)
  int         height;      // initial window height (default 720)
} app_config_t;

// ---------------------------------------------------------------------------
// Opaque application state
// ---------------------------------------------------------------------------
typedef struct app_state app_state_t;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Create window, SDL renderer, and Nuklear context.
// Returns NULL on failure (logs reason via LOG_ERROR).
app_state_t *app_init(const app_config_t *cfg);

// Tear down Nuklear, renderer, window, and SDL subsystems.
void app_shutdown(app_state_t *s);

// ---------------------------------------------------------------------------
// Per-frame API — call in this order each iteration:
//   app_begin_frame → build nk panels → app_end_frame
// ---------------------------------------------------------------------------

// Returns true while the window has not been closed/quit.
bool app_should_close(const app_state_t *s);

// Pump SDL events and start a Nuklear frame. Returns false on fatal error.
bool app_begin_frame(app_state_t *s);

// Render and present; clears the Nuklear command queue.
void app_end_frame(app_state_t *s);

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

// Return the live Nuklear context so callers can build panels.
struct nk_context *app_nk_ctx(app_state_t *s);

// Return current logical window size (updated each frame on resize).
void app_get_size(const app_state_t *s, int *w, int *h);

// Return pixel/logical scale factor (>1 on high-DPI displays).
float app_get_dpi_scale(const app_state_t *s);

// Return the header font handle (larger, for titles). NULL if not available.
struct nk_user_font;
const struct nk_user_font *app_get_header_font(const app_state_t *s);

// ---------------------------------------------------------------------------
// Keyboard event callback
// ---------------------------------------------------------------------------

// Callback invoked for each SDL_EVENT_KEY_DOWN / SDL_EVENT_KEY_UP.
// scancode: SDL_Scancode value (integer).
// modifiers: SDL_Keymod bitmask.
// pressed: true = key down, false = key up.
typedef void (*app_key_fn)(int scancode, int modifiers, bool pressed, void *ctx);

// Register a callback for raw SDL key events.  Pass NULL to remove.
void app_set_key_handler(app_state_t *s, app_key_fn fn, void *ctx);
