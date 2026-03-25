// ---------------------------------------------------------------------------
// GUI application scaffold — SDL3 window + Nuklear SDL3-renderer backend
// ---------------------------------------------------------------------------

// Nuklear configuration — must come before any nuklear include
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA
#define NK_IMPLEMENTATION
#include <nuklear.h>

#include <SDL3/SDL.h>

#define NK_SDL3_RENDERER_IMPLEMENTATION
#include <nuklear_sdl3_renderer.h>

#include "gui/app.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------
struct app_state {
  SDL_Window    *window;
  SDL_Renderer  *renderer;
  struct nk_context *nk;
  bool           should_close;
  int            win_w, win_h;     // logical window size (points)
  int            pixel_w, pixel_h; // pixel size (for high-DPI)
  float          dpi_scale;        // pixel_w / win_w
  app_key_fn     key_fn;
  void          *key_ctx;
  struct nk_font *font_header;
};

// ---------------------------------------------------------------------------
// app_init
// ---------------------------------------------------------------------------
app_state_t *app_init(const app_config_t *cfg) {
  const char *title  = (cfg && cfg->title)  ? cfg->title  : "Volatile";
  int         width  = (cfg && cfg->width  > 0) ? cfg->width  : 1280;
  int         height = (cfg && cfg->height > 0) ? cfg->height : 720;

  if (!SDL_Init(SDL_INIT_VIDEO)) {
    LOG_ERROR("SDL_Init failed: %s", SDL_GetError());
    return NULL;
  }

  SDL_Window *win = SDL_CreateWindow(title, width, height, SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
  if (!win) {
    LOG_ERROR("SDL_CreateWindow failed: %s", SDL_GetError());
    SDL_Quit();
    return NULL;
  }

  SDL_Renderer *ren = SDL_CreateRenderer(win, NULL);
  if (!ren) {
    LOG_ERROR("SDL_CreateRenderer failed: %s", SDL_GetError());
    SDL_DestroyWindow(win);
    SDL_Quit();
    return NULL;
  }
  SDL_SetRenderVSync(ren, 1);

  struct nk_allocator alloc = nk_sdl_allocator();
  struct nk_context *nk = nk_sdl_init(win, ren, alloc);
  if (!nk) {
    LOG_ERROR("nk_sdl_init failed");
    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    return NULL;
  }

  // Query window dimensions (logical + pixel for high-DPI)
  int lw = width, lh = height;
  SDL_GetWindowSize(win, &lw, &lh);
  int pw = lw, ph = lh;
  SDL_GetWindowSizeInPixels(win, &pw, &ph);
  float dpi = (lw > 0) ? (float)pw / (float)lw : 1.0f;

  // Bake fonts: body (15px) + header (22px) scaled to DPI
  struct nk_font_atlas *atlas = nk_sdl_font_stash_begin(nk);
  struct nk_font *font_body = nk_font_atlas_add_default(atlas, 15.0f * dpi, NULL);
  struct nk_font *font_header = nk_font_atlas_add_default(atlas, 22.0f * dpi, NULL);
  nk_sdl_font_stash_end(nk);
  if (font_body) nk_style_set_font(nk, &font_body->handle);

  // Dark theme with accent colors
  struct nk_color bg     = nk_rgb(30, 30, 35);
  struct nk_color panel  = nk_rgb(40, 42, 48);
  struct nk_color border = nk_rgb(60, 63, 70);
  struct nk_color text   = nk_rgb(220, 222, 228);
  struct nk_color accent = nk_rgb(75, 130, 220);
  struct nk_color hover  = nk_rgb(90, 145, 235);
  struct nk_color active = nk_rgb(55, 110, 200);
  struct nk_color header_bg = nk_rgb(50, 55, 65);

  nk->style.window.background            = bg;
  nk->style.window.fixed_background      = nk_style_item_color(panel);
  nk->style.window.border_color          = border;
  nk->style.window.border                = 1.0f;
  nk->style.window.header.normal         = nk_style_item_color(header_bg);
  nk->style.window.header.hover          = nk_style_item_color(header_bg);
  nk->style.window.header.active         = nk_style_item_color(header_bg);
  nk->style.window.header.label_normal   = text;
  nk->style.window.header.label_hover    = text;
  nk->style.window.header.label_active   = text;
  nk->style.window.header.padding        = nk_vec2(6, 4);
  nk->style.window.padding               = nk_vec2(8, 6);
  nk->style.window.spacing               = nk_vec2(6, 4);

  nk->style.button.normal                = nk_style_item_color(accent);
  nk->style.button.hover                 = nk_style_item_color(hover);
  nk->style.button.active                = nk_style_item_color(active);
  nk->style.button.text_normal           = nk_rgb(255, 255, 255);
  nk->style.button.text_hover            = nk_rgb(255, 255, 255);
  nk->style.button.text_active           = nk_rgb(255, 255, 255);
  nk->style.button.border_color          = border;
  nk->style.button.border                = 1.0f;
  nk->style.button.rounding              = 4.0f;
  nk->style.button.padding               = nk_vec2(8, 4);

  nk->style.text.color                   = text;

  nk->style.slider.bar_normal            = nk_rgb(50, 55, 65);
  nk->style.slider.bar_hover             = nk_rgb(55, 60, 70);
  nk->style.slider.bar_active            = nk_rgb(55, 60, 70);
  nk->style.slider.cursor_normal         = nk_style_item_color(accent);
  nk->style.slider.cursor_hover          = nk_style_item_color(hover);
  nk->style.slider.cursor_active         = nk_style_item_color(active);

  nk->style.edit.normal                  = nk_style_item_color(nk_rgb(35, 38, 45));
  nk->style.edit.hover                   = nk_style_item_color(nk_rgb(40, 43, 50));
  nk->style.edit.active                  = nk_style_item_color(nk_rgb(45, 48, 55));
  nk->style.edit.text_normal             = text;
  nk->style.edit.text_hover              = text;
  nk->style.edit.text_active             = text;
  nk->style.edit.border_color            = border;
  nk->style.edit.border                  = 1.0f;
  nk->style.edit.rounding                = 3.0f;

  app_state_t *s = malloc(sizeof(*s));
  REQUIRE(s != NULL, "out of memory");
  s->window       = win;
  s->renderer     = ren;
  s->nk           = nk;
  s->should_close = false;
  s->win_w        = lw;
  s->win_h        = lh;
  s->pixel_w      = pw;
  s->pixel_h      = ph;
  s->dpi_scale    = dpi;
  s->font_header  = font_header;
  s->key_fn       = NULL;
  s->key_ctx      = NULL;

  LOG_INFO("GUI init: %s %dx%d", title, width, height);
  return s;
}

// ---------------------------------------------------------------------------
// app_shutdown
// ---------------------------------------------------------------------------
void app_shutdown(app_state_t *s) {
  if (!s) return;
  nk_sdl_shutdown(s->nk);
  SDL_DestroyRenderer(s->renderer);
  SDL_DestroyWindow(s->window);
  SDL_Quit();
  free(s);
  LOG_INFO("GUI shutdown");
}

// ---------------------------------------------------------------------------
// app_should_close
// ---------------------------------------------------------------------------
bool app_should_close(const app_state_t *s) {
  return s ? s->should_close : true;
}

// ---------------------------------------------------------------------------
// app_begin_frame
// ---------------------------------------------------------------------------
bool app_begin_frame(app_state_t *s) {
  if (!s) return false;

  nk_input_begin(s->nk);
  SDL_Event ev;
  while (SDL_PollEvent(&ev)) {
    if (ev.type == SDL_EVENT_QUIT) {
      s->should_close = true;
    }
    if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.key == SDLK_ESCAPE) {
      s->should_close = true;
    }
    if (s->key_fn && (ev.type == SDL_EVENT_KEY_DOWN || ev.type == SDL_EVENT_KEY_UP)) {
      s->key_fn((int)ev.key.scancode, (int)ev.key.mod,
                ev.type == SDL_EVENT_KEY_DOWN, s->key_ctx);
    }
    if (ev.type == SDL_EVENT_WINDOW_RESIZED) {
      s->win_w = ev.window.data1;
      s->win_h = ev.window.data2;
      SDL_GetWindowSizeInPixels(s->window, &s->pixel_w, &s->pixel_h);
      float new_dpi = (s->win_w > 0) ? (float)s->pixel_w / (float)s->win_w : 1.0f;
      if (new_dpi != s->dpi_scale) {
        s->dpi_scale = new_dpi;
        struct nk_font_atlas *atlas = nk_sdl_font_stash_begin(s->nk);
        struct nk_font *font = nk_font_atlas_add_default(atlas, 13.0f * new_dpi, NULL);
        nk_sdl_font_stash_end(s->nk);
        if (font) nk_style_set_font(s->nk, &font->handle);
      }
    }
    nk_sdl_handle_event(s->nk, &ev);
  }
  nk_sdl_update_TextInput(s->nk);
  nk_input_end(s->nk);
  return true;
}

// ---------------------------------------------------------------------------
// app_end_frame
// ---------------------------------------------------------------------------
void app_end_frame(app_state_t *s) {
  if (!s) return;

  SDL_SetRenderDrawColor(s->renderer, 30, 30, 30, 255);
  SDL_RenderClear(s->renderer);

  nk_sdl_render(s->nk, NK_ANTI_ALIASING_ON);

  SDL_RenderPresent(s->renderer);
}

// ---------------------------------------------------------------------------
// app_nk_ctx
// ---------------------------------------------------------------------------
struct nk_context *app_nk_ctx(app_state_t *s) {
  return s ? s->nk : NULL;
}

// ---------------------------------------------------------------------------
// app_get_size
// ---------------------------------------------------------------------------
void app_get_size(const app_state_t *s, int *w, int *h) {
  if (w) *w = s ? s->win_w : 0;
  if (h) *h = s ? s->win_h : 0;
}

// ---------------------------------------------------------------------------
// app_set_key_handler
// ---------------------------------------------------------------------------
void app_set_key_handler(app_state_t *s, app_key_fn fn, void *ctx) {
  if (!s) return;
  s->key_fn  = fn;
  s->key_ctx = ctx;
}

// ---------------------------------------------------------------------------
// app_get_dpi_scale
// ---------------------------------------------------------------------------
float app_get_dpi_scale(const app_state_t *s) {
  return s ? s->dpi_scale : 1.0f;
}

const struct nk_user_font *app_get_header_font(const app_state_t *s) {
  return (s && s->font_header) ? &s->font_header->handle : NULL;
}
