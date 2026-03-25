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

  // Bake default font
  struct nk_font_atlas *atlas = nk_sdl_font_stash_begin(nk);
  struct nk_font *font = nk_font_atlas_add_default(atlas, 13.0f, NULL);
  nk_sdl_font_stash_end(nk);
  if (font) nk_style_set_font(nk, &font->handle);

  app_state_t *s = malloc(sizeof(*s));
  REQUIRE(s != NULL, "out of memory");
  s->window       = win;
  s->renderer     = ren;
  s->nk           = nk;
  s->should_close = false;

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
