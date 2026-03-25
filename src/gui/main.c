// Volatile GUI - functional with dialogs
#define _POSIX_C_SOURCE 200809L

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wdouble-promotion"
#define NK_INCLUDE_FIXED_TYPES
#define NK_INCLUDE_STANDARD_IO
#define NK_INCLUDE_STANDARD_VARARGS
#define NK_INCLUDE_DEFAULT_ALLOCATOR
#define NK_INCLUDE_VERTEX_BUFFER_OUTPUT
#define NK_INCLUDE_FONT_BAKING
#define NK_INCLUDE_DEFAULT_FONT
#define NK_INCLUDE_COMMAND_USERDATA
#include <nuklear.h>
#pragma GCC diagnostic pop

#include "gui/app.h"
#include "core/log.h"
#include "core/vol.h"
#include <stdlib.h>
#include <string.h>

// Simple state
static int show_s3_input = 0;
static int show_url_input = 0;
static char url_buf[2048] = {0};
static char s3_bucket[256] = {0};
static char s3_key[1024] = {0};
static char s3_access_key[128] = {0};
static char s3_secret_key[128] = {0};
static char s3_region[64] = "us-east-1";
static volume *g_vol = NULL;

int main(int argc, char **argv) {
  (void)argc; (void)argv;
  log_set_level(LOG_INFO);
  LOG_INFO("Volatile v%s starting", volatile_version());

  app_config_t cfg = {.title = "Volatile", .width = 1280, .height = 800};
  app_state_t *app = app_init(&cfg);
  if (!app) return 1;

  LOG_INFO("GUI ready — click buttons to open data");

  while (!app_should_close(app)) {
    app_begin_frame(app);
    int w, h;
    app_get_size(app, &w, &h);
    struct nk_context *ctx = app_nk_ctx(app);

    // === Single main window ===
    if (nk_begin(ctx, "Volatile", nk_rect(0, 0, (float)w, (float)h),
                 NK_WINDOW_NO_SCROLLBAR)) {

      // --- Header ---
      nk_layout_row_dynamic(ctx, 35, 1);
      nk_label(ctx, g_vol ? "Volume Loaded" : "Welcome to Volatile",
               NK_TEXT_CENTERED);

      if (g_vol) {
        // --- Volume info ---
        nk_layout_row_dynamic(ctx, 22, 1);
        char info[256];
        snprintf(info, sizeof(info), "Path: %s", vol_path(g_vol));
        nk_label(ctx, info, NK_TEXT_LEFT);
        snprintf(info, sizeof(info), "Levels: %d", vol_num_levels(g_vol));
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_label(ctx, info, NK_TEXT_LEFT);

        int64_t shape[8];
        vol_shape(g_vol, 0, shape);
        snprintf(info, sizeof(info), "Shape: %lld x %lld x %lld",
                 (long long)shape[0], (long long)shape[1], (long long)shape[2]);
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_label(ctx, info, NK_TEXT_LEFT);

        nk_layout_row_dynamic(ctx, 10, 1);
        nk_spacing(ctx, 1);

        nk_layout_row_dynamic(ctx, 30, 1);
        if (nk_button_label(ctx, "Close Volume")) {
          vol_free(g_vol);
          g_vol = NULL;
        }
      } else {
        // --- Open buttons ---
        nk_layout_row_dynamic(ctx, 10, 1);
        nk_spacing(ctx, 1);

        nk_layout_row_dynamic(ctx, 45, 3);
        if (nk_button_label(ctx, "Open Local Zarr")) {
          LOG_INFO("Use URL bar below with a local path");
        }
        if (nk_button_label(ctx, "Open volpkg")) {
          LOG_INFO("Use URL bar below with a volpkg path");
        }
        if (nk_button_label(ctx, "Browse S3")) {
          show_s3_input = !show_s3_input;
          show_url_input = 0;
        }

        nk_layout_row_dynamic(ctx, 10, 1);
        nk_spacing(ctx, 1);

        nk_layout_row_dynamic(ctx, 45, 2);
        if (nk_button_label(ctx, "Open HTTP URL")) {
          show_url_input = !show_url_input;
          show_s3_input = 0;
        }
        if (nk_button_label(ctx, "Open Local Path")) {
          show_url_input = !show_url_input;
          show_s3_input = 0;
        }
      }

      nk_layout_row_dynamic(ctx, 15, 1);
      nk_spacing(ctx, 1);

      // --- S3 input form ---
      if (show_s3_input) {
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_label(ctx, "S3 Connection:", NK_TEXT_LEFT);

        nk_layout_row(ctx, NK_DYNAMIC, 25, 2, (float[]){0.25f, 0.75f});
        nk_label(ctx, "Bucket:", NK_TEXT_LEFT);
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, s3_bucket, sizeof(s3_bucket), nk_filter_default);

        nk_layout_row(ctx, NK_DYNAMIC, 25, 2, (float[]){0.25f, 0.75f});
        nk_label(ctx, "Key/Path:", NK_TEXT_LEFT);
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, s3_key, sizeof(s3_key), nk_filter_default);

        nk_layout_row(ctx, NK_DYNAMIC, 25, 2, (float[]){0.25f, 0.75f});
        nk_label(ctx, "Access Key:", NK_TEXT_LEFT);
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, s3_access_key, sizeof(s3_access_key), nk_filter_default);

        nk_layout_row(ctx, NK_DYNAMIC, 25, 2, (float[]){0.25f, 0.75f});
        nk_label(ctx, "Secret Key:", NK_TEXT_LEFT);
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, s3_secret_key, sizeof(s3_secret_key), nk_filter_default);

        nk_layout_row(ctx, NK_DYNAMIC, 25, 2, (float[]){0.25f, 0.75f});
        nk_label(ctx, "Region:", NK_TEXT_LEFT);
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, s3_region, sizeof(s3_region), nk_filter_default);

        nk_layout_row_dynamic(ctx, 35, 2);
        if (nk_button_label(ctx, "Connect & Open")) {
          // Set env vars for S3 auth
          if (s3_access_key[0]) setenv("AWS_ACCESS_KEY_ID", s3_access_key, 1);
          if (s3_secret_key[0]) setenv("AWS_SECRET_ACCESS_KEY", s3_secret_key, 1);
          if (s3_region[0]) setenv("AWS_REGION", s3_region, 1);

          char s3_url[2048];
          snprintf(s3_url, sizeof(s3_url), "s3://%s/%s", s3_bucket, s3_key);
          LOG_INFO("Connecting to %s", s3_url);
          g_vol = vol_open(s3_url);
          if (g_vol) {
            show_s3_input = 0;
            LOG_INFO("S3 volume opened!");
          } else {
            LOG_WARN("Failed to open S3 volume. Check credentials and path.");
          }
        }
        if (nk_button_label(ctx, "Cancel")) {
          show_s3_input = 0;
        }
      }

      // --- URL input form ---
      if (show_url_input) {
        nk_layout_row_dynamic(ctx, 22, 1);
        nk_label(ctx, "Enter path or URL:", NK_TEXT_LEFT);

        nk_layout_row(ctx, NK_DYNAMIC, 30, 2, (float[]){0.8f, 0.2f});
        nk_edit_string_zero_terminated(ctx, NK_EDIT_FIELD, url_buf, sizeof(url_buf), nk_filter_default);
        if (nk_button_label(ctx, "Open")) {
          if (url_buf[0]) {
            LOG_INFO("Opening: %s", url_buf);
            g_vol = vol_open(url_buf);
            if (g_vol) {
              show_url_input = 0;
              LOG_INFO("Volume opened!");
            } else {
              LOG_WARN("Failed to open: %s", url_buf);
            }
          }
        }

        nk_layout_row_dynamic(ctx, 25, 1);
        nk_label(ctx, "Examples:", NK_TEXT_LEFT);
        nk_layout_row_dynamic(ctx, 20, 1);
        nk_label(ctx, "  /path/to/volume.zarr", NK_TEXT_LEFT);
        nk_label(ctx, "  https://server/volume.zarr/", NK_TEXT_LEFT);
        nk_label(ctx, "  s3://bucket/path/to/volume/", NK_TEXT_LEFT);
      }

      // --- Status / Log area ---
      nk_layout_row_dynamic(ctx, 10, 1);
      nk_spacing(ctx, 1);
      nk_layout_row_dynamic(ctx, 2, 1);
      nk_rule_horizontal(ctx, ctx->style.window.border_color, nk_false);
      nk_layout_row_dynamic(ctx, 18, 1);
      nk_label(ctx, "Status: Ready", NK_TEXT_LEFT);
    }
    nk_end(ctx);

    app_end_frame(app);
  }

  vol_free(g_vol);
  app_shutdown(app);
  return 0;
}
