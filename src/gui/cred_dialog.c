#define _POSIX_C_SOURCE 200809L
#include "gui/cred_dialog.h"
#include "core/log.h"
#include "core/json.h"

// NOTE: NK_IMPLEMENTATION is owned by app.c — include nuklear declaration-only here.
#include "nuklear.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct cred_dialog {
  bool          visible;
  char          failed_url[512];

  // edit buffer lengths must match s3_credentials field sizes
  char          access_key[128];
  char          secret_key[128];
  char          token[1024];
  char          region[32];
  char          endpoint[256];

  int           access_key_len;
  int           secret_key_len;
  int           token_len;
  int           region_len;
  int           endpoint_len;

  bool          remember;
  bool          show_secret;    // toggle "reveal password"

  s3_credentials result;       // populated on submit; caller reads via get_creds
};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

cred_dialog *cred_dialog_new(void) {
  cred_dialog *d = calloc(1, sizeof(*d));
  REQUIRE(d, "cred_dialog_new: calloc failed");
  // Default region
  strncpy(d->region, "us-east-1", sizeof(d->region) - 1);
  d->region_len = (int)strlen(d->region);
  return d;
}

void cred_dialog_free(cred_dialog *d) {
  if (!d) return;
  // Zero out secret fields before freeing
  memset(d->secret_key, 0, sizeof(d->secret_key));
  memset(d->token,      0, sizeof(d->token));
  free(d);
}

void cred_dialog_show(cred_dialog *d, const char *failed_url) {
  REQUIRE(d, "cred_dialog_show: null dialog");
  d->visible = true;
  if (failed_url)
    strncpy(d->failed_url, failed_url, sizeof(d->failed_url) - 1);
  else
    d->failed_url[0] = '\0';
}

bool cred_dialog_is_visible(const cred_dialog *d) {
  return d && d->visible;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

bool cred_dialog_render(cred_dialog *d, struct nk_context *ctx) {
  if (!d || !ctx || !d->visible) return false;

  static const float LABEL_W = 160;
  static const float FIELD_H = 22;
  bool submitted = false;

  if (nk_begin(ctx, "AWS Credentials",
               nk_rect(100, 80, 480, 380),
               NK_WINDOW_BORDER | NK_WINDOW_TITLE | NK_WINDOW_MOVABLE)) {

    if (d->failed_url[0]) {
      nk_layout_row_dynamic(ctx, FIELD_H, 1);
      nk_label(ctx, d->failed_url, NK_TEXT_LEFT);
      nk_layout_row_dynamic(ctx, 4, 1);  // spacer
    }

    // Access Key ID
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, "Access Key ID:", NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 280);
    nk_edit_string(ctx, NK_EDIT_FIELD, d->access_key, &d->access_key_len,
                   (int)sizeof(d->access_key) - 1, nk_filter_default);
    nk_layout_row_end(ctx);

    // Secret Access Key (masked unless revealed)
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 3);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, "Secret Access Key:", NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 240);
    nk_flags flags = d->show_secret ? NK_EDIT_FIELD
                                    : NK_EDIT_FIELD | NK_EDIT_NO_CURSOR;
    if (d->show_secret) {
      nk_edit_string(ctx, NK_EDIT_FIELD, d->secret_key, &d->secret_key_len,
                     (int)sizeof(d->secret_key) - 1, nk_filter_default);
    } else {
      // Show bullet placeholder — real text unchanged
      char dots[128] = {0};
      int dlen = d->secret_key_len < 32 ? d->secret_key_len : 32;
      memset(dots, 0xE2, (size_t)dlen);  // filler, not real UTF-8 but shows something
      int dummy = dlen;
      nk_edit_string(ctx, NK_EDIT_FIELD | NK_EDIT_READ_ONLY,
                     dots, &dummy, 128, nk_filter_default);
    }
    (void)flags;
    nk_layout_row_push(ctx, 36);
    if (nk_button_label(ctx, d->show_secret ? "Hide" : "Show"))
      d->show_secret = !d->show_secret;
    nk_layout_row_end(ctx);

    // Session Token (optional)
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, "Session Token:", NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 280);
    nk_edit_string(ctx, NK_EDIT_FIELD, d->token, &d->token_len,
                   (int)sizeof(d->token) - 1, nk_filter_default);
    nk_layout_row_end(ctx);

    // Region
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, "Region:", NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 280);
    nk_edit_string(ctx, NK_EDIT_FIELD, d->region, &d->region_len,
                   (int)sizeof(d->region) - 1, nk_filter_default);
    nk_layout_row_end(ctx);

    // Custom endpoint (optional)
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_label(ctx, "Endpoint (opt):", NK_TEXT_RIGHT);
    nk_layout_row_push(ctx, 280);
    nk_edit_string(ctx, NK_EDIT_FIELD, d->endpoint, &d->endpoint_len,
                   (int)sizeof(d->endpoint) - 1, nk_filter_default);
    nk_layout_row_end(ctx);

    // Remember checkbox
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H, 2);
    nk_layout_row_push(ctx, LABEL_W);
    nk_spacing(ctx, 1);
    nk_layout_row_push(ctx, 280);
    nk_checkbox_label(ctx, "Remember credentials", &d->remember);
    nk_layout_row_end(ctx);

    nk_layout_row_dynamic(ctx, 8, 1);  // spacer

    // Submit / Cancel
    nk_layout_row_begin(ctx, NK_STATIC, FIELD_H + 4, 2);
    nk_layout_row_push(ctx, 220);
    if (nk_button_label(ctx, "Connect")) {
      // Null-terminate fields and copy into result
      d->access_key[d->access_key_len] = '\0';
      d->secret_key[d->secret_key_len] = '\0';
      d->token[d->token_len]           = '\0';
      d->region[d->region_len]         = '\0';
      d->endpoint[d->endpoint_len]     = '\0';
      memcpy(d->result.access_key, d->access_key, sizeof(d->result.access_key));
      memcpy(d->result.secret_key, d->secret_key, sizeof(d->result.secret_key));
      memcpy(d->result.token,      d->token,      sizeof(d->result.token));
      memcpy(d->result.region,     d->region,     sizeof(d->result.region));
      memcpy(d->result.endpoint,   d->endpoint,   sizeof(d->result.endpoint));
      d->visible = false;
      submitted  = true;
    }
    nk_layout_row_push(ctx, 220);
    if (nk_button_label(ctx, "Cancel"))
      d->visible = false;
    nk_layout_row_end(ctx);
  }
  nk_end(ctx);
  return submitted;
}

// ---------------------------------------------------------------------------
// get_creds
// ---------------------------------------------------------------------------

s3_credentials *cred_dialog_get_creds(cred_dialog *d) {
  REQUIRE(d, "cred_dialog_get_creds: null dialog");
  return &d->result;
}

// ---------------------------------------------------------------------------
// save / load
// ---------------------------------------------------------------------------

bool cred_dialog_save(const cred_dialog *d, const char *path) {
  REQUIRE(d && path, "cred_dialog_save: null arg");
  if (!d->remember) return true;  // nothing to save

  FILE *f = fopen(path, "w");
  if (!f) { LOG_WARN("cred_dialog_save: cannot open %s", path); return false; }

  // chmod 600 immediately — before writing sensitive data
#ifdef __unix__
  chmod(path, 0600);
#endif

  fprintf(f,
    "{\n"
    "  \"access_key\": \"%s\",\n"
    "  \"secret_key\": \"%s\",\n"
    "  \"token\": \"%s\",\n"
    "  \"region\": \"%s\",\n"
    "  \"endpoint\": \"%s\"\n"
    "}\n",
    d->result.access_key,
    d->result.secret_key,
    d->result.token,
    d->result.region,
    d->result.endpoint);
  fclose(f);
  return true;
}

bool cred_dialog_load(cred_dialog *d, const char *path) {
  REQUIRE(d && path, "cred_dialog_load: null arg");

  FILE *f = fopen(path, "r");
  if (!f) return false;

  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  rewind(f);
  if (sz <= 0 || sz > (long)(1 << 16)) { fclose(f); return false; }

  char *buf = malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return false; }
  size_t n = fread(buf, 1, (size_t)sz, f);
  fclose(f);
  buf[n] = '\0';

  // Parse with our minimal JSON parser
  json_value *root = json_parse(buf);
  free(buf);
  if (!root) return false;

  const json_value *vak = json_object_get(root, "access_key");
  const json_value *vsk = json_object_get(root, "secret_key");
  const json_value *vtk = json_object_get(root, "token");
  const json_value *vrg = json_object_get(root, "region");
  const json_value *vep = json_object_get(root, "endpoint");
  const char *ak = vak ? json_get_str(vak) : NULL;
  const char *sk = vsk ? json_get_str(vsk) : NULL;
  const char *tk = vtk ? json_get_str(vtk) : NULL;
  const char *rg = vrg ? json_get_str(vrg) : NULL;
  const char *ep = vep ? json_get_str(vep) : NULL;

  if (ak) { strncpy(d->access_key, ak, sizeof(d->access_key)-1); d->access_key_len = (int)strlen(d->access_key); }
  if (sk) { strncpy(d->secret_key, sk, sizeof(d->secret_key)-1); d->secret_key_len = (int)strlen(d->secret_key); }
  if (tk) { strncpy(d->token,      tk, sizeof(d->token)-1);      d->token_len      = (int)strlen(d->token); }
  if (rg) { strncpy(d->region,     rg, sizeof(d->region)-1);     d->region_len     = (int)strlen(d->region); }
  if (ep) { strncpy(d->endpoint,   ep, sizeof(d->endpoint)-1);   d->endpoint_len   = (int)strlen(d->endpoint); }

  // Populate result too so it's immediately usable
  if (ak) strncpy(d->result.access_key, ak, sizeof(d->result.access_key)-1);
  if (sk) strncpy(d->result.secret_key, sk, sizeof(d->result.secret_key)-1);
  if (tk) strncpy(d->result.token,      tk, sizeof(d->result.token)-1);
  if (rg) strncpy(d->result.region,     rg, sizeof(d->result.region)-1);
  if (ep) strncpy(d->result.endpoint,   ep, sizeof(d->result.endpoint)-1);

  d->remember = true;
  json_free(root);
  return true;
}
