// WIDGET TYPE: WINDOW — renders its own nk_begin/nk_end window, call OUTSIDE any nk_begin block.
#pragma once
#include "core/net.h"
#include <stdbool.h>

// ---------------------------------------------------------------------------
// cred_dialog — AWS credential input dialog (Nuklear)
//
// Shown when an S3 volume open fails due to missing/expired credentials.
// Prompts for: Access Key ID, Secret Key, Session Token (opt),
//              Region, Custom Endpoint (opt), "Remember" checkbox.
// ---------------------------------------------------------------------------

typedef struct cred_dialog cred_dialog;

cred_dialog    *cred_dialog_new(void);
void            cred_dialog_free(cred_dialog *d);

// Open dialog, optionally showing the URL that triggered the auth failure.
void            cred_dialog_show(cred_dialog *d, const char *failed_url);

// Render one frame. Returns true on submit (credentials are ready).
// ctx may be NULL — returns false immediately without crashing.
bool            cred_dialog_render(cred_dialog *d, struct nk_context *ctx);

// Returns filled credentials; caller does NOT free — owned by dialog.
// Only valid after cred_dialog_render returns true.
s3_credentials *cred_dialog_get_creds(cred_dialog *d);

bool            cred_dialog_is_visible(const cred_dialog *d);

// Persist / restore remembered credentials (plain JSON, chmod 600).
bool            cred_dialog_save(const cred_dialog *d, const char *path);
bool            cred_dialog_load(cred_dialog *d, const char *path);
