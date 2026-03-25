#pragma once
#include <stdbool.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Layered configuration: global (~/.config/volatile/config.json) overlaid
// by project-local (.volatile/config.json).  Project layer takes priority.
// ---------------------------------------------------------------------------

typedef struct settings settings;

// Open layered settings.  project_dir may be NULL for global-only mode.
settings *settings_open(const char *project_dir);
void      settings_close(settings *s);

// Getters — search project layer first, then global, then return def.
const char *settings_get_str  (settings *s, const char *key, const char *def);
int         settings_get_int  (settings *s, const char *key, int         def);
float       settings_get_float(settings *s, const char *key, float       def);
bool        settings_get_bool (settings *s, const char *key, bool        def);

// Setters — write to project layer when available, else global.
void settings_set_str(settings *s, const char *key, const char *val);
void settings_set_int(settings *s, const char *key, int         val);

// Persist the writable layer to disk.  Returns true on success.
bool settings_save(settings *s);

// Print all keys and values to out (debug aid).
void settings_dump(const settings *s, FILE *out);
