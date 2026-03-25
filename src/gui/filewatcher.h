#pragma once
#include <stdbool.h>

typedef struct file_watcher file_watcher;
typedef void (*file_changed_fn)(const char *path, void *ctx);

file_watcher *file_watcher_new(void);
void          file_watcher_free(file_watcher *w);

// watch a file or directory for changes
bool file_watcher_add(file_watcher *w, const char *path, file_changed_fn callback, void *ctx);
bool file_watcher_remove(file_watcher *w, const char *path);

// poll for changes non-blocking; returns number of callbacks fired
int  file_watcher_poll(file_watcher *w);

// like file_watcher_add but suppresses repeated callbacks within min_interval_ms
bool file_watcher_add_debounced(file_watcher *w, const char *path,
                                 file_changed_fn callback, void *ctx,
                                 int min_interval_ms);
