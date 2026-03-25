#pragma once
// Shared inotify buffer size constant for filewatcher.c and plugin.c.
#include <sys/inotify.h>
#include <limits.h>
#define INOTIFY_BUF  (sizeof(struct inotify_event) + NAME_MAX + 1)
