#define _POSIX_C_SOURCE 200809L

#include "cli_mirror.h"
#include "cli_progress.h"
#include "core/vol_mirror.h"
#include "core/vol.h"
#include "core/log.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmd_mirror(int argc, char **argv) {
  if (argc < 1 || strcmp(argv[0], "--help") == 0) {
    puts("usage: volatile mirror <remote_url> [options]");
    puts("");
    puts("  Cache a remote Zarr volume to local disk.");
    puts("");
    puts("  --cache-dir DIR      Local cache root (default ~/.cache/volatile/)");
    puts("  --level N            Cache only this level (default: all levels)");
    puts("  --rechunk Z,Y,X      Rechunk cached data to new chunk shape");
    puts("  --compress4d         Recompress cached data with compress4d");
    return argc < 1 ? 1 : 0;
  }

  const char *url        = argv[0];
  const char *cache_dir  = NULL;
  int         level_only = -1;   // -1 = all levels
  bool        do_rechunk = false;
  bool        do_compress4d = false;
  int64_t     rechunk[3] = {64, 64, 64};

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--cache-dir") == 0) {
      cache_dir = argv[++i];
    } else if (strcmp(argv[i], "--level") == 0) {
      level_only = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--rechunk") == 0) {
      long long z, y, x;
      if (sscanf(argv[++i], "%lld,%lld,%lld", &z, &y, &x) == 3) {
        rechunk[0] = z; rechunk[1] = y; rechunk[2] = x;
        do_rechunk = true;
      }
    } else if (strcmp(argv[i], "--compress4d") == 0) {
      do_compress4d = true;
      --i;  // flag has no argument
    }
  }
  // handle trailing flag (no value after it)
  if (argc >= 2 && strcmp(argv[argc - 1], "--compress4d") == 0)
    do_compress4d = true;

  mirror_config cfg = {
    .remote_url      = url,
    .local_cache_dir = cache_dir,
    .auto_rechunk    = do_rechunk,
    .auto_compress4d = do_compress4d,
    .max_cache_bytes = 0,   // use default
    .prefetch_radius = 0,   // use default
  };

  vol_mirror *mirror = vol_mirror_new(cfg);
  if (!mirror) {
    fprintf(stderr, "error: cannot open remote volume: %s\n", url);
    return 1;
  }

  volume *rv = vol_mirror_volume(mirror);
  int nlevels = rv ? vol_num_levels(rv) : 1;
  int start   = (level_only >= 0) ? level_only : 0;
  int end     = (level_only >= 0) ? level_only + 1 : nlevels;

  for (int lvl = start; lvl < end; lvl++) {
    int total = vol_mirror_chunks_total(mirror, lvl);
    fprintf(stderr, "caching level %d (%d chunks)...\n", lvl, total);

    // vol_mirror_cache_level does all the work; we just show a start/end bar
    cli_progress(0, 1, "downloading");
    bool ok = vol_mirror_cache_level(mirror, lvl);
    cli_progress(1, 1, "downloading");

    if (!ok) {
      fprintf(stderr, "warning: level %d cache incomplete\n", lvl);
    } else {
      fprintf(stderr, "level %d: %d chunks cached (%.1f MB)\n",
              lvl, vol_mirror_chunks_cached(mirror),
              (double)vol_mirror_cached_bytes(mirror) / 1e6);
    }
  }

  if (do_rechunk) {
    fprintf(stderr, "rechunking to [%lld,%lld,%lld]...\n",
            (long long)rechunk[0], (long long)rechunk[1], (long long)rechunk[2]);
    if (!vol_mirror_rechunk(mirror, rechunk))
      fprintf(stderr, "warning: rechunk failed\n");
  }

  if (do_compress4d) {
    fputs("recompressing to compress4d...\n", stderr);
    if (!vol_mirror_recompress(mirror))
      fprintf(stderr, "warning: recompress failed\n");
  }

  printf("cache dir:  %s\n", cache_dir ? cache_dir : "~/.cache/volatile/");
  printf("chunks:     %d cached\n", vol_mirror_chunks_cached(mirror));
  printf("bytes:      %.1f MB\n", (double)vol_mirror_cached_bytes(mirror) / 1e6);
  printf("hit rate:   %.1f%%\n",  (double)vol_mirror_cache_hit_rate(mirror) * 100.0);

  vol_mirror_free(mirror);
  return 0;
}
