#pragma once
#include <stdio.h>

// Print a progress bar to stderr:  [=====>    ] 50% label
// current and total must be >= 0; total == 0 suppresses output.
static inline void cli_progress(int current, int total, const char *label) {
  if (total <= 0) return;
  int pct = (int)((long long)current * 100 / total);
  if (pct > 100) pct = 100;
  const int width = 30;
  int filled = pct * width / 100;
  fprintf(stderr, "\r[");
  for (int i = 0; i < width; i++) {
    if (i < filled)           fputc('=', stderr);
    else if (i == filled)     fputc('>', stderr);
    else                      fputc(' ', stderr);
  }
  fprintf(stderr, "] %3d%% %s", pct, label ? label : "");
  if (pct == 100) fputc('\n', stderr);
  fflush(stderr);
}
