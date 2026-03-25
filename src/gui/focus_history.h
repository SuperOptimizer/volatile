#pragma once
#include "core/math.h"
#include <stdbool.h>

typedef struct focus_history focus_history;

// Create a circular focus history with capacity max_entries (use 0 for default 100).
focus_history *focus_history_new(int max_entries);
void           focus_history_free(focus_history *h);

// Record a new focus position. Truncates any forward history beyond current position.
void focus_history_push(focus_history *h, vec3f position, int pyramid_level);

// Navigate back/forward. Returns false if no entry in that direction.
bool focus_history_back   (focus_history *h, vec3f *pos_out, int *level_out);
bool focus_history_forward(focus_history *h, vec3f *pos_out, int *level_out);

bool focus_history_can_back   (const focus_history *h);
bool focus_history_can_forward(const focus_history *h);
int  focus_history_count      (const focus_history *h);
