#pragma once
#include <stdbool.h>

typedef struct slice_viewer    slice_viewer;
typedef struct slice_controller slice_controller;

// Create / destroy --------------------------------------------------------

slice_controller *slice_controller_new(slice_viewer *viewer);
void              slice_controller_free(slice_controller *c);

// Orientation -------------------------------------------------------------

// Set axis-aligned orientation: 0=XY (normal Z), 1=XZ (normal Y), 2=YZ (normal X).
void slice_controller_set_axis(slice_controller *c, int axis);

// Rotate the slice plane around its current normal by the given angle.
void slice_controller_rotate(slice_controller *c, float angle_degrees);

// Mouse interaction -------------------------------------------------------

void slice_controller_on_mouse_down(slice_controller *c, float x, float y);
void slice_controller_on_mouse_drag(slice_controller *c, float x, float y);
void slice_controller_on_mouse_up(slice_controller *c);

// Returns the current 3x3 rotation matrix (column-major, row-major output).
// The matrix maps slice-local (u, v, n) to world axes.
void slice_controller_get_transform(const slice_controller *c, float *mat3x3_out);

// Tick / debounce ---------------------------------------------------------

// Call each frame with elapsed milliseconds.  Flushes accumulated rotation
// to the viewer after a 200 ms quiet period.
void slice_controller_tick(slice_controller *c, float dt_ms);
