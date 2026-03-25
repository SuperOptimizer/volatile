#pragma once
#include "core/math.h"
#include "core/geom.h"
#include <stdbool.h>

// ---------------------------------------------------------------------------
// neural_tracer — manages a Python ML subprocess over a Unix domain socket.
// Replaces villa's NeuralTraceServiceManager / LasagnaServiceManager.
// ---------------------------------------------------------------------------

typedef struct neural_tracer neural_tracer;

// Lifecycle
neural_tracer *neural_tracer_new(const char *model_path);
void           neural_tracer_free(neural_tracer *t);

// Subprocess management
bool neural_tracer_start(neural_tracer *t);   // fork+exec the Python service
bool neural_tracer_stop(neural_tracer *t);    // send shutdown, wait for exit
bool neural_tracer_is_running(const neural_tracer *t);

// ---------------------------------------------------------------------------
// trace_result — predicted 3D displacement vectors from the ML service
// ---------------------------------------------------------------------------

typedef struct {
  vec3f *displacements;  // count displacement vectors (caller frees via trace_result_free)
  int    count;
} trace_result;

// Predict: send volume patch + current surface points; receive displacements.
// patch is a flat float array of d*h*w values.
// Returns NULL on failure (service not running, timeout, parse error).
trace_result *neural_tracer_predict(neural_tracer *t,
                                    const float *patch, int d, int h, int w,
                                    const vec3f *current_points, int npoints);
void trace_result_free(trace_result *r);

// ---------------------------------------------------------------------------
// opt_service — connection to an external optimisation service (Lasagna-style)
// ---------------------------------------------------------------------------

typedef struct opt_service opt_service;

// Connect (TCP); returns NULL on failure.
opt_service *opt_service_connect(const char *host, int port);
void         opt_service_free(opt_service *s);

// Submit a surface for optimisation with JSON-encoded parameters.
bool opt_service_submit(opt_service *s, quad_surface *surface,
                        const char *params_json);

// Poll: fills status_out with a short status string ("pending","running","done","error").
bool opt_service_poll_status(opt_service *s, char *status_out, int maxlen);

// Retrieve the optimised surface once status=="done". Returns NULL if not ready.
// Caller owns and must free the returned quad_surface.
quad_surface *opt_service_get_result(opt_service *s);
