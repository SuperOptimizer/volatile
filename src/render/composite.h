#pragma once
#include <stdint.h>
#include <stddef.h>

typedef enum {
  COMPOSITE_MAX,          // Maximum Intensity Projection
  COMPOSITE_MIN,          // Minimum Intensity Projection
  COMPOSITE_MEAN,         // Average
  COMPOSITE_ALPHA,        // Front-to-back alpha compositing
  COMPOSITE_BEER_LAMBERT, // Beer-Lambert absorption model
  COMPOSITE_SUM,          // Additive
  COMPOSITE_COUNT
} composite_mode;

typedef struct {
  composite_mode mode;
  float alpha_min, alpha_max;   // for alpha mode: value -> opacity mapping
  float alpha_opacity;          // base opacity multiplier
  float extinction;             // Beer-Lambert extinction coefficient
  int num_layers_front;         // layers in front of surface
  int num_layers_behind;        // layers behind surface
} composite_params;

// Initialize with sensible defaults
void composite_params_default(composite_params *p);

// Composite a stack of slices into a single output
// slices: array of num_slices float buffers, each width*height
// out: pre-allocated float buffer width*height
void composite_slices(const float **slices, int num_slices,
                      float *out, int width, int height,
                      const composite_params *params);

// Single-pixel composite (for testing/reference)
float composite_pixel(const float *values, int count, const composite_params *params);

const char *composite_mode_name(composite_mode m);
