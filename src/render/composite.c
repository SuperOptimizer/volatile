#include "composite.h"

#include <math.h>
#include <float.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

void composite_params_default(composite_params *p) {
  p->mode             = COMPOSITE_MAX;
  p->alpha_min        = 0.0f;
  p->alpha_max        = 1.0f;
  p->alpha_opacity    = 1.0f;
  p->extinction       = 1.0f;
  p->num_layers_front = 0;
  p->num_layers_behind = 0;
}

// ---------------------------------------------------------------------------
// Single-pixel composite
// ---------------------------------------------------------------------------

float composite_pixel(const float *values, int count, const composite_params *params) {
  if (count <= 0) return 0.0f;

  switch (params->mode) {
    case COMPOSITE_MAX: {
      float m = values[0];
      for (int i = 1; i < count; i++) if (values[i] > m) m = values[i];
      return m;
    }

    case COMPOSITE_MIN: {
      float m = values[0];
      for (int i = 1; i < count; i++) if (values[i] < m) m = values[i];
      return m;
    }

    case COMPOSITE_MEAN: {
      float s = 0.0f;
      for (int i = 0; i < count; i++) s += values[i];
      return s / (float)count;
    }

    case COMPOSITE_SUM: {
      float s = 0.0f;
      for (int i = 0; i < count; i++) s += values[i];
      return s;
    }

    case COMPOSITE_ALPHA: {
      // Front-to-back alpha compositing: value maps linearly to opacity in [alpha_min, alpha_max]
      // accumulated_color += (1 - accumulated_alpha) * opacity * value
      // accumulated_alpha += (1 - accumulated_alpha) * opacity
      float range = params->alpha_max - params->alpha_min;
      float inv_range = (range != 0.0f) ? (1.0f / range) : 0.0f;
      float acc_color = 0.0f;
      float acc_alpha = 0.0f;
      for (int i = 0; i < count; i++) {
        float t = (values[i] - params->alpha_min) * inv_range;
        if (t < 0.0f) t = 0.0f;
        else if (t > 1.0f) t = 1.0f;
        float opacity = t * params->alpha_opacity;
        float transmit = 1.0f - acc_alpha;
        acc_color += transmit * opacity * values[i];
        acc_alpha += transmit * opacity;
        if (acc_alpha >= 1.0f) break;
      }
      return acc_color;
    }

    case COMPOSITE_BEER_LAMBERT: {
      // I = I0 * exp(-extinction * sum(values)), I0 = 1
      float sum = 0.0f;
      for (int i = 0; i < count; i++) sum += values[i];
      return expf(-params->extinction * sum);
    }

    default:
      return 0.0f;
  }
}

// ---------------------------------------------------------------------------
// Slice compositing
// ---------------------------------------------------------------------------

// Each mode gets a dedicated inner loop for auto-vectorization friendliness.

static void composite_max(const float **slices, int num_slices,
                          float *restrict out, int n) {
  // Initialize from first slice
  const float *restrict s0 = slices[0];
  for (int i = 0; i < n; i++) out[i] = s0[i];
  for (int k = 1; k < num_slices; k++) {
    const float *restrict sk = slices[k];
    for (int i = 0; i < n; i++) {
      if (sk[i] > out[i]) out[i] = sk[i];
    }
  }
}

static void composite_min(const float **slices, int num_slices,
                          float *restrict out, int n) {
  const float *restrict s0 = slices[0];
  for (int i = 0; i < n; i++) out[i] = s0[i];
  for (int k = 1; k < num_slices; k++) {
    const float *restrict sk = slices[k];
    for (int i = 0; i < n; i++) {
      if (sk[i] < out[i]) out[i] = sk[i];
    }
  }
}

static void composite_mean(const float **slices, int num_slices,
                           float *restrict out, int n) {
  memset(out, 0, (size_t)n * sizeof(float));
  for (int k = 0; k < num_slices; k++) {
    const float *restrict sk = slices[k];
    for (int i = 0; i < n; i++) out[i] += sk[i];
  }
  float inv = 1.0f / (float)num_slices;
  for (int i = 0; i < n; i++) out[i] *= inv;
}

static void composite_sum(const float **slices, int num_slices,
                          float *restrict out, int n) {
  memset(out, 0, (size_t)n * sizeof(float));
  for (int k = 0; k < num_slices; k++) {
    const float *restrict sk = slices[k];
    for (int i = 0; i < n; i++) out[i] += sk[i];
  }
}

static void composite_alpha(const float **slices, int num_slices,
                            float *restrict out, int n,
                            const composite_params *params) {
  float range = params->alpha_max - params->alpha_min;
  float inv_range = (range != 0.0f) ? (1.0f / range) : 0.0f;
  float base_opacity = params->alpha_opacity;

  // acc_color and acc_alpha per pixel — process pixel-by-pixel (no easy vectorization across slices)
  for (int i = 0; i < n; i++) {
    float acc_color = 0.0f;
    float acc_alpha = 0.0f;
    for (int k = 0; k < num_slices; k++) {
      float v = slices[k][i];
      float t = (v - params->alpha_min) * inv_range;
      if (t < 0.0f) t = 0.0f;
      else if (t > 1.0f) t = 1.0f;
      float opacity = t * base_opacity;
      float transmit = 1.0f - acc_alpha;
      acc_color += transmit * opacity * v;
      acc_alpha += transmit * opacity;
      if (acc_alpha >= 1.0f) break;
    }
    out[i] = acc_color;
  }
}

static void composite_beer_lambert(const float **slices, int num_slices,
                                   float *restrict out, int n,
                                   const composite_params *params) {
  // sum values per pixel first, then apply exp once
  memset(out, 0, (size_t)n * sizeof(float));
  for (int k = 0; k < num_slices; k++) {
    const float *restrict sk = slices[k];
    for (int i = 0; i < n; i++) out[i] += sk[i];
  }
  float ext = params->extinction;
  for (int i = 0; i < n; i++) out[i] = expf(-ext * out[i]);
}

void composite_slices(const float **slices, int num_slices,
                      float *out, int width, int height,
                      const composite_params *params) {
  if (num_slices <= 0 || !slices || !out) return;

  int n = width * height;

  switch (params->mode) {
    case COMPOSITE_MAX:          composite_max(slices, num_slices, out, n);                   break;
    case COMPOSITE_MIN:          composite_min(slices, num_slices, out, n);                   break;
    case COMPOSITE_MEAN:         composite_mean(slices, num_slices, out, n);                  break;
    case COMPOSITE_SUM:          composite_sum(slices, num_slices, out, n);                   break;
    case COMPOSITE_ALPHA:        composite_alpha(slices, num_slices, out, n, params);         break;
    case COMPOSITE_BEER_LAMBERT: composite_beer_lambert(slices, num_slices, out, n, params);  break;
    default: memset(out, 0, (size_t)n * sizeof(float)); break;
  }
}

// ---------------------------------------------------------------------------
// Mode name
// ---------------------------------------------------------------------------

const char *composite_mode_name(composite_mode m) {
  switch (m) {
    case COMPOSITE_MAX:          return "max";
    case COMPOSITE_MIN:          return "min";
    case COMPOSITE_MEAN:         return "mean";
    case COMPOSITE_ALPHA:        return "alpha";
    case COMPOSITE_BEER_LAMBERT: return "beer_lambert";
    case COMPOSITE_SUM:          return "sum";
    default:                     return "unknown";
  }
}
