#include "render/cmap.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline double clamp01(double v) {
  return v < 0.0 ? 0.0 : (v > 1.0 ? 1.0 : v);
}

static inline uint8_t u8(double v) {
  return (uint8_t)(clamp01(v) * 255.0 + 0.5);
}

// ---------------------------------------------------------------------------
// Control-point tables for perceptual colormaps.
// Each row: { t, r, g, b } with t in [0,1].
// Sourced from matplotlib's reference implementations (public domain data).
// ---------------------------------------------------------------------------

typedef struct { float t, r, g, b; } cp;

// --- Viridis (16 control points) ---
static const cp viridis_cp[] = {
  {0.000f, 0.267f, 0.005f, 0.329f},
  {0.067f, 0.283f, 0.141f, 0.458f},
  {0.133f, 0.254f, 0.265f, 0.530f},
  {0.200f, 0.208f, 0.374f, 0.553f},
  {0.267f, 0.163f, 0.471f, 0.558f},
  {0.333f, 0.125f, 0.563f, 0.551f},
  {0.400f, 0.108f, 0.648f, 0.522f},
  {0.467f, 0.135f, 0.727f, 0.473f},
  {0.533f, 0.205f, 0.797f, 0.401f},
  {0.600f, 0.310f, 0.851f, 0.311f},
  {0.667f, 0.429f, 0.893f, 0.200f},
  {0.733f, 0.558f, 0.920f, 0.108f},
  {0.800f, 0.682f, 0.937f, 0.093f},
  {0.867f, 0.797f, 0.940f, 0.165f},
  {0.933f, 0.905f, 0.934f, 0.306f},
  {1.000f, 0.993f, 0.906f, 0.144f},
};

// --- Magma (16 control points) ---
static const cp magma_cp[] = {
  {0.000f, 0.001f, 0.000f, 0.014f},
  {0.067f, 0.028f, 0.016f, 0.097f},
  {0.133f, 0.092f, 0.021f, 0.202f},
  {0.200f, 0.183f, 0.026f, 0.296f},
  {0.267f, 0.278f, 0.032f, 0.365f},
  {0.333f, 0.378f, 0.053f, 0.406f},
  {0.400f, 0.478f, 0.087f, 0.417f},
  {0.467f, 0.576f, 0.124f, 0.413f},
  {0.533f, 0.671f, 0.167f, 0.398f},
  {0.600f, 0.762f, 0.229f, 0.375f},
  {0.667f, 0.843f, 0.318f, 0.344f},
  {0.733f, 0.906f, 0.441f, 0.318f},
  {0.800f, 0.948f, 0.585f, 0.349f},
  {0.867f, 0.973f, 0.726f, 0.459f},
  {0.933f, 0.989f, 0.859f, 0.620f},
  {1.000f, 0.988f, 0.992f, 0.749f},
};

// --- Inferno (16 control points) ---
static const cp inferno_cp[] = {
  {0.000f, 0.001f, 0.000f, 0.014f},
  {0.067f, 0.037f, 0.010f, 0.111f},
  {0.133f, 0.120f, 0.013f, 0.216f},
  {0.200f, 0.222f, 0.016f, 0.287f},
  {0.267f, 0.325f, 0.028f, 0.310f},
  {0.333f, 0.427f, 0.056f, 0.300f},
  {0.400f, 0.527f, 0.098f, 0.263f},
  {0.467f, 0.622f, 0.153f, 0.203f},
  {0.533f, 0.710f, 0.219f, 0.127f},
  {0.600f, 0.789f, 0.298f, 0.048f},
  {0.667f, 0.856f, 0.392f, 0.007f},
  {0.733f, 0.908f, 0.502f, 0.030f},
  {0.800f, 0.945f, 0.625f, 0.122f},
  {0.867f, 0.968f, 0.754f, 0.272f},
  {0.933f, 0.983f, 0.882f, 0.475f},
  {1.000f, 0.988f, 1.000f, 0.645f},
};

// --- Plasma (16 control points) ---
static const cp plasma_cp[] = {
  {0.000f, 0.050f, 0.030f, 0.528f},
  {0.067f, 0.177f, 0.006f, 0.583f},
  {0.133f, 0.275f, 0.003f, 0.613f},
  {0.200f, 0.365f, 0.019f, 0.626f},
  {0.267f, 0.449f, 0.047f, 0.624f},
  {0.333f, 0.529f, 0.082f, 0.608f},
  {0.400f, 0.607f, 0.117f, 0.579f},
  {0.467f, 0.682f, 0.154f, 0.539f},
  {0.533f, 0.752f, 0.196f, 0.490f},
  {0.600f, 0.818f, 0.244f, 0.432f},
  {0.667f, 0.877f, 0.302f, 0.366f},
  {0.733f, 0.926f, 0.373f, 0.292f},
  {0.800f, 0.960f, 0.460f, 0.211f},
  {0.867f, 0.979f, 0.561f, 0.131f},
  {0.933f, 0.980f, 0.672f, 0.068f},
  {1.000f, 0.940f, 0.975f, 0.131f},
};

// --- Turbo (16 control points) ---
static const cp turbo_cp[] = {
  {0.000f, 0.190f, 0.072f, 0.232f},
  {0.067f, 0.274f, 0.314f, 0.697f},
  {0.133f, 0.240f, 0.524f, 0.906f},
  {0.200f, 0.133f, 0.718f, 0.907f},
  {0.267f, 0.028f, 0.867f, 0.764f},
  {0.333f, 0.028f, 0.954f, 0.549f},
  {0.400f, 0.241f, 0.990f, 0.324f},
  {0.467f, 0.489f, 0.981f, 0.123f},
  {0.533f, 0.714f, 0.920f, 0.028f},
  {0.600f, 0.889f, 0.809f, 0.028f},
  {0.667f, 0.991f, 0.671f, 0.028f},
  {0.733f, 0.994f, 0.502f, 0.028f},
  {0.800f, 0.942f, 0.321f, 0.028f},
  {0.867f, 0.847f, 0.163f, 0.028f},
  {0.933f, 0.702f, 0.052f, 0.028f},
  {1.000f, 0.479f, 0.016f, 0.010f},
};

// ---------------------------------------------------------------------------
// Interpolate a control-point table into a 256-entry LUT.
// ---------------------------------------------------------------------------

static void build_lut(const cp *pts, size_t n_pts, cmap_rgb lut[256]) {
  for (int i = 0; i < 256; i++) {
    double t = i / 255.0;
    // find bracketing segment
    size_t lo = 0;
    for (size_t k = 0; k + 1 < n_pts; k++) {
      if ((double)pts[k + 1].t >= t) { lo = k; break; }
      lo = k;
    }
    size_t hi = lo + 1;
    if (hi >= n_pts) hi = n_pts - 1;
    double span = (double)(pts[hi].t - pts[lo].t);
    double f = (span > 0.0) ? (t - (double)pts[lo].t) / span : 0.0;
    lut[i].r = u8((double)pts[lo].r + f * ((double)pts[hi].r - (double)pts[lo].r));
    lut[i].g = u8((double)pts[lo].g + f * ((double)pts[hi].g - (double)pts[lo].g));
    lut[i].b = u8((double)pts[lo].b + f * ((double)pts[hi].b - (double)pts[lo].b));
  }
}

// ---------------------------------------------------------------------------
// Static LUTs (populated once at first use via init_luts())
// ---------------------------------------------------------------------------

static cmap_rgb lut_viridis[256];
static cmap_rgb lut_magma[256];
static cmap_rgb lut_inferno[256];
static cmap_rgb lut_plasma[256];
static cmap_rgb lut_turbo[256];
static bool     luts_ready = false;

#define CP_LEN(arr) (sizeof(arr) / sizeof((arr)[0]))

static void init_luts(void) {
  if (luts_ready) return;
  build_lut(viridis_cp, CP_LEN(viridis_cp), lut_viridis);
  build_lut(magma_cp,   CP_LEN(magma_cp),   lut_magma);
  build_lut(inferno_cp, CP_LEN(inferno_cp),  lut_inferno);
  build_lut(plasma_cp,  CP_LEN(plasma_cp),   lut_plasma);
  build_lut(turbo_cp,   CP_LEN(turbo_cp),    lut_turbo);
  luts_ready = true;
}

// ---------------------------------------------------------------------------
// Formula-based maps (computed inline, no LUT needed)
// ---------------------------------------------------------------------------

static cmap_rgb apply_grayscale(double t) {
  uint8_t v = u8(t);
  return (cmap_rgb){v, v, v};
}

static cmap_rgb apply_hot(double t) {
  // black -> red -> yellow -> white
  return (cmap_rgb){
    u8(t * 3.0),
    u8(t * 3.0 - 1.0),
    u8(t * 3.0 - 2.0),
  };
}

static cmap_rgb apply_cool(double t) {
  // cyan -> magenta
  return (cmap_rgb){u8(t), u8(1.0 - t), 255};
}

static cmap_rgb apply_bone(double t) {
  // blue-grey tint of grayscale
  return (cmap_rgb){
    u8(t * 7.0 / 8.0 + (t >= 3.0 / 4.0 ? (t - 3.0 / 4.0) * 4.0 / 3.0 : 0.0)),
    u8(t * 7.0 / 8.0 + (t >= 3.0 / 8.0 && t < 3.0 / 4.0
        ? (t - 3.0 / 8.0) * 4.0 / 3.0
        : (t >= 3.0 / 4.0 ? 1.0 / 8.0 + (t - 3.0 / 4.0) * 4.0 / 3.0 : 0.0))),
    u8(t * 7.0 / 8.0 + (t < 3.0 / 8.0 ? t * 4.0 / 3.0 : 1.0 / 8.0)),
  };
}

static cmap_rgb apply_jet(double t) {
  // classic rainbow: blue -> cyan -> green -> yellow -> red
  double r = clamp01(1.5 - fabs(4.0 * t - 3.0));
  double g = clamp01(1.5 - fabs(4.0 * t - 2.0));
  double b = clamp01(1.5 - fabs(4.0 * t - 1.0));
  return (cmap_rgb){u8(r), u8(g), u8(b)};
}

static cmap_rgb apply_lut(const cmap_rgb lut[256], double t) {
  int idx = (int)(clamp01(t) * 255.0 + 0.5);
  return lut[idx];
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

cmap_rgb cmap_apply(cmap_id id, double value) {
  init_luts();
  switch (id) {
    case CMAP_GRAYSCALE: return apply_grayscale(value);
    case CMAP_VIRIDIS:   return apply_lut(lut_viridis, value);
    case CMAP_MAGMA:     return apply_lut(lut_magma,   value);
    case CMAP_INFERNO:   return apply_lut(lut_inferno,  value);
    case CMAP_PLASMA:    return apply_lut(lut_plasma,   value);
    case CMAP_HOT:       return apply_hot(value);
    case CMAP_COOL:      return apply_cool(value);
    case CMAP_BONE:      return apply_bone(value);
    case CMAP_JET:       return apply_jet(value);
    case CMAP_TURBO:     return apply_lut(lut_turbo,    value);
    default:             return (cmap_rgb){0, 0, 0};
  }
}

void cmap_apply_buf(cmap_id id, const double *values, cmap_rgb *out, size_t n) {
  init_luts();
  for (size_t i = 0; i < n; i++)
    out[i] = cmap_apply(id, values[i]);
}

const char *cmap_name(cmap_id id) {
  static const char *names[CMAP_COUNT] = {
    "grayscale", "viridis", "magma", "inferno", "plasma",
    "hot", "cool", "bone", "jet", "turbo",
  };
  if ((int)id < 0 || id >= CMAP_COUNT) return NULL;
  return names[id];
}

int cmap_count(void) {
  return (int)CMAP_COUNT;
}
