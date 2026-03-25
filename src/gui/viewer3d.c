#define _POSIX_C_SOURCE 200809L

#include "gui/viewer3d.h"
#include "core/log.h"
#include "core/vol.h"
#include "core/math.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <float.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Constants / defaults
// ---------------------------------------------------------------------------

#define DEFAULT_FOV      45.0f
#define DEFAULT_STEP     0.5f
#define MAX_STEPS        2000
#define ALPHA_THRESHOLD  0.99f   // early-exit for transfer func when nearly opaque

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

struct viewer3d {
  viewer3d_config cfg;
  volume         *vol;

  // Camera
  vec3f eye;
  vec3f target;
  vec3f up;

  // Derived each frame from eye/target/up + fov
  // (recomputed in render and screen_ray)
};

// ---------------------------------------------------------------------------
// Colormaps (same IDs as window_level.comp: 0=gray,1=hot,2=cool,3=viridis)
// ---------------------------------------------------------------------------

static uint32_t apply_colormap(int cmap_id, float t) {
  t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
  float r, g, b;

  switch (cmap_id) {
    case 1: // hot
      r = t < 0.333f ? t * 3.0f : 1.0f;
      g = t < 0.333f ? 0.0f : (t < 0.666f ? (t - 0.333f) * 3.0f : 1.0f);
      b = t < 0.666f ? 0.0f : (t - 0.666f) * 3.0f;
      break;
    case 2: // cool
      r = t; g = 1.0f - t; b = 1.0f;
      break;
    case 3: { // viridis (piecewise linear, same knots as GLSL shader)
      static const float vr[] = {0.267f, 0.190f, 0.128f, 0.369f, 0.993f};
      static const float vg[] = {0.005f, 0.407f, 0.566f, 0.718f, 0.906f};
      static const float vb[] = {0.329f, 0.574f, 0.551f, 0.389f, 0.144f};
      float s = t * 4.0f;
      int lo = (int)s; if (lo > 3) lo = 3;
      float f = s - (float)lo;
      r = vr[lo] + f * (vr[lo+1] - vr[lo]);
      g = vg[lo] + f * (vg[lo+1] - vg[lo]);
      b = vb[lo] + f * (vb[lo+1] - vb[lo]);
      break;
    }
    default: // grayscale
      r = g = b = t;
      break;
  }

  uint32_t ri = (uint32_t)(r * 255.0f + 0.5f);
  uint32_t gi = (uint32_t)(g * 255.0f + 0.5f);
  uint32_t bi = (uint32_t)(b * 255.0f + 0.5f);
  if (ri > 255) ri = 255;
  if (gi > 255) gi = 255;
  if (bi > 255) bi = 255;
  return (255u << 24) | (bi << 16) | (gi << 8) | ri;  // RGBA8 little-endian
}

// ---------------------------------------------------------------------------
// Camera helpers
// ---------------------------------------------------------------------------

// Build an orthonormal camera frame {right, up_actual, forward} from eye/target/up.
static void camera_frame(vec3f eye, vec3f target, vec3f up,
                         vec3f *right_out, vec3f *up_out, vec3f *fwd_out) {
  vec3f fwd   = vec3f_normalize(vec3f_sub(target, eye));
  vec3f right = vec3f_normalize(vec3f_cross(fwd, up));
  vec3f up_a  = vec3f_cross(right, fwd);
  *right_out = right;
  *up_out    = up_a;
  *fwd_out   = fwd;
}

// ---------------------------------------------------------------------------
// Ray/volume helpers
// ---------------------------------------------------------------------------

// Intersect ray (origin + t*dir) with the axis-aligned unit cube [0,1]^3.
// Returns true and sets t_near, t_far if an intersection exists.
static bool ray_aabb_unit(vec3f origin, vec3f dir, float *t_near, float *t_far) {
  float tmin = -FLT_MAX, tmax = FLT_MAX;
  const float *o = &origin.x;
  const float *d = &dir.x;
  for (int i = 0; i < 3; i++) {
    if (fabsf(d[i]) < 1e-7f) {
      if (o[i] < 0.0f || o[i] > 1.0f) return false;
    } else {
      float inv = 1.0f / d[i];
      float t1  = (0.0f - o[i]) * inv;
      float t2  = (1.0f - o[i]) * inv;
      if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
      tmin = tmin > t1 ? tmin : t1;
      tmax = tmax < t2 ? tmax : t2;
    }
    if (tmin > tmax) return false;
  }
  if (tmax < 0.0f) return false;
  *t_near = tmin > 0.0f ? tmin : 0.0f;
  *t_far  = tmax;
  return true;
}

// Sample the volume in normalised [0,1]^3 coords, returning a value in [0,1].
static float sample_vol_norm(const viewer3d *v, vec3f pos) {
  const zarr_level_meta *m = vol_level_meta(v->vol, 0);
  // vol_sample expects (level, z, y, x) in voxel space
  float zv = pos.z * (float)(m->shape[0] - 1);
  float yv = pos.y * (float)(m->shape[1] - 1);
  float xv = pos.x * (float)(m->shape[2] - 1);
  return vol_sample(v->vol, 0, zv, yv, xv);
}

// ---------------------------------------------------------------------------
// Ray integrators
// ---------------------------------------------------------------------------

// Window/level map: raw intensity -> [0,1]
static float wl_map(float val, float window, float level) {
  float half = window * 0.5f;
  float t = (val - (level - half)) / window;
  return t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
}

static uint32_t integrate_mip(const viewer3d *v, vec3f origin, vec3f dir, float step) {
  float t_near, t_far;
  if (!ray_aabb_unit(origin, dir, &t_near, &t_far)) return 0;

  float max_val = 0.0f;
  float t = t_near;
  while (t <= t_far) {
    vec3f pos = vec3f_add(origin, vec3f_scale(dir, t));
    float val = sample_vol_norm(v, pos);
    if (val > max_val) max_val = val;
    t += step;
  }

  float mapped = wl_map(max_val, v->cfg.window, v->cfg.level);
  return apply_colormap(v->cfg.cmap_id, mapped);
}

static uint32_t integrate_iso(const viewer3d *v, vec3f origin, vec3f dir, float step) {
  float t_near, t_far;
  if (!ray_aabb_unit(origin, dir, &t_near, &t_far)) return 0;

  float iso = v->cfg.iso_value;
  float prev = -1.0f;
  float t    = t_near;

  while (t <= t_far) {
    vec3f pos = vec3f_add(origin, vec3f_scale(dir, t));
    float val = sample_vol_norm(v, pos);

    if (prev >= 0.0f && ((prev < iso && val >= iso) || (prev >= iso && val < iso))) {
      // Linear interpolation to find the crossing point
      float frac = (iso - prev) / (val - prev + 1e-9f);
      float t_cross = t - step * (1.0f - frac);
      vec3f hit = vec3f_add(origin, vec3f_scale(dir, t_cross));

      // Simple diffuse shading: use position as a proxy for surface normal colour
      float shade = 0.3f + 0.7f * fabsf(hit.z - 0.5f) * 2.0f;
      float mapped = wl_map(iso, v->cfg.window, v->cfg.level);
      // Blend colormap with diffuse shading
      uint32_t base = apply_colormap(v->cfg.cmap_id, mapped);
      uint8_t  ch[4];
      memcpy(ch, &base, 4);
      for (int i = 0; i < 3; i++) ch[i] = (uint8_t)(ch[i] * shade);
      uint32_t out; memcpy(&out, ch, 4);
      return out;
    }
    prev = val;
    t   += step;
  }
  return 0;
}

// Front-to-back alpha compositing transfer function.
// Transfer function: alpha = val^2 (simple density-based), colour from colormap.
static uint32_t integrate_transfer(const viewer3d *v, vec3f origin, vec3f dir, float step) {
  float t_near, t_far;
  if (!ray_aabb_unit(origin, dir, &t_near, &t_far)) return 0;

  float acc_r = 0.0f, acc_g = 0.0f, acc_b = 0.0f, acc_a = 0.0f;
  float t = t_near;

  while (t <= t_far && acc_a < ALPHA_THRESHOLD) {
    vec3f pos = vec3f_add(origin, vec3f_scale(dir, t));
    float val  = sample_vol_norm(v, pos);
    float mapped = wl_map(val, v->cfg.window, v->cfg.level);

    // Opacity from mapped intensity (quadratic transfer function)
    float alpha = mapped * mapped * step * 10.0f;
    if (alpha > 1.0f) alpha = 1.0f;

    uint32_t colour = apply_colormap(v->cfg.cmap_id, mapped);
    float cr = (float)((colour >>  0) & 0xFF) / 255.0f;
    float cg = (float)((colour >>  8) & 0xFF) / 255.0f;
    float cb = (float)((colour >> 16) & 0xFF) / 255.0f;

    // Front-to-back compositing
    float one_minus_a = 1.0f - acc_a;
    acc_r += one_minus_a * alpha * cr;
    acc_g += one_minus_a * alpha * cg;
    acc_b += one_minus_a * alpha * cb;
    acc_a += one_minus_a * alpha;

    t += step;
  }

  if (acc_a < 1e-4f) return 0;

  // Pre-multiplied -> straight alpha
  float inv_a = 1.0f / (acc_a > 1.0f ? 1.0f : acc_a);
  uint32_t r = (uint32_t)(acc_r * inv_a * 255.0f + 0.5f); if (r > 255) r = 255;
  uint32_t g = (uint32_t)(acc_g * inv_a * 255.0f + 0.5f); if (g > 255) g = 255;
  uint32_t b = (uint32_t)(acc_b * inv_a * 255.0f + 0.5f); if (b > 255) b = 255;
  uint32_t a = (uint32_t)(acc_a * 255.0f + 0.5f);         if (a > 255) a = 255;
  return (a << 24) | (b << 16) | (g << 8) | r;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

viewer3d *viewer3d_new(viewer3d_config cfg) {
  viewer3d *v = calloc(1, sizeof(*v));
  REQUIRE(v, "viewer3d_new: calloc failed");

  if (cfg.step_size  <= 0.0f) cfg.step_size  = DEFAULT_STEP;
  if (cfg.fov_degrees <= 0.0f) cfg.fov_degrees = DEFAULT_FOV;
  if (cfg.window     <= 0.0f) cfg.window      = 1.0f;

  v->cfg    = cfg;
  v->eye    = (vec3f){ 0.0f, 0.0f, -3.0f };
  v->target = (vec3f){ 0.5f, 0.5f,  0.5f };
  v->up     = (vec3f){ 0.0f, 1.0f,  0.0f };
  return v;
}

void viewer3d_free(viewer3d *v) {
  free(v);
}

void viewer3d_set_volume(viewer3d *v, volume *vol) {
  REQUIRE(v, "viewer3d_set_volume: null viewer");
  v->vol = vol;
}

void viewer3d_set_camera(viewer3d *v, vec3f eye, vec3f target, vec3f up) {
  REQUIRE(v, "viewer3d_set_camera: null viewer");
  v->eye    = eye;
  v->target = target;
  v->up     = up;
}

void viewer3d_orbit(viewer3d *v, float yaw_delta, float pitch_delta) {
  REQUIRE(v, "viewer3d_orbit: null viewer");

  // Rotate eye around target
  vec3f arm = vec3f_sub(v->eye, v->target);

  // Yaw around world-up
  mat4f yaw = mat4f_rotate((vec3f){0.0f, 1.0f, 0.0f}, yaw_delta);
  arm = mat4f_transform_vec(yaw, arm);

  // Pitch around camera right
  vec3f right, up_a, fwd;
  camera_frame(v->eye, v->target, v->up, &right, &up_a, &fwd);
  mat4f pitch = mat4f_rotate(right, pitch_delta);
  arm = mat4f_transform_vec(pitch, arm);

  v->eye = vec3f_add(v->target, arm);
}

void viewer3d_dolly(viewer3d *v, float delta) {
  REQUIRE(v, "viewer3d_dolly: null viewer");
  vec3f fwd = vec3f_normalize(vec3f_sub(v->target, v->eye));
  v->eye    = vec3f_add(v->eye, vec3f_scale(fwd, delta));
}

void viewer3d_screen_ray(const viewer3d *v, float sx, float sy, int w, int h,
                         vec3f *origin, vec3f *dir) {
  REQUIRE(v && origin && dir, "viewer3d_screen_ray: null arg");

  vec3f right, up_a, fwd;
  camera_frame(v->eye, v->target, v->up, &right, &up_a, &fwd);

  float aspect = (h > 0) ? (float)w / (float)h : 1.0f;
  float tan_half = tanf(v->cfg.fov_degrees * (float)M_PI / 360.0f);

  // NDC in [-1, 1]
  float nx = (2.0f * (sx + 0.5f) / (float)w - 1.0f) * aspect * tan_half;
  float ny = (1.0f - 2.0f * (sy + 0.5f) / (float)h) * tan_half;

  *origin = v->eye;
  *dir    = vec3f_normalize(
    vec3f_add(fwd, vec3f_add(vec3f_scale(right, nx), vec3f_scale(up_a, ny)))
  );
}

void viewer3d_render_cpu(viewer3d *v, uint8_t *pixels, int width, int height) {
  REQUIRE(v && pixels && width > 0 && height > 0, "viewer3d_render_cpu: invalid args");

  if (!v->vol) {
    memset(pixels, 0, (size_t)width * (size_t)height * 4);
    return;
  }

  const zarr_level_meta *m = vol_level_meta(v->vol, 0);
  // Step size in normalised [0,1] space: step_size voxels / max_dim
  int max_dim = (int)m->shape[0];
  if (m->shape[1] > max_dim) max_dim = (int)m->shape[1];
  if (m->shape[2] > max_dim) max_dim = (int)m->shape[2];
  float step = v->cfg.step_size / (float)(max_dim > 0 ? max_dim : 1);

  // Translate eye/target to normalised [0,1] volume space.
  // Convention: volume occupies the unit cube; camera coords are already
  // expressed relative to it (default eye at z=-3).
  for (int py = 0; py < height; py++) {
    for (int px = 0; px < width; px++) {
      vec3f origin, dir;
      viewer3d_screen_ray(v, (float)px, (float)py, width, height, &origin, &dir);

      uint32_t colour;
      switch (v->cfg.mode) {
        case RENDER3D_ISO_SURFACE:
          colour = integrate_iso(v, origin, dir, step);
          break;
        case RENDER3D_TRANSFER_FUNC:
          colour = integrate_transfer(v, origin, dir, step);
          break;
        default: // RENDER3D_MIP
          colour = integrate_mip(v, origin, dir, step);
          break;
      }

      uint8_t *p = pixels + (py * width + px) * 4;
      memcpy(p, &colour, 4);
    }
  }
}
