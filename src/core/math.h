#pragma once
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef struct { float x, y; }          vec2f;
typedef struct { float x, y, z; }       vec3f;
typedef struct { float x, y, z, w; }    vec4f;
typedef struct { float m[9]; }          mat3f;   // column-major 3x3
typedef struct { float m[16]; }         mat4f;   // column-major 4x4
typedef struct { float x, y, z, w; }    quatf;   // w is scalar part

// ---------------------------------------------------------------------------
// vec2f
// ---------------------------------------------------------------------------

static inline vec2f vec2f_add(vec2f a, vec2f b)         { return (vec2f){a.x+b.x, a.y+b.y}; }
static inline vec2f vec2f_sub(vec2f a, vec2f b)         { return (vec2f){a.x-b.x, a.y-b.y}; }
static inline vec2f vec2f_scale(vec2f v, float s)       { return (vec2f){v.x*s, v.y*s}; }
static inline float vec2f_dot(vec2f a, vec2f b)         { return a.x*b.x + a.y*b.y; }
static inline float vec2f_len(vec2f v)                  { return sqrtf(vec2f_dot(v, v)); }
static inline vec2f vec2f_normalize(vec2f v) {
  float l = vec2f_len(v);
  return l > 0.0f ? vec2f_scale(v, 1.0f/l) : v;
}
static inline vec2f vec2f_lerp(vec2f a, vec2f b, float t) {
  return vec2f_add(vec2f_scale(a, 1.0f-t), vec2f_scale(b, t));
}

// ---------------------------------------------------------------------------
// vec3f
// ---------------------------------------------------------------------------

static inline vec3f vec3f_add(vec3f a, vec3f b)         { return (vec3f){a.x+b.x, a.y+b.y, a.z+b.z}; }
static inline vec3f vec3f_sub(vec3f a, vec3f b)         { return (vec3f){a.x-b.x, a.y-b.y, a.z-b.z}; }
static inline vec3f vec3f_scale(vec3f v, float s)       { return (vec3f){v.x*s, v.y*s, v.z*s}; }
static inline float vec3f_dot(vec3f a, vec3f b)         { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float vec3f_len(vec3f v)                  { return sqrtf(vec3f_dot(v, v)); }
static inline vec3f vec3f_normalize(vec3f v) {
  float l = vec3f_len(v);
  return l > 0.0f ? vec3f_scale(v, 1.0f/l) : v;
}
static inline vec3f vec3f_cross(vec3f a, vec3f b) {
  return (vec3f){a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static inline vec3f vec3f_lerp(vec3f a, vec3f b, float t) {
  return vec3f_add(vec3f_scale(a, 1.0f-t), vec3f_scale(b, t));
}
static inline bool vec3f_eq(vec3f a, vec3f b, float eps) {
  return fabsf(a.x-b.x) < eps && fabsf(a.y-b.y) < eps && fabsf(a.z-b.z) < eps;
}

// ---------------------------------------------------------------------------
// vec4f
// ---------------------------------------------------------------------------

static inline vec4f vec4f_add(vec4f a, vec4f b)         { return (vec4f){a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w}; }
static inline vec4f vec4f_sub(vec4f a, vec4f b)         { return (vec4f){a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w}; }
static inline vec4f vec4f_scale(vec4f v, float s)       { return (vec4f){v.x*s, v.y*s, v.z*s, v.w*s}; }
static inline float vec4f_dot(vec4f a, vec4f b)         { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline float vec4f_len(vec4f v)                  { return sqrtf(vec4f_dot(v, v)); }
static inline vec4f vec4f_normalize(vec4f v) {
  float l = vec4f_len(v);
  return l > 0.0f ? vec4f_scale(v, 1.0f/l) : v;
}
static inline vec4f vec4f_lerp(vec4f a, vec4f b, float t) {
  return vec4f_add(vec4f_scale(a, 1.0f-t), vec4f_scale(b, t));
}

// ---------------------------------------------------------------------------
// mat4f — column-major; column j starts at m[j*4]
// ---------------------------------------------------------------------------

mat4f mat4f_identity(void);
mat4f mat4f_mul(mat4f a, mat4f b);
mat4f mat4f_transpose(mat4f m);
mat4f mat4f_inverse(mat4f m);

mat4f mat4f_translate(float tx, float ty, float tz);
mat4f mat4f_rotate(vec3f axis, float angle_rad);
mat4f mat4f_scale(float sx, float sy, float sz);
mat4f mat4f_perspective(float fovy_rad, float aspect, float znear, float zfar);
mat4f mat4f_ortho(float left, float right, float bottom, float top, float znear, float zfar);
mat4f mat4f_lookat(vec3f eye, vec3f center, vec3f up);

vec3f mat4f_transform_point(mat4f m, vec3f p);   // applies translation
vec3f mat4f_transform_vec(mat4f m, vec3f v);     // ignores translation

// ---------------------------------------------------------------------------
// mat3f — column-major; column j starts at m[j*3]
// ---------------------------------------------------------------------------

mat3f mat3f_from_mat4(mat4f m);
mat3f mat3f_transpose(mat3f m);
mat3f mat3f_inverse(mat3f m);

// ---------------------------------------------------------------------------
// quatf  (x,y,z = vector part, w = scalar)
// ---------------------------------------------------------------------------

quatf quatf_from_axis_angle(vec3f axis, float angle_rad);
quatf quatf_mul(quatf a, quatf b);
quatf quatf_normalize(quatf q);
mat4f quatf_to_mat4(quatf q);
quatf quatf_slerp(quatf a, quatf b, float t);

// ---------------------------------------------------------------------------
// Scalar utilities
// ---------------------------------------------------------------------------

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ---------------------------------------------------------------------------
// Volume / signal utilities
// ---------------------------------------------------------------------------

// Trilinear interpolation into a float volume of size sx*sy*sz (row-major: z fastest).
float trilinear_interp(const float *data, int sx, int sy, int sz, float x, float y, float z);

// Lanczos-3 kernel weight for a single sample offset x.
float lanczos3_weight(float x);
