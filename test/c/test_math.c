#include "greatest.h"
#include "core/math.h"

#include <math.h>

#define EPS 1e-5f
#define EPS_LOOSE 1e-4f

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool f_eq(float a, float b, float eps) { return fabsf(a - b) < eps; }

static bool mat4f_eq(mat4f a, mat4f b, float eps) {
  for (int i = 0; i < 16; i++)
    if (fabsf(a.m[i] - b.m[i]) >= eps) return false;
  return true;
}

static bool quatf_eq(quatf a, quatf b, float eps) {
  return fabsf(a.x-b.x)<eps && fabsf(a.y-b.y)<eps &&
         fabsf(a.z-b.z)<eps && fabsf(a.w-b.w)<eps;
}

// ---------------------------------------------------------------------------
// vec ops
// ---------------------------------------------------------------------------

TEST test_vec3f_add_sub(void) {
  vec3f a = {1,2,3}, b = {4,5,6};
  vec3f s = vec3f_add(a, b);
  ASSERT(f_eq(s.x,5,EPS) && f_eq(s.y,7,EPS) && f_eq(s.z,9,EPS));
  vec3f d = vec3f_sub(a, b);
  ASSERT(f_eq(d.x,-3,EPS) && f_eq(d.y,-3,EPS) && f_eq(d.z,-3,EPS));
  PASS();
}

TEST test_vec3f_dot_cross(void) {
  vec3f x = {1,0,0}, y = {0,1,0}, z = {0,0,1};
  ASSERT(f_eq(vec3f_dot(x,y), 0, EPS));
  ASSERT(f_eq(vec3f_dot(x,x), 1, EPS));
  vec3f c = vec3f_cross(x, y);
  ASSERT(vec3f_eq(c, z, EPS));
  PASS();
}

TEST test_vec3f_normalize(void) {
  vec3f v = {3, 0, 0};
  vec3f n = vec3f_normalize(v);
  ASSERT(f_eq(vec3f_len(n), 1.0f, EPS));
  // Zero vector returns itself without divide-by-zero
  vec3f zero = {0,0,0};
  vec3f nz = vec3f_normalize(zero);
  ASSERT(f_eq(nz.x,0,EPS) && f_eq(nz.y,0,EPS) && f_eq(nz.z,0,EPS));
  PASS();
}

TEST test_vec3f_lerp(void) {
  vec3f a = {0,0,0}, b = {1,1,1};
  vec3f m = vec3f_lerp(a, b, 0.5f);
  ASSERT(vec3f_eq(m, (vec3f){0.5f,0.5f,0.5f}, EPS));
  PASS();
}

TEST test_vec4f_ops(void) {
  vec4f a = {1,2,3,4}, b = {1,1,1,1};
  vec4f s = vec4f_add(a, b);
  ASSERT(f_eq(s.w, 5, EPS));
  ASSERT(f_eq(vec4f_len(b), 2.0f, EPS));
  PASS();
}

TEST test_vec2f_normalize(void) {
  vec2f v = {3, 4};
  vec2f n = vec2f_normalize(v);
  ASSERT(f_eq(vec2f_len(n), 1.0f, EPS));
  PASS();
}

// ---------------------------------------------------------------------------
// mat4f identity + mul
// ---------------------------------------------------------------------------

TEST test_mat4f_identity(void) {
  mat4f I = mat4f_identity();
  // diagonal should be 1, off-diagonal 0
  for (int c = 0; c < 4; c++)
    for (int r = 0; r < 4; r++)
      ASSERT(f_eq(I.m[c*4+r], (c==r)?1.0f:0.0f, EPS));
  PASS();
}

TEST test_mat4f_mul_identity(void) {
  mat4f I = mat4f_identity();
  mat4f T = mat4f_translate(1, 2, 3);
  mat4f r1 = mat4f_mul(T, I);
  mat4f r2 = mat4f_mul(I, T);
  ASSERT(mat4f_eq(r1, T, EPS));
  ASSERT(mat4f_eq(r2, T, EPS));
  PASS();
}

TEST test_mat4f_translate_point(void) {
  mat4f T = mat4f_translate(5, -3, 2);
  vec3f p = {1, 1, 1};
  vec3f r = mat4f_transform_point(T, p);
  ASSERT(f_eq(r.x, 6,  EPS));
  ASSERT(f_eq(r.y, -2, EPS));
  ASSERT(f_eq(r.z, 3,  EPS));
  PASS();
}

TEST test_mat4f_translate_vec(void) {
  // Directions should not be translated
  mat4f T = mat4f_translate(5, -3, 2);
  vec3f v = {1, 0, 0};
  vec3f r = mat4f_transform_vec(T, v);
  ASSERT(vec3f_eq(r, v, EPS));
  PASS();
}

TEST test_mat4f_scale(void) {
  mat4f S = mat4f_scale(2, 3, 4);
  vec3f p = {1, 1, 1};
  vec3f r = mat4f_transform_point(S, p);
  ASSERT(f_eq(r.x,2,EPS) && f_eq(r.y,3,EPS) && f_eq(r.z,4,EPS));
  PASS();
}

TEST test_mat4f_rotate_90_y(void) {
  // Rotating (1,0,0) by 90 degrees around Y should give (0,0,-1).
  mat4f R = mat4f_rotate((vec3f){0,1,0}, (float)M_PI_2);
  vec3f r = mat4f_transform_vec(R, (vec3f){1,0,0});
  ASSERT(f_eq(r.x,  0, EPS_LOOSE));
  ASSERT(f_eq(r.y,  0, EPS_LOOSE));
  ASSERT(f_eq(r.z, -1, EPS_LOOSE));
  PASS();
}

TEST test_mat4f_transpose(void) {
  mat4f T = mat4f_translate(1, 2, 3);
  mat4f Tt = mat4f_transpose(T);
  mat4f Ttt = mat4f_transpose(Tt);
  ASSERT(mat4f_eq(T, Ttt, EPS));
  PASS();
}

// ---------------------------------------------------------------------------
// mat4f inverse
// ---------------------------------------------------------------------------

TEST test_mat4f_inverse_identity(void) {
  mat4f I = mat4f_identity();
  mat4f inv = mat4f_inverse(I);
  ASSERT(mat4f_eq(inv, I, EPS));
  PASS();
}

TEST test_mat4f_inverse_translate(void) {
  mat4f T = mat4f_translate(3, -2, 7);
  mat4f inv = mat4f_inverse(T);
  mat4f prod = mat4f_mul(T, inv);
  ASSERT(mat4f_eq(prod, mat4f_identity(), EPS_LOOSE));
  PASS();
}

TEST test_mat4f_inverse_general(void) {
  // Scale + rotate + translate composite
  mat4f M = mat4f_mul(mat4f_mul(
    mat4f_translate(1, 2, 3),
    mat4f_rotate((vec3f){1,1,0}, 0.7f)),
    mat4f_scale(2, 3, 1));
  mat4f inv = mat4f_inverse(M);
  mat4f prod = mat4f_mul(M, inv);
  ASSERT(mat4f_eq(prod, mat4f_identity(), EPS_LOOSE));
  PASS();
}

// ---------------------------------------------------------------------------
// mat3f
// ---------------------------------------------------------------------------

TEST test_mat3f_inverse(void) {
  mat4f M4 = mat4f_scale(2, 3, 4);
  mat3f M = mat3f_from_mat4(M4);
  mat3f inv = mat3f_inverse(M);
  mat3f prod;
  for (int c = 0; c < 3; c++)
    for (int r = 0; r < 3; r++) {
      float s = 0;
      for (int k = 0; k < 3; k++) s += M.m[k*3+r] * inv.m[c*3+k];
      prod.m[c*3+r] = s;
    }
  // prod should be identity
  for (int c = 0; c < 3; c++)
    for (int r = 0; r < 3; r++)
      ASSERT(f_eq(prod.m[c*3+r], (c==r)?1.0f:0.0f, EPS_LOOSE));
  PASS();
}

// ---------------------------------------------------------------------------
// Quaternion
// ---------------------------------------------------------------------------

TEST test_quatf_identity_rotation(void) {
  quatf q = quatf_from_axis_angle((vec3f){0,1,0}, 0.0f);
  mat4f M = quatf_to_mat4(q);
  ASSERT(mat4f_eq(M, mat4f_identity(), EPS_LOOSE));
  PASS();
}

TEST test_quatf_roundtrip(void) {
  // 90 degree rotation around Z, applied via quaternion and matrix_rotate should match.
  vec3f axis = {0, 0, 1};
  float angle = (float)M_PI_2;
  mat4f Mq = quatf_to_mat4(quatf_from_axis_angle(axis, angle));
  mat4f Mr = mat4f_rotate(axis, angle);
  ASSERT(mat4f_eq(Mq, Mr, EPS_LOOSE));
  PASS();
}

TEST test_quatf_mul_compose(void) {
  // Two 90° rotations around Z should equal 180° around Z.
  quatf q90  = quatf_from_axis_angle((vec3f){0,0,1}, (float)M_PI_2);
  quatf q180 = quatf_from_axis_angle((vec3f){0,0,1}, (float)M_PI);
  quatf composed = quatf_normalize(quatf_mul(q90, q90));
  // Both quaternions represent the same rotation; they may differ by sign.
  bool same = quatf_eq(composed, q180, EPS_LOOSE)
           || quatf_eq(composed, (quatf){-q180.x,-q180.y,-q180.z,-q180.w}, EPS_LOOSE);
  ASSERT(same);
  PASS();
}

TEST test_quatf_slerp_endpoints(void) {
  quatf a = quatf_from_axis_angle((vec3f){1,0,0}, 0.0f);
  quatf b = quatf_from_axis_angle((vec3f){1,0,0}, (float)M_PI_2);
  quatf s0 = quatf_slerp(a, b, 0.0f);
  quatf s1 = quatf_slerp(a, b, 1.0f);
  ASSERT(quatf_eq(quatf_normalize(s0), quatf_normalize(a), EPS_LOOSE));
  ASSERT(quatf_eq(quatf_normalize(s1), quatf_normalize(b), EPS_LOOSE));
  PASS();
}

TEST test_quatf_slerp_midpoint(void) {
  // Midpoint between identity and 90° around X should be 45° around X.
  quatf a   = quatf_from_axis_angle((vec3f){1,0,0}, 0.0f);
  quatf b   = quatf_from_axis_angle((vec3f){1,0,0}, (float)M_PI_2);
  quatf mid = quatf_slerp(a, b, 0.5f);
  quatf exp = quatf_from_axis_angle((vec3f){1,0,0}, (float)M_PI_4);
  ASSERT(quatf_eq(quatf_normalize(mid), quatf_normalize(exp), EPS_LOOSE));
  PASS();
}

// ---------------------------------------------------------------------------
// Trilinear interpolation
// ---------------------------------------------------------------------------

TEST test_trilinear_corners(void) {
  // 2x2x2 volume with known values at each corner.
  float data[8] = {0,1, 2,3, 4,5, 6,7};
  // layout: data[z*4 + y*2 + x]
  // corner (0,0,0) = 0, (1,0,0) = 1, (0,1,0) = 2, etc.
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 0,0,0), 0.0f, EPS));
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 1,0,0), 1.0f, EPS));
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 0,1,0), 2.0f, EPS));
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 1,1,0), 3.0f, EPS));
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 0,0,1), 4.0f, EPS));
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 1,1,1), 7.0f, EPS));
  PASS();
}

TEST test_trilinear_midpoint(void) {
  // Uniform volume — midpoint must equal that value.
  float data[8] = {1,1, 1,1, 1,1, 1,1};
  ASSERT(f_eq(trilinear_interp(data,2,2,2, 0.5f,0.5f,0.5f), 1.0f, EPS));

  // Linear ramp in x: x=0 → 0, x=1 → 1. Mid should be 0.5.
  float ramp[8] = {0,1, 0,1, 0,1, 0,1};
  ASSERT(f_eq(trilinear_interp(ramp,2,2,2, 0.5f,0.0f,0.0f), 0.5f, EPS));
  PASS();
}

TEST test_trilinear_clamp(void) {
  float data[8] = {1,2,3,4,5,6,7,8};
  // Out-of-bounds coordinates should clamp without crashing.
  float v = trilinear_interp(data,2,2,2, -1, -1, -1);
  ASSERT(f_eq(v, data[0], EPS));
  float v2 = trilinear_interp(data,2,2,2, 99, 99, 99);
  ASSERT(f_eq(v2, data[7], EPS));
  PASS();
}

// ---------------------------------------------------------------------------
// Lanczos-3
// ---------------------------------------------------------------------------

TEST test_lanczos3_weight(void) {
  // Center weight must be 1.
  ASSERT(f_eq(lanczos3_weight(0.0f), 1.0f, EPS));
  // Weights at integer offsets beyond 0 must be 0 (sinc property).
  ASSERT(f_eq(lanczos3_weight(1.0f), 0.0f, EPS));
  ASSERT(f_eq(lanczos3_weight(2.0f), 0.0f, EPS));
  ASSERT(f_eq(lanczos3_weight(-1.0f), 0.0f, EPS));
  // Beyond the window: 0.
  ASSERT(f_eq(lanczos3_weight(4.0f), 0.0f, EPS));
  ASSERT(f_eq(lanczos3_weight(-4.0f), 0.0f, EPS));
  // Symmetry.
  ASSERT(f_eq(lanczos3_weight(0.5f), lanczos3_weight(-0.5f), EPS));
  ASSERT(f_eq(lanczos3_weight(1.5f), lanczos3_weight(-1.5f), EPS));
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(vec_suite) {
  RUN_TEST(test_vec3f_add_sub);
  RUN_TEST(test_vec3f_dot_cross);
  RUN_TEST(test_vec3f_normalize);
  RUN_TEST(test_vec3f_lerp);
  RUN_TEST(test_vec4f_ops);
  RUN_TEST(test_vec2f_normalize);
}

SUITE(mat4_suite) {
  RUN_TEST(test_mat4f_identity);
  RUN_TEST(test_mat4f_mul_identity);
  RUN_TEST(test_mat4f_translate_point);
  RUN_TEST(test_mat4f_translate_vec);
  RUN_TEST(test_mat4f_scale);
  RUN_TEST(test_mat4f_rotate_90_y);
  RUN_TEST(test_mat4f_transpose);
  RUN_TEST(test_mat4f_inverse_identity);
  RUN_TEST(test_mat4f_inverse_translate);
  RUN_TEST(test_mat4f_inverse_general);
  RUN_TEST(test_mat3f_inverse);
}

SUITE(quat_suite) {
  RUN_TEST(test_quatf_identity_rotation);
  RUN_TEST(test_quatf_roundtrip);
  RUN_TEST(test_quatf_mul_compose);
  RUN_TEST(test_quatf_slerp_endpoints);
  RUN_TEST(test_quatf_slerp_midpoint);
}

SUITE(volume_suite) {
  RUN_TEST(test_trilinear_corners);
  RUN_TEST(test_trilinear_midpoint);
  RUN_TEST(test_trilinear_clamp);
  RUN_TEST(test_lanczos3_weight);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(vec_suite);
  RUN_SUITE(mat4_suite);
  RUN_SUITE(quat_suite);
  RUN_SUITE(volume_suite);
  GREATEST_MAIN_END();
}
