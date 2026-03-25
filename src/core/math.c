#include "core/math.h"
#include <string.h>

// ---------------------------------------------------------------------------
// mat4f helpers
// ---------------------------------------------------------------------------

// Element access: column-major, col _c, row _r → m[_c*4 + _r]
#define M4(_m, _c, _r) ((_m).m[(_c)*4 + (_r)])

mat4f mat4f_identity(void) {
  mat4f r = {0};
  M4(r,0,0) = M4(r,1,1) = M4(r,2,2) = M4(r,3,3) = 1.0f;
  return r;
}

mat4f mat4f_mul(mat4f a, mat4f b) {
  mat4f r = {0};
  for (int c = 0; c < 4; c++) {
    for (int row = 0; row < 4; row++) {
      float s = 0.0f;
      for (int k = 0; k < 4; k++) s += M4(a, k, row) * M4(b, c, k);
      M4(r, c, row) = s;
    }
  }
  return r;
}

mat4f mat4f_transpose(mat4f m) {
  mat4f r;
  for (int c = 0; c < 4; c++)
    for (int row = 0; row < 4; row++)
      M4(r, c, row) = M4(m, row, c);
  return r;
}

// Gauss-Jordan inverse; returns identity if singular.
mat4f mat4f_inverse(mat4f m) {
  float aug[4][8];
  for (int r = 0; r < 4; r++) {
    for (int c = 0; c < 4; c++) aug[r][c]   = M4(m, c, r);
    for (int c = 0; c < 4; c++) aug[r][4+c] = (r == c) ? 1.0f : 0.0f;
  }
  for (int col = 0; col < 4; col++) {
    // Partial pivot
    int pivot = col;
    for (int r = col+1; r < 4; r++)
      if (fabsf(aug[r][col]) > fabsf(aug[pivot][col])) pivot = r;
    if (pivot != col) {
      float tmp[8];
      memcpy(tmp, aug[col], sizeof(tmp));
      memcpy(aug[col], aug[pivot], sizeof(tmp));
      memcpy(aug[pivot], tmp, sizeof(tmp));
    }
    float diag = aug[col][col];
    if (fabsf(diag) < 1e-8f) return mat4f_identity(); // singular fallback
    float inv_diag = 1.0f / diag;
    for (int c = 0; c < 8; c++) aug[col][c] *= inv_diag;
    for (int r = 0; r < 4; r++) {
      if (r == col) continue;
      float factor = aug[r][col];
      for (int c = 0; c < 8; c++) aug[r][c] -= factor * aug[col][c];
    }
  }
  mat4f result;
  for (int r = 0; r < 4; r++)
    for (int c = 0; c < 4; c++)
      M4(result, c, r) = aug[r][4+c];
  return result;
}

// ---------------------------------------------------------------------------
// mat4f transform constructors
// ---------------------------------------------------------------------------

mat4f mat4f_translate(float tx, float ty, float tz) {
  mat4f r = mat4f_identity();
  M4(r,3,0) = tx; M4(r,3,1) = ty; M4(r,3,2) = tz;
  return r;
}

mat4f mat4f_rotate(vec3f axis, float angle_rad) {
  vec3f a = vec3f_normalize(axis);
  float c = cosf(angle_rad), s = sinf(angle_rad), t = 1.0f - c;
  mat4f r = mat4f_identity();
  M4(r,0,0) = t*a.x*a.x + c;       M4(r,0,1) = t*a.x*a.y + s*a.z; M4(r,0,2) = t*a.x*a.z - s*a.y;
  M4(r,1,0) = t*a.x*a.y - s*a.z;   M4(r,1,1) = t*a.y*a.y + c;     M4(r,1,2) = t*a.y*a.z + s*a.x;
  M4(r,2,0) = t*a.x*a.z + s*a.y;   M4(r,2,1) = t*a.y*a.z - s*a.x; M4(r,2,2) = t*a.z*a.z + c;
  return r;
}

mat4f mat4f_scale(float sx, float sy, float sz) {
  mat4f r = mat4f_identity();
  M4(r,0,0) = sx; M4(r,1,1) = sy; M4(r,2,2) = sz;
  return r;
}

// Reversed-Z, right-handed perspective (standard OpenGL convention).
mat4f mat4f_perspective(float fovy_rad, float aspect, float znear, float zfar) {
  float f = 1.0f / tanf(fovy_rad * 0.5f);
  float dz = znear - zfar;
  mat4f r = {0};
  M4(r,0,0) = f / aspect;
  M4(r,1,1) = f;
  M4(r,2,2) = (zfar + znear) / dz;
  M4(r,2,3) = -1.0f;
  M4(r,3,2) = (2.0f * zfar * znear) / dz;
  return r;
}

mat4f mat4f_ortho(float left, float right, float bottom, float top, float znear, float zfar) {
  mat4f r = {0};
  M4(r,0,0) =  2.0f / (right - left);
  M4(r,1,1) =  2.0f / (top - bottom);
  M4(r,2,2) = -2.0f / (zfar - znear);
  M4(r,3,0) = -(right + left)   / (right - left);
  M4(r,3,1) = -(top   + bottom) / (top   - bottom);
  M4(r,3,2) = -(zfar  + znear)  / (zfar  - znear);
  M4(r,3,3) = 1.0f;
  return r;
}

mat4f mat4f_lookat(vec3f eye, vec3f center, vec3f up) {
  vec3f f = vec3f_normalize(vec3f_sub(center, eye));
  vec3f s = vec3f_normalize(vec3f_cross(f, up));
  vec3f u = vec3f_cross(s, f);
  mat4f r = mat4f_identity();
  M4(r,0,0) =  s.x; M4(r,1,0) =  s.y; M4(r,2,0) =  s.z;
  M4(r,0,1) =  u.x; M4(r,1,1) =  u.y; M4(r,2,1) =  u.z;
  M4(r,0,2) = -f.x; M4(r,1,2) = -f.y; M4(r,2,2) = -f.z;
  M4(r,3,0) = -vec3f_dot(s, eye);
  M4(r,3,1) = -vec3f_dot(u, eye);
  M4(r,3,2) =  vec3f_dot(f, eye);
  return r;
}

vec3f mat4f_transform_point(mat4f m, vec3f p) {
  float w = M4(m,0,3)*p.x + M4(m,1,3)*p.y + M4(m,2,3)*p.z + M4(m,3,3);
  float inv_w = (w != 0.0f) ? 1.0f/w : 1.0f;
  return (vec3f){
    (M4(m,0,0)*p.x + M4(m,1,0)*p.y + M4(m,2,0)*p.z + M4(m,3,0)) * inv_w,
    (M4(m,0,1)*p.x + M4(m,1,1)*p.y + M4(m,2,1)*p.z + M4(m,3,1)) * inv_w,
    (M4(m,0,2)*p.x + M4(m,1,2)*p.y + M4(m,2,2)*p.z + M4(m,3,2)) * inv_w,
  };
}

vec3f mat4f_transform_vec(mat4f m, vec3f v) {
  return (vec3f){
    M4(m,0,0)*v.x + M4(m,1,0)*v.y + M4(m,2,0)*v.z,
    M4(m,0,1)*v.x + M4(m,1,1)*v.y + M4(m,2,1)*v.z,
    M4(m,0,2)*v.x + M4(m,1,2)*v.y + M4(m,2,2)*v.z,
  };
}

// ---------------------------------------------------------------------------
// mat3f
// ---------------------------------------------------------------------------

#define M3(_m, _c, _r) ((_m).m[(_c)*3 + (_r)])

mat3f mat3f_from_mat4(mat4f m) {
  mat3f r;
  for (int c = 0; c < 3; c++)
    for (int row = 0; row < 3; row++)
      M3(r, c, row) = M4(m, c, row);
  return r;
}

mat3f mat3f_transpose(mat3f m) {
  mat3f r;
  for (int c = 0; c < 3; c++)
    for (int row = 0; row < 3; row++)
      M3(r, c, row) = M3(m, row, c);
  return r;
}

mat3f mat3f_inverse(mat3f m) {
  float a = M3(m,0,0), b = M3(m,1,0), c = M3(m,2,0);
  float d = M3(m,0,1), e = M3(m,1,1), f = M3(m,2,1);
  float g = M3(m,0,2), h = M3(m,1,2), k = M3(m,2,2);
  float det = a*(e*k - f*h) - b*(d*k - f*g) + c*(d*h - e*g);
  if (fabsf(det) < 1e-8f) {
    mat3f id = {0};
    M3(id,0,0) = M3(id,1,1) = M3(id,2,2) = 1.0f;
    return id;
  }
  float inv = 1.0f / det;
  mat3f r;
  M3(r,0,0) =  (e*k - f*h) * inv;  M3(r,1,0) = -(b*k - c*h) * inv;  M3(r,2,0) =  (b*f - c*e) * inv;
  M3(r,0,1) = -(d*k - f*g) * inv;  M3(r,1,1) =  (a*k - c*g) * inv;  M3(r,2,1) = -(a*f - c*d) * inv;
  M3(r,0,2) =  (d*h - e*g) * inv;  M3(r,1,2) = -(a*h - b*g) * inv;  M3(r,2,2) =  (a*e - b*d) * inv;
  return r;
}

// ---------------------------------------------------------------------------
// quatf
// ---------------------------------------------------------------------------

quatf quatf_normalize(quatf q) {
  float l = sqrtf(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
  if (l < 1e-8f) return (quatf){0,0,0,1};
  float inv = 1.0f / l;
  return (quatf){q.x*inv, q.y*inv, q.z*inv, q.w*inv};
}

quatf quatf_from_axis_angle(vec3f axis, float angle_rad) {
  vec3f a = vec3f_normalize(axis);
  float half = angle_rad * 0.5f;
  float s = sinf(half);
  return (quatf){a.x*s, a.y*s, a.z*s, cosf(half)};
}

quatf quatf_mul(quatf a, quatf b) {
  return (quatf){
    a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
    a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
    a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
    a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
  };
}

mat4f quatf_to_mat4(quatf q) {
  q = quatf_normalize(q);
  float x = q.x, y = q.y, z = q.z, w = q.w;
  mat4f r = mat4f_identity();
  M4(r,0,0) = 1 - 2*(y*y + z*z);  M4(r,0,1) = 2*(x*y + z*w);     M4(r,0,2) = 2*(x*z - y*w);
  M4(r,1,0) = 2*(x*y - z*w);      M4(r,1,1) = 1 - 2*(x*x + z*z); M4(r,1,2) = 2*(y*z + x*w);
  M4(r,2,0) = 2*(x*z + y*w);      M4(r,2,1) = 2*(y*z - x*w);     M4(r,2,2) = 1 - 2*(x*x + y*y);
  return r;
}

quatf quatf_slerp(quatf a, quatf b, float t) {
  a = quatf_normalize(a);
  b = quatf_normalize(b);
  float dot = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
  // Ensure shortest path
  if (dot < 0.0f) {
    b = (quatf){-b.x, -b.y, -b.z, -b.w};
    dot = -dot;
  }
  // Fall back to lerp for nearly identical quaternions
  if (dot > 0.9995f) {
    quatf r = {
      a.x + t*(b.x - a.x), a.y + t*(b.y - a.y),
      a.z + t*(b.z - a.z), a.w + t*(b.w - a.w),
    };
    return quatf_normalize(r);
  }
  float theta0 = acosf(dot);
  float theta  = theta0 * t;
  float sin0   = sinf(theta0);
  float sa = sinf(theta0 - theta) / sin0;
  float sb = sinf(theta)          / sin0;
  return (quatf){
    sa*a.x + sb*b.x, sa*a.y + sb*b.y,
    sa*a.z + sb*b.z, sa*a.w + sb*b.w,
  };
}

// ---------------------------------------------------------------------------
// Volume sampling
// ---------------------------------------------------------------------------

// NOTE: layout is data[z*sy*sx + y*sx + x] — z is slowest axis.
float trilinear_interp(const float *data, int sx, int sy, int sz, float x, float y, float z) {
  // Clamp to valid range
  if (x < 0.0f) x = 0.0f; else if (x > sx-1) x = (float)(sx-1);
  if (y < 0.0f) y = 0.0f; else if (y > sy-1) y = (float)(sy-1);
  if (z < 0.0f) z = 0.0f; else if (z > sz-1) z = (float)(sz-1);

  int x0 = (int)x, y0 = (int)y, z0 = (int)z;
  int x1 = x0 < sx-1 ? x0+1 : x0;
  int y1 = y0 < sy-1 ? y0+1 : y0;
  int z1 = z0 < sz-1 ? z0+1 : z0;

  float xd = x - x0, yd = y - y0, zd = z - z0;
  int stride_y = sx, stride_z = sx * sy;

#define IDX(xi, yi, zi) ((zi)*stride_z + (yi)*stride_y + (xi))
  float c00 = data[IDX(x0,y0,z0)] * (1-xd) + data[IDX(x1,y0,z0)] * xd;
  float c01 = data[IDX(x0,y0,z1)] * (1-xd) + data[IDX(x1,y0,z1)] * xd;
  float c10 = data[IDX(x0,y1,z0)] * (1-xd) + data[IDX(x1,y1,z0)] * xd;
  float c11 = data[IDX(x0,y1,z1)] * (1-xd) + data[IDX(x1,y1,z1)] * xd;
  float c0  = c00 * (1-yd) + c10 * yd;
  float c1  = c01 * (1-yd) + c11 * yd;
#undef IDX
  return c0 * (1-zd) + c1 * zd;
}

// ---------------------------------------------------------------------------
// Lanczos-3 kernel
// ---------------------------------------------------------------------------

// NOTE: M_PI not guaranteed in C23 strict mode; define locally.
#ifndef M_PI_F
#  define M_PI_F 3.14159265358979323846f
#endif

float lanczos3_weight(float x) {
  if (x == 0.0f) return 1.0f;
  if (x < -3.0f || x > 3.0f) return 0.0f;
  float px = M_PI_F * x;
  return sinf(px) * sinf(px / 3.0f) / (px * px / 3.0f);
}
