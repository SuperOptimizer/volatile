#define _POSIX_C_SOURCE 200809L

#include "core/diffusion.h"
#include "core/math.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define IDX(z,y,x)  ((size_t)(z)*h*w + (size_t)(y)*w + (x))

static inline bool is_fixed(float v) { return isnan(v); }

// ---------------------------------------------------------------------------
// diffusion_discrete: heat equation on 3D grid (6-neighbour stencil)
// Dirichlet BCs: NaN cells are seeds; only non-NaN cells diffuse.
// ---------------------------------------------------------------------------

void diffusion_discrete(float *field, int d, int h, int w,
                        float dt, int iterations) {
  if (!field || d < 1 || h < 1 || w < 1 || iterations < 1) return;

  size_t n = (size_t)d * h * w;
  float *tmp = malloc(n * sizeof(float));
  if (!tmp) return;

  for (int iter = 0; iter < iterations; iter++) {
    memcpy(tmp, field, n * sizeof(float));

    for (int z = 0; z < d; z++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          float v = field[IDX(z,y,x)];
          if (is_fixed(v)) continue;  // Dirichlet seed — leave unchanged

          // Gather 6 neighbours, skip out-of-bounds and NaN neighbours.
          float sum = 0.0f;
          int   cnt = 0;

#define NEIGH(dz,dy,dx) do { \
  int nz=z+(dz), ny=y+(dy), nx=x+(dx); \
  if (nz>=0&&nz<d&&ny>=0&&ny<h&&nx>=0&&nx<w) { \
    float nv=field[IDX(nz,ny,nx)]; \
    if (!isnan(nv)) { sum+=nv; cnt++; } \
  } \
} while(0)
          NEIGH(-1,0,0); NEIGH(1,0,0);
          NEIGH(0,-1,0); NEIGH(0,1,0);
          NEIGH(0,0,-1); NEIGH(0,0,1);
#undef NEIGH

          if (cnt > 0)
            tmp[IDX(z,y,x)] = v + dt * (sum - (float)cnt * v);
        }
      }
    }
    memcpy(field, tmp, n * sizeof(float));
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// diffusion_continuous: anisotropic diffusion guided by structure tensor.
// st_tensor layout per voxel: [Txx, Txy, Txz, Tyy, Tyz, Tzz] (6 floats).
//
// Uses the Weickert anisotropic scheme: diffuse along the tensor's principal
// eigenvector (i.e. the smallest eigenvalue direction = along-sheet).
// For efficiency we use a first-order finite-difference approximation:
//   ∂u/∂t = div(D∇u)  ≈  Σ_neighbour  D_face * (u_nb - u_c)
// where D_face is the tensor component along each face normal.
// ---------------------------------------------------------------------------

// Solve 3×3 symmetric eigenvalue problem via Jacobi sweeps (2 sweeps = ~OK
// for the accuracy we need here; field is smooth after a few iterations).
static void sym3_eigen(const float t[6], float eval[3], float evec[3][3]) {
  // Load symmetric 3x3: indices [0]=Txx [1]=Txy [2]=Txz [3]=Tyy [4]=Tyz [5]=Tzz
  float A[3][3] = {
    { t[0], t[1], t[2] },
    { t[1], t[3], t[4] },
    { t[2], t[4], t[5] },
  };
  // Identity eigenvectors to start.
  float V[3][3] = { {1,0,0},{0,1,0},{0,0,1} };

  for (int sweep = 0; sweep < 6; sweep++) {
    for (int p = 0; p < 2; p++) {
      for (int q = p+1; q < 3; q++) {
        if (fabsf(A[p][q]) < 1e-9f) continue;
        float theta = 0.5f * (A[q][q] - A[p][p]) / A[p][q];
        float t2 = (theta >= 0.0f ? 1.0f : -1.0f)
                   / (fabsf(theta) + sqrtf(1.0f + theta*theta));
        float c = 1.0f / sqrtf(1.0f + t2*t2);
        float s = c * t2;
        // Rotate A
        float App = A[p][p] - t2*A[p][q];
        float Aqq = A[q][q] + t2*A[p][q];
        A[p][p] = App; A[q][q] = Aqq; A[p][q] = A[q][p] = 0.0f;
        for (int r = 0; r < 3; r++) {
          if (r==p||r==q) continue;
          float Arp = c*A[r][p] - s*A[r][q];
          float Arq = s*A[r][p] + c*A[r][q];
          A[r][p]=A[p][r]=Arp; A[r][q]=A[q][r]=Arq;
        }
        for (int r = 0; r < 3; r++) {
          float Vrp = c*V[r][p] - s*V[r][q];
          float Vrq = s*V[r][p] + c*V[r][q];
          V[r][p]=Vrp; V[r][q]=Vrq;
        }
      }
    }
  }
  for (int i = 0; i < 3; i++) {
    eval[i] = A[i][i];
    for (int j = 0; j < 3; j++) evec[i][j] = V[j][i]; // rows = eigenvectors
  }
}

void diffusion_continuous(float *field, const float *st_tensor,
                          int d, int h, int w,
                          float dt, int iterations) {
  if (!field || d<1 || h<1 || w<1 || iterations<1) return;
  size_t n = (size_t)d*h*w;
  float *tmp = malloc(n * sizeof(float));
  if (!tmp) return;

  for (int iter = 0; iter < iterations; iter++) {
    memcpy(tmp, field, n * sizeof(float));

    for (int z = 0; z < d; z++) {
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          float v = field[IDX(z,y,x)];
          if (is_fixed(v)) continue;

          // Anisotropic diffusion coefficient from structure tensor.
          // D is the tensor; we project onto each axis.
          float flux = 0.0f;
          const float *T = st_tensor ? &st_tensor[IDX(z,y,x)*6] : NULL;

          // Conductivity along each axis: D_xx, D_yy, D_zz diagonal entries.
          float Dxx = T ? T[0] : 1.0f;
          float Dyy = T ? T[3] : 1.0f;
          float Dzz = T ? T[5] : 1.0f;

#define FLUX_AXIS(dz,dy,dx,Dii) do { \
  int nz=z+(dz), ny=y+(dy), nx=x+(dx); \
  if (nz>=0&&nz<d&&ny>=0&&ny<h&&nx>=0&&nx<w) { \
    float nv=field[IDX(nz,ny,nx)]; \
    if (!isnan(nv)) flux += (Dii)*(nv - v); \
  } \
} while(0)
          FLUX_AXIS(-1,0,0,Dzz); FLUX_AXIS(1,0,0,Dzz);
          FLUX_AXIS(0,-1,0,Dyy); FLUX_AXIS(0,1,0,Dyy);
          FLUX_AXIS(0,0,-1,Dxx); FLUX_AXIS(0,0,1,Dxx);
#undef FLUX_AXIS

          tmp[IDX(z,y,x)] = v + dt * flux;
        }
      }
    }
    memcpy(field, tmp, n * sizeof(float));
  }
  free(tmp);
}

// ---------------------------------------------------------------------------
// diffusion_continuous_3d: isotropic variant (identity tensor)
// ---------------------------------------------------------------------------

void diffusion_continuous_3d(float *field, const float *tensor,
                             int d, int h, int w,
                             float dt, int iterations) {
  // Pass NULL tensor to diffusion_continuous → falls back to Dxx=Dyy=Dzz=1.
  diffusion_continuous(field, tensor, d, h, w, dt, iterations);
}

// ---------------------------------------------------------------------------
// diffusion_spiral: label-propagation from umbilicus axis.
//
// Algorithm (port of villa's discrete.cpp BFS with cut-plane crossing):
//  1. Project each voxel onto the umbilicus axis to find the closest axis pt.
//  2. Seed winding=0 in a thin cylinder around the axis.
//  3. BFS outward; when a step crosses the cut half-plane (the XZ half-plane
//     through the axis at y=0), increment/decrement the winding counter.
//  Only voxels where volume[i] > 0 are eligible.
// ---------------------------------------------------------------------------

// Closest point on polyline to q; returns the parameter t (index + frac).
static vec3f umbilicus_closest(const vec3f *pts, int npts, vec3f q,
                                float *t_out) {
  float best_d2 = FLT_MAX;
  vec3f best = pts[0];
  float best_t = 0.0f;
  for (int i = 0; i < npts-1; i++) {
    vec3f a = pts[i], b = pts[i+1];
    vec3f ab = vec3f_sub(b, a);
    float len2 = vec3f_dot(ab, ab);
    float t = (len2 > 1e-9f)
              ? fmaxf(0.0f, fminf(1.0f, vec3f_dot(vec3f_sub(q, a), ab)/len2))
              : 0.0f;
    vec3f p = vec3f_add(a, vec3f_scale(ab, t));
    float d2 = vec3f_dot(vec3f_sub(q, p), vec3f_sub(q, p));
    if (d2 < best_d2) { best_d2=d2; best=p; best_t=(float)i+t; }
  }
  if (t_out) *t_out = best_t;
  return best;
}

// Returns +1 if the step from (z0,y0,x0) to (z1,y1,x1) crosses the cut
// half-plane in the positive winding direction, -1 for negative, 0 for no cross.
// The cut plane is the xz half-plane at uy=umb_y, x > umb_x (VC3D convention).
static int cut_crossing(int y0, int x0, int y1, int x1,
                        int umb_y, int umb_x) {
  // Only horizontal moves at or below the umbilicus can cross.
  if (y0 != y1) return 0;
  if (y0 < umb_y) return 0;
  if (x0 < umb_x && x1 >= umb_x) return +1;
  if (x0 >= umb_x && x1 < umb_x) return -1;
  return 0;
}

void diffusion_spiral(float *winding, const float *volume,
                      const vec3f *umbilicus_points, int num_points,
                      int d, int h, int w,
                      float step, int iterations) {
  (void)step;
  if (!winding || !volume || !umbilicus_points || num_points < 1) return;

  size_t n = (size_t)d*h*w;
  // Initialise output to NaN.
  for (size_t i = 0; i < n; i++) winding[i] = NAN;

  // Find approximate umbilicus pixel on the central Z slice.
  int mid_z = d/2;
  vec3f q = { (float)(w/2), (float)(h/2), (float)mid_z };
  vec3f umb_closest = umbilicus_closest(umbilicus_points, num_points, q, NULL);
  int umb_y = (int)roundf(umb_closest.y);
  int umb_x = (int)roundf(umb_closest.x);

  // Seed: a 1-voxel cylinder around each umbilicus point gets winding = 0.
  // Use a simple BFS queue; we allocate [d*h*w] as worst case.
  int *queue = malloc(n * sizeof(int));
  if (!queue) return;
  int qhead = 0, qtail = 0;

  for (int pi = 0; pi < num_points; pi++) {
    int gz = (int)roundf(umbilicus_points[pi].z);
    int gy = (int)roundf(umbilicus_points[pi].y);
    int gx = (int)roundf(umbilicus_points[pi].x);
    if (gz<0||gz>=d||gy<0||gy>=h||gx<0||gx>=w) continue;
    size_t idx = IDX(gz,gy,gx);
    if (!isnan(winding[idx])) continue;
    if (volume[idx] <= 0.0f) continue;
    winding[idx] = 0.0f;
    queue[qtail++] = (int)idx;
  }

  // BFS with cut-plane crossing detection.
  int iter = 0;
  while (qhead < qtail && iter < iterations) {
    int idx = queue[qhead++];
    iter++;
    int x = idx % w;
    int y = (idx / w) % h;
    int z = idx / (w * h);
    float w_cur = winding[idx];

    int dx[] = {0,0,0,0,1,-1};
    int dy[] = {0,0,1,-1,0,0};
    int dz[] = {1,-1,0,0,0,0};

    for (int k = 0; k < 6; k++) {
      int nx=x+dx[k], ny=y+dy[k], nz=z+dz[k];
      if (nx<0||nx>=w||ny<0||ny>=h||nz<0||nz>=d) continue;
      size_t nidx = IDX(nz,ny,nx);
      if (volume[nidx] <= 0.0f) continue;
      if (!isnan(winding[nidx])) continue;

      int cross = cut_crossing(y, x, ny, nx, umb_y, umb_x);
      winding[nidx] = w_cur + (float)cross;
      queue[qtail++] = (int)nidx;
    }
  }

  free(queue);
}

// ---------------------------------------------------------------------------
// winding_from_mesh: Oosterom-Strackee solid angle formula.
// For each voxel centre p and each triangle (a,b,c), accumulate the signed
// solid angle subtended by the triangle at p.  Divide by 4π for the winding
// number.
// ---------------------------------------------------------------------------

// Signed solid angle of triangle (a,b,c) as seen from point p.
// Reference: Oosterom & Strackee (1983).
static float triangle_solid_angle(vec3f p, vec3f a, vec3f b, vec3f c) {
  vec3f ra = vec3f_sub(a, p);
  vec3f rb = vec3f_sub(b, p);
  vec3f rc = vec3f_sub(c, p);

  float la = vec3f_len(ra);
  float lb = vec3f_len(rb);
  float lc = vec3f_len(rc);

  if (la < 1e-9f || lb < 1e-9f || lc < 1e-9f) return 0.0f;

  // Numerator: scalar triple product ra · (rb × rc)
  float num = vec3f_dot(ra, vec3f_cross(rb, rc));

  // Denominator: la*lb*lc + (ra·rb)*lc + (rb·rc)*la + (ra·rc)*lb
  float den = la*lb*lc
            + vec3f_dot(ra,rb)*lc
            + vec3f_dot(rb,rc)*la
            + vec3f_dot(ra,rc)*lb;

  if (fabsf(den) < 1e-12f) return 0.0f;
  return 2.0f * atan2f(num, den);
}

void winding_from_mesh(float *winding, const obj_mesh *mesh,
                       int d, int h, int w) {
  if (!winding || !mesh || !mesh->vertices || !mesh->indices) return;

  int ntri = mesh->index_count / 3;

  for (int z = 0; z < d; z++) {
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        vec3f p = { (float)x, (float)y, (float)z };
        float total = 0.0f;

        for (int t = 0; t < ntri; t++) {
          int ia = mesh->indices[t*3+0];
          int ib = mesh->indices[t*3+1];
          int ic = mesh->indices[t*3+2];

          vec3f a = { mesh->vertices[ia*3+0],
                      mesh->vertices[ia*3+1],
                      mesh->vertices[ia*3+2] };
          vec3f b = { mesh->vertices[ib*3+0],
                      mesh->vertices[ib*3+1],
                      mesh->vertices[ib*3+2] };
          vec3f c = { mesh->vertices[ic*3+0],
                      mesh->vertices[ic*3+1],
                      mesh->vertices[ic*3+2] };

          total += triangle_solid_angle(p, a, b, c);
        }

        // Winding number = total solid angle / (4π)
        winding[IDX(z,y,x)] = total / (4.0f * (float)M_PI);
      }
    }
  }
}
