#include "geom.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// ---------------------------------------------------------------------------
// tri_mesh lifecycle
// ---------------------------------------------------------------------------

tri_mesh *tri_mesh_new(int num_verts, int num_faces) {
  assert(num_verts > 0 && num_faces > 0);
  tri_mesh *m = calloc(1, sizeof(*m));
  if (!m) return NULL;
  m->verts   = calloc((size_t)num_verts, sizeof(vec3f));
  m->indices = calloc((size_t)(num_faces * 3), sizeof(int));
  if (!m->verts || !m->indices) { free(m->verts); free(m->indices); free(m); return NULL; }
  m->num_verts = num_verts;
  m->num_faces = num_faces;
  return m;
}

void tri_mesh_free(tri_mesh *m) {
  if (!m) return;
  free(m->verts);
  free(m->indices);
  free(m);
}

// ---------------------------------------------------------------------------
// mesh_voxelize — ray casting (X-axis ray per voxel center)
// ---------------------------------------------------------------------------

// Returns true if point p is inside the mesh by counting X-ray crossings.
// Uses the "top-left" edge rule to avoid double-counting on shared edges:
// edges with f_i == 0 are only counted if the edge goes in a specific direction.
static bool point_in_mesh(const tri_mesh *m, vec3f p) {
  int crossings = 0;
  for (int f = 0; f < m->num_faces; f++) {
    vec3f a = m->verts[m->indices[f*3+0]];
    vec3f b = m->verts[m->indices[f*3+1]];
    vec3f c = m->verts[m->indices[f*3+2]];

    // Shoot a ray from p in +X direction.
    float ay = a.y - p.y, by = b.y - p.y, cy = c.y - p.y;
    float az = a.z - p.z, bz = b.z - p.z, cz = c.z - p.z;

    // Edge function signs for the YZ projection.
    float f0 = ay * bz - az * by;
    float f1 = by * cz - bz * cy;
    float f2 = cy * az - cz * ay;

    // Use strict inequality for "top-left" rule: ties go to > 0 direction.
    // This avoids double-counting at shared edges.
    bool pass_pos = (f0 > 0 || (f0 == 0.0f && f1 > 0) || (f0 == 0.0f && f1 == 0.0f && f2 > 0));
    bool all_pos  = (f0 >= 0 && f1 >= 0 && f2 >= 0 && pass_pos);
    bool all_neg  = (f0 <= 0 && f1 <= 0 && f2 <= 0 && !pass_pos && (f0 < 0 || f1 < 0 || f2 < 0));

    if (!all_pos && !all_neg) continue;

    float det = f0 + f1 + f2;
    if (fabsf(det) < 1e-12f) continue;
    float t = (f0 * a.x + f1 * b.x + f2 * c.x) / det;
    if (t > p.x) crossings++;
  }
  return (crossings & 1) != 0;
}

uint8_t *mesh_voxelize(const tri_mesh *m, int d, int h, int w) {
  assert(m && d > 0 && h > 0 && w > 0);
  uint8_t *mask = calloc((size_t)(d * h * w), 1);
  if (!mask) return NULL;

  // Compute mesh bounding box to map voxel centers to world coords.
  vec3f lo = m->verts[0], hi = m->verts[0];
  for (int i = 1; i < m->num_verts; i++) {
    vec3f v = m->verts[i];
    if (v.x < lo.x) lo.x = v.x; if (v.x > hi.x) hi.x = v.x;
    if (v.y < lo.y) lo.y = v.y; if (v.y > hi.y) hi.y = v.y;
    if (v.z < lo.z) lo.z = v.z; if (v.z > hi.z) hi.z = v.z;
  }
  float sx = (hi.x - lo.x) / (float)w;
  float sy = (hi.y - lo.y) / (float)h;
  float sz = (hi.z - lo.z) / (float)d;

  for (int iz = 0; iz < d; iz++) {
    for (int iy = 0; iy < h; iy++) {
      for (int ix = 0; ix < w; ix++) {
        vec3f p = { lo.x + (ix + 0.5f) * sx,
                    lo.y + (iy + 0.5f) * sy,
                    lo.z + (iz + 0.5f) * sz };
        if (point_in_mesh(m, p))
          mask[iz * h * w + iy * w + ix] = 1;
      }
    }
  }
  return mask;
}

// ---------------------------------------------------------------------------
// mesh_quality
// ---------------------------------------------------------------------------

static float angle_between(vec3f a, vec3f b) {
  float d = vec3f_dot(a, b) / (vec3f_len(a) * vec3f_len(b) + 1e-12f);
  if (d >  1.0f) d =  1.0f;
  if (d < -1.0f) d = -1.0f;
  return acosf(d) * (180.0f / 3.14159265f);
}

mesh_quality_t mesh_quality(const tri_mesh *m) {
  assert(m);
  mesh_quality_t q = { .min_angle_deg = 180.0f, .max_angle_deg = 0.0f };
  double angle_sum = 0.0;
  int angle_count = 0;
  float worst_aspect = 0.0f;

  for (int f = 0; f < m->num_faces; f++) {
    vec3f a = m->verts[m->indices[f*3+0]];
    vec3f b = m->verts[m->indices[f*3+1]];
    vec3f c = m->verts[m->indices[f*3+2]];

    vec3f ab = vec3f_sub(b, a), ac = vec3f_sub(c, a);
    vec3f bc = vec3f_sub(c, b), ba = vec3f_sub(a, b);
    vec3f ca = vec3f_sub(a, c), cb = vec3f_sub(b, c);

    float ang[3] = { angle_between(ab, ac), angle_between(ba, bc), angle_between(ca, cb) };
    for (int i = 0; i < 3; i++) {
      if (ang[i] < q.min_angle_deg) q.min_angle_deg = ang[i];
      if (ang[i] > q.max_angle_deg) q.max_angle_deg = ang[i];
      angle_sum += ang[i];
      angle_count++;
    }

    float e0 = vec3f_len(ab), e1 = vec3f_len(bc), e2 = vec3f_len(ca);
    float emax = e0 > e1 ? (e0 > e2 ? e0 : e2) : (e1 > e2 ? e1 : e2);
    float emin = e0 < e1 ? (e0 < e2 ? e0 : e2) : (e1 < e2 ? e1 : e2);
    float aspect = emin > 1e-12f ? emax / emin : 1e6f;
    if (aspect > worst_aspect) worst_aspect = aspect;
  }

  q.avg_angle_deg   = angle_count > 0 ? (float)(angle_sum / angle_count) : 0.0f;
  q.max_aspect_ratio = worst_aspect;

  // Self-intersection check: O(n^2) triangle pair test, capped at 10000 pairs.
  int max_pairs = 10000, pairs = 0;
  for (int i = 0; i < m->num_faces && pairs < max_pairs; i++) {
    for (int j = i + 1; j < m->num_faces && pairs < max_pairs; j++, pairs++) {
      // Check bounding-box overlap as a fast reject.
      vec3f a0 = m->verts[m->indices[i*3+0]];
      vec3f a1 = m->verts[m->indices[i*3+1]];
      vec3f a2 = m->verts[m->indices[i*3+2]];
      vec3f b0 = m->verts[m->indices[j*3+0]];
      vec3f b1 = m->verts[m->indices[j*3+1]];
      vec3f b2 = m->verts[m->indices[j*3+2]];
      // AABB overlap test (one axis sufficient for quick reject).
      float aminx = fminf(a0.x, fminf(a1.x, a2.x));
      float amaxx = fmaxf(a0.x, fmaxf(a1.x, a2.x));
      float bminx = fminf(b0.x, fminf(b1.x, b2.x));
      float bmaxx = fmaxf(b0.x, fmaxf(b1.x, b2.x));
      if (amaxx < bminx || bmaxx < aminx) continue;
      // Shared-vertex neighbors can't be self-intersecting in the geometric sense.
      bool shared = false;
      for (int p = 0; p < 3 && !shared; p++)
        for (int r = 0; r < 3 && !shared; r++)
          if (m->indices[i*3+p] == m->indices[j*3+r]) shared = true;
      if (shared) continue;
      // Two-sided signed-volume separating-plane test (with small epsilon to
      // avoid counting coplanar-touching faces as self-intersecting).
      const float eps = 1e-4f;
      vec3f na = vec3f_normalize(vec3f_cross(vec3f_sub(a1,a0), vec3f_sub(a2,a0)));
      float d0 = vec3f_dot(vec3f_sub(b0,a0),na);
      float d1 = vec3f_dot(vec3f_sub(b1,a0),na);
      float d2 = vec3f_dot(vec3f_sub(b2,a0),na);
      if ((d0>eps&&d1>eps&&d2>eps)||(d0<-eps&&d1<-eps&&d2<-eps)) continue;
      // Also test B's plane against A's vertices.
      vec3f nb = vec3f_normalize(vec3f_cross(vec3f_sub(b1,b0), vec3f_sub(b2,b0)));
      float e0 = vec3f_dot(vec3f_sub(a0,b0),nb);
      float e1 = vec3f_dot(vec3f_sub(a1,b0),nb);
      float e2 = vec3f_dot(vec3f_sub(a2,b0),nb);
      if ((e0>eps&&e1>eps&&e2>eps)||(e0<-eps&&e1<-eps&&e2<-eps)) continue;
      q.self_intersections++;
    }
  }
  return q;
}

// ---------------------------------------------------------------------------
// mesh_simplify — quadric error metric edge collapse
// ---------------------------------------------------------------------------

// 4x4 symmetric quadric matrix stored as upper triangle (10 floats).
typedef struct { float q[10]; } quadric;

static quadric quadric_zero(void) { quadric r; memset(r.q, 0, sizeof(r.q)); return r; }

static quadric quadric_from_plane(float a, float b, float c, float d) {
  // Q = [aa ab ac ad; ab bb bc bd; ac bc cc cd; ad bd cd dd]
  quadric r;
  r.q[0]=a*a; r.q[1]=a*b; r.q[2]=a*c; r.q[3]=a*d;
  r.q[4]=b*b; r.q[5]=b*c; r.q[6]=b*d;
  r.q[7]=c*c; r.q[8]=c*d;
  r.q[9]=d*d;
  return r;
}

static void quadric_add(quadric *dst, const quadric *src) {
  for (int i = 0; i < 10; i++) dst->q[i] += src->q[i];
}

static float quadric_eval(const quadric *q, vec3f v) {
  float x=v.x, y=v.y, z=v.z;
  return q->q[0]*x*x + 2*q->q[1]*x*y + 2*q->q[2]*x*z + 2*q->q[3]*x
       + q->q[4]*y*y + 2*q->q[5]*y*z + 2*q->q[6]*y
       + q->q[7]*z*z + 2*q->q[8]*z
       + q->q[9];
}

tri_mesh *mesh_simplify(const tri_mesh *m, int target_faces) {
  assert(m && target_faces > 0);
  if (m->num_faces <= target_faces) return tri_mesh_new(m->num_verts, m->num_faces);

  // Copy vertex/index arrays (we'll collapse in-place then compact).
  int nv = m->num_verts, nf = m->num_faces;
  vec3f *verts = malloc((size_t)nv * sizeof(vec3f));
  int   *idx   = malloc((size_t)(nf * 3) * sizeof(int));
  int   *remap = malloc((size_t)nv * sizeof(int));   // remap[i] = canonical vertex
  if (!verts || !idx || !remap) { free(verts); free(idx); free(remap); return NULL; }
  memcpy(verts, m->verts,   (size_t)nv * sizeof(vec3f));
  memcpy(idx,   m->indices, (size_t)(nf * 3) * sizeof(int));
  for (int i = 0; i < nv; i++) remap[i] = i;

  // Build per-vertex quadrics from incident faces.
  quadric *qs = calloc((size_t)nv, sizeof(quadric));
  if (!qs) { free(verts); free(idx); free(remap); return NULL; }
  for (int f = 0; f < nf; f++) {
    vec3f a = verts[idx[f*3+0]], b = verts[idx[f*3+1]], c = verts[idx[f*3+2]];
    vec3f n = vec3f_normalize(vec3f_cross(vec3f_sub(b,a), vec3f_sub(c,a)));
    float d = -vec3f_dot(n, a);
    quadric q = quadric_from_plane(n.x, n.y, n.z, d);
    quadric_add(&qs[idx[f*3+0]], &q);
    quadric_add(&qs[idx[f*3+1]], &q);
    quadric_add(&qs[idx[f*3+2]], &q);
  }

  int active_faces = nf;
  // Greedy collapse: find the cheapest edge each iteration.
  while (active_faces > target_faces) {
    float best_cost = 1e30f;
    int best_i = -1, best_j = -1;
    vec3f best_v = {0,0,0};

    for (int f = 0; f < nf && best_i == -1; f++) {
      if (idx[f*3] < 0) continue;
      for (int e = 0; e < 3; e++) {
        int vi = remap[idx[f*3 + e]];
        int vj = remap[idx[f*3 + (e+1)%3]];
        if (vi == vj) continue;
        quadric combined = qs[vi];
        quadric_add(&combined, &qs[vj]);
        vec3f mid = vec3f_scale(vec3f_add(verts[vi], verts[vj]), 0.5f);
        float cost = quadric_eval(&combined, mid);
        if (cost < best_cost) {
          best_cost = cost; best_i = vi; best_j = vj; best_v = mid;
        }
      }
    }
    if (best_i < 0) break;

    // Collapse best_j into best_i.
    verts[best_i] = best_v;
    quadric_add(&qs[best_i], &qs[best_j]);
    // Remap all references to best_j -> best_i.
    for (int i = 0; i < nv; i++) if (remap[i] == best_j) remap[i] = best_i;
    // Degenerate faces (two or more identical vertices after remap) -> mark removed.
    for (int f = 0; f < nf; f++) {
      if (idx[f*3] < 0) continue;
      int a = remap[idx[f*3]], b = remap[idx[f*3+1]], c = remap[idx[f*3+2]];
      if (a == b || b == c || a == c) { idx[f*3] = -1; active_faces--; }
    }
  }

  // Compact into a new mesh.
  int *new_vmap = calloc((size_t)nv, sizeof(int));
  if (!new_vmap) { free(verts); free(idx); free(remap); free(qs); return NULL; }
  int new_nv = 0;
  for (int i = 0; i < nv; i++) if (remap[i] == i) new_vmap[i] = new_nv++;

  tri_mesh *out = tri_mesh_new(new_nv > 0 ? new_nv : 1, active_faces > 0 ? active_faces : 1);
  if (!out) { free(verts); free(idx); free(remap); free(qs); free(new_vmap); return NULL; }
  out->num_verts = new_nv;
  out->num_faces = active_faces;
  for (int i = 0; i < nv; i++) if (remap[i] == i) out->verts[new_vmap[i]] = verts[i];
  int fi = 0;
  for (int f = 0; f < nf; f++) {
    if (idx[f*3] < 0) continue;
    out->indices[fi*3+0] = new_vmap[remap[idx[f*3+0]]];
    out->indices[fi*3+1] = new_vmap[remap[idx[f*3+1]]];
    out->indices[fi*3+2] = new_vmap[remap[idx[f*3+2]]];
    fi++;
  }
  free(verts); free(idx); free(remap); free(qs); free(new_vmap);
  return out;
}

// ---------------------------------------------------------------------------
// mesh_smooth — Laplacian smoothing
// ---------------------------------------------------------------------------

void mesh_smooth(tri_mesh *m, int iterations, float lambda) {
  assert(m && iterations >= 0);
  if (iterations == 0 || lambda == 0.0f) return;

  // Build adjacency: sum of neighbor positions and count per vertex.
  vec3f *tmp = malloc((size_t)m->num_verts * sizeof(vec3f));
  int   *cnt = malloc((size_t)m->num_verts * sizeof(int));
  if (!tmp || !cnt) { free(tmp); free(cnt); return; }

  for (int it = 0; it < iterations; it++) {
    memset(tmp, 0, (size_t)m->num_verts * sizeof(vec3f));
    memset(cnt, 0, (size_t)m->num_verts * sizeof(int));

    for (int f = 0; f < m->num_faces; f++) {
      int a = m->indices[f*3+0], b = m->indices[f*3+1], c = m->indices[f*3+2];
      tmp[a] = vec3f_add(tmp[a], m->verts[b]); tmp[a] = vec3f_add(tmp[a], m->verts[c]); cnt[a] += 2;
      tmp[b] = vec3f_add(tmp[b], m->verts[a]); tmp[b] = vec3f_add(tmp[b], m->verts[c]); cnt[b] += 2;
      tmp[c] = vec3f_add(tmp[c], m->verts[a]); tmp[c] = vec3f_add(tmp[c], m->verts[b]); cnt[c] += 2;
    }

    for (int i = 0; i < m->num_verts; i++) {
      if (cnt[i] == 0) continue;
      vec3f avg = vec3f_scale(tmp[i], 1.0f / (float)cnt[i]);
      m->verts[i] = vec3f_add(m->verts[i], vec3f_scale(vec3f_sub(avg, m->verts[i]), lambda));
    }
  }
  free(tmp); free(cnt);
}
