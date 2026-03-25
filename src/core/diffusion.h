#pragma once
#include "core/geom.h"
#include "core/io.h"
#include <stdint.h>

// ---------------------------------------------------------------------------
// Diffusion and winding-number solvers
// Port of villa's volume-cartographer/apps/diffusion algorithms.
// ---------------------------------------------------------------------------

// Discrete Laplacian diffusion: heat equation ∂u/∂t = Δu on a 3D grid.
// field: flat float array [d*h*w], modified in-place.
// NaN values are treated as Dirichlet boundaries (fixed, not updated).
void diffusion_discrete(float *field, int d, int h, int w,
                        float dt, int iterations);

// Continuous anisotropic diffusion guided by a structure tensor.
// st_tensor: flat array [d*h*w*6], symmetric 3x3 tensor per voxel
//            stored as (Txx,Txy,Txz,Tyy,Tyz,Tzz).
// Diffuses along the dominant tensor eigenvector (edge-preserving).
// NaN values treated as Dirichlet boundaries.
void diffusion_continuous(float *field, const float *st_tensor,
                          int d, int h, int w,
                          float dt, int iterations);

// 3D continuous diffusion — isotropic variant when no tensor is available.
// Equivalent to diffusion_continuous with an identity tensor everywhere.
void diffusion_continuous_3d(float *field, const float *tensor,
                             int d, int h, int w,
                             float dt, int iterations);

// Spiral winding solver: propagates winding numbers outward from an
// umbilicus axis, incrementing by 1 each time the cut-plane is crossed.
// winding: flat float output [d*h*w], pre-filled with NaN; on return each
//          voxel where volume[i] > 0 has its winding number assigned.
// umbilicus_points: 3D polyline defining the scroll axis.
// step: voxel step along normal direction for propagation.
void diffusion_spiral(float *winding, const float *volume,
                      const vec3f *umbilicus_points, int num_points,
                      int d, int h, int w,
                      float step, int iterations);

// Winding number field from a closed triangle mesh (Oosterom-Strackee solid
// angle formula).  Each voxel centre is assigned the generalised winding
// number w.r.t. the mesh.
// winding: flat float output [d*h*w].
void winding_from_mesh(float *winding, const obj_mesh *mesh,
                       int d, int h, int w);
