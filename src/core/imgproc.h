#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

// 1D Gaussian kernel (used for separable 2D/3D blur)
typedef struct {
  float *weights;
  int radius;  // kernel extends from -radius to +radius
  int size;    // 2*radius + 1
} gauss_kernel;

gauss_kernel *gauss_kernel_new(float sigma);
void gauss_kernel_free(gauss_kernel *k);

// Separable Gaussian blur on a 3D float buffer
// data: row-major float array of shape [depth][height][width]
// out: pre-allocated output of same shape
void gaussian_blur_3d(const float *restrict data, float *restrict out,
                      int depth, int height, int width, float sigma);

// 2D Gaussian blur (single slice)
void gaussian_blur_2d(const float *restrict data, float *restrict out,
                      int height, int width, float sigma);

// Structure tensor (3D) using Pavel Holoborodko's derivative kernel
// Input: float volume [depth][height][width]
// Output: 6-channel tensor [depth][height][width][6] = {Jzz, Jzy, Jzx, Jyy, Jyx, Jxx}
void structure_tensor_3d(const float *restrict data, float *restrict out,
                         int depth, int height, int width,
                         float deriv_sigma, float smooth_sigma);

// Euclidean Distance Transform (3D, exact)
// Input: binary mask (uint8, 0=background, nonzero=foreground)
// Output: float distance field
void edt_3d(const uint8_t *restrict mask, float *restrict dist,
            int depth, int height, int width);

// Histogram
typedef struct {
  int num_bins;
  float min_val, max_val, bin_width;
  uint32_t *bins;
  size_t total;
} histogram;

histogram *histogram_new(const float *data, size_t n, int num_bins);
float histogram_percentile(const histogram *h, float p);  // p in [0,1]
float histogram_mean(const histogram *h);
void histogram_free(histogram *h);

// Window/level (contrast adjustment)
void window_level(const float *restrict in, uint8_t *restrict out,
                  size_t n, float window, float level);

// Coherence-Enhancing Diffusion (2D)
// Anisotropic diffusion: smooths along edges, sharpens across them.
// sigma: pre-smoothing for structure tensor; rho: integration scale.
// iterations: number of diffusion steps (5-10 typical).
void ced_2d(const float *restrict data, float *restrict out,
            int height, int width, float sigma, float rho, int iterations);

// Frangi vesselness filter (3D)
// Multi-scale Hessian eigenvalue analysis for tubular/sheet structures.
// sigmas[n_sigmas]: scales to evaluate. alpha, beta, gamma: sensitivity params.
// out: float volume [depth][height][width], values in [0,1]
void frangi_3d(const float *restrict data, float *restrict out,
               int depth, int height, int width,
               const float *sigmas, int n_sigmas,
               float alpha, float beta, float gamma);

// 3D topological thinning (skeletonization) — Lee94 algorithm
// mask: uint8 binary volume (nonzero = foreground)
// skeleton: pre-allocated uint8 output of same shape
void thinning_3d(const uint8_t *restrict mask, uint8_t *restrict skeleton,
                 int depth, int height, int width);

// 3D connected component labeling using union-find (6-connectivity).
// mask:   uint8 input  [depth*height*width], nonzero = foreground
// labels: int32 output [depth*height*width], 0 = background, 1..N = component
// Returns number of components found.
int connected_components_3d(const uint8_t *restrict mask, int *restrict labels,
                             int depth, int height, int width);

// 3D Dijkstra shortest path on a cost volume (6-connected).
// cost:      float input  [depth*height*width], must be >= 0
// start_idx: flat index of the source voxel
// dist:      float output [depth*height*width], filled with shortest distances
//            (FLT_MAX for unreachable voxels)
void dijkstra_3d(const float *restrict cost, int start_idx,
                 float *restrict dist, int depth, int height, int width);
