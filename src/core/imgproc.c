#include "imgproc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Gaussian kernel
// ---------------------------------------------------------------------------

gauss_kernel *gauss_kernel_new(float sigma) {
  int radius = (int)ceilf(3.0f * sigma);
  if (radius < 1) radius = 1;
  int size = 2 * radius + 1;

  gauss_kernel *k = malloc(sizeof(gauss_kernel));
  if (!k) return NULL;
  k->weights = malloc((size_t)size * sizeof(float));
  if (!k->weights) { free(k); return NULL; }
  k->radius = radius;
  k->size = size;

  float sum = 0.0f;
  for (int i = 0; i < size; i++) {
    float x = (float)(i - radius);
    float w = expf(-0.5f * x * x / (sigma * sigma));
    k->weights[i] = w;
    sum += w;
  }
  // normalize
  float inv = 1.0f / sum;
  for (int i = 0; i < size; i++) k->weights[i] *= inv;

  return k;
}

void gauss_kernel_free(gauss_kernel *k) {
  if (!k) return;
  free(k->weights);
  free(k);
}

// ---------------------------------------------------------------------------
// Internal: 1D convolution with replicate-edge clamping
// src[n], dst[n], kernel radius r, weights w[2r+1]
// ---------------------------------------------------------------------------

static void conv1d(const float *restrict src, float *restrict dst, int n,
                   const float *restrict w, int r) {
  for (int i = 0; i < n; i++) {
    float acc = 0.0f;
    for (int j = -r; j <= r; j++) {
      int idx = i + j;
      if (idx < 0) idx = 0;
      else if (idx >= n) idx = n - 1;
      acc += src[idx] * w[j + r];
    }
    dst[i] = acc;
  }
}

// ---------------------------------------------------------------------------
// 2D Gaussian blur (single slice)
// ---------------------------------------------------------------------------

void gaussian_blur_2d(const float *restrict data, float *restrict out,
                      int height, int width, float sigma) {
  gauss_kernel *k = gauss_kernel_new(sigma);
  if (!k) return;

  int n = height * width;
  float *tmp = malloc((size_t)n * sizeof(float));
  if (!tmp) { gauss_kernel_free(k); return; }

  // Pass 1: convolve along X (rows)
  for (int y = 0; y < height; y++) {
    conv1d(data + (size_t)y * width, tmp + (size_t)y * width, width, k->weights, k->radius);
  }

  // Pass 2: convolve along Y (columns) – gather/scatter via temp row
  float *col_in  = malloc((size_t)height * sizeof(float));
  float *col_out = malloc((size_t)height * sizeof(float));
  if (!col_in || !col_out) { free(col_in); free(col_out); free(tmp); gauss_kernel_free(k); return; }

  for (int x = 0; x < width; x++) {
    for (int y = 0; y < height; y++) col_in[y] = tmp[(size_t)y * width + x];
    conv1d(col_in, col_out, height, k->weights, k->radius);
    for (int y = 0; y < height; y++) out[(size_t)y * width + x] = col_out[y];
  }

  free(col_in);
  free(col_out);
  free(tmp);
  gauss_kernel_free(k);
}

// ---------------------------------------------------------------------------
// 3D Gaussian blur (separable: X, Y, Z passes)
// ---------------------------------------------------------------------------

void gaussian_blur_3d(const float *restrict data, float *restrict out,
                      int depth, int height, int width, float sigma) {
  gauss_kernel *k = gauss_kernel_new(sigma);
  if (!k) return;

  size_t vol = (size_t)depth * height * width;
  float *tmp1 = malloc(vol * sizeof(float));
  float *tmp2 = malloc(vol * sizeof(float));
  if (!tmp1 || !tmp2) { free(tmp1); free(tmp2); gauss_kernel_free(k); return; }

  // Pass 1: X
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      size_t off = ((size_t)z * height + y) * width;
      conv1d(data + off, tmp1 + off, width, k->weights, k->radius);
    }
  }

  // Pass 2: Y
  float *col = malloc((size_t)height * sizeof(float));
  float *col2 = malloc((size_t)height * sizeof(float));
  if (!col || !col2) { free(col); free(col2); free(tmp1); free(tmp2); gauss_kernel_free(k); return; }
  for (int z = 0; z < depth; z++) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) col[y] = tmp1[((size_t)z * height + y) * width + x];
      conv1d(col, col2, height, k->weights, k->radius);
      for (int y = 0; y < height; y++) tmp2[((size_t)z * height + y) * width + x] = col2[y];
    }
  }
  free(col);
  free(col2);

  // Pass 3: Z
  float *zcol  = malloc((size_t)depth * sizeof(float));
  float *zcol2 = malloc((size_t)depth * sizeof(float));
  if (!zcol || !zcol2) { free(zcol); free(zcol2); free(tmp1); free(tmp2); gauss_kernel_free(k); return; }
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int z = 0; z < depth; z++) zcol[z] = tmp2[((size_t)z * height + y) * width + x];
      conv1d(zcol, zcol2, depth, k->weights, k->radius);
      for (int z = 0; z < depth; z++) out[((size_t)z * height + y) * width + x] = zcol2[z];
    }
  }
  free(zcol);
  free(zcol2);
  free(tmp1);
  free(tmp2);
  gauss_kernel_free(k);
}

// ---------------------------------------------------------------------------
// Structure tensor (3D)
// Pavel Holoborodko derivative kernel: {2,1,-16,-27,0,27,16,-1,-2}/(96*16*16)
// Smoothing kernel (binomial): {1,4,6,4,1}/16
// Derivative is applied per-axis, then Jij = di * dj smoothed over each axis.
// ---------------------------------------------------------------------------

// Derivative kernel weights (unnormalized numerators, denom = 96*16*16 = 24576)
static const float DERIV_NUM[9] = { 2.0f, 1.0f, -16.0f, -27.0f, 0.0f, 27.0f, 16.0f, -1.0f, -2.0f };
#define DERIV_DENOM 24576.0f
#define DERIV_RADIUS 4

// Smoothing kernel {1,4,6,4,1}/16, radius 2
static const float SMOOTH_W[5] = { 1.0f/16.0f, 4.0f/16.0f, 6.0f/16.0f, 4.0f/16.0f, 1.0f/16.0f };
#define SMOOTH_RADIUS 2

// Apply derivative along X for entire volume, result stored in out
static void deriv_x(const float *restrict vol, float *restrict out,
                    int depth, int height, int width) {
  float w[9];
  for (int i = 0; i < 9; i++) w[i] = DERIV_NUM[i] / DERIV_DENOM;

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      size_t off = ((size_t)z * height + y) * width;
      for (int x = 0; x < width; x++) {
        float acc = 0.0f;
        for (int j = -DERIV_RADIUS; j <= DERIV_RADIUS; j++) {
          int ix = x + j;
          if (ix < 0) ix = 0;
          else if (ix >= width) ix = width - 1;
          acc += vol[off + ix] * w[j + DERIV_RADIUS];
        }
        out[off + x] = acc;
      }
    }
  }
}

static void deriv_y(const float *restrict vol, float *restrict out,
                    int depth, int height, int width) {
  float w[9];
  for (int i = 0; i < 9; i++) w[i] = DERIV_NUM[i] / DERIV_DENOM;

  for (int z = 0; z < depth; z++) {
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        float acc = 0.0f;
        for (int j = -DERIV_RADIUS; j <= DERIV_RADIUS; j++) {
          int iy = y + j;
          if (iy < 0) iy = 0;
          else if (iy >= height) iy = height - 1;
          acc += vol[((size_t)z * height + iy) * width + x] * w[j + DERIV_RADIUS];
        }
        out[((size_t)z * height + y) * width + x] = acc;
      }
    }
  }
}

static void deriv_z(const float *restrict vol, float *restrict out,
                    int depth, int height, int width) {
  float w[9];
  for (int i = 0; i < 9; i++) w[i] = DERIV_NUM[i] / DERIV_DENOM;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int z = 0; z < depth; z++) {
        float acc = 0.0f;
        for (int j = -DERIV_RADIUS; j <= DERIV_RADIUS; j++) {
          int iz = z + j;
          if (iz < 0) iz = 0;
          else if (iz >= depth) iz = depth - 1;
          acc += vol[((size_t)iz * height + y) * width + x] * w[j + DERIV_RADIUS];
        }
        out[((size_t)z * height + y) * width + x] = acc;
      }
    }
  }
}

// Smooth a volume in-place using the binomial {1,4,6,4,1}/16 kernel along all 3 axes
static void smooth_volume(float *restrict vol, float *restrict tmp,
                          int depth, int height, int width) {
  size_t vol_size = (size_t)depth * height * width;

  // X pass
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      size_t off = ((size_t)z * height + y) * width;
      for (int x = 0; x < width; x++) {
        float acc = 0.0f;
        for (int j = -SMOOTH_RADIUS; j <= SMOOTH_RADIUS; j++) {
          int ix = x + j;
          if (ix < 0) ix = 0;
          else if (ix >= width) ix = width - 1;
          acc += vol[off + ix] * SMOOTH_W[j + SMOOTH_RADIUS];
        }
        tmp[off + x] = acc;
      }
    }
  }

  // Y pass
  float *col = malloc((size_t)height * sizeof(float));
  float *col2 = malloc((size_t)height * sizeof(float));
  if (col && col2) {
    for (int z = 0; z < depth; z++) {
      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) col[y] = tmp[((size_t)z * height + y) * width + x];
        conv1d(col, col2, height, SMOOTH_W, SMOOTH_RADIUS);
        for (int y = 0; y < height; y++) vol[((size_t)z * height + y) * width + x] = col2[y];
      }
    }
  }
  free(col); free(col2);

  // Z pass
  float *zcol  = malloc((size_t)depth * sizeof(float));
  float *zcol2 = malloc((size_t)depth * sizeof(float));
  if (zcol && zcol2) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int z = 0; z < depth; z++) zcol[z] = vol[((size_t)z * height + y) * width + x];
        conv1d(zcol, zcol2, depth, SMOOTH_W, SMOOTH_RADIUS);
        for (int z = 0; z < depth; z++) tmp[((size_t)z * height + y) * width + x] = zcol2[z];
      }
    }
  }
  free(zcol); free(zcol2);

  // copy tmp back to vol
  memcpy(vol, tmp, vol_size * sizeof(float));
}

void structure_tensor_3d(const float *restrict data, float *restrict out,
                         int depth, int height, int width,
                         float deriv_sigma, float smooth_sigma) {
  size_t vol = (size_t)depth * height * width;

  // Smooth input first with deriv_sigma
  float *smoothed = malloc(vol * sizeof(float));
  float *tmp      = malloc(vol * sizeof(float));
  if (!smoothed || !tmp) { free(smoothed); free(tmp); return; }

  gaussian_blur_3d(data, smoothed, depth, height, width, deriv_sigma);

  float *dz = malloc(vol * sizeof(float));
  float *dy = malloc(vol * sizeof(float));
  float *dx = malloc(vol * sizeof(float));
  if (!dz || !dy || !dx) { free(dz); free(dy); free(dx); free(smoothed); free(tmp); return; }

  deriv_z(smoothed, dz, depth, height, width);
  deriv_y(smoothed, dy, depth, height, width);
  deriv_x(smoothed, dx, depth, height, width);
  free(smoothed);

  // Compute outer products: Jzz, Jzy, Jzx, Jyy, Jyx, Jxx
  // Store each component as a separate vol-sized buffer, then smooth, then interleave into out
  float *Jzz = malloc(vol * sizeof(float));
  float *Jzy = malloc(vol * sizeof(float));
  float *Jzx = malloc(vol * sizeof(float));
  float *Jyy = malloc(vol * sizeof(float));
  float *Jyx = malloc(vol * sizeof(float));
  float *Jxx = malloc(vol * sizeof(float));
  if (!Jzz || !Jzy || !Jzx || !Jyy || !Jyx || !Jxx) {
    free(Jzz); free(Jzy); free(Jzx); free(Jyy); free(Jyx); free(Jxx);
    free(dz); free(dy); free(dx); free(tmp);
    return;
  }

  for (size_t i = 0; i < vol; i++) {
    Jzz[i] = dz[i] * dz[i];
    Jzy[i] = dz[i] * dy[i];
    Jzx[i] = dz[i] * dx[i];
    Jyy[i] = dy[i] * dy[i];
    Jyx[i] = dy[i] * dx[i];
    Jxx[i] = dx[i] * dx[i];
  }
  free(dz); free(dy); free(dx);

  // Smooth each component
  smooth_volume(Jzz, tmp, depth, height, width);
  smooth_volume(Jzy, tmp, depth, height, width);
  smooth_volume(Jzx, tmp, depth, height, width);
  smooth_volume(Jyy, tmp, depth, height, width);
  smooth_volume(Jyx, tmp, depth, height, width);
  smooth_volume(Jxx, tmp, depth, height, width);
  free(tmp);

  // Apply smooth_sigma Gaussian to each component
  float *buf = malloc(vol * sizeof(float));
  if (buf) {
    gaussian_blur_3d(Jzz, buf, depth, height, width, smooth_sigma); memcpy(Jzz, buf, vol * sizeof(float));
    gaussian_blur_3d(Jzy, buf, depth, height, width, smooth_sigma); memcpy(Jzy, buf, vol * sizeof(float));
    gaussian_blur_3d(Jzx, buf, depth, height, width, smooth_sigma); memcpy(Jzx, buf, vol * sizeof(float));
    gaussian_blur_3d(Jyy, buf, depth, height, width, smooth_sigma); memcpy(Jyy, buf, vol * sizeof(float));
    gaussian_blur_3d(Jyx, buf, depth, height, width, smooth_sigma); memcpy(Jyx, buf, vol * sizeof(float));
    gaussian_blur_3d(Jxx, buf, depth, height, width, smooth_sigma); memcpy(Jxx, buf, vol * sizeof(float));
    free(buf);
  }

  // Interleave into out[vol * 6]: {Jzz, Jzy, Jzx, Jyy, Jyx, Jxx}
  for (size_t i = 0; i < vol; i++) {
    out[i * 6 + 0] = Jzz[i];
    out[i * 6 + 1] = Jzy[i];
    out[i * 6 + 2] = Jzx[i];
    out[i * 6 + 3] = Jyy[i];
    out[i * 6 + 4] = Jyx[i];
    out[i * 6 + 5] = Jxx[i];
  }

  free(Jzz); free(Jzy); free(Jzx); free(Jyy); free(Jyx); free(Jxx);
}

// ---------------------------------------------------------------------------
// Euclidean Distance Transform (3D) – Saito's separable algorithm
// Phase 1: per-voxel squared distance in X
// Phase 2: parabolic envelope in Y
// Phase 3: parabolic envelope in Z
// ---------------------------------------------------------------------------

// 1D squared EDT via parabola envelope (Meijster/Felzenszwalb style)
// g[i] = squared partial distance at each position along row
static void edt_1d(float *restrict g, int n) {
  // g already contains squared distances from the previous phase.
  // We compute the lower envelope of parabolas f_i(x) = (x-i)^2 + g[i].
  int *v  = malloc((size_t)n * sizeof(int));
  float *z = malloc(((size_t)n + 1) * sizeof(float));
  if (!v || !z) { free(v); free(z); return; }

  int k = 0;
  v[0] = 0;
  z[0] = -FLT_MAX;
  z[1] =  FLT_MAX;

  for (int q = 1; q < n; q++) {
    float sq = (float)q;
    float sv = (float)v[k];
    // intersection of parabola at q and parabola at v[k]
    float s = ((g[q] + sq * sq) - (g[v[k]] + sv * sv)) / (2.0f * (sq - sv));
    while (k > 0 && s <= z[k]) {
      k--;
      sv = (float)v[k];
      s = ((g[q] + sq * sq) - (g[v[k]] + sv * sv)) / (2.0f * (sq - sv));
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = FLT_MAX;
  }

  k = 0;
  for (int q = 0; q < n; q++) {
    while (z[k + 1] < (float)q) k++;
    float d = (float)(q - v[k]);
    g[q] = d * d + g[v[k]];
  }

  free(v);
  free(z);
}

void edt_3d(const uint8_t *restrict mask, float *restrict dist,
            int depth, int height, int width) {
  size_t vol = (size_t)depth * height * width;
  float *sq = malloc(vol * sizeof(float));
  if (!sq) return;

  // Phase 1: X – initialize squared distance along each row via two-pass linear scan
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      size_t off = ((size_t)z * height + y) * width;
      float *row = sq + off;
      const uint8_t *mrow = mask + off;

      // left pass
      float dist_left = (float)(width + 1);
      for (int x = 0; x < width; x++) {
        if (mrow[x]) dist_left = 0.0f;
        else dist_left += 1.0f;
        row[x] = dist_left * dist_left;
      }
      // right pass
      float dist_right = (float)(width + 1);
      for (int x = width - 1; x >= 0; x--) {
        if (mrow[x]) dist_right = 0.0f;
        else dist_right += 1.0f;
        float dr2 = dist_right * dist_right;
        if (dr2 < row[x]) row[x] = dr2;
      }
    }
  }

  // Phase 2: Y – parabolic envelope along columns
  float *col = malloc((size_t)height * sizeof(float));
  if (col) {
    for (int z = 0; z < depth; z++) {
      for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) col[y] = sq[((size_t)z * height + y) * width + x];
        edt_1d(col, height);
        for (int y = 0; y < height; y++) sq[((size_t)z * height + y) * width + x] = col[y];
      }
    }
    free(col);
  }

  // Phase 3: Z – parabolic envelope along depth
  float *zcol = malloc((size_t)depth * sizeof(float));
  if (zcol) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        for (int z = 0; z < depth; z++) zcol[z] = sq[((size_t)z * height + y) * width + x];
        edt_1d(zcol, depth);
        for (int z = 0; z < depth; z++) dist[((size_t)z * height + y) * width + x] = sqrtf(zcol[z]);
      }
    }
    free(zcol);
  }

  free(sq);
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

histogram *histogram_new(const float *data, size_t n, int num_bins) {
  if (n == 0 || num_bins < 1) return NULL;

  float min_val = data[0], max_val = data[0];
  for (size_t i = 1; i < n; i++) {
    if (data[i] < min_val) min_val = data[i];
    if (data[i] > max_val) max_val = data[i];
  }

  histogram *h = malloc(sizeof(histogram));
  if (!h) return NULL;
  h->bins = calloc((size_t)num_bins, sizeof(uint32_t));
  if (!h->bins) { free(h); return NULL; }

  h->num_bins = num_bins;
  h->min_val  = min_val;
  h->max_val  = max_val;
  h->total    = n;

  float range = max_val - min_val;
  if (range == 0.0f) {
    h->bin_width = 0.0f;  // sentinel: all values are identical
    h->bins[0] = (uint32_t)n;
    return h;
  }

  h->bin_width = range / (float)num_bins;
  float inv_bw = 1.0f / h->bin_width;

  for (size_t i = 0; i < n; i++) {
    int bin = (int)((data[i] - min_val) * inv_bw);
    if (bin < 0) bin = 0;
    else if (bin >= num_bins) bin = num_bins - 1;
    h->bins[bin]++;
  }

  return h;
}

float histogram_percentile(const histogram *h, float p) {
  if (!h || h->total == 0) return 0.0f;
  if (p <= 0.0f) return h->min_val;
  if (p >= 1.0f) return h->max_val;

  size_t target = (size_t)(p * (float)h->total);
  size_t cum = 0;
  for (int i = 0; i < h->num_bins; i++) {
    cum += h->bins[i];
    if (cum >= target) {
      return h->min_val + (float)(i + 1) * h->bin_width;
    }
  }
  return h->max_val;
}

float histogram_mean(const histogram *h) {
  if (!h || h->total == 0) return 0.0f;
  if (h->bin_width == 0.0f) return h->min_val;  // all values identical
  double acc = 0.0;
  for (int i = 0; i < h->num_bins; i++) {
    float bin_center = h->min_val + ((float)i + 0.5f) * h->bin_width;
    acc += (double)h->bins[i] * bin_center;
  }
  return (float)(acc / (double)h->total);
}

void histogram_free(histogram *h) {
  if (!h) return;
  free(h->bins);
  free(h);
}

// ---------------------------------------------------------------------------
// Window / level
// ---------------------------------------------------------------------------

void window_level(const float *restrict in, uint8_t *restrict out,
                  size_t n, float window, float level) {
  float half = window * 0.5f;
  float lo   = level - half;
  float inv  = (window > 0.0f) ? (255.0f / window) : 0.0f;

  for (size_t i = 0; i < n; i++) {
    float v = (in[i] - lo) * inv;
    if (v < 0.0f) v = 0.0f;
    else if (v > 255.0f) v = 255.0f;
    out[i] = (uint8_t)v;
  }
}

// ---------------------------------------------------------------------------
// Coherence-Enhancing Diffusion (2D)
//
// At each pixel, we compute the 2D structure tensor J = [Jxx Jxy; Jxy Jyy],
// find its eigenvectors (edge direction v1, along-edge v2) and eigenvalues
// (l1 >= l2), then construct a diffusion tensor D that:
//   - has large diffusivity alpha1 along v2 (along edge)
//   - has small diffusivity alpha2 across v1 (across edge), inversely
//     proportional to coherence (l1-l2)^2
// One explicit diffusion step: u += dt * div(D * grad(u))
// ---------------------------------------------------------------------------

void ced_2d(const float *restrict data, float *restrict out,
            int height, int width, float sigma, float rho, int iterations) {
  size_t n = (size_t)height * width;
  float *cur  = malloc(n * sizeof(float));
  float *next = malloc(n * sizeof(float));
  float *Jxx  = malloc(n * sizeof(float));
  float *Jxy  = malloc(n * sizeof(float));
  float *Jyy  = malloc(n * sizeof(float));
  if (!cur || !next || !Jxx || !Jxy || !Jyy) {
    free(cur); free(next); free(Jxx); free(Jxy); free(Jyy); return;
  }

  memcpy(cur, data, n * sizeof(float));

  float dt = 0.1f;  // explicit step, stable for dt <= 0.25

  for (int iter = 0; iter < iterations; iter++) {
    // Compute smoothed gradients via Gaussian derivatives
    float *sm = malloc(n * sizeof(float));
    float *gx = malloc(n * sizeof(float));
    float *gy = malloc(n * sizeof(float));
    if (!sm || !gx || !gy) { free(sm); free(gx); free(gy); break; }

    gaussian_blur_2d(cur, sm, height, width, sigma);

    // Central differences for gradient
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int xp = x < width  - 1 ? x + 1 : x;
        int xm = x > 0          ? x - 1 : x;
        int yp = y < height - 1 ? y + 1 : y;
        int ym = y > 0          ? y - 1 : y;
        gx[y * width + x] = (sm[y  * width + xp] - sm[y  * width + xm]) * 0.5f;
        gy[y * width + x] = (sm[yp * width + x]  - sm[ym * width + x])  * 0.5f;
      }
    }
    free(sm);

    // Structure tensor components (outer product of gradient, smoothed by rho)
    for (size_t i = 0; i < n; i++) {
      Jxx[i] = gx[i] * gx[i];
      Jxy[i] = gx[i] * gy[i];
      Jyy[i] = gy[i] * gy[i];
    }
    free(gx); free(gy);

    float *tjxx = malloc(n * sizeof(float));
    float *tjxy = malloc(n * sizeof(float));
    float *tjyy = malloc(n * sizeof(float));
    if (tjxx && tjxy && tjyy) {
      gaussian_blur_2d(Jxx, tjxx, height, width, rho);
      gaussian_blur_2d(Jxy, tjxy, height, width, rho);
      gaussian_blur_2d(Jyy, tjyy, height, width, rho);
      memcpy(Jxx, tjxx, n * sizeof(float));
      memcpy(Jxy, tjxy, n * sizeof(float));
      memcpy(Jyy, tjyy, n * sizeof(float));
    }
    free(tjxx); free(tjxy); free(tjyy);

    // Diffusion step: for each pixel compute D from eigenvectors of J
    for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
        size_t i = (size_t)y * width + x;
        float jxx = Jxx[i], jxy = Jxy[i], jyy = Jyy[i];

        // 2x2 symmetric eigendecomposition
        float trace = jxx + jyy;
        float diff  = jxx - jyy;
        float disc  = sqrtf(diff * diff + 4.0f * jxy * jxy);
        float l1    = 0.5f * (trace + disc);   // larger eigenvalue
        float l2    = 0.5f * (trace - disc);   // smaller eigenvalue

        // eigenvector for l1: (jxy, l1-jxx) normalised
        float vx = jxy, vy = l1 - jxx;
        float vlen = sqrtf(vx * vx + vy * vy);
        if (vlen < 1e-8f) { vx = 1.0f; vy = 0.0f; } else { vx /= vlen; vy /= vlen; }
        // perpendicular eigenvector
        float wx = -vy, wy = vx;

        // diffusivities: large along edge (d2), small across edge (d1)
        float coh = (l1 - l2) * (l1 - l2);
        float d1  = 0.001f;  // alpha_min across edge
        float d2  = (coh < 1e-10f) ? 1.0f : 1.0f - expf(-3.315f / (coh * coh * coh * coh));

        // D = d1*(v x v^T) + d2*(w x w^T)
        float Dxx = d1 * vx * vx + d2 * wx * wx;
        float Dxy = d1 * vx * vy + d2 * wx * wy;
        float Dyy = d1 * vy * vy + d2 * wy * wy;

        // div(D grad u) via finite differences (explicit)
        float u   = cur[i];
        float uxp = cur[i + 1], uxm = cur[i - 1];
        float uyp = cur[i + width], uym = cur[i - width];
        float uxpy = cur[i + width + 1], uxmy = cur[i + width - 1];
        float uxnym = cur[i - width + 1], uxymp = cur[i - width - 1];

        float div_val =
          Dxx * (uxp - 2.0f * u + uxm) +
          Dyy * (uyp - 2.0f * u + uym) +
          0.5f * Dxy * (uxpy - uxmy - uxnym + uxymp);

        next[i] = u + dt * div_val;
      }
    }
    // copy border pixels unchanged
    for (int x = 0; x < width; x++) {
      next[x] = cur[x];
      next[(height - 1) * width + x] = cur[(height - 1) * width + x];
    }
    for (int y = 0; y < height; y++) {
      next[(size_t)y * width] = cur[(size_t)y * width];
      next[(size_t)y * width + width - 1] = cur[(size_t)y * width + width - 1];
    }

    float *swap = cur; cur = next; next = swap;
  }

  memcpy(out, cur, n * sizeof(float));
  free(cur); free(next); free(Jxx); free(Jxy); free(Jyy);
}

// ---------------------------------------------------------------------------
// Frangi vesselness filter (3D)
//
// At each scale sigma: smooth with Gaussian(sigma), compute 3x3 Hessian
// (second derivatives), find eigenvalues l1<=l2<=l3, compute vesselness:
//   Rb = |l1| / sqrt(|l2*l3|)   (blobness)
//   Ra = |l2| / |l3|             (plate vs tube)
//   S  = sqrt(l1^2+l2^2+l3^2)   (structure magnitude)
//   V  = exp(-Ra^2/(2a^2)) * (1-exp(-Rb^2/(2b^2))) * (1-exp(-S^2/(2g^2)))
// Response taken only when l2<0 and l3<0 (dark tube on bright background).
// Final output = max over scales.
// ---------------------------------------------------------------------------

// Compute 3x3 symmetric Hessian second derivatives via central differences
static void hessian_3d(const float *restrict vol, float *restrict H,
                       int depth, int height, int width) {
  // H: [d*h*w * 6] = {Hzz, Hzy, Hzx, Hyy, Hyx, Hxx}
  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int zp = z < depth  - 1 ? z + 1 : z, zm = z > 0 ? z - 1 : z;
        int yp = y < height - 1 ? y + 1 : y, ym = y > 0 ? y - 1 : y;
        int xp = x < width  - 1 ? x + 1 : x, xm = x > 0 ? x - 1 : x;
        float c  = vol[((size_t)z  * height + y)  * width + x];
        float fzp = vol[((size_t)zp * height + y)  * width + x];
        float fzm = vol[((size_t)zm * height + y)  * width + x];
        float fyp = vol[((size_t)z  * height + yp) * width + x];
        float fym = vol[((size_t)z  * height + ym) * width + x];
        float fxp = vol[((size_t)z  * height + y)  * width + xp];
        float fxm = vol[((size_t)z  * height + y)  * width + xm];
        // mixed: approximate with corner differences
        float fzpyp = vol[((size_t)zp * height + yp) * width + x];
        float fzmym = vol[((size_t)zm * height + ym) * width + x];
        float fzpym = vol[((size_t)zp * height + ym) * width + x];
        float fzmyp = vol[((size_t)zm * height + yp) * width + x];
        float fypxp = vol[((size_t)z  * height + yp) * width + xp];
        float fymxm = vol[((size_t)z  * height + ym) * width + xm];
        float fypxm = vol[((size_t)z  * height + yp) * width + xm];
        float fymxp = vol[((size_t)z  * height + ym) * width + xp];
        float fzpxp = vol[((size_t)zp * height + y)  * width + xp];
        float fzmxm = vol[((size_t)zm * height + y)  * width + xm];
        float fzpxm = vol[((size_t)zp * height + y)  * width + xm];
        float fzmxp = vol[((size_t)zm * height + y)  * width + xp];

        size_t idx = (((size_t)z * height + y) * width + x) * 6;
        H[idx + 0] = fzp - 2.0f * c + fzm;                         // Hzz
        H[idx + 1] = 0.25f * (fzpyp - fzmyp - fzpym + fzmym);      // Hzy
        H[idx + 2] = 0.25f * (fzpxp - fzmxp - fzpxm + fzmxm);      // Hzx
        H[idx + 3] = fyp - 2.0f * c + fym;                          // Hyy
        H[idx + 4] = 0.25f * (fypxp - fymxp - fypxm + fymxm);      // Hyx
        H[idx + 5] = fxp - 2.0f * c + fxm;                          // Hxx
      }
    }
  }
}

// Eigenvalues of 3x3 symmetric matrix via analytical method (sorted |l1|<=|l2|<=|l3|)
static void eig3x3(float a00, float a01, float a02,
                              float a11, float a12,
                                         float a22,
                   float *l1, float *l2, float *l3) {
  // Cardano's method
  float p1 = a01*a01 + a02*a02 + a12*a12;
  if (p1 < 1e-12f) {
    float ev[3] = { a00, a11, a22 };
    // sort by absolute value
    if (fabsf(ev[0]) > fabsf(ev[1])) { float t = ev[0]; ev[0] = ev[1]; ev[1] = t; }
    if (fabsf(ev[1]) > fabsf(ev[2])) { float t = ev[1]; ev[1] = ev[2]; ev[2] = t; }
    if (fabsf(ev[0]) > fabsf(ev[1])) { float t = ev[0]; ev[0] = ev[1]; ev[1] = t; }
    *l1 = ev[0]; *l2 = ev[1]; *l3 = ev[2];
    return;
  }
  float q = (a00 + a11 + a22) / 3.0f;
  float b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
  float p2 = b00*b00 + b11*b11 + b22*b22 + 2.0f * p1;
  float p  = sqrtf(p2 / 6.0f);
  float r  = (b00*(b11*b22 - a12*a12) - a01*(a01*b22 - a12*a02) + a02*(a01*a12 - b11*a02)) / (2.0f * p*p*p);
  r = r < -1.0f ? -1.0f : r > 1.0f ? 1.0f : r;
  float phi = acosf(r) / 3.0f;
  float ev0 = q + 2.0f * p * cosf(phi);
  float ev2 = q + 2.0f * p * cosf(phi + 2.0f * 3.14159265f / 3.0f);
  float ev1 = 3.0f * q - ev0 - ev2;
  // sort by absolute value ascending
  float ev[3] = { ev0, ev1, ev2 };
  if (fabsf(ev[0]) > fabsf(ev[1])) { float t = ev[0]; ev[0] = ev[1]; ev[1] = t; }
  if (fabsf(ev[1]) > fabsf(ev[2])) { float t = ev[1]; ev[1] = ev[2]; ev[2] = t; }
  if (fabsf(ev[0]) > fabsf(ev[1])) { float t = ev[0]; ev[0] = ev[1]; ev[1] = t; }
  *l1 = ev[0]; *l2 = ev[1]; *l3 = ev[2];
}

void frangi_3d(const float *restrict data, float *restrict out,
               int depth, int height, int width,
               const float *sigmas, int n_sigmas,
               float alpha, float beta, float gamma) {
  size_t vol = (size_t)depth * height * width;

  float *sm  = malloc(vol * sizeof(float));
  float *H   = malloc(vol * 6 * sizeof(float));
  if (!sm || !H) { free(sm); free(H); return; }

  // initialise output to zero
  memset(out, 0, vol * sizeof(float));

  float inv2a2 = 1.0f / (2.0f * alpha * alpha);
  float inv2b2 = 1.0f / (2.0f * beta  * beta);
  float inv2g2 = 1.0f / (2.0f * gamma * gamma);

  for (int si = 0; si < n_sigmas; si++) {
    float s = sigmas[si];
    gaussian_blur_3d(data, sm, depth, height, width, s);
    hessian_3d(sm, H, depth, height, width);

    for (size_t i = 0; i < vol; i++) {
      float hzz = H[i*6+0], hzy = H[i*6+1], hzx = H[i*6+2];
      float hyy = H[i*6+3], hyx = H[i*6+4], hxx = H[i*6+5];
      float l1, l2, l3;
      eig3x3(hzz, hzy, hzx, hyy, hyx, hxx, &l1, &l2, &l3);

      // vesselness only for dark tubes: l2<0, l3<0
      if (l2 >= 0.0f || l3 >= 0.0f) continue;

      float Ra = fabsf(l2) / (fabsf(l3) + 1e-10f);
      float Rb = fabsf(l1) / (sqrtf(fabsf(l2) * fabsf(l3)) + 1e-10f);
      float S  = sqrtf(l1*l1 + l2*l2 + l3*l3);

      float v = (1.0f - expf(-Ra*Ra * inv2a2)) *
                expf(-Rb*Rb * inv2b2) *
                (1.0f - expf(-S*S * inv2g2));

      if (v > out[i]) out[i] = v;
    }
  }

  free(sm); free(H);
}

// ---------------------------------------------------------------------------
// 3D Topological Thinning (skeletonization) — Lee94 algorithm
//
// Simple-point check: a point p is simple if removing it does not change the
// topology of the object (connected components of foreground and background).
// We use the 26-connectivity criterion via index table for the 26-neighbor cube.
// Iteratively peel border voxels that are simple, in 6 directional sub-iterations.
// ---------------------------------------------------------------------------

// 6-face neighbor directions
static const int FACE6[6][3] = {
  {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}
};

// Count 26-connected components in 3x3x3 neighborhood using union-find
static int count_c26(const uint8_t nb[27]) {
  // nb[13] is the center (always excluded)
  int roots[27];
  for (int i = 0; i < 27; i++) roots[i] = i;
  #define FIND(x) ({ int _r = (x); while (roots[_r] != _r) { roots[_r] = roots[roots[_r]]; _r = roots[_r]; } _r; })

  for (int i = 0; i < 27; i++) {
    if (i == 13 || !nb[i]) continue;
    int iz = i / 9, iy = (i / 3) % 3, ix = i % 3;
    for (int j = i + 1; j < 27; j++) {
      if (j == 13 || !nb[j]) continue;
      int jz = j / 9, jy = (j / 3) % 3, jx = j % 3;
      if (abs(iz-jz) <= 1 && abs(iy-jy) <= 1 && abs(ix-jx) <= 1) {
        int ri = FIND(i), rj = FIND(j);
        if (ri != rj) roots[ri] = rj;
      }
    }
  }
  int count = 0;
  for (int i = 0; i < 27; i++) {
    if (i == 13 || !nb[i]) continue;
    if (FIND(i) == i) count++;
  }
  #undef FIND
  return count;
}

// A voxel is a simple point if:
//   C26(N26*(p)) == 1  (foreground stays connected in 26-nbhd)
//   C6(N6*(p))   == 1  (background stays connected in 6-nbhd)
// We use a compact 3x3x3 lookup for the 6-connectivity background check.
static bool is_simple(const uint8_t *vol, int z, int y, int x,
                      int depth, int height, int width) {
  uint8_t nb[27];
  // Fill 3x3x3 neighborhood (center = index 13)
  for (int dz = -1; dz <= 1; dz++) {
    for (int dy = -1; dy <= 1; dy++) {
      for (int dx = -1; dx <= 1; dx++) {
        int nz = z+dz, ny = y+dy, nx = x+dx;
        int idx = (dz+1)*9 + (dy+1)*3 + (dx+1);
        if (nz < 0 || nz >= depth || ny < 0 || ny >= height || nx < 0 || nx >= width)
          nb[idx] = 0;
        else
          nb[idx] = vol[((size_t)nz * height + ny) * width + nx];
      }
    }
  }
  nb[13] = 0;  // exclude center from neighbor checks

  // Check foreground 26-connectivity
  if (count_c26(nb) != 1) return false;

  // Check background 6-connectivity: invert and check 6-neighbors
  // Simple heuristic: count 6-connected background components in the 3x3x3 cube
  uint8_t bgnb[27];
  for (int i = 0; i < 27; i++) bgnb[i] = nb[i] ? 0 : 1;
  bgnb[13] = 0;  // center excluded

  // Use 6-connectivity for background: only 6-neighbors of voxels matter
  int bg_roots[27];
  for (int i = 0; i < 27; i++) bg_roots[i] = i;
  #define BGFIND(x) ({ int _r = (x); while (bg_roots[_r] != _r) { bg_roots[_r] = bg_roots[bg_roots[_r]]; _r = bg_roots[_r]; } _r; })

  static const int adj6[6][3] = {{-1,0,0},{1,0,0},{0,-1,0},{0,1,0},{0,0,-1},{0,0,1}};
  for (int i = 0; i < 27; i++) {
    if (i == 13 || !bgnb[i]) continue;
    int iz = i/9, iy = (i/3)%3, ix = i%3;
    for (int d = 0; d < 6; d++) {
      int jz = iz+adj6[d][0], jy = iy+adj6[d][1], jx = ix+adj6[d][2];
      if (jz<0||jz>2||jy<0||jy>2||jx<0||jx>2) continue;
      int j = jz*9+jy*3+jx;
      if (j == 13 || !bgnb[j]) continue;
      int ri = BGFIND(i), rj = BGFIND(j);
      if (ri != rj) bg_roots[ri] = rj;
    }
  }
  int bg_count = 0;
  for (int i = 0; i < 27; i++) {
    if (i == 13 || !bgnb[i]) continue;
    if (BGFIND(i) == i) bg_count++;
  }
  #undef BGFIND
  return bg_count == 1;
}

void thinning_3d(const uint8_t *restrict mask, uint8_t *restrict skeleton,
                 int depth, int height, int width) {
  size_t vol = (size_t)depth * height * width;

  // Work buffer (we modify in-place across iterations)
  uint8_t *cur = malloc(vol);
  uint8_t *del = malloc(vol);
  if (!cur || !del) { free(cur); free(del); return; }
  memcpy(cur, mask, vol);

  bool changed = true;
  while (changed) {
    changed = false;
    // 6 directional sub-iterations
    for (int dir = 0; dir < 6; dir++) {
      int dz = FACE6[dir][0], dy = FACE6[dir][1], dx = FACE6[dir][2];
      memset(del, 0, vol);

      for (int z = 0; z < depth; z++) {
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            size_t i = ((size_t)z * height + y) * width + x;
            if (!cur[i]) continue;

            // Border check: must have background in the current direction
            int nz = z+dz, ny = y+dy, nx = x+dx;
            bool border_bg = (nz < 0 || nz >= depth || ny < 0 || ny >= height ||
                              nx < 0 || nx >= width)
                             ? true
                             : cur[((size_t)nz * height + ny) * width + nx] == 0;
            if (!border_bg) continue;

            // Endpoint preservation (Lee94): never remove voxels with <= 1
            // foreground 26-neighbor (endpoints or isolated voxels).
            int fg_nbrs = 0;
            for (int dz2 = -1; dz2 <= 1; dz2++)
              for (int dy2 = -1; dy2 <= 1; dy2++)
                for (int dx2 = -1; dx2 <= 1; dx2++) {
                  if (dz2 == 0 && dy2 == 0 && dx2 == 0) continue;
                  int nz2 = z+dz2, ny2 = y+dy2, nx2 = x+dx2;
                  if (nz2 >= 0 && nz2 < depth && ny2 >= 0 && ny2 < height &&
                      nx2 >= 0 && nx2 < width &&
                      cur[((size_t)nz2 * height + ny2) * width + nx2])
                    fg_nbrs++;
                }
            if (fg_nbrs <= 1) continue;

            if (is_simple(cur, z, y, x, depth, height, width)) del[i] = 1;
          }
        }
      }

      for (size_t i = 0; i < vol; i++) {
        if (del[i]) { cur[i] = 0; changed = true; }
      }
    }
  }

  memcpy(skeleton, cur, vol);
  free(cur); free(del);
}

// ---------------------------------------------------------------------------
// Connected components 3D — union-find with path compression + rank
// ---------------------------------------------------------------------------

static int uf_find(int *parent, int i) {
  while (parent[i] != i) { parent[i] = parent[parent[i]]; i = parent[i]; }
  return i;
}

static void uf_union(int *parent, int *rank, int a, int b) {
  a = uf_find(parent, a); b = uf_find(parent, b);
  if (a == b) return;
  if (rank[a] < rank[b]) { int t = a; a = b; b = t; }
  parent[b] = a;
  if (rank[a] == rank[b]) rank[a]++;
}

int connected_components_3d(const uint8_t *restrict mask, int *restrict labels,
                             int depth, int height, int width) {
  size_t vol = (size_t)depth * height * width;
  int *parent = malloc(vol * sizeof(int));
  int *rnk    = calloc(vol, sizeof(int));
  if (!parent || !rnk) { free(parent); free(rnk); return 0; }

  for (size_t i = 0; i < vol; i++) parent[i] = (int)i;

  for (int z = 0; z < depth; z++) {
    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        int idx = (z * height + y) * width + x;
        if (!mask[idx]) continue;
        if (x > 0 && mask[idx - 1])                    uf_union(parent, rnk, idx, idx - 1);
        if (y > 0 && mask[idx - width])                 uf_union(parent, rnk, idx, idx - width);
        if (z > 0 && mask[idx - height * width])        uf_union(parent, rnk, idx, idx - height * width);
      }
    }
  }

  int *root_label = calloc(vol, sizeof(int));
  if (!root_label) { free(parent); free(rnk); return 0; }
  int num_components = 0;
  for (size_t i = 0; i < vol; i++) {
    labels[i] = 0;
    if (!mask[i]) continue;
    int root = uf_find(parent, (int)i);
    if (!root_label[root]) root_label[root] = ++num_components;
    labels[i] = root_label[root];
  }

  free(root_label); free(parent); free(rnk);
  return num_components;
}

// ---------------------------------------------------------------------------
// Dijkstra 3D — binary min-heap over flat voxel indices
// ---------------------------------------------------------------------------

typedef struct { float dist; int idx; } _dijk_node;

static void _dijk_push(_dijk_node *h, int *sz, float d, int idx) {
  int i = (*sz)++;
  h[i] = (_dijk_node){d, idx};
  while (i > 0) {
    int p = (i - 1) / 2;
    if (h[p].dist <= h[i].dist) break;
    _dijk_node tmp = h[p]; h[p] = h[i]; h[i] = tmp;
    i = p;
  }
}

static _dijk_node _dijk_pop(_dijk_node *h, int *sz) {
  _dijk_node top = h[0];
  h[0] = h[--(*sz)];
  int i = 0;
  for (;;) {
    int l = 2*i+1, r = 2*i+2, s = i;
    if (l < *sz && h[l].dist < h[s].dist) s = l;
    if (r < *sz && h[r].dist < h[s].dist) s = r;
    if (s == i) break;
    _dijk_node tmp = h[i]; h[i] = h[s]; h[s] = tmp;
    i = s;
  }
  return top;
}

void dijkstra_3d(const float *restrict cost, int start_idx,
                 float *restrict dist, int depth, int height, int width) {
  size_t vol = (size_t)depth * height * width;
  for (size_t i = 0; i < vol; i++) dist[i] = FLT_MAX;
  dist[start_idx] = 0.0f;

  _dijk_node *heap = malloc(6 * vol * sizeof(_dijk_node));
  if (!heap) return;
  int sz = 0;
  _dijk_push(heap, &sz, 0.0f, start_idx);

  const int ddx[6] = {1,-1, 0, 0, 0, 0};
  const int ddy[6] = {0, 0, 1,-1, 0, 0};
  const int ddz[6] = {0, 0, 0, 0, 1,-1};

  while (sz > 0) {
    _dijk_node cur = _dijk_pop(heap, &sz);
    if (cur.dist > dist[cur.idx]) continue;

    int z = cur.idx / (height * width);
    int rem = cur.idx % (height * width);
    int y = rem / width, x = rem % width;

    for (int d = 0; d < 6; d++) {
      int nx = x + ddx[d], ny = y + ddy[d], nz = z + ddz[d];
      if (nx < 0 || nx >= width || ny < 0 || ny >= height || nz < 0 || nz >= depth) continue;
      int nidx = (nz * height + ny) * width + nx;
      float nd = cur.dist + cost[nidx];
      if (nd < dist[nidx]) { dist[nidx] = nd; _dijk_push(heap, &sz, nd, nidx); }
    }
  }
  free(heap);
}
