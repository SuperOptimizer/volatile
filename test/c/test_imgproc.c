#include "greatest.h"
#include "core/imgproc.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Gaussian kernel tests
// ---------------------------------------------------------------------------

TEST test_gauss_kernel_sum(void) {
  gauss_kernel *k = gauss_kernel_new(1.5f);
  ASSERT(k != NULL);
  ASSERT(k->size == 2 * k->radius + 1);

  float sum = 0.0f;
  for (int i = 0; i < k->size; i++) sum += k->weights[i];

  ASSERT_IN_RANGE(1.0f, sum, 1e-5f);
  gauss_kernel_free(k);
  PASS();
}

TEST test_gauss_kernel_sum_small_sigma(void) {
  gauss_kernel *k = gauss_kernel_new(0.5f);
  ASSERT(k != NULL);

  float sum = 0.0f;
  for (int i = 0; i < k->size; i++) sum += k->weights[i];

  ASSERT_IN_RANGE(1.0f, sum, 1e-5f);
  gauss_kernel_free(k);
  PASS();
}

TEST test_gauss_kernel_sum_large_sigma(void) {
  gauss_kernel *k = gauss_kernel_new(5.0f);
  ASSERT(k != NULL);

  float sum = 0.0f;
  for (int i = 0; i < k->size; i++) sum += k->weights[i];

  ASSERT_IN_RANGE(1.0f, sum, 1e-5f);
  gauss_kernel_free(k);
  PASS();
}

// ---------------------------------------------------------------------------
// gaussian_blur_2d: impulse response should be Gaussian-shaped
// ---------------------------------------------------------------------------

TEST test_gauss_blur_2d_impulse(void) {
  int H = 31, W = 31;
  float *src = calloc((size_t)H * W, sizeof(float));
  float *dst = calloc((size_t)H * W, sizeof(float));
  ASSERT(src && dst);

  // single-pixel impulse at center
  src[15 * W + 15] = 1.0f;

  float sigma = 2.0f;
  gaussian_blur_2d(src, dst, H, W, sigma);

  // output should sum to ~1.0 (energy preserving)
  float sum = 0.0f;
  for (int i = 0; i < H * W; i++) sum += dst[i];
  ASSERT_IN_RANGE(1.0f, sum, 0.01f);

  // peak should be at center
  float peak = dst[15 * W + 15];
  ASSERT(peak > 0.0f);
  for (int i = 0; i < H * W; i++) {
    ASSERT(dst[i] <= peak + 1e-6f);
  }

  // output should be symmetric: dst[cy+dy][cx+dx] == dst[cy-dy][cx-dx]
  for (int dy = -5; dy <= 5; dy++) {
    for (int dx = -5; dx <= 5; dx++) {
      float a = dst[(15 + dy) * W + (15 + dx)];
      float b = dst[(15 - dy) * W + (15 - dx)];
      ASSERT_IN_RANGE(a, b, 1e-5f);
    }
  }

  // verify Gaussian shape: compare against expected value at (15, 15+r)
  for (int r = 1; r <= 5; r++) {
    float expected = dst[15 * W + 15] * expf(-0.5f * (float)(r * r) / (sigma * sigma));
    float actual = dst[15 * W + 15 + r];
    ASSERT_IN_RANGE(expected, actual, 0.005f);
  }

  free(src); free(dst);
  PASS();
}

TEST test_gauss_blur_2d_constant_field(void) {
  int H = 10, W = 10;
  float *src = malloc((size_t)H * W * sizeof(float));
  float *dst = malloc((size_t)H * W * sizeof(float));
  ASSERT(src && dst);

  // constant input should produce constant output (up to edge effects with clamp)
  for (int i = 0; i < H * W; i++) src[i] = 3.0f;
  gaussian_blur_2d(src, dst, H, W, 1.0f);

  // interior pixels should be exactly 3.0
  for (int y = 2; y < H - 2; y++) {
    for (int x = 2; x < W - 2; x++) {
      ASSERT_IN_RANGE(3.0f, dst[y * W + x], 1e-4f);
    }
  }

  free(src); free(dst);
  PASS();
}

// ---------------------------------------------------------------------------
// Histogram tests
// ---------------------------------------------------------------------------

TEST test_histogram_uniform(void) {
  int n = 100;
  float data[100];
  for (int i = 0; i < n; i++) data[i] = (float)i;  // 0..99

  histogram *h = histogram_new(data, n, 10);
  ASSERT(h != NULL);
  ASSERT_EQ(10, h->num_bins);
  ASSERT_IN_RANGE(0.0f,  h->min_val, 1e-5f);
  ASSERT_IN_RANGE(99.0f, h->max_val, 1e-5f);

  // total count
  size_t total = 0;
  for (int i = 0; i < h->num_bins; i++) total += h->bins[i];
  ASSERT_EQ((size_t)n, total);

  // median should be near 50
  float med = histogram_percentile(h, 0.5f);
  ASSERT(med >= 40.0f && med <= 60.0f);

  histogram_free(h);
  PASS();
}

TEST test_histogram_mean(void) {
  // known data: 0, 1, 2, 3, 4 -> mean = 2.0
  float data[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  histogram *h = histogram_new(data, 5, 5);
  ASSERT(h != NULL);

  float m = histogram_mean(h);
  ASSERT_IN_RANGE(2.0f, m, 0.5f);  // bin approximation

  histogram_free(h);
  PASS();
}

TEST test_histogram_percentile_extremes(void) {
  float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  histogram *h = histogram_new(data, 5, 5);
  ASSERT(h != NULL);

  ASSERT_IN_RANGE(h->min_val, histogram_percentile(h, 0.0f), 1e-5f);
  ASSERT_IN_RANGE(h->max_val, histogram_percentile(h, 1.0f), 1e-5f);

  histogram_free(h);
  PASS();
}

TEST test_histogram_single_value(void) {
  float data[] = {7.0f, 7.0f, 7.0f};
  histogram *h = histogram_new(data, 3, 4);
  ASSERT(h != NULL);
  ASSERT_IN_RANGE(7.0f, histogram_mean(h), 1e-4f);
  histogram_free(h);
  PASS();
}

// ---------------------------------------------------------------------------
// window_level tests
// ---------------------------------------------------------------------------

TEST test_window_level_basic(void) {
  float in[] = {0.0f, 50.0f, 100.0f, 150.0f, 200.0f};
  uint8_t out[5];

  // window=200, level=100 -> range [0, 200], full scale
  window_level(in, out, 5, 200.0f, 100.0f);

  ASSERT_EQ(0,   out[0]);
  ASSERT_EQ(63,  out[1]);  // 50/200*255 = 63.75
  ASSERT_EQ(127, out[2]);  // 100/200*255 = 127.5
  ASSERT_EQ(191, out[3]);  // 150/200*255 = 191.25
  ASSERT_EQ(255, out[4]);

  PASS();
}

TEST test_window_level_clamp(void) {
  float in[] = {-100.0f, 50.0f, 300.0f};
  uint8_t out[3];

  // window=200, level=100 -> range [0, 200]
  window_level(in, out, 3, 200.0f, 100.0f);

  ASSERT_EQ(0,   out[0]);   // below range -> 0
  ASSERT_EQ(63,  out[1]);   // 50/200*255 = 63.75 -> 63
  ASSERT_EQ(255, out[2]);   // above range -> 255

  PASS();
}

TEST test_window_level_identity(void) {
  // window=255, level=127.5 -> range [0, 255]
  float in[256];
  uint8_t out[256];
  for (int i = 0; i < 256; i++) in[i] = (float)i;

  window_level(in, out, 256, 255.0f, 127.5f);

  ASSERT_EQ(0,   out[0]);
  ASSERT_EQ(255, out[255]);
  for (int i = 0; i < 256; i++) {
    ASSERT(abs((int)out[i] - i) <= 1);
  }

  PASS();
}

// ---------------------------------------------------------------------------
// EDT 3D tests
// ---------------------------------------------------------------------------

TEST test_edt_3d_foreground_zero(void) {
  int D = 5, H = 5, W = 5;
  uint8_t mask[125];
  float dist[125];

  memset(mask, 0xFF, sizeof(mask));  // all foreground
  edt_3d(mask, dist, D, H, W);

  for (int i = 0; i < D * H * W; i++) {
    ASSERT_IN_RANGE(0.0f, dist[i], 1e-5f);
  }

  PASS();
}

TEST test_edt_3d_single_foreground_voxel(void) {
  int D = 5, H = 5, W = 5;
  uint8_t mask[125];
  float dist[125];

  memset(mask, 0, sizeof(mask));
  // set center voxel as foreground
  mask[2 * H * W + 2 * W + 2] = 1;
  edt_3d(mask, dist, D, H, W);

  // center voxel should have distance 0
  ASSERT_IN_RANGE(0.0f, dist[2 * H * W + 2 * W + 2], 1e-5f);

  // corner (0,0,0) should have distance sqrt(4+4+4) = sqrt(12) = 2*sqrt(3) ~ 3.464
  float expected_corner = sqrtf(4.0f + 4.0f + 4.0f);
  ASSERT_IN_RANGE(expected_corner, dist[0], 0.01f);

  // immediate neighbor should have distance 1
  ASSERT_IN_RANGE(1.0f, dist[2 * H * W + 2 * W + 3], 0.01f);

  // diagonal neighbor in 2D should have distance sqrt(2) ~ 1.414
  float expected_diag = sqrtf(2.0f);
  ASSERT_IN_RANGE(expected_diag, dist[2 * H * W + 3 * W + 3], 0.01f);

  PASS();
}

TEST test_edt_3d_slab(void) {
  // foreground plane at z=0; distance at z=k should be k
  int D = 8, H = 4, W = 4;
  uint8_t *mask = calloc((size_t)D * H * W, 1);
  float *dist = malloc((size_t)D * H * W * sizeof(float));
  ASSERT(mask && dist);

  // fill z=0 plane with foreground
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      mask[0 * H * W + y * W + x] = 1;
    }
  }

  edt_3d(mask, dist, D, H, W);

  // interior pixels (not near x/y edges) should have distance == z
  for (int z = 0; z < D; z++) {
    float expected = (float)z;
    float actual = dist[(size_t)z * H * W + 2 * W + 2];
    ASSERT_IN_RANGE(expected, actual, 0.01f);
  }

  free(mask); free(dist);
  PASS();
}

// ---------------------------------------------------------------------------
// CED 2D tests
// ---------------------------------------------------------------------------

TEST test_ced_2d_preserves_edge(void) {
  // Step edge: left half = 0, right half = 1
  // After CED, the edge should still be present (not blurred away).
  int H = 20, W = 20;
  float *src = calloc((size_t)H * W, sizeof(float));
  float *dst = calloc((size_t)H * W, sizeof(float));
  ASSERT(src && dst);

  for (int y = 0; y < H; y++)
    for (int x = W / 2; x < W; x++)
      src[y * W + x] = 1.0f;

  ced_2d(src, dst, H, W, 1.0f, 2.0f, 5);

  // Edge column should still show a transition
  float left_val  = dst[H/2 * W + W/4];    // well inside left half
  float right_val = dst[H/2 * W + 3*W/4];  // well inside right half
  ASSERT(right_val > left_val + 0.1f);

  // Output should not have grown wildly outside [0,1] range
  for (int i = 0; i < H * W; i++) {
    ASSERT(dst[i] >= -0.1f && dst[i] <= 1.1f);
  }

  free(src); free(dst);
  PASS();
}

TEST test_ced_2d_constant_unchanged(void) {
  // Constant input: diffusion should not change values
  int H = 16, W = 16;
  float *src = malloc((size_t)H * W * sizeof(float));
  float *dst = malloc((size_t)H * W * sizeof(float));
  ASSERT(src && dst);
  for (int i = 0; i < H * W; i++) src[i] = 0.5f;

  ced_2d(src, dst, H, W, 1.0f, 1.0f, 3);

  for (int y = 1; y < H-1; y++)
    for (int x = 1; x < W-1; x++)
      ASSERT_IN_RANGE(0.5f, dst[y * W + x], 0.01f);

  free(src); free(dst);
  PASS();
}

// ---------------------------------------------------------------------------
// Frangi 3D tests
// ---------------------------------------------------------------------------

TEST test_frangi_3d_tube_detected(void) {
  // Synthetic bright tube along Z axis through center
  int D = 12, H = 12, W = 12;
  float *data = calloc((size_t)D * H * W, sizeof(float));
  float *out  = calloc((size_t)D * H * W, sizeof(float));
  ASSERT(data && out);

  // Tube: cylinder of radius 2 along Z, centered at (6,6)
  for (int z = 0; z < D; z++)
    for (int y = 0; y < H; y++)
      for (int x = 0; x < W; x++) {
        float dy = y - 5.5f, dx = x - 5.5f;
        if (dy*dy + dx*dx <= 4.0f) data[((size_t)z*H+y)*W+x] = 1.0f;
      }

  float sigmas[] = { 1.5f, 2.0f };
  frangi_3d(data, out, D, H, W, sigmas, 2, 0.5f, 0.5f, 0.5f);

  // Tube center should have higher vesselness than background
  float center_val = out[((size_t)(D/2)*H + H/2)*W + W/2];
  float bg_val     = out[0];  // corner = background

  ASSERT(center_val > bg_val);

  free(data); free(out);
  PASS();
}

TEST test_frangi_3d_background_zero(void) {
  // Uniform background: vesselness should be near zero
  int D = 6, H = 6, W = 6;
  float *data = malloc((size_t)D * H * W * sizeof(float));
  float *out  = calloc((size_t)D * H * W, sizeof(float));
  ASSERT(data && out);
  for (int i = 0; i < D*H*W; i++) data[i] = 0.5f;

  float sigmas[] = { 1.0f };
  frangi_3d(data, out, D, H, W, sigmas, 1, 0.5f, 0.5f, 0.5f);

  for (int i = 0; i < D*H*W; i++) ASSERT_IN_RANGE(0.0f, out[i], 0.01f);

  free(data); free(out);
  PASS();
}

// ---------------------------------------------------------------------------
// Thinning 3D tests
// ---------------------------------------------------------------------------

TEST test_thinning_3d_single_voxel_preserved(void) {
  // A single isolated voxel is its own skeleton
  int D = 3, H = 3, W = 3;
  uint8_t mask[27] = {0}, skel[27] = {0};
  mask[1*H*W + 1*W + 1] = 1;

  thinning_3d(mask, skel, D, H, W);

  ASSERT_EQ(1, skel[1*H*W + 1*W + 1]);

  // Count total skeleton voxels
  int total = 0;
  for (int i = 0; i < 27; i++) total += skel[i];
  ASSERT_EQ(1, total);

  PASS();
}

TEST test_thinning_3d_solid_block_produces_skeleton(void) {
  // A solid 7x7x7 block should thin to a much smaller skeleton
  int D = 7, H = 7, W = 7;
  int n = D * H * W;
  uint8_t *mask = malloc(n);
  uint8_t *skel = calloc(n, 1);
  ASSERT(mask && skel);
  memset(mask, 1, n);

  thinning_3d(mask, skel, D, H, W);

  int fg_orig = n;
  int fg_skel = 0;
  for (int i = 0; i < n; i++) fg_skel += skel[i];

  // Skeleton must be strictly smaller than original
  ASSERT(fg_skel < fg_orig);
  // Skeleton must be non-empty
  ASSERT(fg_skel > 0);

  free(mask); free(skel);
  PASS();
}

TEST test_thinning_3d_line_preserved(void) {
  // A 1-voxel-wide line along X: interior voxels must be preserved
  // (endpoints may be peeled since they are topological endpoints)
  int D = 3, H = 3, W = 9;
  int n = D * H * W;
  uint8_t *mask = calloc(n, 1);
  uint8_t *skel = calloc(n, 1);
  ASSERT(mask && skel);

  // Line at (z=1, y=1, x=0..8)
  for (int x = 0; x < W; x++) mask[1*H*W + 1*W + x] = 1;

  thinning_3d(mask, skel, D, H, W);

  // Skeleton must be non-empty
  int total = 0;
  for (int i = 0; i < n; i++) total += skel[i];
  ASSERT(total > 0);

  // Interior voxels (not near ends) should survive — they're not simple points
  for (int x = 2; x < W - 2; x++) {
    ASSERT_EQ(1, skel[1*H*W + 1*W + x]);
  }

  free(mask); free(skel);
  PASS();
}

// ---------------------------------------------------------------------------
// Test suites
// ---------------------------------------------------------------------------

SUITE(suite_gauss_kernel) {
  RUN_TEST(test_gauss_kernel_sum);
  RUN_TEST(test_gauss_kernel_sum_small_sigma);
  RUN_TEST(test_gauss_kernel_sum_large_sigma);
}

SUITE(suite_gauss_blur) {
  RUN_TEST(test_gauss_blur_2d_impulse);
  RUN_TEST(test_gauss_blur_2d_constant_field);
}

SUITE(suite_histogram) {
  RUN_TEST(test_histogram_uniform);
  RUN_TEST(test_histogram_mean);
  RUN_TEST(test_histogram_percentile_extremes);
  RUN_TEST(test_histogram_single_value);
}

SUITE(suite_window_level) {
  RUN_TEST(test_window_level_basic);
  RUN_TEST(test_window_level_clamp);
  RUN_TEST(test_window_level_identity);
}

SUITE(suite_edt) {
  RUN_TEST(test_edt_3d_foreground_zero);
  RUN_TEST(test_edt_3d_single_foreground_voxel);
  RUN_TEST(test_edt_3d_slab);
}

SUITE(suite_ced) {
  RUN_TEST(test_ced_2d_preserves_edge);
  RUN_TEST(test_ced_2d_constant_unchanged);
}

SUITE(suite_frangi) {
  RUN_TEST(test_frangi_3d_tube_detected);
  RUN_TEST(test_frangi_3d_background_zero);
}

SUITE(suite_thinning) {
  RUN_TEST(test_thinning_3d_single_voxel_preserved);
  RUN_TEST(test_thinning_3d_solid_block_produces_skeleton);
  RUN_TEST(test_thinning_3d_line_preserved);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();

  RUN_SUITE(suite_gauss_kernel);
  RUN_SUITE(suite_gauss_blur);
  RUN_SUITE(suite_histogram);
  RUN_SUITE(suite_window_level);
  RUN_SUITE(suite_edt);
  RUN_SUITE(suite_ced);
  RUN_SUITE(suite_frangi);
  RUN_SUITE(suite_thinning);

  GREATEST_MAIN_END();
}
