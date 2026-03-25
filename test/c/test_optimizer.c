#include "greatest.h"
#include "core/optimizer.h"

#include <math.h>
#include <stdlib.h>

// ---------------------------------------------------------------------------
// Test 1: Rosenbrock function
//   f(x,y) = (1-x)^2 + 100*(y-x^2)^2
// Minimum at (1,1) with f=0.
// Expressed as least-squares: r0 = 1-x, r1 = 10*(y-x^2)
// ---------------------------------------------------------------------------

static void rosenbrock_res(const float *p, int np, float *r, int nr, void *ctx) {
  (void)np; (void)nr; (void)ctx;
  r[0] = 1.0f - p[0];
  r[1] = 10.0f * (p[1] - p[0] * p[0]);
}

static void rosenbrock_jac(const float *p, int np, float *J, int nr, void *ctx) {
  (void)np; (void)nr; (void)ctx;
  // J[i*np+j] = d r_i / d p_j
  J[0 * 2 + 0] = -1.0f;
  J[0 * 2 + 1] =  0.0f;
  J[1 * 2 + 0] = -20.0f * p[0];
  J[1 * 2 + 1] =  10.0f;
}

TEST test_rosenbrock_with_jacobian(void) {
  optimizer_params cfg = optimizer_default_params();
  cfg.max_iterations = 200;
  cfg.tolerance = 1e-7f;

  optimizer *o = optimizer_new(2, 2, cfg);
  ASSERT(o != NULL);

  float params[2] = { -1.2f, 1.0f };  // standard starting point
  int iters = optimizer_solve(o, params, rosenbrock_res, rosenbrock_jac, NULL);

  ASSERT(iters >= 0);
  ASSERT_IN_RANGE(0.99f, params[0], 1.01f);
  ASSERT_IN_RANGE(0.99f, params[1], 1.01f);
  ASSERT(optimizer_final_cost(o) < 1e-8f);

  optimizer_free(o);
  PASS();
}

TEST test_rosenbrock_fd_jacobian(void) {
  optimizer_params cfg = optimizer_default_params();
  cfg.max_iterations = 200;
  cfg.tolerance = 1e-6f;

  optimizer *o = optimizer_new(2, 2, cfg);
  ASSERT(o != NULL);

  float params[2] = { 0.0f, 0.0f };
  int iters = optimizer_solve(o, params, rosenbrock_res, NULL, NULL);

  ASSERT(iters >= 0);
  ASSERT_IN_RANGE(0.98f, params[0], 1.02f);
  ASSERT_IN_RANGE(0.98f, params[1], 1.02f);
  ASSERT(optimizer_final_cost(o) < 1e-6f);

  optimizer_free(o);
  PASS();
}

// ---------------------------------------------------------------------------
// Test 2: linear least-squares  A*x ≈ b
//   r_i = (A*x - b)_i
// We fit a line y = a*t + b through noisy data.
// Minimum is the least-squares line fit.
// ---------------------------------------------------------------------------

typedef struct { int n; float *t; float *y; } line_ctx;

static void line_res(const float *p, int np, float *r, int nr, void *ctx) {
  (void)np;
  line_ctx *lc = ctx;
  for (int i = 0; i < nr; i++)
    r[i] = p[0] * lc->t[i] + p[1] - lc->y[i];
}

TEST test_line_fit(void) {
  // Generate data y = 2*t + 3 + small noise
  int n = 10;
  float t[10], y[10];
  for (int i = 0; i < n; i++) {
    t[i] = (float)i;
    y[i] = 2.0f * t[i] + 3.0f;  // exact (no noise) — LM should find exact solution
  }
  line_ctx lc = { n, t, y };

  optimizer_params cfg = optimizer_default_params();
  optimizer *o = optimizer_new(2, n, cfg);
  ASSERT(o != NULL);

  float params[2] = { 0.0f, 0.0f };  // start at zero
  int iters = optimizer_solve(o, params, line_res, NULL, &lc);

  ASSERT(iters >= 0);
  ASSERT_IN_RANGE(1.99f, params[0], 2.01f);  // slope ≈ 2
  ASSERT_IN_RANGE(2.99f, params[1], 3.01f);  // intercept ≈ 3
  ASSERT(optimizer_final_cost(o) < 1e-6f);

  optimizer_free(o);
  PASS();
}

// ---------------------------------------------------------------------------
// Test 3: quadratic minimum — r = p^2, minimum at p=0
// ---------------------------------------------------------------------------

static void quad_res(const float *p, int np, float *r, int nr, void *ctx) {
  (void)np; (void)nr; (void)ctx;
  r[0] = p[0];
}

TEST test_quadratic_minimum(void) {
  optimizer_params cfg = optimizer_default_params();
  optimizer *o = optimizer_new(1, 1, cfg);
  ASSERT(o != NULL);

  float params[1] = { 5.0f };
  int iters = optimizer_solve(o, params, quad_res, NULL, NULL);

  ASSERT(iters >= 0);
  ASSERT_IN_RANGE(0.0f, params[0], 0.01f);
  ASSERT(optimizer_converged(o));

  optimizer_free(o);
  PASS();
}

// ---------------------------------------------------------------------------
// Test 4: null/edge-case safety
// ---------------------------------------------------------------------------

TEST test_null_safety(void) {
  optimizer_params cfg = optimizer_default_params();

  // NULL vol
  optimizer *o = optimizer_new(0, 0, cfg);
  ASSERT(o == NULL);

  optimizer_free(NULL);  // must not crash
  ASSERT_EQ(optimizer_final_cost(NULL), 0.0f);
  ASSERT(!optimizer_converged(NULL));

  PASS();
}

// ---------------------------------------------------------------------------
// Test 5: default params have sensible values
// ---------------------------------------------------------------------------

TEST test_default_params(void) {
  optimizer_params p = optimizer_default_params();
  ASSERT(p.max_iterations > 0);
  ASSERT(p.tolerance > 0.0f);
  ASSERT(p.lambda_init > 0.0f);
  ASSERT(p.lambda_up > 1.0f);
  ASSERT(p.lambda_down > 1.0f);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

SUITE(optimizer_suite) {
  RUN_TEST(test_default_params);
  RUN_TEST(test_null_safety);
  RUN_TEST(test_quadratic_minimum);
  RUN_TEST(test_line_fit);
  RUN_TEST(test_rosenbrock_with_jacobian);
  RUN_TEST(test_rosenbrock_fd_jacobian);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(optimizer_suite);
  GREATEST_MAIN_END();
}
