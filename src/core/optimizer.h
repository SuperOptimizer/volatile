#pragma once
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Levenberg-Marquardt nonlinear least-squares optimizer
//
// Minimises  F(p) = 0.5 * sum_i  r_i(p)^2
//
// Usage:
//   optimizer *o = optimizer_new(num_params, num_residuals, opts);
//   int iters    = optimizer_solve(o, params, res_fn, jac_fn, ctx);
//   float cost   = optimizer_final_cost(o);
//   optimizer_free(o);
// ---------------------------------------------------------------------------

typedef struct optimizer optimizer;

// Compute residuals r[0..num_residuals-1] from params p[0..num_params-1].
typedef void (*residual_fn)(const float *params, int num_params,
                             float *residuals, int num_residuals, void *ctx);

// Compute Jacobian J[i * num_params + j] = d r_i / d p_j (row-major).
// Pass NULL to use central finite differences (eps = 1e-5).
typedef void (*jacobian_fn)(const float *params, int num_params,
                             float *jacobian, int num_residuals, void *ctx);

typedef struct {
  int   max_iterations; // default: 100
  float tolerance;      // stop when |gradient|_inf < tolerance (default: 1e-6)
  float lambda_init;    // initial damping factor (default: 1e-3)
  float lambda_up;      // multiply lambda on step failure (default: 10)
  float lambda_down;    // divide   lambda on step success (default: 10)
} optimizer_params;

// Returns an optimizer_params with sensible defaults.
optimizer_params optimizer_default_params(void);

optimizer *optimizer_new(int num_params, int num_residuals, optimizer_params p);
void       optimizer_free(optimizer *o);

// Solve in place. params[] is initial guess on entry, solution on exit.
// Returns number of iterations used, or -1 on allocation failure.
int  optimizer_solve(optimizer *o, float *params,
                     residual_fn res_fn, jacobian_fn jac_fn, void *ctx);

float optimizer_final_cost(const optimizer *o);
bool  optimizer_converged(const optimizer *o);
