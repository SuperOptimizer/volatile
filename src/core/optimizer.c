#include "core/optimizer.h"
#include "core/log.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct optimizer {
  int   np;   // num_params
  int   nr;   // num_residuals
  optimizer_params cfg;

  // Scratch buffers (allocated once in optimizer_new)
  float *r;       // residuals [nr]
  float *r_try;   // residuals at trial point [nr]
  float *J;       // Jacobian [nr x np], row-major
  float *JtJ;     // normal matrix [np x np]
  float *Jtr;     // gradient -J^T r [np]
  float *delta;   // step [np]
  float *p_try;   // trial params [np]

  float final_cost;
  bool  converged;
};

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------

optimizer_params optimizer_default_params(void) {
  return (optimizer_params){
    .max_iterations = 100,
    .tolerance      = 1e-6f,
    .lambda_init    = 1e-3f,
    .lambda_up      = 10.0f,
    .lambda_down    = 10.0f,
  };
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

optimizer *optimizer_new(int np, int nr, optimizer_params cfg) {
  if (np <= 0 || nr <= 0) return NULL;
  optimizer *o = calloc(1, sizeof(*o));
  if (!o) return NULL;
  o->np  = np;
  o->nr  = nr;
  o->cfg = cfg;

  o->r     = malloc((size_t)nr * sizeof(float));
  o->r_try = malloc((size_t)nr * sizeof(float));
  o->J     = malloc((size_t)nr * np * sizeof(float));
  o->JtJ   = malloc((size_t)np * np * sizeof(float));
  o->Jtr   = malloc((size_t)np * sizeof(float));
  o->delta = malloc((size_t)np * sizeof(float));
  o->p_try = malloc((size_t)np * sizeof(float));

  if (!o->r || !o->r_try || !o->J || !o->JtJ || !o->Jtr || !o->delta || !o->p_try) {
    optimizer_free(o);
    return NULL;
  }
  return o;
}

void optimizer_free(optimizer *o) {
  if (!o) return;
  free(o->r);
  free(o->r_try);
  free(o->J);
  free(o->JtJ);
  free(o->Jtr);
  free(o->delta);
  free(o->p_try);
  free(o);
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// 0.5 * ||r||^2
static float cost(const float *r, int nr) {
  float s = 0.0f;
  for (int i = 0; i < nr; i++) s += r[i] * r[i];
  return 0.5f * s;
}

// Finite-difference Jacobian (central differences, eps=1e-5)
static void fd_jacobian(const float *p, int np, float *J, int nr,
                        residual_fn res_fn, void *ctx) {
  const float eps = 1e-5f;
  float *ptmp = malloc((size_t)np * sizeof(float));
  float *rp   = malloc((size_t)nr * sizeof(float));
  float *rm   = malloc((size_t)nr * sizeof(float));
  if (!ptmp || !rp || !rm) { free(ptmp); free(rp); free(rm); return; }
  memcpy(ptmp, p, (size_t)np * sizeof(float));

  for (int j = 0; j < np; j++) {
    ptmp[j] = p[j] + eps;
    res_fn(ptmp, np, rp, nr, ctx);
    ptmp[j] = p[j] - eps;
    res_fn(ptmp, np, rm, nr, ctx);
    ptmp[j] = p[j];
    float inv2e = 0.5f / eps;
    for (int i = 0; i < nr; i++)
      J[i * np + j] = (rp[i] - rm[i]) * inv2e;
  }
  free(ptmp); free(rp); free(rm);
}

// Cholesky solve: A*x = b, A is np x np symmetric positive semi-definite.
// Returns false if factorisation fails (singular / not PD).
static bool cholesky_solve(float *A, float *b, float *x, int n) {
  // In-place lower Cholesky on A (overwrites upper triangle too but we only use lower)
  for (int j = 0; j < n; j++) {
    float s = A[j * n + j];
    for (int k = 0; k < j; k++) s -= A[j * n + k] * A[j * n + k];
    if (s <= 0.0f) return false;
    A[j * n + j] = sqrtf(s);
    float inv = 1.0f / A[j * n + j];
    for (int i = j + 1; i < n; i++) {
      float t = A[i * n + j];
      for (int k = 0; k < j; k++) t -= A[i * n + k] * A[j * n + k];
      A[i * n + j] = t * inv;
    }
  }
  // Forward substitution L*y = b
  for (int i = 0; i < n; i++) {
    float t = b[i];
    for (int k = 0; k < i; k++) t -= A[i * n + k] * x[k];
    x[i] = t / A[i * n + i];
  }
  // Back substitution L^T * x = y
  for (int i = n - 1; i >= 0; i--) {
    float t = x[i];
    for (int k = i + 1; k < n; k++) t -= A[k * n + i] * x[k];
    x[i] = t / A[i * n + i];
  }
  return true;
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

int optimizer_solve(optimizer *o, float *params,
                    residual_fn res_fn, jacobian_fn jac_fn, void *ctx) {
  if (!o || !params || !res_fn) return -1;
  int np = o->np, nr = o->nr;

  float lambda = o->cfg.lambda_init;
  o->converged = false;

  res_fn(params, np, o->r, nr, ctx);
  float F = cost(o->r, nr);

  int iter;
  for (iter = 0; iter < o->cfg.max_iterations; iter++) {
    // Build Jacobian
    if (jac_fn)
      jac_fn(params, np, o->J, nr, ctx);
    else
      fd_jacobian(params, np, o->J, nr, res_fn, ctx);

    // JtJ = J^T J,  Jtr = J^T r
    for (int i = 0; i < np; i++) {
      float g = 0.0f;
      for (int k = 0; k < nr; k++) g += o->J[k * np + i] * o->r[k];
      o->Jtr[i] = g;
      for (int j = 0; j <= i; j++) {
        float v = 0.0f;
        for (int k = 0; k < nr; k++) v += o->J[k * np + i] * o->J[k * np + j];
        o->JtJ[i * np + j] = o->JtJ[j * np + i] = v;
      }
    }

    // Check gradient convergence
    float gmax = 0.0f;
    for (int i = 0; i < np; i++) {
      float g = fabsf(o->Jtr[i]);
      if (g > gmax) gmax = g;
    }
    if (gmax < o->cfg.tolerance) { o->converged = true; break; }

    // LM step: (JtJ + lambda * diag(JtJ)) * delta = -Jtr
    // Use a local copy of JtJ for factorisation
    float *A = malloc((size_t)np * np * sizeof(float));
    if (!A) { o->final_cost = F; return -1; }
    memcpy(A, o->JtJ, (size_t)np * np * sizeof(float));

    float *rhs = malloc((size_t)np * sizeof(float));
    if (!rhs) { free(A); o->final_cost = F; return -1; }

    bool step_accepted = false;
    for (int attempt = 0; attempt < 8; attempt++) {
      // Apply damping
      for (int i = 0; i < np; i++) {
        float d = o->JtJ[i * np + i];
        A[i * np + i] = d + lambda * (d > 0.0f ? d : 1.0f);
      }
      // rhs = -Jtr
      for (int i = 0; i < np; i++) rhs[i] = -o->Jtr[i];

      if (cholesky_solve(A, rhs, o->delta, np)) {
        // Trial step
        for (int i = 0; i < np; i++) o->p_try[i] = params[i] + o->delta[i];
        res_fn(o->p_try, np, o->r_try, nr, ctx);
        float F_try = cost(o->r_try, nr);

        if (F_try < F) {
          memcpy(params, o->p_try, (size_t)np * sizeof(float));
          memcpy(o->r,   o->r_try, (size_t)nr * sizeof(float));
          F = F_try;
          lambda /= o->cfg.lambda_down;
          if (lambda < 1e-10f) lambda = 1e-10f;
          step_accepted = true;
          break;
        }
      }
      // Step rejected: increase damping and retry
      lambda *= o->cfg.lambda_up;
      if (lambda > 1e10f) break;
      // Restore A from JtJ for next attempt
      memcpy(A, o->JtJ, (size_t)np * np * sizeof(float));
    }
    free(rhs);
    free(A);

    if (!step_accepted) {
      LOG_DEBUG("optimizer: step rejected at iter %d, lambda=%.3e", iter, (double)lambda);
      break;
    }
  }

  o->final_cost = F;
  return iter;
}

float optimizer_final_cost(const optimizer *o) { return o ? o->final_cost : 0.0f; }
bool  optimizer_converged(const optimizer *o)  { return o && o->converged; }
