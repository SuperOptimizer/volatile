from __future__ import annotations

"""
Learning-rate schedulers for tinygrad training.

All schedulers are pure Python (no torch dependency) and operate by returning
a float LR for a given step, or by updating an optimizer in place via `step()`.

Schedulers:
  CosineAnnealingWarmRestarts  — cosine decay with periodic warm restarts (SGDR)
  PolynomialLR                 — polynomial decay with optional linear warmup
  OneCycleLR                   — 1-cycle policy (warmup → peak → annealing)

All can be used standalone::

    sched = CosineAnnealingWarmRestarts(lr=1e-3, T_0=10)
    for epoch in range(100):
        lr = sched.get_lr(epoch)
        ...

Or with an optimizer::

    sched = PolynomialLR(lr=1e-3, total_steps=100, warmup_steps=5)
    sched.attach(optimizer)
    for epoch in range(100):
        sched.step()     # updates optimizer.lr in place
"""

import math
from typing import Optional

try:
  _TINYGRAD = True
  from tinygrad import nn as _tg_nn
except ImportError:
  _TINYGRAD = False


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class _Scheduler:
  """Abstract base — subclasses implement `get_lr(step)`."""

  def __init__(self, lr: float):
    self._initial_lr = lr
    self._step = 0
    self._optimizer = None

  def attach(self, optimizer) -> "_Scheduler":
    """Bind this scheduler to a tinygrad optimizer."""
    self._optimizer = optimizer
    return self

  def step(self) -> float:
    """Advance internal counter, update optimizer LR, return new LR."""
    lr = self.get_lr(self._step)
    self._step += 1
    if self._optimizer is not None:
      self._optimizer.lr = lr
    return lr

  def get_lr(self, step: int) -> float:
    raise NotImplementedError

  @property
  def last_lr(self) -> float:
    return self.get_lr(max(0, self._step - 1))

  @property
  def initial_lr(self) -> float:
    return self._initial_lr

  def reset(self) -> None:
    self._step = 0


# ---------------------------------------------------------------------------
# CosineAnnealingWarmRestarts
# ---------------------------------------------------------------------------

class CosineAnnealingWarmRestarts(_Scheduler):
  """
  Cosine annealing with warm restarts (SGDR).

  LR decays from `lr` to `eta_min` over `T_0` steps following a cosine curve,
  then restarts.  After each restart the cycle length is multiplied by `T_mult`.

  Args:
    lr:       peak (initial) learning rate
    T_0:      steps in the first cycle
    T_mult:   cycle-length multiplier after each restart (default 1 = constant cycles)
    eta_min:  minimum LR (default 1e-6)
    warmup_steps: linear warmup applied at the start of *each* cycle (default 0)
    gamma:    decay applied to peak LR after each restart (default 1.0 = no decay)
  """

  def __init__(
    self,
    lr: float,
    T_0: int,
    T_mult: float = 1.0,
    eta_min: float = 1e-6,
    warmup_steps: int = 0,
    gamma: float = 1.0,
  ):
    super().__init__(lr)
    if T_0 <= 0:
      raise ValueError(f"T_0 must be > 0, got {T_0}")
    self.T_0 = T_0
    self.T_mult = float(T_mult)
    self.eta_min = eta_min
    self.warmup_steps = int(warmup_steps)
    self.gamma = gamma

  def _cycle_info(self, step: int):
    """Return (cycle_index, step_within_cycle, cycle_length)."""
    if self.T_mult == 1.0:
      cycle = step // self.T_0
      t_i = self.T_0
      t_cur = step % self.T_0
    else:
      # Geometric series: T_0, T_0*T_mult, T_0*T_mult^2, ...
      # Total steps up to end of cycle i: T_0 * (T_mult^(i+1) - 1) / (T_mult - 1)
      cycle = 0
      remaining = step
      t_i = self.T_0
      while remaining >= t_i:
        remaining -= t_i
        cycle += 1
        t_i = max(1, int(t_i * self.T_mult))
      t_cur = remaining
    return cycle, t_cur, t_i

  def get_lr(self, step: int) -> float:
    cycle, t_cur, t_i = self._cycle_info(step)
    peak = self._initial_lr * (self.gamma ** cycle)
    if self.warmup_steps > 0 and t_cur < self.warmup_steps:
      return self.eta_min + (peak - self.eta_min) * (t_cur / self.warmup_steps)
    cos_inner = t_cur - self.warmup_steps
    cos_len = max(1, t_i - self.warmup_steps)
    return self.eta_min + 0.5 * (peak - self.eta_min) * (1.0 + math.cos(math.pi * cos_inner / cos_len))


# ---------------------------------------------------------------------------
# PolynomialLR
# ---------------------------------------------------------------------------

class PolynomialLR(_Scheduler):
  """
  Polynomial LR decay with optional linear warmup.

  During warmup (steps 0..warmup_steps-1) LR increases linearly from 0 to `lr`.
  After warmup it decays as::

      lr * (1 - t / total_steps) ** exponent

  where t = step - warmup_steps and total_steps is measured after warmup end.

  Args:
    lr:           peak learning rate (reached at end of warmup / start of decay)
    total_steps:  total training steps (including warmup)
    warmup_steps: linear warmup duration (default 0)
    exponent:     polynomial exponent (default 0.9, nnU-Net style)
    eta_min:      floor LR (default 0.0)
  """

  def __init__(
    self,
    lr: float,
    total_steps: int,
    warmup_steps: int = 0,
    exponent: float = 0.9,
    eta_min: float = 0.0,
  ):
    super().__init__(lr)
    self.total_steps = int(total_steps)
    self.warmup_steps = int(warmup_steps)
    self.exponent = float(exponent)
    self.eta_min = float(eta_min)

  def get_lr(self, step: int) -> float:
    if step < self.warmup_steps:
      return self._initial_lr * (step / max(1, self.warmup_steps))
    t = step - self.warmup_steps
    decay_steps = max(1, self.total_steps - self.warmup_steps)
    frac = min(1.0, t / decay_steps)
    return self.eta_min + (self._initial_lr - self.eta_min) * ((1.0 - frac) ** self.exponent)


# ---------------------------------------------------------------------------
# OneCycleLR
# ---------------------------------------------------------------------------

class OneCycleLR(_Scheduler):
  """
  1-cycle learning rate policy.

  Three phases:
  1. Warmup  (0 → peak_lr) over `pct_start * total_steps` steps — cosine ramp
  2. Annealing (peak_lr → eta_min) over the remaining steps — cosine decay
  3. (Optionally) final very low LR for last `final_div_factor` fraction

  Args:
    lr:               base learning rate (= peak_lr / div_factor)
    total_steps:      total training steps
    pct_start:        fraction of steps used for warmup (default 0.3)
    div_factor:       peak_lr = lr * div_factor (default 25.0)
    final_div_factor: eta_min = lr / final_div_factor (default 1e4)
    anneal_strategy:  'cos' or 'linear' (default 'cos')
  """

  def __init__(
    self,
    lr: float,
    total_steps: int,
    pct_start: float = 0.3,
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    anneal_strategy: str = 'cos',
  ):
    super().__init__(lr)
    self.total_steps = int(total_steps)
    self.pct_start = float(pct_start)
    self.peak_lr = lr * div_factor
    self.eta_min = lr / final_div_factor
    self.anneal_strategy = anneal_strategy.lower()
    self._warmup_steps = int(math.ceil(pct_start * total_steps))

  def _anneal(self, start: float, end: float, frac: float) -> float:
    if self.anneal_strategy == 'cos':
      return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))
    # linear
    return start + frac * (end - start)

  def get_lr(self, step: int) -> float:
    step = min(step, self.total_steps - 1)
    if step <= self._warmup_steps:
      frac = step / max(1, self._warmup_steps)
      return self._anneal(self._initial_lr, self.peak_lr, 1.0 - frac)
    frac = (step - self._warmup_steps) / max(1, self.total_steps - self._warmup_steps)
    return self._anneal(self.peak_lr, self.eta_min, frac)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_scheduler(name: str, lr: float, total_steps: int, **kwargs) -> _Scheduler:
  """
  Build a scheduler by name.

  Supported names: 'cosine_warm_restarts', 'poly', 'one_cycle'

  Extra keyword args are forwarded to the scheduler constructor.
  """
  name = name.lower().replace('-', '_')
  if name in ('cosine_warm_restarts', 'cosine_warmup_restarts', 'sgdr'):
    T_0 = kwargs.pop('T_0', total_steps)
    return CosineAnnealingWarmRestarts(lr=lr, T_0=T_0, **kwargs)
  elif name in ('poly', 'polynomial'):
    return PolynomialLR(lr=lr, total_steps=total_steps, **kwargs)
  elif name in ('one_cycle', 'onecycle'):
    return OneCycleLR(lr=lr, total_steps=total_steps, **kwargs)
  raise ValueError(f"unknown scheduler '{name}'; choose: cosine_warm_restarts, poly, one_cycle")
