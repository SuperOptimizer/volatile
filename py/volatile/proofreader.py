"""proofreader.py — Review and approve/reject predicted labels.

Usage:
  pr = Proofreader("predictions.npy", ground_truth_path="gt.npy")
  decision = pr.review_patch(patch_id=0)   # "approve" | "reject" | "skip"
  pr.export_approved("approved_labels.npy")
  print(pr.stats())
"""
from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Decisions stored per patch
_APPROVE = "approve"
_REJECT  = "reject"
_SKIP    = "skip"
_PENDING = "pending"

# ---------------------------------------------------------------------------
# Patch extraction helpers
# ---------------------------------------------------------------------------

def _extract_patch(volume: np.ndarray, patch_id: int, patch_size: int) -> np.ndarray:
  """Extract a cubic patch from a flat (D, H, W) volume by linear index."""
  d, h, w = volume.shape[-3], volume.shape[-2], volume.shape[-1]
  cols = w // patch_size
  rows = h // patch_size
  deps = d // patch_size
  total = deps * rows * cols
  if patch_id < 0 or patch_id >= total:
    raise IndexError(f"patch_id {patch_id} out of range [0, {total})")

  di = patch_id // (rows * cols)
  ri = (patch_id % (rows * cols)) // cols
  ci = patch_id % cols
  zs, ze = di * patch_size, (di + 1) * patch_size
  ys, ye = ri * patch_size, (ri + 1) * patch_size
  xs, xe = ci * patch_size, (ci + 1) * patch_size
  return volume[..., zs:ze, ys:ye, xs:xe]


def _patch_count(volume: np.ndarray, patch_size: int) -> int:
  d, h, w = volume.shape[-3], volume.shape[-2], volume.shape[-1]
  return (d // patch_size) * (h // patch_size) * (w // patch_size)


# ---------------------------------------------------------------------------
# Proofreader
# ---------------------------------------------------------------------------

class Proofreader:
  """Review and approve/reject predicted label patches.

  Decisions are stored in memory and can be persisted via save_state /
  load_state.  export_approved() writes only approved patches back to a
  volume of the same shape as predictions.
  """

  def __init__(
    self,
    predictions_path: str,
    ground_truth_path: Optional[str] = None,
    patch_size: int = 64,
    state_path: Optional[str] = None,
  ):
    self.predictions_path = predictions_path
    self.ground_truth_path = ground_truth_path
    self.patch_size = patch_size
    self.state_path = state_path

    log.info("Loading predictions from %s", predictions_path)
    self._predictions: np.ndarray = np.load(predictions_path)
    self._ground_truth: Optional[np.ndarray] = None
    if ground_truth_path:
      log.info("Loading ground truth from %s", ground_truth_path)
      self._ground_truth = np.load(ground_truth_path)

    n = _patch_count(self._predictions, self.patch_size)
    # decisions[patch_id] = "approve" | "reject" | "skip" | "pending"
    self._decisions: Dict[int, str] = {i: _PENDING for i in range(n)}

    if state_path and os.path.isfile(state_path):
      self.load_state(state_path)

  # ---- review ------------------------------------------------------------

  def review_patch(
    self,
    patch_id: int,
    show: bool = False,
  ) -> str:
    """Return information about a patch and prompt for a decision.

    In interactive mode (show=True, napari available) opens a small viewer.
    In headless mode prompts via stdin.  Returns the stored decision string.

    Args:
      patch_id: linear patch index.
      show: if True, attempt to open a napari viewer for the patch.
    Returns:
      Decision string: "approve", "reject", or "skip".
    """
    patch = _extract_patch(self._predictions, patch_id, self.patch_size)
    gt_patch: Optional[np.ndarray] = None
    if self._ground_truth is not None:
      gt_patch = _extract_patch(self._ground_truth, patch_id, self.patch_size)

    log.info(
      "Patch %d: shape=%s  pred_sum=%.1f%s",
      patch_id, patch.shape, float(patch.sum()),
      f"  gt_sum={float(gt_patch.sum()):.1f}" if gt_patch is not None else "",
    )

    if show:
      decision = self._show_napari(patch_id, patch, gt_patch)
    else:
      decision = self._prompt_cli(patch_id, patch, gt_patch)

    self._decisions[patch_id] = decision
    if self.state_path:
      self.save_state(self.state_path)
    return decision

  def _prompt_cli(
    self,
    patch_id: int,
    patch: np.ndarray,
    gt_patch: Optional[np.ndarray],
  ) -> str:
    """Prompt the user in CLI mode; returns decision."""
    print(f"\nPatch {patch_id}: shape={patch.shape}  pred_sum={patch.sum():.1f}")
    if gt_patch is not None:
      overlap = float(np.logical_and(patch > 0, gt_patch > 0).sum())
      union   = float(np.logical_or(patch > 0, gt_patch > 0).sum())
      iou = overlap / union if union > 0 else 0.0
      print(f"  IoU with ground truth: {iou:.3f}")
    while True:
      ans = input("  Decision [a=approve / r=reject / s=skip]: ").strip().lower()
      if ans in ("a", "approve"):
        return _APPROVE
      if ans in ("r", "reject"):
        return _REJECT
      if ans in ("s", "skip"):
        return _SKIP
      print("  Please enter 'a', 'r', or 's'.")

  def _show_napari(
    self,
    patch_id: int,
    patch: np.ndarray,
    gt_patch: Optional[np.ndarray],
  ) -> str:
    """Open napari for visual review; decision set via a widget."""
    try:
      import napari
    except ImportError:
      log.warning("napari not available; falling back to CLI prompt")
      return self._prompt_cli(patch_id, patch, gt_patch)

    decision_holder: List[str] = [_SKIP]

    viewer = napari.Viewer(title=f"Patch {patch_id}")
    viewer.add_image(patch, name="prediction", colormap="magma")
    if gt_patch is not None:
      viewer.add_labels(gt_patch.astype(np.int32), name="ground_truth")

    from napari.qt.threading import create_worker

    def _set(d: str):
      decision_holder[0] = d
      viewer.close()

    # Simple keybindings: a/r/s
    @viewer.bind_key("a")
    def _approve(v): _set(_APPROVE)

    @viewer.bind_key("r")
    def _reject(v): _set(_REJECT)

    @viewer.bind_key("s")
    def _skip_key(v): _set(_SKIP)

    napari.run()
    return decision_holder[0]

  # ---- set_decision (programmatic) ----------------------------------------

  def set_decision(self, patch_id: int, decision: str) -> None:
    """Set the decision for a patch without interactive review."""
    if decision not in (_APPROVE, _REJECT, _SKIP, _PENDING):
      raise ValueError(f"Unknown decision: {decision!r}")
    if patch_id not in self._decisions:
      raise IndexError(f"patch_id {patch_id} out of range")
    self._decisions[patch_id] = decision

  # ---- export ------------------------------------------------------------

  def export_approved(self, output_path: str) -> np.ndarray:
    """Write approved patches to a new volume; rejected/pending patches are zeroed.

    Returns the assembled output array.
    """
    out = np.zeros_like(self._predictions)
    ps = self.patch_size
    d, h, w = self._predictions.shape[-3], self._predictions.shape[-2], self._predictions.shape[-1]
    cols = w // ps
    rows = h // ps
    deps = d // ps

    for patch_id, decision in self._decisions.items():
      if decision != _APPROVE:
        continue
      di = patch_id // (rows * cols)
      ri = (patch_id % (rows * cols)) // cols
      ci = patch_id % cols
      zs, ze = di * ps, (di + 1) * ps
      ys, ye = ri * ps, (ri + 1) * ps
      xs, xe = ci * ps, (ci + 1) * ps
      out[..., zs:ze, ys:ye, xs:xe] = self._predictions[..., zs:ze, ys:ye, xs:xe]

    np.save(output_path, out)
    log.info("Exported approved patches to %s", output_path)
    return out

  # ---- stats -------------------------------------------------------------

  def stats(self) -> Dict[str, int]:
    """Return counts: approved, rejected, skipped, pending, total."""
    counts = {_APPROVE: 0, _REJECT: 0, _SKIP: 0, _PENDING: 0}
    for d in self._decisions.values():
      counts[d] = counts.get(d, 0) + 1
    return {
      "approved": counts[_APPROVE],
      "rejected": counts[_REJECT],
      "skipped":  counts[_SKIP],
      "pending":  counts[_PENDING],
      "total":    len(self._decisions),
    }

  # ---- persistence -------------------------------------------------------

  def save_state(self, path: str) -> None:
    """Persist decisions to a JSON file."""
    with open(path, "w") as f:
      json.dump({str(k): v for k, v in self._decisions.items()}, f, indent=2)
    log.debug("State saved to %s", path)

  def load_state(self, path: str) -> None:
    """Load decisions from a JSON file (merges into current decisions)."""
    with open(path) as f:
      data = json.load(f)
    for k, v in data.items():
      pid = int(k)
      if pid in self._decisions:
        self._decisions[pid] = v
    log.info("State loaded from %s", path)
