"""napari_plugin.py — Stub for future napari integration.

When napari becomes an optional dependency, this module will register
volatile as a napari plugin for interactive training and labeling via the
npe2 plugin system.
"""
from __future__ import annotations


def register_napari_plugin() -> None:
  """Register volatile as a napari plugin for interactive training/labeling.

  This is a stub. Full implementation requires:
    - napari and npe2 installed (`pip install volatile[napari]`)
    - Widget classes for ink-detection overlay, surface browsing,
      and annotation editing (see volatile.seg.QuadSurface)
    - Entry-point declaration in pyproject.toml:
        [project.entry-points."napari.manifest"]
        volatile = "volatile.napari_plugin:napari.yaml"
  """
  try:
    import napari  # noqa: F401
  except ImportError:
    raise ImportError(
      "napari is required for register_napari_plugin(); "
      "install it with: pip install volatile[napari]"
    )
  # Future: call napari.run() or register widget contributions here.
  pass


def get_napari_widget_contributions() -> list[dict]:
  """Return napari widget contribution specs (npe2 format).

  Returns an empty list until widgets are implemented.
  """
  return []
