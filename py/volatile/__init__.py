from __future__ import annotations
__version__ = "0.1.0"

# try to import the C extension
try:
  from volatile._core import (
    version as _c_version,
    log_set_level,
    log_get_level,
    vol_open,
    vol_free,
    vol_num_levels,
    vol_shape,
    vol_sample,
  )
except ImportError:
  _c_version = None
  def log_set_level(level: int) -> None: pass
  def log_get_level() -> int: return 1
  def vol_open(path: str): raise ImportError("volatile._core not built")
  def vol_free(vol) -> None: raise ImportError("volatile._core not built")
  def vol_num_levels(vol) -> int: raise ImportError("volatile._core not built")
  def vol_shape(vol, level: int = 0) -> tuple: raise ImportError("volatile._core not built")
  def vol_sample(vol, level: int = 0, z: float = 0, y: float = 0, x: float = 0) -> float: raise ImportError("volatile._core not built")
