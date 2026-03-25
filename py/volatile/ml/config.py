from __future__ import annotations

from typing import Any

try:
  from tinygrad import Tensor, nn
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

from .model import UNet, ResUNet


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
  "in_channels": 1,
  "out_channels": 4,
  "base_channels": 32,
  "num_levels": 4,
  "block_type": "residual",   # "conv" | "residual"
  "use_se": False,            # Squeeze-and-Excitation on encoder levels
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_model(config: dict[str, Any]):
  """
  Build a UNet (or ResUNet) from a plain-dict config.

  Supported keys (all optional, fall back to DEFAULT_CONFIG):
    in_channels   : int   — input feature channels
    out_channels  : int   — output class channels
    base_channels : int   — channels at first encoder level (doubles per level)
    num_levels    : int   — encoder/decoder depth (excluding bottleneck)
    block_type    : str   — "conv" for plain ConvBlocks, "residual" for ResBlocks
    use_se        : bool  — add Squeeze-and-Excitation attention to encoder

  Returns a UNet or ResUNet instance.

  Example::

    config = {
      "in_channels": 1, "out_channels": 4, "base_channels": 32,
      "num_levels": 5, "block_type": "residual", "use_se": True
    }
    model = build_model(config)
  """
  if not _TINYGRAD:
    raise ImportError("tinygrad is required for build_model")

  cfg = {**DEFAULT_CONFIG, **config}

  in_ch = int(cfg["in_channels"])
  out_ch = int(cfg["out_channels"])
  base_ch = int(cfg["base_channels"])
  num_levels = int(cfg["num_levels"])
  block_type = str(cfg["block_type"]).lower()
  use_se = bool(cfg["use_se"])

  if block_type == "conv":
    return UNet(
      in_channels=in_ch,
      out_channels=out_ch,
      base_channels=base_ch,
      num_levels=num_levels,
    )
  elif block_type == "residual":
    return ResUNet(
      in_channels=in_ch,
      out_channels=out_ch,
      base_channels=base_ch,
      num_levels=num_levels,
      use_se=use_se,
    )
  else:
    raise ValueError(f"unknown block_type '{block_type}'; choose 'conv' or 'residual'")
