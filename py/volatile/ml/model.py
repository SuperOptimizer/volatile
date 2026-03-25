from __future__ import annotations

try:
  from tinygrad import Tensor, nn
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False


class ConvBlock:
  """Conv2d + ReLU building block."""

  def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for ConvBlock")
    self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.conv(x).relu()

  def parameters(self) -> list:
    return nn.state.get_parameters(self)


class _EncoderLevel:
  """Two ConvBlocks at one encoder level."""

  def __init__(self, in_ch: int, out_ch: int):
    self.c1 = ConvBlock(in_ch, out_ch)
    self.c2 = ConvBlock(out_ch, out_ch)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.c2(self.c1(x))


class _DecoderLevel:
  """Upsample + skip-concat + two ConvBlocks at one decoder level."""

  def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
    # in_ch is channels coming up from below; skip_ch is from the encoder skip
    self.c1 = ConvBlock(in_ch + skip_ch, out_ch)
    self.c2 = ConvBlock(out_ch, out_ch)

  def __call__(self, x: "Tensor", skip: "Tensor") -> "Tensor":
    # bilinear upsample to match skip spatial dims
    h, w = skip.shape[2], skip.shape[3]
    x = x.interpolate((h, w))
    x = Tensor.cat(x, skip, dim=1)
    return self.c2(self.c1(x))


class UNet:
  """
  2-D UNet with configurable depth and channel width.

  Args:
    in_channels:   number of input feature channels (e.g. 1 for grayscale)
    out_channels:  number of output classes / channels
    base_channels: channels at the first encoder level; doubles each level
    num_levels:    number of encoder/decoder levels (excluding bottleneck)
  """

  def __init__(
    self,
    in_channels: int = 1,
    out_channels: int = 4,
    base_channels: int = 32,
    num_levels: int = 4,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for UNet")

    self.num_levels = num_levels

    # encoder levels: channels go base, 2*base, 4*base, ...
    enc_channels = [base_channels * (2 ** i) for i in range(num_levels)]
    self.encoders: list[_EncoderLevel] = []
    ch_in = in_channels
    for ch_out in enc_channels:
      self.encoders.append(_EncoderLevel(ch_in, ch_out))
      ch_in = ch_out

    # bottleneck
    bn_ch = base_channels * (2 ** num_levels)
    self.bottleneck = _EncoderLevel(ch_in, bn_ch)

    # decoder levels: mirror of encoder, skip channels come from encoders in reverse
    self.decoders: list[_DecoderLevel] = []
    ch_up = bn_ch
    for ch_skip in reversed(enc_channels):
      ch_out = ch_skip  # decoder output matches encoder width at this level
      self.decoders.append(_DecoderLevel(ch_up, ch_skip, ch_out))
      ch_up = ch_out

    # final 1×1 projection
    self.final_conv = nn.Conv2d(ch_up, out_channels, kernel_size=1)

  def __call__(self, x: "Tensor") -> "Tensor":
    # encoder pass — collect feature maps before pooling as skip connections
    skips: list["Tensor"] = []
    for enc in self.encoders:
      x = enc(x)
      skips.append(x)
      x = x.max_pool2d(2)

    x = self.bottleneck(x)

    # decoder pass — skips in reverse order
    for dec, skip in zip(self.decoders, reversed(skips)):
      x = dec(x, skip)

    return self.final_conv(x)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------

class SEBlock:
  """
  Channel Squeeze-and-Excitation block.

  Recalibrates channel-wise feature responses by learning a per-channel gating
  vector via global average pooling → FC → ReLU → FC → Sigmoid.

  Args:
    channels:        number of input/output channels
    reduction_ratio: channel reduction factor (default 16; clamped to ≥ 1)
  """

  def __init__(self, channels: int, reduction_ratio: int = 16):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for SEBlock")
    reduced = max(1, channels // reduction_ratio)
    self.fc1 = nn.Linear(channels, reduced)
    self.fc2 = nn.Linear(reduced, channels)

  def __call__(self, x: "Tensor") -> "Tensor":
    # Global average pool → (B, C)
    s = x.mean(axis=(2, 3))
    s = self.fc1(s).relu()
    s = self.fc2(s).sigmoid()
    # Rescale: (B, C) → (B, C, 1, 1)
    return x * s.reshape(s.shape[0], s.shape[1], 1, 1)


# ---------------------------------------------------------------------------
# Residual block
# ---------------------------------------------------------------------------

class ResBlock:
  """
  Pre-activation residual block (two 3×3 convolutions with a skip connection).

  When `in_ch != out_ch` a 1×1 projection is applied to the skip path.
  Optionally applies a Squeeze-and-Excitation block to the residual branch.

  Args:
    in_ch:  input channels
    out_ch: output channels
    use_se: attach a SEBlock to the residual branch (default False)
  """

  def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for ResBlock")
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
    self.bn1 = nn.BatchNorm(out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
    self.bn2 = nn.BatchNorm(out_ch)
    self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else None
    self.se = SEBlock(out_ch) if use_se else None

  def __call__(self, x: "Tensor") -> "Tensor":
    skip = x if self.proj is None else self.proj(x)
    out = self.bn1(self.conv1(x)).relu()
    out = self.bn2(self.conv2(out))
    if self.se is not None:
      out = self.se(out)
    return (out + skip).relu()


# ---------------------------------------------------------------------------
# Residual encoder / decoder levels
# ---------------------------------------------------------------------------

class _ResEncoderLevel:
  """Two ResBlocks at one encoder level."""

  def __init__(self, in_ch: int, out_ch: int, use_se: bool = False):
    self.r1 = ResBlock(in_ch, out_ch, use_se=use_se)
    self.r2 = ResBlock(out_ch, out_ch, use_se=use_se)

  def __call__(self, x: "Tensor") -> "Tensor":
    return self.r2(self.r1(x))


class _ResDecoderLevel:
  """Upsample + skip-concat + two ResBlocks at one decoder level."""

  def __init__(self, in_ch: int, skip_ch: int, out_ch: int, use_se: bool = False):
    self.r1 = ResBlock(in_ch + skip_ch, out_ch, use_se=use_se)
    self.r2 = ResBlock(out_ch, out_ch, use_se=use_se)

  def __call__(self, x: "Tensor", skip: "Tensor") -> "Tensor":
    h, w = skip.shape[2], skip.shape[3]
    x = x.interpolate((h, w))
    x = Tensor.cat(x, skip, dim=1)
    return self.r2(self.r1(x))


# ---------------------------------------------------------------------------
# ResUNet
# ---------------------------------------------------------------------------

class ResUNet:
  """
  Residual UNet with optional Squeeze-and-Excitation on every encoder block.

  Drop-in replacement for UNet; accepts the same interface plus `use_se`.

  Args:
    in_channels:   input feature channels
    out_channels:  output classes / channels
    base_channels: channels at first encoder level; doubles each level
    num_levels:    encoder/decoder depth (excluding bottleneck)
    use_se:        attach SEBlock to each ResBlock (default False)
  """

  def __init__(
    self,
    in_channels: int = 1,
    out_channels: int = 4,
    base_channels: int = 32,
    num_levels: int = 4,
    use_se: bool = False,
  ):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for ResUNet")

    self.num_levels = num_levels

    enc_channels = [base_channels * (2 ** i) for i in range(num_levels)]
    self.encoders: list[_ResEncoderLevel] = []
    ch_in = in_channels
    for ch_out in enc_channels:
      self.encoders.append(_ResEncoderLevel(ch_in, ch_out, use_se=use_se))
      ch_in = ch_out

    bn_ch = base_channels * (2 ** num_levels)
    self.bottleneck = _ResEncoderLevel(ch_in, bn_ch, use_se=use_se)

    self.decoders: list[_ResDecoderLevel] = []
    ch_up = bn_ch
    for ch_skip in reversed(enc_channels):
      ch_out = ch_skip
      self.decoders.append(_ResDecoderLevel(ch_up, ch_skip, ch_out, use_se=use_se))
      ch_up = ch_out

    self.final_conv = nn.Conv2d(ch_up, out_channels, kernel_size=1)

  def __call__(self, x: "Tensor") -> "Tensor":
    skips: list["Tensor"] = []
    for enc in self.encoders:
      x = enc(x)
      skips.append(x)
      x = x.max_pool2d(2)

    x = self.bottleneck(x)

    for dec, skip in zip(self.decoders, reversed(skips)):
      x = dec(x, skip)

    return self.final_conv(x)
