from __future__ import annotations

import math
import numpy as np
from typing import List, Tuple, Union

try:
  from tinygrad import Tensor
  _TINYGRAD = True
except ImportError:
  _TINYGRAD = False

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Transform:
  """Base class for all augmentation transforms."""
  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Compose / Random
# ---------------------------------------------------------------------------

class Compose(Transform):
  """Apply a list of transforms sequentially."""

  def __init__(self, transforms: List[Transform]):
    self.transforms = transforms

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    for t in self.transforms:
      image, mask = t(image, mask)
    return image, mask


class RandomApply(Transform):
  """Apply a single transform with probability p."""

  def __init__(self, transform: Transform, p: float = 0.5):
    self.transform = transform
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() < self.p:
      return self.transform(image, mask)
    return image, mask


# ---------------------------------------------------------------------------
# Spatial transforms — operate on (C, H, W) numpy arrays
# ---------------------------------------------------------------------------

class RandomFlip(Transform):
  """Randomly flip image (and mask) along spatial axes."""

  def __init__(self, axes: Tuple[int, ...] = (1, 2), p: float = 0.5):
    self.axes = axes  # spatial axes relative to (C, H, W)
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    for ax in self.axes:
      if np.random.random() < self.p:
        image = np.flip(image, axis=ax).copy()
        if mask is not None:
          # image is (C, H, W) so axis 1=H, 2=W; mask is (H, W) so axis 0=H, 1=W
          mask_ax = ax - 1 if mask.ndim < image.ndim else ax
          if 0 <= mask_ax < mask.ndim:
            mask = np.flip(mask, axis=mask_ax).copy()
    return image, mask


class RandomRotate90(Transform):
  """Randomly rotate by 0/90/180/270 degrees in the H-W plane."""

  def __init__(self, p: float = 0.75):
    self.p = p  # probability of applying *any* rotation (k != 0)

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    k = np.random.randint(1, 4)  # 1..3 quarter-turns
    # image is (C, H, W): rotate in axes (1, 2)
    image = np.rot90(image, k=k, axes=(1, 2)).copy()
    if mask is not None:
      ndim = mask.ndim
      if ndim == 2:  # (H, W)
        mask = np.rot90(mask, k=k, axes=(0, 1)).copy()
      else:  # (C, H, W) or (H, W, ...)
        mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
    return image, mask


class RandomRotate(Transform):
  """
  Rotate image (and mask) by a random angle via bilinear resampling.

  Uses a simple affine warp implemented in numpy.  For integer masks the
  nearest-neighbour fallback is used.
  """

  def __init__(self, angle_range: Tuple[float, float] = (-30.0, 30.0), p: float = 0.5):
    self.angle_range = angle_range  # degrees
    self.p = p

  @staticmethod
  def _rotate_2d(arr2d: np.ndarray, angle_deg: float, order: int = 1) -> np.ndarray:
    """Rotate a 2-D array by angle_deg around its centre."""
    H, W = arr2d.shape
    cy, cx = H / 2.0, W / 2.0
    angle_rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    ys = np.arange(H, dtype=np.float32) - cy
    xs = np.arange(W, dtype=np.float32) - cx
    grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
    src_y = cos_a * grid_y + sin_a * grid_x + cy
    src_x = -sin_a * grid_y + cos_a * grid_x + cx
    if order == 0:
      src_y = np.clip(np.round(src_y).astype(np.int32), 0, H - 1)
      src_x = np.clip(np.round(src_x).astype(np.int32), 0, W - 1)
      return arr2d[src_y, src_x]
    # bilinear
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, H - 2)
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, W - 2)
    dy = (src_y - y0).astype(np.float32)
    dx = (src_x - x0).astype(np.float32)
    out = (arr2d[y0, x0] * (1 - dy) * (1 - dx) +
           arr2d[y0 + 1, x0] * dy * (1 - dx) +
           arr2d[y0, x0 + 1] * (1 - dy) * dx +
           arr2d[y0 + 1, x0 + 1] * dy * dx)
    return out.astype(arr2d.dtype)

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    angle = float(np.random.uniform(*self.angle_range))
    out_image = np.stack([self._rotate_2d(image[c], angle, order=1) for c in range(image.shape[0])], axis=0)
    out_mask = None
    if mask is not None:
      if mask.ndim == 2:
        out_mask = self._rotate_2d(mask, angle, order=0)
      else:
        out_mask = np.stack([self._rotate_2d(mask[c], angle, order=0) for c in range(mask.shape[0])], axis=0)
    return out_image, out_mask


class RandomScale(Transform):
  """
  Rescale image (and mask) by a random factor, then crop/pad back to original size.

  Implements zoom-in (factor > 1) and zoom-out (factor < 1) with centre-crop and
  constant padding respectively.
  """

  def __init__(self, scale_range: Tuple[float, float] = (0.8, 1.2), p: float = 0.5):
    self.scale_range = scale_range
    self.p = p

  @staticmethod
  def _rescale_channel(arr2d: np.ndarray, scale: float) -> np.ndarray:
    H, W = arr2d.shape
    nH, nW = max(1, int(round(H * scale))), max(1, int(round(W * scale)))
    # map output coords → input coords
    src_y = (np.arange(nH, dtype=np.float32) + 0.5) / scale - 0.5
    src_x = (np.arange(nW, dtype=np.float32) + 0.5) / scale - 0.5
    gy, gx = np.meshgrid(src_y, src_x, indexing='ij')
    y0 = np.clip(np.floor(gy).astype(np.int32), 0, H - 2)
    x0 = np.clip(np.floor(gx).astype(np.int32), 0, W - 2)
    dy = (gy - y0).astype(np.float32)
    dx = (gx - x0).astype(np.float32)
    rescaled = (arr2d[y0, x0] * (1 - dy) * (1 - dx) +
                arr2d[y0 + 1, x0] * dy * (1 - dx) +
                arr2d[y0, x0 + 1] * (1 - dy) * dx +
                arr2d[y0 + 1, x0 + 1] * dy * dx).astype(arr2d.dtype)
    # crop or pad to original H, W
    out = np.zeros((H, W), dtype=arr2d.dtype)
    ch = min(nH, H)
    cw = min(nW, W)
    y_start_src = (nH - ch) // 2
    x_start_src = (nW - cw) // 2
    y_start_dst = (H - ch) // 2
    x_start_dst = (W - cw) // 2
    out[y_start_dst:y_start_dst + ch, x_start_dst:x_start_dst + cw] = (
      rescaled[y_start_src:y_start_src + ch, x_start_src:x_start_src + cw]
    )
    return out

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    scale = float(np.random.uniform(*self.scale_range))
    out_image = np.stack([self._rescale_channel(image[c], scale) for c in range(image.shape[0])], axis=0)
    out_mask = None
    if mask is not None:
      if mask.ndim == 2:
        out_mask = self._rescale_channel(mask, scale)
      else:
        out_mask = np.stack([self._rescale_channel(mask[c], scale) for c in range(mask.shape[0])], axis=0)
    return out_image, out_mask


class ElasticDeformation(Transform):
  """
  Random elastic deformation using a coarse displacement field smoothed by a Gaussian blur.

  Only applied to the image (float); mask is deformed with nearest-neighbour sampling.
  """

  def __init__(self, alpha: float = 50.0, sigma: float = 6.0, p: float = 0.3):
    self.alpha = alpha  # displacement magnitude
    self.sigma = sigma  # smoothing sigma (pixels at coarse grid)
    self.p = p

  @staticmethod
  def _gaussian_blur_1d(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur along axis=-1 preserving array shape via 'full' convolution + centre crop."""
    n = arr.shape[-1]
    radius = int(math.ceil(3 * sigma))
    ks = 2 * radius + 1
    k = np.exp(-0.5 * (np.arange(ks) - radius) ** 2 / sigma ** 2).astype(np.float32)
    k /= k.sum()

    def _blur_row(row: np.ndarray) -> np.ndarray:
      full = np.convolve(row, k, mode='full')  # length = n + ks - 1
      start = radius
      return full[start:start + n]

    return np.apply_along_axis(_blur_row, axis=-1, arr=arr)

  def _smooth_field(self, H: int, W: int) -> np.ndarray:
    raw = np.random.randn(H, W).astype(np.float32)
    blurred = self._gaussian_blur_1d(raw, self.sigma)         # blur along W axis
    blurred = self._gaussian_blur_1d(blurred.T, self.sigma).T  # blur along H axis
    return blurred * self.alpha

  @staticmethod
  def _warp_channel(arr2d: np.ndarray, dy: np.ndarray, dx: np.ndarray, order: int) -> np.ndarray:
    H, W = arr2d.shape
    ys = np.clip(np.arange(H, dtype=np.float32)[:, None] + dy, 0, H - 1.001)
    xs = np.clip(np.arange(W, dtype=np.float32)[None, :] + dx, 0, W - 1.001)
    if order == 0:
      return arr2d[np.round(ys).astype(np.int32), np.round(xs).astype(np.int32)]
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    dy_frac = (ys - y0).astype(np.float32)
    dx_frac = (xs - x0).astype(np.float32)
    y1 = np.minimum(y0 + 1, H - 1)
    x1 = np.minimum(x0 + 1, W - 1)
    out = (arr2d[y0, x0] * (1 - dy_frac) * (1 - dx_frac) +
           arr2d[y1, x0] * dy_frac * (1 - dx_frac) +
           arr2d[y0, x1] * (1 - dy_frac) * dx_frac +
           arr2d[y1, x1] * dy_frac * dx_frac)
    return out.astype(arr2d.dtype)

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    C, H, W = image.shape
    dy = self._smooth_field(H, W)
    dx = self._smooth_field(H, W)
    out_image = np.stack([self._warp_channel(image[c], dy, dx, order=1) for c in range(C)], axis=0)
    out_mask = None
    if mask is not None:
      if mask.ndim == 2:
        out_mask = self._warp_channel(mask, dy, dx, order=0)
      else:
        out_mask = np.stack([self._warp_channel(mask[c], dy, dx, order=0) for c in range(mask.shape[0])], axis=0)
    return out_image, out_mask


# ---------------------------------------------------------------------------
# Intensity transforms — operate on image only (float arrays)
# ---------------------------------------------------------------------------

class GaussianNoise(Transform):
  """Add i.i.d. Gaussian noise scaled by `std`."""

  def __init__(self, std: float = 0.05, p: float = 0.5):
    self.std = std
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    noise = np.random.randn(*image.shape).astype(np.float32) * self.std
    return (image + noise).astype(image.dtype), mask


class BrightnessJitter(Transform):
  """Additive brightness jitter: add uniform noise in [-delta, +delta]."""

  def __init__(self, delta: float = 0.2, p: float = 0.5):
    self.delta = delta
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    shift = float(np.random.uniform(-self.delta, self.delta))
    return (image + shift).astype(image.dtype), mask


class ContrastJitter(Transform):
  """Multiplicative contrast jitter: multiply by factor in [1-delta, 1+delta]."""

  def __init__(self, delta: float = 0.3, p: float = 0.5):
    self.delta = delta
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    factor = float(np.random.uniform(1.0 - self.delta, 1.0 + self.delta))
    mean = image.mean()
    return ((image - mean) * factor + mean).astype(image.dtype), mask


class GammaCorrection(Transform):
  """Random gamma correction: out = in^gamma, with gamma ~ Uniform(lo, hi)."""

  def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.5), p: float = 0.3):
    self.gamma_range = gamma_range
    self.p = p

  def __call__(self, image: np.ndarray, mask: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray | None]:
    if np.random.random() > self.p:
      return image, mask
    gamma = float(np.random.uniform(*self.gamma_range))
    mn, mx = image.min(), image.max()
    if mx - mn < 1e-7:
      return image, mask
    normed = (image - mn) / (mx - mn)
    out = np.power(np.clip(normed, 0.0, 1.0), gamma) * (mx - mn) + mn
    return out.astype(image.dtype), mask


# ---------------------------------------------------------------------------
# Tensor-aware wrappers
# ---------------------------------------------------------------------------

class TensorCompose:
  """
  Wrap a Compose pipeline so it can accept tinygrad Tensors (B, C, H, W).

  Converts each sample to numpy, runs the augmentation pipeline, then
  converts back to a stacked Tensor.  Labels/masks are optional.
  """

  def __init__(self, pipeline: Compose):
    if not _TINYGRAD:
      raise ImportError("tinygrad is required for TensorCompose")
    self.pipeline = pipeline

  def __call__(
    self,
    images: "Tensor",  # (B, C, H, W)
    masks: "Tensor | None" = None,  # (B, H, W) or (B, C, H, W)
  ) -> Tuple["Tensor", "Tensor | None"]:
    imgs_np = images.numpy()  # (B, C, H, W)
    msks_np = masks.numpy() if masks is not None else None

    out_imgs = []
    out_msks = []
    B = imgs_np.shape[0]
    for i in range(B):
      img_i = imgs_np[i]  # (C, H, W)
      msk_i = msks_np[i] if msks_np is not None else None
      img_i, msk_i = self.pipeline(img_i, msk_i)
      out_imgs.append(img_i)
      if msk_i is not None:
        out_msks.append(msk_i)

    out_img_tensor = Tensor(np.stack(out_imgs, axis=0).astype(np.float32))
    out_msk_tensor = None
    if out_msks:
      out_msk_tensor = Tensor(np.stack(out_msks, axis=0))
    return out_img_tensor, out_msk_tensor


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def default_train_augmentation(p_flip: float = 0.5, p_rot90: float = 0.75, noise_std: float = 0.05) -> Compose:
  """Return a standard lightweight 2-D training augmentation pipeline."""
  return Compose([
    RandomFlip(axes=(1, 2), p=p_flip),
    RandomRotate90(p=p_rot90),
    GaussianNoise(std=noise_std, p=0.3),
    BrightnessJitter(delta=0.1, p=0.3),
    ContrastJitter(delta=0.2, p=0.3),
    GammaCorrection(gamma_range=(0.8, 1.2), p=0.2),
  ])
