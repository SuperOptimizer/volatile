# Python API

Install the package:

```sh
pip install -e py/
pip install tinygrad   # required for ML features
```

---

## volatile — Core Volume Access

The `volatile` package wraps the C core via a CPython extension (`_core.so`).

```python
import volatile

# Open a volume (local path or http/s3 URL)
vol = volatile.vol_open("/data/scroll1.zarr")
vol = volatile.vol_open("https://dl.ash2txt.org/full-scrolls/Scroll1/")
vol = volatile.vol_open("s3://my-bucket/scroll1.zarr")

# Inspect
n = volatile.vol_num_levels(vol)       # number of pyramid levels
shape = volatile.vol_shape(vol, level=0)   # (z, y, x) as a tuple of ints

# Sample a voxel (trilinear interpolation)
val = volatile.vol_sample(vol, level=0, z=1000.0, y=2000.0, x=3000.0)

# Free when done
volatile.vol_free(vol)
```

### Logging

```python
volatile.log_set_level(0)  # 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR
level = volatile.log_get_level()
```

---

## volatile.seg — Surface Editing

Pure Python surface representation with undo support.

```python
from volatile.seg import QuadSurface, SurfaceEdit

# Create a flat surface
surf = QuadSurface(rows=256, cols=256)

# Set/get vertices
surf.set(row=10, col=20, xyz=(100.0, 200.0, 300.0))
xyz = surf.get(row=10, col=20)   # (x, y, z) tuple

# Apply brush deformation (Gaussian falloff)
edit = surf.brush(row=128, col=128, radius=20.0, strength=2.0, normal=(0, 0, 1))

# Undo
edit.undo(surf)

# Compute normals
surf.compute_normals()

# Area
area = surf.area()

# Serialize to dict (JSON-serializable)
d = surf.to_dict()
surf2 = QuadSurface.from_dict(d)
```

### Growth from seed

```python
from volatile.seg import grow_surface

# Grow a surface from a seed voxel using structure-tensor guidance
surf = grow_surface(
    vol=vol,
    seed_z=1000, seed_y=2000, seed_x=3000,
    max_area=500_000,
    smoothing_iterations=5,
)
```

---

## volatile.ink — Ink Detection

```python
from volatile.ink import detect_ink

# Run ink detection on a surface
# Returns a (rows, cols) float32 array with ink probability in [0, 1]
probs = detect_ink(
    volume=vol,
    surface=surf,
    model_path="checkpoints/unet_best.safetensors",
)

# Without a model (returns zeros — useful for testing the pipeline)
probs = detect_ink(vol, surf)
```

---

## volatile.ml.model — Neural Network Models

Requires tinygrad.

### UNet

```python
from volatile.ml.model import UNet

model = UNet(
    in_channels=1,
    out_channels=1,
    base_channels=32,    # doubles each level
    num_levels=4,        # encoder/decoder levels
)

# Forward pass
from tinygrad import Tensor
x = Tensor.zeros(1, 1, 256, 256)
y = model(x)   # (1, 1, 256, 256)
```

### nnUNet

```python
from volatile.ml.nnunet import nnUNet

model = nnUNet(
    in_channels=1,
    out_channels=2,
    patch_size=(64, 256, 256),   # 3D
)
```

---

## volatile.ml.train — Training

```python
from volatile.ml.train import Trainer, save_checkpoint, load_checkpoint
from volatile.ml.model import UNet

model = UNet(in_channels=1, out_channels=1)
trainer = Trainer(
    model=model,
    lr=1e-4,
    weight_decay=1e-5,
    epochs=100,
    checkpoint_dir="checkpoints/",
)

# Training loop with dataset
from volatile.ml.data import ScrollDataset
dataset = ScrollDataset(volume_path="/data/scroll1.zarr", surface_path="surface.obj")
trainer.fit(dataset)

# Manual checkpoint
save_checkpoint(model, trainer.optimizer, epoch=10, path="checkpoints/epoch10.safetensors")
model, epoch = load_checkpoint(model, trainer.optimizer, "checkpoints/epoch10.safetensors")

# LR schedule
from volatile.ml.train import cosine_annealing_lr
lr = cosine_annealing_lr(initial_lr=1e-4, epoch=50, total_epochs=100)
```

---

## volatile.ml.infer — Tiled Inference

```python
from volatile.ml.infer import tiled_infer
import numpy as np

# volume_np: (H, W), (C, H, W), (Z, H, W), or (Z, C, H, W)
volume_np = np.random.rand(256, 256).astype(np.float32)

result = tiled_infer(
    model=model,
    volume_np=volume_np,
    tile_h=256,
    tile_w=256,
    overlap=64,
    batch_size=4,
)
# result shape matches input spatial dims, with model output channels
```

---

## volatile.imgproc — Image Processing

Pure Python image ops (no OpenCV dependency).

```python
from volatile.imgproc import (
    gaussian_blur,
    structure_tensor,
    window_level,
    normalize,
)
import numpy as np

data = np.random.rand(64, 64).astype(np.float32)

# Gaussian blur (2D or 3D)
blurred = gaussian_blur(data, sigma=2.0)

# Structure tensor (returns Jxx, Jxy, Jyy components)
tensor = structure_tensor(data, sigma=1.5, rho=3.0)

# Window/level mapping to uint8
display = window_level(data, window=30000.0, level=15000.0)

# Normalize to [0, 1]
norm = normalize(data)
```

---

## volatile.ml.data — Datasets

```python
from volatile.ml.data import ScrollDataset, PatchSampler

# Dataset backed by a zarr volume + surface
dataset = ScrollDataset(
    volume_path="/data/scroll1.zarr",
    surface_path="surface.obj",
    patch_size=(64, 64),
    patches_per_surface=1000,
    augment=True,
)

# len + indexing
print(len(dataset))
patch, label = dataset[0]   # (C, H, W) tensors

# Sampler for balanced positive/negative patches
sampler = PatchSampler(dataset, positive_fraction=0.5)
```

---

## volatile.ml.augment — Data Augmentation

```python
from volatile.ml.augment import augment_patch

import numpy as np
patch = np.random.rand(1, 64, 64).astype(np.float32)
label = np.zeros((1, 64, 64), dtype=np.float32)

aug_patch, aug_label = augment_patch(
    patch, label,
    flip=True,
    rotate=True,
    brightness_jitter=0.1,
    noise_std=0.02,
)
```

---

## volatile.fit — Surface Fitting

Port of the exps_2d_model surface fitting utilities.

```python
from volatile.fit import fit_surface_to_points

# Fit a quad surface to a cloud of 3D points
points = np.random.rand(10000, 3).astype(np.float32)
surf = fit_surface_to_points(points, rows=256, cols=256, iterations=50)
```

---

## volatile.zarr_tasks — Async Zarr Operations

```python
from volatile.zarr_tasks import rechunk_async, mirror_async
import asyncio

async def main():
    await rechunk_async(
        input_path="/data/scroll1.zarr",
        output_path="/data/scroll1_128.zarr",
        chunk_shape=(128, 128, 128),
    )

asyncio.run(main())
```

---

## Error Handling

Most functions raise `RuntimeError` on failure (propagated from C via the extension).
`vol_open` raises `FileNotFoundError` for missing local paths and `ConnectionError` for unreachable remote URLs.

```python
try:
    vol = volatile.vol_open("/nonexistent.zarr")
except FileNotFoundError as e:
    print(f"Volume not found: {e}")
```

When tinygrad is not installed, ML functions raise `ImportError` with a clear message.
Volume access and CLI tools do not require tinygrad.
