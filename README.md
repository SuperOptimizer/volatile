# volatile

A high-performance volume visualization and segmentation platform for the Vesuvius Challenge.
Complete rewrite of [villa](https://github.com/ScrollPrize/villa) in pure C23 + Python.

## Features

- OME-Zarr v2/v3 + sharding: local, HTTP, S3
- compress4d: progressive multiscale codec (rANS + Lanczos residuals)
- GPU-accelerated rendering via Vulkan 1.3 compute (Metal/DX12 backends)
- Full segmentation toolkit: brush, line, push-pull, surface growth
- Multi-user server: binary TCP, SQLite segment DB, git-backed storage
- 25+ CLI tools for format conversion, stats, masking, winding numbers, video
- Python ML stack (tinygrad): UNet, nnUNet, training, tiled inference

Zero heavyweight dependencies. No Qt, OpenCV, Eigen, CGAL, PyTorch, or numpy in the C core.

## Build

**Requirements:** clang 16+ or gcc 13+, CMake 3.25+, Ninja, libcurl, blosc2, SQLite3, Vulkan SDK.

```sh
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang
cmake --build build -j4

# Run tests
ctest --test-dir build --timeout 10
```

**Optional:** sanitizers, coverage, fuzzing:

```sh
cmake -B build -G Ninja -DVOLATILE_SANITIZE=asan
cmake -B build -G Ninja -DVOLATILE_COVERAGE=ON
cmake -B build -G Ninja -DVOLATILE_FUZZ=ON
```

**Python package:**

```sh
pip install -e py/
```

Requires tinygrad for ML features (`pip install tinygrad`).

## Quick Start

```sh
# Inspect a volume
volatile info /data/scroll1.zarr

# Sample a voxel
volatile sample /data/scroll1.zarr 1000 2000 3000 --level 0

# Convert TIFF stack to Zarr
volatile convert scan.tif scroll.zarr

# Compute statistics
volatile stats /data/scroll1.zarr

# Mirror a remote volume locally
volatile mirror https://dl.ash2txt.org/full-scrolls/Scroll1/ --cache-dir ./scroll1_cache

# Grow a surface from a seed
volatile grow /data/scroll1.zarr --seed 1000,2000,3000 --output surface.obj

# Render surface to TIFF image
volatile render surface.obj --volume /data/scroll1.zarr --output ink.tiff

# Start multi-user server
volatile serve --port 8765 --data /data --db segments.db
```

## CLI

See [docs/cli.md](docs/cli.md) for the full reference.

| Command      | Description |
|---|---|
| `info`       | Show volume metadata (shape, chunks, dtype, codec) |
| `sample`     | Sample a single voxel value |
| `convert`    | Convert between zarr/tiff/nrrd |
| `rechunk`    | Re-chunk a zarr volume |
| `stats`      | Min/max/mean/std/percentiles |
| `compress`   | Re-compress with compress4d |
| `mirror`     | Cache a remote volume locally |
| `downsample` | 2x mean downsampling |
| `threshold`  | Binarize or clip voxel values |
| `merge`      | Element-wise combine two volumes |
| `extract`    | Crop a sub-region bbox |
| `flatten`    | UV-flatten a quad surface (LSCM) |
| `grow`       | Grow a surface from a seed point |
| `render`     | Render surface pixels to TIFF |
| `metrics`    | Surface area, curvature, smoothness |
| `mask`       | Binary mask from surface |
| `normals`    | Per-voxel normals via structure tensor |
| `winding`    | Winding number field |
| `inpaint`    | Telea hole-filling inpainting |
| `diff`       | Per-vertex distance stats between surfaces |
| `transform`  | Apply 4x4 matrix to a surface |
| `video`      | Render slice-sweep video |
| `serve`      | Start the multi-user server |
| `connect`    | Test a server connection |

## Python API

See [docs/python.md](docs/python.md) for the full reference.

```python
import volatile

# Volume access (via C extension)
vol = volatile.vol_open("/data/scroll1.zarr")
print(volatile.vol_shape(vol, level=0))   # (z, y, x)
val = volatile.vol_sample(vol, level=0, z=1000, y=2000, x=3000)
volatile.vol_free(vol)

# ML inference
from volatile.ml.model import UNet
from volatile.ml.infer import tiled_infer
import numpy as np

model = UNet(in_channels=1, out_channels=1)
# load checkpoint...
result = tiled_infer(model, volume_np, tile_h=256, tile_w=256, overlap=64)
```

## Architecture

```
volatile/
  src/
    core/       C23 — vol, chunk, cache, compress4d, math, geom, imgproc,
                      io, net, hash, json, thread, log, sparse, profile
    gpu/        Vulkan 1.3 compute + Metal/DX12 backends, abstract gpu.h
    render/     tile renderer, compositing, colormaps, overlays, camera
    gui/        SDL3 + Nuklear — viewer, seg tools, annotations, settings
    server/     binary TCP server, protocol, SQLite DB, git storage
    cli/        25+ subcommands
  py/
    volatile/   CPython C extension + pure Python ML layer (tinygrad)
  test/
    c/          greatest.h unit tests (one file per module)
    py/         pytest
```

See [docs/api.md](docs/api.md) for the C API reference.

## License

TBD
