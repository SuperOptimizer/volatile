# CLI Reference

All commands follow the pattern `volatile <command> [args]`.

---

## Volume Inspection

### `info`

Show volume metadata: shape, chunk shape, dtype, compressor.

```sh
volatile info /data/scroll1.zarr
volatile info s3://my-bucket/scroll1.zarr
```

Output:
```
path:    /data/scroll1.zarr
source:  local
levels:  5

level 0:
  ndim:   3
  shape:  [7888, 8096, 7457]
  chunks: [128, 128, 128]
  dtype:  u16
  codec:  blosc/zstd (level 5, shuffle 1)
```

### `sample`

Sample a single voxel value at floating-point coordinates (trilinear interpolation).

```sh
volatile sample /data/scroll1.zarr 1000 2000 3000
volatile sample /data/scroll1.zarr 1000 2000 3000 --level 2
```

---

## Format Conversion

### `convert`

Convert between zarr, tiff, and nrrd formats. Auto-detects input format.

```sh
volatile convert scan.tif scroll.zarr
volatile convert scroll.zarr output.nrrd
volatile convert input.zarr output.zarr --format zarr
```

### `rechunk`

Re-chunk an existing zarr volume with a new chunk shape.

```sh
volatile rechunk /data/scroll1.zarr --output /data/scroll1_128.zarr --chunk-size 128,128,128
```

### `compress`

Re-compress a zarr volume using compress4d (rANS + Lanczos progressive codec).

```sh
volatile compress /data/scroll1.zarr
volatile compress /data/scroll1.zarr --output /data/scroll1_c4d.zarr
```

---

## Statistics

### `stats`

Compute voxel statistics: min, max, mean, std, and percentiles (p1/p5/p25/p50/p75/p95/p99).

```sh
volatile stats /data/scroll1.zarr
volatile stats /data/scroll1.zarr --level 2
```

Output:
```
min:    0.000
max:    65535.000
mean:   12847.324
std:    8921.445
p01:    0.000
p05:    102.000
p25:    5842.000
p50:    11024.000
p75:    19331.000
p95:    31220.000
p99:    42100.000
```

---

## Zarr Operations

### `downsample`

2x mean downsampling along all spatial axes.

```sh
volatile downsample /data/scroll1.zarr --output /data/scroll1_2x.zarr
volatile downsample /data/scroll1.zarr --output /data/scroll1_4x.zarr --factor 4
```

### `threshold`

Binarize or clip voxel values.

```sh
# binarize: values >= 128 become 1, below become 0
volatile threshold /data/mask.zarr --output /data/binary.zarr --low 128

# clip to range [100, 60000]
volatile threshold /data/scroll1.zarr --output /data/clipped.zarr --low 100 --high 60000
```

### `merge`

Element-wise operation between two volumes. Both must have the same shape.

```sh
volatile merge a.zarr b.zarr --output merged.zarr --op max
volatile merge base.zarr overlay.zarr --output combined.zarr --op add
volatile merge volume.zarr mask.zarr --output masked.zarr --op mask
```

Ops: `max`, `add`, `mask` (zero where mask==0).

### `extract`

Crop a sub-region by bounding box.

```sh
volatile extract /data/scroll1.zarr \
  --bbox 0,0,0,500,500,500 \
  --output /data/crop.zarr
```

Bbox format: `z0,y0,x0,z1,y1,x1` (exclusive end).

---

## Remote / Mirroring

### `mirror`

Mirror a remote OME-Zarr volume to local disk with optional rechunking and recompression.

```sh
# basic mirror
volatile mirror https://dl.ash2txt.org/full-scrolls/Scroll1/ --cache-dir ./scroll1

# mirror with rechunking
volatile mirror https://example.com/scroll.zarr \
  --cache-dir ./scroll \
  --rechunk 128,128,128

# mirror + recompress with compress4d
volatile mirror https://example.com/scroll.zarr \
  --cache-dir ./scroll \
  --compress4d

# mirror from S3
volatile mirror s3://my-bucket/scroll1.zarr --cache-dir ./scroll1
```

---

## Surface Tools

### `flatten`

UV-flatten a quad surface using LSCM (Least Squares Conformal Maps).

```sh
volatile flatten surface.obj --output surface_uv.obj
```

### `grow`

Grow a surface from a seed point using structure-tensor guided expansion.

```sh
volatile grow /data/scroll1.zarr \
  --seed 1000,2000,3000 \
  --output surface.obj

volatile grow /data/scroll1.zarr \
  --seed 1000,2000,3000 \
  --output surface.obj \
  --max-area 500000 \
  --smoothing 5
```

### `render`

Render surface pixel values to a TIFF image by sampling the volume.

```sh
volatile render surface.obj \
  --volume /data/scroll1.zarr \
  --output ink.tiff

volatile render surface.obj \
  --volume /data/scroll1.zarr \
  --output ink.tiff \
  --level 0 \
  --radius 32
```

### `metrics`

Compute surface quality metrics: total area, mean/max curvature, smoothness score.

```sh
volatile metrics surface.obj
```

### `mask`

Generate a binary voxel mask from a surface (voxelized surface footprint).

```sh
volatile mask surface.obj \
  --volume /data/scroll1.zarr \
  --output mask.zarr
```

### `diff`

Compute per-vertex distance statistics between two surfaces (e.g., before vs after smoothing).

```sh
volatile diff surface_before.obj surface_after.obj
volatile diff surface_before.obj surface_after.obj --output diff.tiff
```

### `transform`

Apply a 4x4 affine matrix (JSON file) to all surface vertices.

```sh
volatile transform surface.obj \
  --matrix transform.json \
  --output surface_transformed.obj
```

Matrix JSON format:
```json
[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
```

---

## Volume Analysis

### `normals`

Compute per-voxel surface normals via 3D structure tensor eigenvectors.

```sh
volatile normals /data/scroll1.zarr --output normals.zarr
```

Output is a 4-channel zarr (nx, ny, nz, confidence).

### `winding`

Compute the winding number field of a closed mesh relative to a volume grid.
Useful for inside/outside tests and ink mask generation.

```sh
volatile winding surface.obj \
  --output winding.zarr \
  --shape 7888,8096,7457
```

### `inpaint`

Fill holes in a 2D image using Telea inpainting.

```sh
volatile inpaint ink.tiff \
  --mask hole_mask.tiff \
  --output inpainted.tiff
```

---

## Video

### `video`

Render a slice-sweep video through a volume along a surface.

```sh
volatile video surface.obj \
  --volume /data/scroll1.zarr \
  --output fly_through.mp4

volatile video surface.obj \
  --volume /data/scroll1.zarr \
  --output fly_through.mp4 \
  --fps 30 \
  --depth-range -20,20
```

---

## Server

### `serve`

Start the multi-user volatile server (binary TCP).

```sh
volatile serve
volatile serve --port 8765 --data /data --db segments.db
volatile serve --port 8765 --data /data --db segments.db --workers 8
```

Default port: 8765.

### `connect`

Test a connection to a running volatile server.

```sh
volatile connect localhost:8765
volatile connect myserver:8765 --volume scroll1
```

---

## Misc

### `version`

```sh
volatile version
# volatile 0.1.0
```

### `help`

```sh
volatile help
volatile <command> --help
```
