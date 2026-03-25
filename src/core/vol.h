#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "chunk.h"
#include "io.h"

typedef enum { VOL_LOCAL, VOL_REMOTE } vol_source_t;

typedef struct {
  int ndim;
  int64_t shape[8];          // full volume shape at this level
  int64_t chunk_shape[8];    // chunk dimensions
  int dtype;                 // DTYPE_U8, DTYPE_U16, etc (from io.h enum)
  char compressor_id[32];    // "blosc", "zstd", etc
  char compressor_cname[32]; // sub-codec: "zstd", "lz4", etc
  int compressor_clevel;     // compression level
  int compressor_shuffle;    // 0=none, 1=byte, 2=bit
  char order;                // 'C' or 'F'
  char chunk_sep;            // chunk key separator: '.' (default v2) or '/' (dimension_separator="/")
  // Zarr v3 extensions
  int zarr_version;          // 2 or 3
  bool sharded;              // true when outermost codec is sharding_indexed
  int64_t shard_shape[8];    // shard dimensions (outer chunk grid); 0 if not sharded
} zarr_level_meta;

typedef struct volume volume;

// open a volume from a local path or remote URL
// path can be: /path/to/volume.zarr or https://server/volume.zarr
volume *vol_open(const char *path);
void vol_free(volume *vol);

// metadata
int vol_num_levels(const volume *v);
const zarr_level_meta *vol_level_meta(const volume *v, int level);
void vol_shape(const volume *v, int level, int64_t *shape_out);  // writes ndim values

// chunk access (raw, decompressed)
// returns a newly allocated buffer that caller must free
// returns NULL if chunk doesn't exist
uint8_t *vol_read_chunk(const volume *v, int level, const int64_t *chunk_coords, size_t *out_size);

// shard access: read all chunks packed in one shard file (Zarr v3 sharding_indexed).
// shard_coords are the coordinates of the shard in the outer chunk grid.
// chunks_out receives a malloc'd array of (ndim * nchunks_per_shard) int64_t coords followed
// by nchunks_per_shard data pointers — caller must free each data pointer, then chunks_out.
// Returns number of chunks successfully extracted, or -1 on error.
// For simpler use, vol_read_chunk transparently handles sharded volumes.
int vol_read_shard(const volume *v, int level, const int64_t *shard_coords,
                   uint8_t ***chunks_out, size_t **chunk_sizes_out, size_t *n_chunks_out);

// sampling (returns interpolated value at fractional coordinates)
float vol_sample(const volume *v, int level, float z, float y, float x);

// info
const char *vol_path(const volume *v);
bool vol_is_remote(const volume *v);
vol_source_t vol_source(const volume *v);

// ---------------------------------------------------------------------------
// Write API
// ---------------------------------------------------------------------------

typedef struct {
  int zarr_version;        // 2 or 3
  int ndim;
  int64_t shape[8];
  int64_t chunk_shape[8];
  dtype_t dtype;
  const char *compressor;  // "blosc", "zstd", or "" / NULL for none
  int clevel;              // compression level (1-9)
  bool sharded;            // v3 sharding_indexed (only meaningful when zarr_version==3)
  int64_t shard_shape[8];  // outer shard dimensions (only used when sharded==true)
} vol_create_params;

// Create a new zarr volume directory, write metadata, return writable volume*.
// Returns NULL on error (e.g. path already exists or bad params).
volume *vol_create(const char *path, vol_create_params params);

// Compress data and write it as a chunk file (v2: z.y.x  v3: c/z/y/x).
// data/size are the raw (uncompressed) bytes for this chunk.
bool vol_write_chunk(volume *v, int level, const int64_t *chunk_coords,
                     const void *data, size_t size);

// Pack num_chunks raw data buffers into one shard file with a binary index.
// chunk_data[i] / chunk_sizes[i]: raw bytes for inner chunk i (NULL = absent).
// Chunks are compressed before packing. Index written at end of shard file.
bool vol_write_shard(volume *v, int level, const int64_t *shard_coords,
                     const void **chunk_data, const size_t *chunk_sizes, int num_chunks);

// Build a downsampled pyramid: for levels 1..max_levels, downsample level i-1
// by 2x in each dimension using mean pooling. Stops when any dimension < 2.
bool vol_build_pyramid(volume *v, int max_levels);

// Write OME-Zarr .zattrs with multiscales coordinate transforms.
bool vol_finalize(volume *v);
