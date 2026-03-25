# C API Reference

Include the relevant header; link against `libvolatile_core`, `libvolatile_render`, `libvolatile_gpu`, or `libvolatile_server` as needed.

---

## core/vol.h — Volume I/O

```c
volume *vol_open(const char *path);           // local path or http/s3 URL
void    vol_free(volume *v);
int     vol_num_levels(const volume *v);
const zarr_level_meta *vol_level_meta(const volume *v, int level);
void    vol_shape(const volume *v, int level, int64_t *shape_out);
float   vol_sample(const volume *v, int level, float z, float y, float x);
uint8_t *vol_read_chunk(const volume *v, int level, const int64_t *chunk_coords, size_t *out_size);

// Writing
volume *vol_create(const char *path, vol_create_params params);
bool    vol_write_chunk(volume *v, int level, const int64_t *chunk_coords,
                        const void *data, size_t len);
bool    vol_build_pyramid(volume *v, int max_levels);
bool    vol_finalize(volume *v);
```

`vol_sample` performs trilinear interpolation at fractional voxel coordinates. Returns 0 for out-of-bounds.

## core/vol_mirror.h — Remote Mirroring

```c
vol_mirror *vol_mirror_new(mirror_config cfg);
void        vol_mirror_free(vol_mirror *m);
volume     *vol_mirror_volume(vol_mirror *m);         // use like a normal volume
bool        vol_mirror_cache_level(vol_mirror *m, int level);
bool        vol_mirror_rechunk(vol_mirror *m, const int64_t *new_chunk_shape);
bool        vol_mirror_recompress(vol_mirror *m);
float       vol_mirror_cache_hit_rate(const vol_mirror *m);
```

## core/chunk.h — Chunked N-D Arrays

```c
chunked_array *chunked_array_new(int ndim, const int64_t *shape,
                                 const int64_t *chunk_shape, size_t elem_size);
void  chunked_array_free(chunked_array *a);
void *chunked_array_get_chunk(const chunked_array *a, const int64_t *chunk_coords);
void  chunked_array_set_chunk(chunked_array *a, const int64_t *chunk_coords, void *data);
float chunked_array_get_f32(const chunked_array *a, const int64_t *indices);
void  chunked_array_set_f32(chunked_array *a, const int64_t *indices, float val);
```

Chunks are owned by the array after `set_chunk`. Indices are element-level (not chunk-level).

## core/cache.h — Tiered Chunk Cache (CLOCK-Pro)

```c
chunk_cache *cache_new(cache_config cfg);
void         cache_free(chunk_cache *c);
chunk_data  *cache_get(chunk_cache *c, chunk_key key);
chunk_data  *cache_get_blocking(chunk_cache *c, chunk_key key, int timeout_ms);
void         cache_put(chunk_cache *c, chunk_key key, chunk_data *data);
void         cache_prefetch(chunk_cache *c, chunk_key key);
void         cache_pin(chunk_cache *c, chunk_key key, chunk_data *data);
size_t       cache_hits(const chunk_cache *c);
size_t       cache_misses(const chunk_cache *c);
```

## core/compress4d.h — Progressive Codec

```c
// ANS entropy coder
ans_table *ans_table_build(const uint8_t *data, size_t len);
uint8_t   *ans_encode(const ans_table *t, const uint8_t *src, size_t len, size_t *out_len);
uint8_t   *ans_decode(const ans_table *t, const uint8_t *src, size_t src_len, size_t orig_len);
void       ans_table_free(ans_table *t);

// Residual codec (Lanczos upsample + ANS encode delta)
uint8_t *compress4d_encode_residual(const float *residual, size_t len, size_t *out_len);
bool     compress4d_decode_residual(const uint8_t *src, size_t src_len,
                                    float *out, size_t out_len);

// Lanczos 3x upsampling
void lanczos3_upsample3d(const float *src, size_t dx, size_t dy, size_t dz,
                         float *dst, size_t odx, size_t ody, size_t odz);
```

## core/math.h — Linear Algebra

All vec ops are `static inline`. mat4f/mat3f are column-major.

```c
// Types
typedef struct { float x, y; }       vec2f;
typedef struct { float x, y, z; }    vec3f;
typedef struct { float x, y, z, w; } vec4f;
typedef struct { float m[9]; }        mat3f;   // column-major
typedef struct { float m[16]; }       mat4f;   // column-major
typedef struct { float x, y, z, w; } quatf;   // w = scalar part

// vec3f ops (same pattern for vec2f, vec4f)
vec3f vec3f_add(vec3f a, vec3f b);
vec3f vec3f_sub(vec3f a, vec3f b);
vec3f vec3f_scale(vec3f v, float s);
float vec3f_dot(vec3f a, vec3f b);
float vec3f_len(vec3f v);
vec3f vec3f_normalize(vec3f v);
vec3f vec3f_cross(vec3f a, vec3f b);
vec3f vec3f_lerp(vec3f a, vec3f b, float t);

// mat4f
mat4f mat4f_identity(void);
mat4f mat4f_mul(mat4f a, mat4f b);
mat4f mat4f_inverse(mat4f m);
mat4f mat4f_transpose(mat4f m);
mat4f mat4f_translate(float tx, float ty, float tz);
mat4f mat4f_rotate(vec3f axis, float angle_rad);
mat4f mat4f_scale(float sx, float sy, float sz);
mat4f mat4f_perspective(float fovy_rad, float aspect, float znear, float zfar);
mat4f mat4f_lookat(vec3f eye, vec3f center, vec3f up);

// quatf
quatf quatf_from_axis_angle(vec3f axis, float angle_rad);
quatf quatf_mul(quatf a, quatf b);
quatf quatf_normalize(quatf q);
quatf quatf_slerp(quatf a, quatf b, float t);
vec3f quatf_rotate(quatf q, vec3f v);
mat4f quatf_to_mat4(quatf q);

// Volume interpolation
float trilinear_interp(const float *grid, int sx, int sy, int sz,
                       float x, float y, float z);
float lanczos3_weight(float x);
```

## core/geom.h — Geometry

```c
// Quad surface (rows x cols grid of 3D points)
quad_surface *quad_surface_new(int rows, int cols);
void          quad_surface_free(quad_surface *s);
vec3f         quad_surface_get(const quad_surface *s, int row, int col);
void          quad_surface_set(quad_surface *s, int row, int col, vec3f point);
float         quad_surface_area(const quad_surface *s);
void          quad_surface_compute_normals(quad_surface *s);

// Plane surface
plane_surface plane_surface_from_normal(vec3f origin, vec3f normal);
vec3f         plane_surface_sample(const plane_surface *p, float u, float v);
float         plane_surface_dist(const plane_surface *p, vec3f point);

// Triangle mesh
tri_mesh *tri_mesh_new(int num_verts, int num_faces);
void      tri_mesh_free(tri_mesh *m);
uint8_t  *mesh_voxelize(const tri_mesh *m, int d, int h, int w);
tri_mesh *mesh_simplify(const tri_mesh *m, int target_faces);
void      mesh_smooth(tri_mesh *m, int iterations, float lambda);

// HAMT (persistent hash array mapped trie, for undo)
hamt_node *hamt_empty(void);
hamt_node *hamt_set(hamt_node *root, uint64_t key, void *val);  // returns new root
void      *hamt_get(const hamt_node *root, uint64_t key);
hamt_node *hamt_del(hamt_node *root, uint64_t key);
void       hamt_release(hamt_node *n);
```

## core/imgproc.h — Image Processing

```c
// Gaussian blur
void gaussian_blur_3d(const float *data, float *out, int d, int h, int w, float sigma);
void gaussian_blur_2d(const float *data, float *out, int h, int w, float sigma);

// Structure tensor
void structure_tensor_3d(const float *data, float *out, int d, int h, int w, float sigma);

// Euclidean distance transform
void edt_3d(const uint8_t *mask, float *dist, int d, int h, int w);

// Window/level mapping (float -> uint8)
void window_level(const float *in, uint8_t *out, int n, float window, float level);

// Advanced filters
void frangi_3d(const float *data, float *out, int d, int h, int w,
               float sigma_min, float sigma_max, int num_scales);
void thinning_3d(const uint8_t *mask, uint8_t *skeleton, int d, int h, int w);
void ced_2d(const float *data, float *out, int h, int w, int iterations, float dt);

// Connected components (6-connectivity, returns number of labels)
int  connected_components_3d(const uint8_t *mask, int *labels,
                              int depth, int height, int width);

// Dijkstra shortest path on cost field
void dijkstra_3d(const float *cost, int start_idx, float *dist,
                 int depth, int height, int width);

// Histogram / statistics
histogram *histogram_new(const float *data, size_t n, int num_bins);
float      histogram_percentile(const histogram *h, float p);  // p in [0, 1]
void       histogram_free(histogram *h);
```

## core/io.h — File I/O

```c
// TIFF
image *tiff_read(const char *path);
bool   tiff_write(const char *path, const image *img);

// NRRD
nrrd_data *nrrd_read(const char *path);
void       nrrd_free(nrrd_data *n);

// PPM/PGM
image *ppm_read(const char *path);
bool   ppm_write(const char *path, const image *img);
bool   pgm_write(const char *path, const uint8_t *data, int width, int height);

// OBJ mesh
obj_mesh *obj_read(const char *path);
void      obj_free(obj_mesh *m);

void image_free(image *img);
```

## core/net.h — HTTP / S3

```c
void http_init(void);
void http_cleanup(void);

http_response *http_get(const char *url, int timeout_ms);
http_response *http_get_range(const char *url, int64_t offset, int64_t len, int timeout_ms);
http_response *http_head(const char *url, int timeout_ms);
void           http_response_free(http_response *r);

// S3
s3_credentials *s3_creds_from_env(void);
s3_credentials *s3_creds_from_file(const char *profile);
http_response  *s3_get_object(const s3_credentials *creds, const char *bucket,
                               const char *key, int timeout_ms);
http_response  *s3_get_object_range(const s3_credentials *creds, const char *bucket,
                                     const char *key, int64_t offset, int64_t len, int timeout_ms);

// Connection pool
http_pool *http_pool_new(int max_connections);
void       http_pool_free(http_pool *p);
http_response *http_pool_get(http_pool *p, const char *url, int timeout_ms);
```

## core/thread.h — Thread Pool

```c
threadpool *threadpool_new(int num_threads);   // 0 = auto (num_cores/2, min 2)
void        threadpool_free(threadpool *p);    // drains then destroys

// Submit with result tracking
future *threadpool_submit(threadpool *p, task_fn fn, void *arg);
void   *future_get(future *f, int timeout_ms);  // NULL on timeout
bool    future_done(const future *f);
void    future_free(future *f);

// Fire-and-forget
void   threadpool_fire(threadpool *p, task_fn fn, void *arg);
void   threadpool_drain(threadpool *p, int timeout_ms);
size_t threadpool_pending(const threadpool *p);
```

## core/hash.h — Hash Maps

```c
// String-keyed map
hash_map *hash_map_new(void);
void      hash_map_free(hash_map *m);
void     *hash_map_get(hash_map *m, const char *key);
bool      hash_map_put(hash_map *m, const char *key, void *val);
bool      hash_map_del(hash_map *m, const char *key);
size_t    hash_map_len(hash_map *m);

hash_map_iter *hash_map_iter_new(hash_map *m);
bool           hash_map_iter_next(hash_map_iter *it, hash_map_entry *out);
void           hash_map_iter_free(hash_map_iter *it);

// uint64-keyed map
hash_map_int *hash_map_int_new(void);
void         *hash_map_int_get(hash_map_int *m, uint64_t key);
bool          hash_map_int_put(hash_map_int *m, uint64_t key, void *val);
bool          hash_map_int_del(hash_map_int *m, uint64_t key);
```

## core/json.h — JSON Parser

```c
json_value *json_parse(const char *str);
void        json_free(json_value *v);

json_type       json_typeof(const json_value *v);
bool            json_get_bool(const json_value *v, bool def);
double          json_get_number(const json_value *v, double def);
int64_t         json_get_int(const json_value *v, int64_t def);
const char     *json_get_str(const json_value *v);
size_t          json_array_len(const json_value *v);
const json_value *json_array_get(const json_value *v, size_t idx);
const json_value *json_object_get(const json_value *v, const char *key);
void json_object_iter(const json_value *v, json_object_iter_fn fn, void *ctx);
```

## core/sparse.h — Sparse Linear Algebra

```c
sparse_mat *sparse_new(int rows, int cols, int nnz_hint);
void        sparse_free(sparse_mat *m);
void        sparse_add(sparse_mat *m, int row, int col, float val);  // accumulates duplicates

// Conjugate gradient solver. Returns iteration count, or -1 if not converged.
int sparse_solve_cg(const sparse_mat *A, const float *b, float *x,
                    int max_iter, float tol);
```

## core/log.h — Logging

```c
void log_set_level(log_level_t level);  // LOG_DEBUG/INFO/WARN/ERROR
void log_set_file(FILE *f);
void log_set_callback(log_callback_fn fn, void *ctx);

// Macros (preferred)
LOG_DEBUG("format %d", val);
LOG_INFO("format %d", val);
LOG_WARN("format %d", val);
LOG_ERROR("format %d", val);
ASSERT(condition, "message");
```

## core/profile.h — Profiling

```c
void profile_init(void);
void profile_shutdown(void);
void profile_begin(const char *name);
void profile_end(void);
void profile_counter_inc(const char *name);
void profile_counter_add(const char *name, int64_t delta);
void profile_frame_begin(void);
void profile_frame_end(void);
bool profile_export_json(const char *path);   // Chrome trace format
void profile_print_summary(FILE *out);
```

---

## render/tile.h — Tiled Async Renderer

```c
#define TILE_PX 256  // pixels per tile side

typedef struct { int col, row; int pyramid_level; uint64_t epoch; } tile_key;
typedef struct { tile_key key; uint8_t *pixels; bool valid; } tile_result;

tile_renderer *tile_renderer_new(int num_threads);
void           tile_renderer_free(tile_renderer *r);

void tile_renderer_submit(tile_renderer *r, tile_key key);
int  tile_renderer_drain(tile_renderer *r, tile_result *out, int max_results);
void tile_renderer_cancel_stale(tile_renderer *r, uint64_t min_epoch);
int  tile_renderer_pending(const tile_renderer *r);
```

`tile_result.pixels` is a 256x256 RGBA buffer (heap-allocated). Caller owns it after drain.

## render/cmap.h — Colormaps

```c
typedef enum { CMAP_GRAYSCALE, CMAP_VIRIDIS, CMAP_MAGMA, CMAP_PLASMA,
               CMAP_INFERNO, CMAP_HOT, CMAP_COOL, CMAP_BONE,
               CMAP_JET, CMAP_TURBO } cmap_id;

cmap_rgb    cmap_apply(cmap_id id, double value);             // value in [0, 1]
void        cmap_apply_buf(cmap_id id, const double *values,
                           cmap_rgb *out, size_t n);
const char *cmap_name(cmap_id id);
int         cmap_count(void);
```

## render/composite.h — Compositing

```c
typedef enum { COMP_MIP, COMP_ALPHA, COMP_BEER_LAMBERT,
               COMP_AVERAGE, COMP_MIN, COMP_CUSTOM } composite_mode;

void  composite_params_default(composite_params *p);
float composite_pixel(const float *values, int count, const composite_params *p);
void  composite_slices(const float **slices, int num_slices,
                       int width, int height, const composite_params *p,
                       float *out);
```

## render/overlay.h — Overlay Rendering

```c
overlay_list *overlay_list_new(void);
void          overlay_list_free(overlay_list *l);
void          overlay_list_clear(overlay_list *l);

void overlay_add_point(overlay_list *l, float x, float y,
                       uint8_t r, uint8_t g, uint8_t b, float radius);
void overlay_add_line(overlay_list *l, float x0, float y0,
                      float x1, float y1, uint8_t r, uint8_t g, uint8_t b, float thickness);
void overlay_add_circle(overlay_list *l, float cx, float cy, float radius,
                        uint8_t r, uint8_t g, uint8_t b);
void overlay_add_text(overlay_list *l, float x, float y, const char *text,
                      uint8_t r, uint8_t g, uint8_t b);
void overlay_render(const overlay_list *l, uint8_t *pixels, int width, int height);
```

---

## gpu/gpu.h — Abstract GPU Interface

```c
typedef enum { GPU_BACKEND_VULKAN, GPU_BACKEND_METAL, GPU_BACKEND_DX12 } gpu_backend_t;

gpu_device *gpu_init(gpu_backend_t preferred);  // falls back automatically
void        gpu_shutdown(gpu_device *dev);
const char *gpu_device_name(const gpu_device *dev);

gpu_buffer  *gpu_buffer_create(gpu_device *dev, size_t size, bool host_visible);
void         gpu_buffer_destroy(gpu_buffer *buf);
void        *gpu_buffer_map(gpu_buffer *buf);
void         gpu_buffer_unmap(gpu_buffer *buf);
void         gpu_buffer_upload(gpu_buffer *buf, const void *data, size_t size);
void         gpu_buffer_download(gpu_buffer *buf, void *data, size_t size);

gpu_pipeline *gpu_pipeline_create(gpu_device *dev, const uint8_t *spirv, size_t spirv_size);
void          gpu_pipeline_destroy(gpu_pipeline *p);
void          gpu_dispatch(gpu_device *dev, gpu_pipeline *p,
                           gpu_buffer **buffers, int num_buffers,
                           uint32_t gx, uint32_t gy, uint32_t gz);
void          gpu_wait(gpu_device *dev);
```

---

## server/srv.h — Multi-User Server

```c
vol_server *server_new(server_config cfg);
void        server_free(vol_server *s);
bool        server_start(vol_server *s);
void        server_stop(vol_server *s);

void server_on(vol_server *s, msg_type_t type, server_handler_fn fn, void *ctx);
bool server_send(vol_server *s, int client_id, msg_type_t type,
                 const void *payload, uint32_t len);
void server_broadcast(vol_server *s, msg_type_t type,
                      const void *payload, uint32_t len);
int  server_client_count(const vol_server *s);
```

## server/protocol.h — Binary Protocol

```c
#define PROTOCOL_HEADER_SZ 16

void protocol_encode_header(const protocol_header_t *header, uint8_t buf[PROTOCOL_HEADER_SZ]);
int  protocol_decode_header(const uint8_t buf[PROTOCOL_HEADER_SZ], protocol_header_t *header);
int  protocol_send(int fd, msg_type_t msg_type, const void *payload, uint32_t len);
int  protocol_recv(int fd, protocol_header_t *header_out, void **payload_out, int timeout_ms);
```
