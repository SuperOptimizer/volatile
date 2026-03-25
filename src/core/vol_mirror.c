#define _POSIX_C_SOURCE 200809L

#include "core/vol_mirror.h"
#include "core/vol_internal.h"
#include "core/net.h"
#include "core/thread.h"
#include "core/compress4d.h"
#include "core/log.h"

#include <blosc2.h>
#include <errno.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <pthread.h>
#include <unistd.h>

#define DEFAULT_MAX_CACHE_BYTES  (INT64_C(50) * 1024 * 1024 * 1024)
#define DEFAULT_PREFETCH_RADIUS  2
#define FETCH_TIMEOUT_MS         30000

// ---------------------------------------------------------------------------
// vol_mirror struct
// ---------------------------------------------------------------------------

struct vol_mirror {
  mirror_config cfg;
  char          cache_dir[VOL_PATH_MAX];  // resolved local cache root
  char          local_zarr[VOL_PATH_MAX]; // cache_dir/<hash_of_url>/
  volume       *local_vol;               // opened after first cache write
  volume       *remote_vol;              // opened from remote URL
  threadpool   *pool;                    // background prefetch pool

  // detected once in vol_mirror_new
  bool          remote_is_c4d;          // remote codec is compress4d
  bool          remote_is_volt_server;  // remote speaks volatile TCP binary protocol

  pthread_mutex_t lock;
  int   chunks_cached;
  int   total_fetches;
  int   cache_hits;
  int   cache_misses;
  int64_t cached_bytes;
};

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

// Simple djb2 hash of a string → hex suffix for cache dir name
static void url_to_dir_name(const char *url, char *out, size_t outsz) {
  uint64_t h = 5381;
  for (const char *p = url; *p; p++) h = h * 33 ^ (uint8_t)*p;
  snprintf(out, outsz, "%016llx", (unsigned long long)h);
}

// Build the local path for a cached chunk given its zarr-format path fragment.
// e.g. cache_dir/local_zarr/0/0.0.0
static void chunk_cache_path(const vol_mirror *m, int level,
                              const int64_t *coords, int ndim,
                              char *out, size_t outsz) {
  int n = snprintf(out, outsz, "%s/%d", m->local_zarr, level);
  for (int d = 0; d < ndim; d++) {
    char sep = (d == 0) ? '/' : '.';
    n += snprintf(out + n, outsz - (size_t)n, "%c%lld", sep, (long long)coords[d]);
  }
}

// Build the remote URL for a chunk. Supports v2 (z.y.x) path convention.
static void chunk_remote_url(const vol_mirror *m, int level,
                              const int64_t *coords, int ndim,
                              char *out, size_t outsz) {
  // strip trailing slash from remote_url
  size_t base_len = strlen(m->cfg.remote_url);
  const char *base = m->cfg.remote_url;
  if (base[base_len - 1] == '/') base_len--;

  int n = snprintf(out, outsz, "%.*s/%d", (int)base_len, base, level);
  for (int d = 0; d < ndim; d++) {
    char sep = (d == 0) ? '/' : '.';
    n += snprintf(out + n, outsz - (size_t)n, "%c%lld", sep, (long long)coords[d]);
  }
}

// ---------------------------------------------------------------------------
// Cache-on-disk: write a fetched chunk to local zarr tree
// ---------------------------------------------------------------------------

static bool write_chunk_to_disk(const char *path, const uint8_t *data, size_t sz) {
  // ensure parent dirs
  char tmp[VOL_PATH_MAX];
  snprintf(tmp, sizeof(tmp), "%s", path);
  char *slash = strrchr(tmp, '/');
  if (slash) { *slash = '\0'; vol_mkdirp(tmp); }

  FILE *f = fopen(path, "wb");
  if (!f) return false;
  bool ok = fwrite(data, 1, sz, f) == sz;
  fclose(f);
  return ok;
}

// ---------------------------------------------------------------------------
// Fetch one chunk from remote and store to local disk cache.
// Returns malloc'd decompressed bytes (caller frees), sets *out_sz.
// ---------------------------------------------------------------------------

// Returns true if the URL is HTTP/HTTPS/S3, false if it's a local path.
static bool is_remote_url(const char *url) {
  return strncmp(url, "http://",  7) == 0 ||
         strncmp(url, "https://", 8) == 0 ||
         strncmp(url, "s3://",    5) == 0;
}

static uint8_t *fetch_and_cache(vol_mirror *m, int level,
                                 const int64_t *coords, int ndim,
                                 size_t *out_sz) {
  char url[VOL_PATH_MAX * 2];
  chunk_remote_url(m, level, coords, ndim, url, sizeof(url));

  char disk_path[VOL_PATH_MAX];
  chunk_cache_path(m, level, coords, ndim, disk_path, sizeof(disk_path));

  uint8_t *raw = NULL;
  size_t   rsz = 0;

  if (is_remote_url(m->cfg.remote_url)) {
    http_response *resp = http_get(url, FETCH_TIMEOUT_MS);
    if (!resp || resp->status_code != 200 || !resp->data || resp->size == 0) {
      http_response_free(resp);
      pthread_mutex_lock(&m->lock);
      m->cache_misses++;
      pthread_mutex_unlock(&m->lock);
      return NULL;
    }
    // store to disk
    write_chunk_to_disk(disk_path, resp->data, resp->size);
    raw = resp->data;
    rsz = resp->size;
    resp->data = NULL;  // take ownership
    http_response_free(resp);
  } else {
    // local "remote": url is a filesystem path — read it directly
    raw = vol_read_file_bytes(url, &rsz);
    if (!raw || rsz == 0) {
      free(raw);
      pthread_mutex_lock(&m->lock);
      m->cache_misses++;
      pthread_mutex_unlock(&m->lock);
      return NULL;
    }
    write_chunk_to_disk(disk_path, raw, rsz);
  }

  // try blosc decompress; if it fails treat as raw
  uint8_t *decompressed = NULL;
  int32_t nbytes = 0, cbytes = 0, blocksize = 0;
  blosc2_cbuffer_sizes(raw, &nbytes, &cbytes, &blocksize);
  if (nbytes > 0 && (size_t)cbytes == rsz) {
    decompressed = malloc((size_t)nbytes);
    if (decompressed) {
      int ret = blosc2_decompress(raw, (int32_t)rsz, decompressed, nbytes);
      if (ret < 0) { free(decompressed); decompressed = NULL; }
    }
  }
  if (!decompressed) {
    // not blosc — treat as raw bytes
    decompressed = malloc(rsz);
    if (decompressed) memcpy(decompressed, raw, rsz);
    *out_sz = rsz;
  } else {
    *out_sz = (size_t)nbytes;
  }

  free(raw);

  pthread_mutex_lock(&m->lock);
  m->chunks_cached++;
  m->cached_bytes += (int64_t)rsz;
  m->total_fetches++;
  pthread_mutex_unlock(&m->lock);

  return decompressed;
}

// ---------------------------------------------------------------------------
// Prefetch task submitted to thread pool
// ---------------------------------------------------------------------------

typedef struct {
  vol_mirror *mirror;
  int level;
  int64_t coords[8];
  int ndim;
} prefetch_task;

static void *prefetch_fn(void *arg) {
  prefetch_task *t = arg;
  // check if already cached on disk
  char disk_path[VOL_PATH_MAX];
  chunk_cache_path(t->mirror, t->level, t->coords, t->ndim, disk_path, sizeof(disk_path));
  struct stat st;
  if (stat(disk_path, &st) == 0) { free(t); return NULL; }

  size_t sz = 0;
  uint8_t *data = fetch_and_cache(t->mirror, t->level, t->coords, t->ndim, &sz);
  free(data);
  free(t);
  return NULL;
}

static void schedule_prefetch(vol_mirror *m, int level, const int64_t *center,
                               int ndim, int radius) {
  if (!m->pool || radius <= 0) return;
  // simple 1D radius along each axis to keep prefetch count bounded
  for (int d = 0; d < ndim; d++) {
    for (int delta = -radius; delta <= radius; delta++) {
      if (delta == 0) continue;
      prefetch_task *t = malloc(sizeof(*t));
      if (!t) continue;
      t->mirror = m;
      t->level  = level;
      t->ndim   = ndim;
      memcpy(t->coords, center, sizeof(int64_t) * (size_t)ndim);
      t->coords[d] += delta;
      if (t->coords[d] < 0) { free(t); continue; }
      threadpool_fire(m->pool, prefetch_fn, t);
    }
  }
}

// ---------------------------------------------------------------------------
// Codec / protocol detection
// ---------------------------------------------------------------------------

// Returns true if the remote volume's level-0 codec is compress4d
// (compressor_id == "compress4d" for v2, or codec name "compress4d" for v3).
static bool detect_compress4d(const volume *v) {
  if (!v) return false;
  const zarr_level_meta *m = vol_level_meta(v, 0);
  if (!m) return false;
  return strcmp(m->compressor_id, "compress4d") == 0;
}

// Probe whether the host:port at the base of the URL speaks the volatile
// binary TCP protocol. Opens a TCP connection, sends a PING header, and
// checks whether the server responds with a PONG bearing PROTOCOL_MAGIC.
// Returns false immediately for non-HTTP URLs or parse failures.
static bool probe_volatile_server(const char *url) {
  // only probe http/https URLs (not local paths or s3://)
  if (strncmp(url, "http://",  7) != 0 &&
      strncmp(url, "https://", 8) != 0) return false;

  parsed_url pu;
  if (!url_parse(url, &pu)) return false;

  // default TCP port for volatile server (same as HTTP default; server listens
  // on both HTTP for zarr metadata and TCP for binary protocol on port+1)
  int port = pu.port > 0 ? pu.port + 1 : 27183;  // 27182 HTTP -> 27183 binary

  struct addrinfo hints = { .ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM };
  struct addrinfo *res = NULL;
  char portstr[8];
  snprintf(portstr, sizeof(portstr), "%d", port);
  if (getaddrinfo(pu.host, portstr, &hints, &res) != 0) return false;

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  bool ok = false;
  if (fd >= 0) {
    // non-blocking connect with short timeout via select
    struct timeval tv = { .tv_sec = 1, .tv_usec = 0 };
    if (connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
      // send a minimal PING: 12-byte header with PROTOCOL_MAGIC
      uint8_t hdr[12] = {0};
      uint32_t magic = 0x564F4C54;  // "VOLT" big-endian
      hdr[0] = (uint8_t)(magic >> 24); hdr[1] = (uint8_t)(magic >> 16);
      hdr[2] = (uint8_t)(magic >> 8);  hdr[3] = (uint8_t)(magic);
      // msg_type MSG_PING = 0, flags = 0, payload_len = 0
      send(fd, hdr, sizeof(hdr), 0);

      // wait up to 1s for a PONG
      fd_set rfds; FD_ZERO(&rfds); FD_SET(fd, &rfds);
      if (select(fd + 1, &rfds, NULL, NULL, &tv) > 0) {
        uint8_t resp[12] = {0};
        ssize_t n = recv(fd, resp, sizeof(resp), 0);
        if (n == 12) {
          uint32_t m2 = ((uint32_t)resp[0] << 24) | ((uint32_t)resp[1] << 16) |
                        ((uint32_t)resp[2] << 8)  |  (uint32_t)resp[3];
          uint16_t mt = (uint16_t)(((uint16_t)resp[4] << 8) | resp[5]);
          ok = (m2 == magic) && (mt == 1 /* MSG_PONG */);
        }
      }
    }
    close(fd);
  }
  freeaddrinfo(res);
  return ok;
}

// Public accessors — check cached detection results.
bool vol_mirror_remote_is_compress4d(const vol_mirror *m) {
  return m && m->remote_is_c4d;
}

bool vol_mirror_remote_is_volatile_server(const vol_mirror *m) {
  return m && m->remote_is_volt_server;
}

// ---------------------------------------------------------------------------
// Binary-protocol chunk fetch (volatile TCP protocol)
// ---------------------------------------------------------------------------

// Fetch a chunk over the volatile binary TCP protocol.
// Returns malloc'd raw bytes (already in the server's native encoding),
// or NULL on failure.
static uint8_t *fetch_via_binary_protocol(const vol_mirror *m, int level,
                                           const int64_t *coords, int ndim,
                                           size_t *out_sz) {
  parsed_url pu;
  if (!url_parse(m->cfg.remote_url, &pu)) return NULL;
  int port = pu.port > 0 ? pu.port + 1 : 27183;

  struct addrinfo hints = { .ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM };
  struct addrinfo *res  = NULL;
  char portstr[8]; snprintf(portstr, sizeof(portstr), "%d", port);
  if (getaddrinfo(pu.host, portstr, &hints, &res) != 0) return NULL;

  int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
  uint8_t *result = NULL;
  if (fd >= 0 && connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
    // Build MSG_CHUNK_REQ payload: level(4) + ndim(4) + coords[ndim](8 each)
    size_t payload_sz = (size_t)(8 + ndim * 8);
    uint8_t *payload  = malloc(payload_sz);
    if (payload) {
      payload[0] = (uint8_t)(level >> 24); payload[1] = (uint8_t)(level >> 16);
      payload[2] = (uint8_t)(level >>  8); payload[3] = (uint8_t)(level);
      payload[4] = (uint8_t)(ndim  >> 24); payload[5] = (uint8_t)(ndim  >> 16);
      payload[6] = (uint8_t)(ndim  >>  8); payload[7] = (uint8_t)(ndim);
      for (int d = 0; d < ndim; d++) {
        uint64_t cv = (uint64_t)coords[d];
        for (int b = 7; b >= 0; b--) { payload[8 + d*8 + b] = (uint8_t)(cv & 0xff); cv >>= 8; }
      }
      // header: magic + MSG_CHUNK_REQ(4) + flags(0) + payload_len
      uint8_t hdr[12];
      uint32_t magic = 0x564F4C54;
      hdr[0]=(uint8_t)(magic>>24); hdr[1]=(uint8_t)(magic>>16);
      hdr[2]=(uint8_t)(magic>>8);  hdr[3]=(uint8_t)(magic);
      hdr[4]=0; hdr[5]=4; // MSG_CHUNK_REQ = 4
      hdr[6]=0; hdr[7]=0; // flags
      uint32_t plen = (uint32_t)payload_sz;
      hdr[8]=(uint8_t)(plen>>24); hdr[9]=(uint8_t)(plen>>16);
      hdr[10]=(uint8_t)(plen>>8); hdr[11]=(uint8_t)(plen);
      send(fd, hdr, 12, 0);
      send(fd, payload, payload_sz, 0);
      free(payload);

      // receive response header
      uint8_t rhdr[12] = {0};
      ssize_t n = recv(fd, rhdr, 12, MSG_WAITALL);
      if (n == 12) {
        uint32_t rlen = ((uint32_t)rhdr[8]<<24)|((uint32_t)rhdr[9]<<16)|
                        ((uint32_t)rhdr[10]<<8)|(uint32_t)rhdr[11];
        if (rlen > 0 && rlen < (uint32_t)(256 * 1024 * 1024)) {
          result = malloc(rlen);
          if (result) {
            ssize_t got = recv(fd, result, rlen, MSG_WAITALL);
            if ((size_t)got == rlen) *out_sz = rlen;
            else { free(result); result = NULL; }
          }
        }
      }
    }
    close(fd);
  }
  freeaddrinfo(res);
  return result;
}

// ---------------------------------------------------------------------------
// vol_mirror_new / free
// ---------------------------------------------------------------------------

vol_mirror *vol_mirror_new(mirror_config cfg) {
  if (!cfg.remote_url || !cfg.remote_url[0]) return NULL;

  vol_mirror *m = calloc(1, sizeof(*m));
  if (!m) return NULL;

  m->cfg = cfg;
  if (!m->cfg.max_cache_bytes) m->cfg.max_cache_bytes = DEFAULT_MAX_CACHE_BYTES;
  if (!m->cfg.prefetch_radius) m->cfg.prefetch_radius = DEFAULT_PREFETCH_RADIUS;

  // resolve cache dir
  const char *base = cfg.local_cache_dir;
  if (!base || !base[0]) {
    const char *home = getenv("HOME");
    if (!home) home = "/tmp";
    snprintf(m->cache_dir, sizeof(m->cache_dir), "%s/.cache/volatile", home);
  } else {
    snprintf(m->cache_dir, sizeof(m->cache_dir), "%s", base);
  }

  char hash_name[32];
  url_to_dir_name(cfg.remote_url, hash_name, sizeof(hash_name));
  snprintf(m->local_zarr, sizeof(m->local_zarr), "%s/%s", m->cache_dir, hash_name);

  if (!vol_mkdirp(m->local_zarr)) {
    LOG_WARN("vol_mirror: cannot create cache dir %s: %s", m->local_zarr, strerror(errno));
    free(m);
    return NULL;
  }

  http_init();

  // try to open the remote volume for metadata
  m->remote_vol = vol_open(cfg.remote_url);
  if (!m->remote_vol) {
    LOG_WARN("vol_mirror: cannot open remote volume: %s", cfg.remote_url);
    free(m);
    return NULL;
  }

  // detect remote codec and protocol
  m->remote_is_c4d = detect_compress4d(m->remote_vol);
  // probe binary protocol only for HTTP URLs and when preferred (default true unless
  // caller explicitly set prefer_binary_protocol = false)
  bool try_binary = cfg.prefer_binary_protocol || !cfg.force_recompress;
  m->remote_is_volt_server = try_binary && probe_volatile_server(cfg.remote_url);

  if (m->remote_is_c4d)
    LOG_INFO("vol_mirror: remote codec is compress4d — raw chunks will be cached as-is");
  if (m->remote_is_volt_server)
    LOG_INFO("vol_mirror: volatile binary protocol detected — will use TCP streaming");

  // try to open existing local cache (may not exist yet)
  m->local_vol = vol_open(m->local_zarr);

  pthread_mutex_init(&m->lock, NULL);
  m->pool = threadpool_new(4);

  LOG_INFO("vol_mirror: mirroring %s → %s", cfg.remote_url, m->local_zarr);
  return m;
}

void vol_mirror_free(vol_mirror *m) {
  if (!m) return;
  if (m->pool) { threadpool_drain(m->pool, 5000); threadpool_free(m->pool); }
  vol_free(m->remote_vol);
  vol_free(m->local_vol);
  pthread_mutex_destroy(&m->lock);
  free(m);
}

volume *vol_mirror_volume(vol_mirror *m) {
  if (!m) return NULL;
  return m->local_vol ? m->local_vol : m->remote_vol;
}

// ---------------------------------------------------------------------------
// vol_mirror_cache_level
// ---------------------------------------------------------------------------

bool vol_mirror_cache_level(vol_mirror *m, int level) {
  if (!m || !m->remote_vol) return false;

  const zarr_level_meta *meta = vol_level_meta(m->remote_vol, level);
  if (!meta) return false;

  int ndim = meta->ndim;
  int64_t nchunks[8], total = 1;
  for (int d = 0; d < ndim; d++) {
    nchunks[d] = (meta->shape[d] + meta->chunk_shape[d] - 1) / meta->chunk_shape[d];
    total *= nchunks[d];
  }

  // write a stub .zarray so vol_open can recognise the local cache
  char level_dir[VOL_PATH_MAX];
  snprintf(level_dir, sizeof(level_dir), "%s/%d", m->local_zarr, level);
  vol_mkdirp(level_dir);

  int64_t coords[8] = {0};
  int64_t done = 0;

  do {
    // skip if already on disk
    char disk_path[VOL_PATH_MAX];
    chunk_cache_path(m, level, coords, ndim, disk_path, sizeof(disk_path));
    struct stat st;
    if (stat(disk_path, &st) != 0) {
      size_t sz = 0;
      uint8_t *data = fetch_and_cache(m, level, coords, ndim, &sz);
      free(data);
    } else {
      pthread_mutex_lock(&m->lock);
      m->cache_hits++;
      pthread_mutex_unlock(&m->lock);
    }

    done++;
    if (done % 100 == 0)
      LOG_INFO("vol_mirror: cached %lld/%lld chunks at level %d",
               (long long)done, (long long)total, level);

    // advance coord
    for (int d = ndim - 1; d >= 0; d--) {
      if (++coords[d] < nchunks[d]) break;
      coords[d] = 0;
    }
  } while (done < total);

  // (re-)open local vol so future reads use disk
  vol_free(m->local_vol);
  m->local_vol = vol_open(m->local_zarr);
  return true;
}

// ---------------------------------------------------------------------------
// vol_mirror_rechunk
// ---------------------------------------------------------------------------

bool vol_mirror_rechunk(vol_mirror *m, const int64_t *new_chunk_shape) {
  if (!m || !new_chunk_shape) return false;
  volume *src = m->local_vol ? m->local_vol : m->remote_vol;
  if (!src) return false;

  const zarr_level_meta *meta = vol_level_meta(src, 0);
  if (!meta) return false;

  char rechunked_path[VOL_PATH_MAX];
  snprintf(rechunked_path, sizeof(rechunked_path), "%s_rechunked", m->local_zarr);

  vol_create_params p = {
    .zarr_version = meta->zarr_version > 0 ? meta->zarr_version : 2,
    .ndim         = meta->ndim,
    .dtype        = (dtype_t)meta->dtype,
    .compressor   = meta->compressor_id[0] ? meta->compressor_id : NULL,
    .clevel       = meta->compressor_clevel > 0 ? meta->compressor_clevel : 5,
  };
  for (int d = 0; d < meta->ndim; d++) {
    p.shape[d]       = meta->shape[d];
    p.chunk_shape[d] = new_chunk_shape[d];
  }

  volume *dst = vol_create(rechunked_path, p);
  if (!dst) return false;

  // iterate new chunk grid, sample from src via vol_sample (accepts any coords)
  int64_t nchunks[8], total = 1;
  for (int d = 0; d < meta->ndim; d++) {
    nchunks[d] = (meta->shape[d] + new_chunk_shape[d] - 1) / new_chunk_shape[d];
    total *= nchunks[d];
  }

  size_t elem = vol_dtype_elem_size(meta->dtype);
  size_t cvol = 1;
  for (int d = 0; d < meta->ndim; d++) cvol *= (size_t)new_chunk_shape[d];
  uint8_t *buf = calloc(cvol, elem);
  if (!buf) { vol_free(dst); return false; }

  int64_t coords[8] = {0};
  do {
    // only 3D for now; generalise if needed
    if (meta->ndim == 3) {
      int64_t oz = coords[0] * new_chunk_shape[0];
      int64_t oy = coords[1] * new_chunk_shape[1];
      int64_t ox = coords[2] * new_chunk_shape[2];
      size_t ci = 0;
      for (int64_t z = oz; z < oz + new_chunk_shape[0]; z++)
        for (int64_t y = oy; y < oy + new_chunk_shape[1]; y++)
          for (int64_t x = ox; x < ox + new_chunk_shape[2]; x++) {
            float v = vol_sample(src, 0, (float)z, (float)y, (float)x);
            if (elem == 1) buf[ci] = (uint8_t)v;
            else           ((uint16_t *)buf)[ci] = (uint16_t)v;
            ci++;
          }
      vol_write_chunk(dst, 0, coords, buf, cvol * elem);
    }
    for (int d = meta->ndim - 1; d >= 0; d--) {
      if (++coords[d] < nchunks[d]) break;
      coords[d] = 0;
    }
    total--;
  } while (total > 0);

  free(buf);
  vol_finalize(dst);
  vol_free(dst);
  LOG_INFO("vol_mirror: rechunked → %s", rechunked_path);
  return true;
}

// ---------------------------------------------------------------------------
// vol_mirror_recompress (blosc → compress4d residuals)
// ---------------------------------------------------------------------------

bool vol_mirror_recompress(vol_mirror *m) {
  if (!m) return false;
  volume *src = m->local_vol ? m->local_vol : m->remote_vol;
  if (!src) return false;

  const zarr_level_meta *meta = vol_level_meta(src, 0);
  if (!meta) return false;

  int64_t nchunks[8];
  for (int d = 0; d < meta->ndim; d++)
    nchunks[d] = (meta->shape[d] + meta->chunk_shape[d] - 1) / meta->chunk_shape[d];

  char c4d_path[VOL_PATH_MAX];
  snprintf(c4d_path, sizeof(c4d_path), "%s_compress4d", m->local_zarr);
  vol_mkdirp(c4d_path);

  int64_t coords[8] = {0};
  int count = 0;
  do {
    size_t raw_sz = 0;
    uint8_t *raw = vol_read_chunk(src, 0, coords, &raw_sz);
    if (raw && raw_sz > 0) {
      size_t nfloats = raw_sz / sizeof(float);
      if (nfloats == 0) nfloats = raw_sz;
      // convert bytes to float for residual encoding
      float *fdata = malloc(nfloats * sizeof(float));
      if (fdata) {
        for (size_t i = 0; i < nfloats; i++)
          fdata[i] = (meta->dtype == DTYPE_U8) ? (float)raw[i] :
                     (float)((uint16_t *)raw)[i];
        size_t enc_sz = 0;
        uint8_t *enc = compress4d_encode_residual(fdata, nfloats, 1.0f / 255.0f, &enc_sz);
        free(fdata);
        if (enc) {
          // write to c4d_path/<level>/<coords>
          char out_path[VOL_PATH_MAX];
          int n = snprintf(out_path, sizeof(out_path), "%s/0", c4d_path);
          vol_mkdirp(out_path);
          n += snprintf(out_path + n, sizeof(out_path) - (size_t)n, "/%lld",
                        (long long)coords[0]);
          for (int d = 1; d < meta->ndim; d++)
            n += snprintf(out_path + n, sizeof(out_path) - (size_t)n,
                          ".%lld", (long long)coords[d]);
          write_chunk_to_disk(out_path, enc, enc_sz);
          free(enc);
          count++;
        }
      }
      free(raw);
    }
    for (int d = meta->ndim - 1; d >= 0; d--) {
      if (++coords[d] < nchunks[d]) break;
      coords[d] = 0;
    }
  } while (coords[0] != 0 || coords[1] != 0 || coords[2] != 0);

  LOG_INFO("vol_mirror: recompressed %d chunks to compress4d at %s", count, c4d_path);
  return true;
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

size_t vol_mirror_cached_bytes(const vol_mirror *m) {
  return m ? (size_t)m->cached_bytes : 0;
}

float vol_mirror_cache_hit_rate(const vol_mirror *m) {
  if (!m) return 0.0f;
  int total = m->cache_hits + m->cache_misses;
  return total > 0 ? (float)m->cache_hits / (float)total : 0.0f;
}

int vol_mirror_chunks_cached(const vol_mirror *m) {
  return m ? m->chunks_cached : 0;
}

int vol_mirror_chunks_total(const vol_mirror *m, int level) {
  if (!m) return 0;
  const volume *src = m->remote_vol ? m->remote_vol : m->local_vol;
  if (!src) return 0;
  const zarr_level_meta *meta = vol_level_meta(src, level);
  if (!meta) return 0;
  int64_t total = 1;
  for (int d = 0; d < meta->ndim; d++)
    total *= (meta->shape[d] + meta->chunk_shape[d] - 1) / meta->chunk_shape[d];
  return (int)total;
}
