#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE   /* be64toh */

#include "cli/cli_serve.h"
#include "core/log.h"
#include "core/io.h"   // dtype_t — required before vol.h
#include "core/vol.h"
#include "server/srv.h"
#include "server/db.h"
#include "server/protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <dirent.h>
#include <sys/stat.h>
#include <inttypes.h>
#include <endian.h>
#include <arpa/inet.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Chunk request wire layout (binary, network byte order for integers)
//
//  [0..63]  volume_id  — NUL-padded ASCII string
//  [64..67] level      — big-endian int32
//  [68..75] iz         — big-endian int64
//  [76..83] iy         — big-endian int64
//  [84..91] ix         — big-endian int64
// ---------------------------------------------------------------------------

#define CHUNK_REQ_SIZE 92

typedef struct {
  char    volume_id[64];
  int32_t level;
  int64_t iz, iy, ix;
} chunk_req;

static bool chunk_req_decode(const uint8_t *buf, uint32_t len, chunk_req *out) {
  if (len < CHUNK_REQ_SIZE) return false;
  memcpy(out->volume_id, buf, 64);
  out->volume_id[63] = '\0';

  uint32_t lv;  memcpy(&lv, buf + 64, 4); out->level = (int32_t)ntohl(lv);
  uint64_t iz;  memcpy(&iz, buf + 68, 8); out->iz    = (int64_t)be64toh(iz);
  uint64_t iy;  memcpy(&iy, buf + 76, 8); out->iy    = (int64_t)be64toh(iy);
  uint64_t ix;  memcpy(&ix, buf + 84, 8); out->ix    = (int64_t)be64toh(ix);
  return true;
}

// ---------------------------------------------------------------------------
// Minimal JSON string-value extractor
// Finds the first occurrence of "key":"value" and copies value into dst.
// ---------------------------------------------------------------------------

static void json_str_extract(const char *js, const char *key, char *dst, size_t dst_len) {
  char needle[128];
  snprintf(needle, sizeof(needle), "\"%s\":", key);
  const char *p = strstr(js, needle);
  if (!p) return;
  p += strlen(needle);
  while (*p == ' ') p++;
  if (*p != '"') return;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i < dst_len - 1) dst[i++] = *p++;
  dst[i] = '\0';
}

// ---------------------------------------------------------------------------
// Server context (shared across handlers)
// ---------------------------------------------------------------------------

#define MAX_VOLUMES 256

typedef struct {
  vol_server *srv;
  seg_db     *db;
  volume     *vols[MAX_VOLUMES];
  char        vol_ids[MAX_VOLUMES][256];
  int         nvols;
} serve_ctx;

static volume *find_vol(serve_ctx *ctx, const char *id) {
  for (int i = 0; i < ctx->nvols; i++) {
    if (strcmp(ctx->vol_ids[i], id) == 0) return ctx->vols[i];
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// MSG_CHUNK_REQ handler
// ---------------------------------------------------------------------------

static void on_chunk_req(vol_server *srv, const server_request *req, void *ud) {
  serve_ctx *ctx = ud;
  chunk_req cr;
  if (!chunk_req_decode(req->payload, req->payload_len, &cr)) {
    LOG_WARN("chunk_req: malformed payload from client %d", req->client_id);
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"bad chunk_req", 13);
    return;
  }

  volume *v = find_vol(ctx, cr.volume_id);
  if (!v) {
    LOG_WARN("chunk_req: unknown volume '%s'", cr.volume_id);
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"unknown volume", 14);
    return;
  }

  int64_t coords[3] = { cr.iz, cr.iy, cr.ix };
  size_t  chunk_size = 0;
  uint8_t *data = vol_read_chunk(v, cr.level, coords, &chunk_size);
  if (!data) {
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"chunk not found", 15);
    return;
  }

  server_send(srv, req->client_id, MSG_CHUNK_RESP, data, (uint32_t)chunk_size);
  free(data);
}

// ---------------------------------------------------------------------------
// MSG_SEG_UPDATE handler — payload: {"volume_id":…,"name":…,"surface_path":…}
// ---------------------------------------------------------------------------

static void on_seg_update(vol_server *srv, const server_request *req, void *ud) {
  serve_ctx *ctx = ud;
  if (!ctx->db || !req->payload || req->payload_len == 0) return;

  char *js = malloc(req->payload_len + 1);
  if (!js) return;
  memcpy(js, req->payload, req->payload_len);
  js[req->payload_len] = '\0';

  char vol_id[128] = {0}, name[256] = {0}, surf[1024] = {0};
  json_str_extract(js, "volume_id",    vol_id, sizeof(vol_id));
  json_str_extract(js, "name",         name,   sizeof(name));
  json_str_extract(js, "surface_path", surf,   sizeof(surf));
  free(js);

  if (!vol_id[0] || !name[0]) {
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"bad seg_update", 14);
    return;
  }

  int64_t id = seg_db_insert_segment(ctx->db, vol_id, name, surf);
  if (id < 0) {
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"db error", 8);
    return;
  }

  char ack[32];
  int n = snprintf(ack, sizeof(ack), "{\"id\":%" PRId64 "}", id);
  server_send(srv, req->client_id, MSG_SEG_ACK,    (uint8_t *)ack,      (uint32_t)n);
  server_broadcast(srv,             MSG_SEG_UPDATE, req->payload, req->payload_len);
}

// ---------------------------------------------------------------------------
// MSG_ANNOT_UPDATE handler — payload: {"segment_id":…,"type":…}
// ---------------------------------------------------------------------------

static void on_annot_update(vol_server *srv, const server_request *req, void *ud) {
  serve_ctx *ctx = ud;
  if (!ctx->db || !req->payload || req->payload_len == 0) return;

  char *js = malloc(req->payload_len + 1);
  if (!js) return;
  memcpy(js, req->payload, req->payload_len);
  js[req->payload_len] = '\0';

  int64_t segment_id = -1;
  const char *p = strstr(js, "\"segment_id\":");
  if (p) { p += 13; while (*p == ' ') p++; segment_id = (int64_t)strtoll(p, NULL, 10); }

  char type_buf[64] = {0};
  json_str_extract(js, "type", type_buf, sizeof(type_buf));
  free(js);

  if (segment_id < 0 || !type_buf[0]) {
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"bad annot_update", 16);
    return;
  }

  int64_t id = seg_db_insert_annotation(ctx->db, segment_id, type_buf,
                                         (char *)req->payload);
  if (id < 0) {
    server_send(srv, req->client_id, MSG_ERROR, (uint8_t *)"db error", 8);
    return;
  }

  char ack[32];
  int n = snprintf(ack, sizeof(ack), "{\"id\":%" PRId64 "}", id);
  server_send(srv, req->client_id, MSG_ANNOT_ACK,    (uint8_t *)ack,      (uint32_t)n);
  server_broadcast(srv,             MSG_ANNOT_UPDATE, req->payload, req->payload_len);
}

// ---------------------------------------------------------------------------
// Volume scanning
// ---------------------------------------------------------------------------

static int scan_volumes(serve_ctx *ctx, const char *dir) {
  DIR *d = opendir(dir);
  if (!d) { fprintf(stderr, "serve: cannot open data dir: %s\n", dir); return 0; }

  struct dirent *ent;
  while ((ent = readdir(d)) != NULL && ctx->nvols < MAX_VOLUMES) {
    if (ent->d_name[0] == '.') continue;

    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", dir, ent->d_name);

    struct stat st;
    if (stat(path, &st) != 0) continue;
    if (!S_ISDIR(st.st_mode)) continue;

    volume *v = vol_open(path);
    if (!v) continue;

    ctx->vols[ctx->nvols] = v;
    strncpy(ctx->vol_ids[ctx->nvols], ent->d_name, 255);
    ctx->vol_ids[ctx->nvols][255] = '\0';
    ctx->nvols++;
    printf("  loaded volume: %s\n", ent->d_name);
  }
  closedir(d);
  return ctx->nvols;
}

// ---------------------------------------------------------------------------
// SIGINT
// ---------------------------------------------------------------------------

static volatile sig_atomic_t g_stop = 0;
static void on_sigint(int sig) { (void)sig; g_stop = 1; }

// ---------------------------------------------------------------------------
// cmd_serve
// ---------------------------------------------------------------------------

int cmd_serve(int argc, char **argv) {
  int         port    = 9876;
  const char *data    = ".";
  const char *db_path = "segments.db";

  for (int i = 0; i < argc - 1; i++) {
    if (strcmp(argv[i], "--port") == 0) port    = atoi(argv[i + 1]);
    if (strcmp(argv[i], "--data") == 0) data    = argv[i + 1];
    if (strcmp(argv[i], "--db")   == 0) db_path = argv[i + 1];
  }

  serve_ctx ctx = {0};

  printf("scanning volumes in: %s\n", data);
  scan_volumes(&ctx, data);
  printf("loaded %d volume(s)\n\n", ctx.nvols);

  ctx.db = seg_db_open(db_path);
  if (!ctx.db) fprintf(stderr, "warning: could not open segment db: %s\n", db_path);

  server_config cfg = {
    .port        = port,
    .max_clients = 64,
    .io_threads  = 4,
    .db_path     = db_path,
  };
  ctx.srv = server_new(cfg);
  if (!ctx.srv) {
    fputs("error: could not create server\n", stderr);
    seg_db_close(ctx.db);
    return 1;
  }

  server_on(ctx.srv, MSG_CHUNK_REQ,    on_chunk_req,    &ctx);
  server_on(ctx.srv, MSG_SEG_UPDATE,   on_seg_update,   &ctx);
  server_on(ctx.srv, MSG_ANNOT_UPDATE, on_annot_update, &ctx);

  if (!server_start(ctx.srv)) {
    fputs("error: server failed to start\n", stderr);
    server_free(ctx.srv);
    seg_db_close(ctx.db);
    return 1;
  }

  printf("volatile server listening on port %d\n", port);
  printf("volumes: %d  |  db: %s\n", ctx.nvols, db_path);
  puts("press Ctrl-C to stop\n");
  fflush(stdout);

  signal(SIGINT, on_sigint);
  while (!g_stop) {
    struct timespec ts = { .tv_sec = 10, .tv_nsec = 0 };
    nanosleep(&ts, NULL);
    if (g_stop) break;
    printf("[status] port=%d  volumes=%d  clients=%d\n",
           port, ctx.nvols, server_client_count(ctx.srv));
    fflush(stdout);
  }

  puts("\nstopping server...");
  server_stop(ctx.srv);
  server_free(ctx.srv);
  seg_db_close(ctx.db);
  for (int i = 0; i < ctx.nvols; i++) vol_free(ctx.vols[i]);
  return 0;
}
