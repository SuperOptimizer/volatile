#define _POSIX_C_SOURCE 200809L

#include "server/srv.h"
#include "core/log.h"
#include "core/thread.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <pthread.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <arpa/inet.h>

// ---------------------------------------------------------------------------
// Limits & defaults
// ---------------------------------------------------------------------------

#define DEFAULT_PORT        9876
#define DEFAULT_MAX_CLIENTS 64
#define DEFAULT_IO_THREADS  4
#define MAX_MSG_TYPES       16   // enough for all msg_type_t values

// ---------------------------------------------------------------------------
// Client slot
// ---------------------------------------------------------------------------

typedef struct {
  int  fd;
  int  id;          // monotonically increasing client id
  bool active;
} client_slot;

// ---------------------------------------------------------------------------
// Per-type handler registration
// ---------------------------------------------------------------------------

typedef struct {
  server_handler_fn fn;
  void             *ctx;
  bool              set;
} handler_entry;

// ---------------------------------------------------------------------------
// Dispatch task args (heap-allocated, freed by worker)
// ---------------------------------------------------------------------------

typedef struct {
  vol_server    *srv;
  server_request req;         // req.payload is malloc'd; worker frees it
} dispatch_args;

// ---------------------------------------------------------------------------
// Server struct
// ---------------------------------------------------------------------------

struct vol_server {
  server_config   cfg;
  int             listen_fd;
  bool            running;

  client_slot    *clients;    // cfg.max_clients slots
  int             next_id;    // next client id to assign
  pthread_mutex_t clients_mu;

  handler_entry   handlers[MAX_MSG_TYPES];

  threadpool     *pool;
  pthread_t       accept_tid;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Find a client slot by id (must hold clients_mu).
static client_slot *find_client_locked(vol_server *s, int id) {
  for (int i = 0; i < s->cfg.max_clients; i++) {
    if (s->clients[i].active && s->clients[i].id == id) return &s->clients[i];
  }
  return NULL;
}

// Find a free slot (must hold clients_mu).
static client_slot *alloc_slot_locked(vol_server *s) {
  for (int i = 0; i < s->cfg.max_clients; i++) {
    if (!s->clients[i].active) return &s->clients[i];
  }
  return NULL;
}

// ---------------------------------------------------------------------------
// Handler dispatch (runs in thread pool)
// ---------------------------------------------------------------------------

static void *dispatch_task(void *arg) {
  dispatch_args *da = arg;
  vol_server    *s  = da->srv;
  server_request req = da->req;
  free(da);

  msg_type_t type = req.msg_type;

  // Built-in: auto-reply PONG to PING.
  if (type == MSG_PING) {
    server_send(s, req.client_id, MSG_PONG, NULL, 0);
  }

  if ((int)type < MAX_MSG_TYPES && s->handlers[type].set) {
    s->handlers[type].fn(s, &req, s->handlers[type].ctx);
  }

  free(req.payload);
  return NULL;
}

// ---------------------------------------------------------------------------
// Per-client reader thread
// ---------------------------------------------------------------------------

typedef struct { vol_server *srv; int client_id; int fd; } reader_args;

static void *client_reader(void *arg) {
  reader_args *ra  = arg;
  vol_server  *s   = ra->srv;
  int          cid = ra->client_id;
  int          fd  = ra->fd;
  free(ra);

  for (;;) {
    protocol_header_t hdr;
    void *payload = NULL;
    int rc = protocol_recv(fd, &hdr, &payload, 0 /* block */);

    if (rc != PROTO_OK) {
      if (rc != PROTO_ERR_IO) {
        LOG_WARN("client %d: protocol_recv error %d", cid, rc);
      }
      free(payload);
      break;
    }

    dispatch_args *da = malloc(sizeof(*da));
    if (!da) { free(payload); break; }

    da->srv           = s;
    da->req.client_id  = cid;
    da->req.msg_type   = (msg_type_t)hdr.msg_type;
    da->req.payload    = (uint8_t *)payload;
    da->req.payload_len = hdr.payload_len;

    threadpool_fire(s->pool, dispatch_task, da);
  }

  // Remove from client table and close fd.
  pthread_mutex_lock(&s->clients_mu);
  client_slot *slot = find_client_locked(s, cid);
  if (slot) { slot->active = false; slot->fd = -1; }
  pthread_mutex_unlock(&s->clients_mu);

  close(fd);
  LOG_DEBUG("client %d disconnected", cid);
  return NULL;
}

// ---------------------------------------------------------------------------
// Accept thread
// ---------------------------------------------------------------------------

static void *accept_thread(void *arg) {
  vol_server *s = arg;

  while (s->running) {
    struct sockaddr_in addr = {0};
    socklen_t addrlen = sizeof(addr);
    int fd = accept(s->listen_fd, (struct sockaddr *)&addr, &addrlen);
    if (fd < 0) {
      if (!s->running) break;  // normal shutdown via shutdown()/close()
      if (errno == EINTR) continue;
      LOG_WARN("accept: %s", strerror(errno));
      continue;
    }

    // If woken by the self-connect during shutdown, discard and exit.
    if (!s->running) { close(fd); break; }

    pthread_mutex_lock(&s->clients_mu);
    client_slot *slot = alloc_slot_locked(s);
    if (!slot) {
      pthread_mutex_unlock(&s->clients_mu);
      LOG_WARN("server: max clients (%d) reached, dropping connection", s->cfg.max_clients);
      close(fd);
      continue;
    }
    int cid     = s->next_id++;
    slot->fd    = fd;
    slot->id    = cid;
    slot->active = true;
    pthread_mutex_unlock(&s->clients_mu);

    LOG_DEBUG("client %d connected from %s", cid, inet_ntoa(addr.sin_addr));

    reader_args *ra = malloc(sizeof(*ra));
    if (!ra) { close(fd); continue; }
    ra->srv       = s;
    ra->client_id = cid;
    ra->fd        = fd;

    // NOTE: Detached reader thread — no need to join; it self-cleans on exit.
    pthread_t tid;
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    if (pthread_create(&tid, &attr, client_reader, ra) != 0) {
      LOG_ERROR("server: pthread_create for client %d failed", cid);
      pthread_mutex_lock(&s->clients_mu);
      slot->active = false;
      pthread_mutex_unlock(&s->clients_mu);
      free(ra);
      close(fd);
    }
    pthread_attr_destroy(&attr);
  }

  return NULL;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

vol_server *server_new(server_config cfg) {
  if (cfg.port        <= 0) cfg.port        = DEFAULT_PORT;
  if (cfg.max_clients <= 0) cfg.max_clients = DEFAULT_MAX_CLIENTS;
  if (cfg.io_threads  <= 0) cfg.io_threads  = DEFAULT_IO_THREADS;

  vol_server *s = calloc(1, sizeof(*s));
  if (!s) return NULL;

  s->cfg      = cfg;
  s->listen_fd = -1;
  s->next_id  = 1;

  s->clients = calloc((size_t)cfg.max_clients, sizeof(client_slot));
  if (!s->clients) { free(s); return NULL; }

  // Mark all slots inactive with invalid fd.
  for (int i = 0; i < cfg.max_clients; i++) s->clients[i].fd = -1;

  if (pthread_mutex_init(&s->clients_mu, NULL) != 0) {
    free(s->clients); free(s); return NULL;
  }

  s->pool = threadpool_new(cfg.io_threads);
  if (!s->pool) {
    pthread_mutex_destroy(&s->clients_mu);
    free(s->clients); free(s); return NULL;
  }

  return s;
}

void server_free(vol_server *s) {
  if (!s) return;
  if (s->running) server_stop(s);
  threadpool_free(s->pool);
  pthread_mutex_destroy(&s->clients_mu);
  free(s->clients);
  free(s);
}

void server_on(vol_server *s, msg_type_t type, server_handler_fn fn, void *ctx) {
  if (!s || (int)type >= MAX_MSG_TYPES) return;
  s->handlers[type] = (handler_entry){ .fn = fn, .ctx = ctx, .set = true };
}

bool server_start(vol_server *s) {
  if (!s || s->running) return false;

  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) { LOG_ERROR("server_start: socket: %s", strerror(errno)); return false; }

  int opt = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in addr = {
    .sin_family      = AF_INET,
    .sin_port        = htons((uint16_t)s->cfg.port),
    .sin_addr.s_addr = INADDR_ANY,
  };

  if (bind(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    LOG_ERROR("server_start: bind port %d: %s", s->cfg.port, strerror(errno));
    close(fd); return false;
  }

  if (listen(fd, 16) < 0) {
    LOG_ERROR("server_start: listen: %s", strerror(errno));
    close(fd); return false;
  }

  s->listen_fd = fd;
  s->running   = true;

  if (pthread_create(&s->accept_tid, NULL, accept_thread, s) != 0) {
    LOG_ERROR("server_start: pthread_create: %s", strerror(errno));
    s->running = false;
    close(fd); s->listen_fd = -1;
    return false;
  }

  LOG_INFO("server listening on port %d", s->cfg.port);
  return true;
}

void server_stop(vol_server *s) {
  if (!s || !s->running) return;
  s->running = false;

  // Unblock accept() — shutdown() makes the kernel return EINVAL/EBADF
  // to the blocked accept() call, then close() releases the fd.
  // A loopback self-connect is belt-and-suspenders in case shutdown alone
  // isn't sufficient on this kernel.
  if (s->listen_fd >= 0) {
    shutdown(s->listen_fd, SHUT_RDWR);
    // Brief self-connect to unblock accept() if still sleeping.
    int wake = socket(AF_INET, SOCK_STREAM, 0);
    if (wake >= 0) {
      struct sockaddr_in addr = {
        .sin_family      = AF_INET,
        .sin_port        = htons((uint16_t)s->cfg.port),
        .sin_addr.s_addr = htonl(INADDR_LOOPBACK),
      };
      connect(wake, (struct sockaddr *)&addr, sizeof(addr));  // ignore result
      close(wake);
    }
    close(s->listen_fd);
    s->listen_fd = -1;
  }
  pthread_join(s->accept_tid, NULL);

  // Close all client fds to unblock their reader threads.
  pthread_mutex_lock(&s->clients_mu);
  for (int i = 0; i < s->cfg.max_clients; i++) {
    if (s->clients[i].active && s->clients[i].fd >= 0) {
      close(s->clients[i].fd);
      s->clients[i].fd     = -1;
      s->clients[i].active = false;
    }
  }
  pthread_mutex_unlock(&s->clients_mu);

  // Drain handler tasks.
  threadpool_drain(s->pool, 2000);
  LOG_INFO("server stopped");
}

bool server_send(vol_server *s, int client_id, msg_type_t type,
                 const uint8_t *payload, uint32_t len) {
  if (!s) return false;

  pthread_mutex_lock(&s->clients_mu);
  client_slot *slot = find_client_locked(s, client_id);
  int fd = slot ? slot->fd : -1;
  pthread_mutex_unlock(&s->clients_mu);

  if (fd < 0) return false;
  return protocol_send(fd, type, payload, len) == PROTO_OK;
}

void server_broadcast(vol_server *s, msg_type_t type,
                      const uint8_t *payload, uint32_t len) {
  if (!s) return;

  pthread_mutex_lock(&s->clients_mu);
  for (int i = 0; i < s->cfg.max_clients; i++) {
    if (s->clients[i].active && s->clients[i].fd >= 0) {
      protocol_send(s->clients[i].fd, type, payload, len);
    }
  }
  pthread_mutex_unlock(&s->clients_mu);
}

int server_client_count(const vol_server *s) {
  if (!s) return 0;

  pthread_mutex_lock((pthread_mutex_t *)&s->clients_mu);
  int count = 0;
  for (int i = 0; i < s->cfg.max_clients; i++) {
    if (s->clients[i].active) count++;
  }
  pthread_mutex_unlock((pthread_mutex_t *)&s->clients_mu);
  return count;
}
