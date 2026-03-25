#pragma once
#include "server/protocol.h"
#include <stdint.h>
#include <stdbool.h>

// ---------------------------------------------------------------------------
// Server types
// ---------------------------------------------------------------------------

typedef struct vol_server vol_server;

typedef struct {
  int         port;        // listen port (default 9876)
  int         max_clients; // max concurrent connections (default 64)
  int         io_threads;  // thread-pool size for handler dispatch (default 4)
  const char *db_path;     // SQLite database path (may be NULL)
} server_config;

typedef struct {
  int       client_id;
  msg_type_t msg_type;
  uint8_t  *payload;
  uint32_t  payload_len;
} server_request;

typedef void (*server_handler_fn)(vol_server *srv, const server_request *req, void *ctx);

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

vol_server *server_new(server_config cfg);
void        server_free(vol_server *s);

// Register a handler for a specific message type. Only one handler per type.
void server_on(vol_server *s, msg_type_t type, server_handler_fn fn, void *ctx);

// Start listening (non-blocking — spawns accept thread internally).
bool server_start(vol_server *s);

// Drain connections and shut down.
void server_stop(vol_server *s);

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

// Send a message to one client by id. Returns false if client not found.
bool server_send(vol_server *s, int client_id, msg_type_t type,
                 const uint8_t *payload, uint32_t len);

// Broadcast to every connected client.
void server_broadcast(vol_server *s, msg_type_t type,
                      const uint8_t *payload, uint32_t len);

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

int server_client_count(const vol_server *s);
