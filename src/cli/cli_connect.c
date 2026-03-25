#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE   /* htobe64 */

#include "cli/cli_connect.h"
#include "core/log.h"
#include "server/protocol.h"
#include "server/db.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <endian.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>

// ---------------------------------------------------------------------------
// Chunk request encoding (must match cli_serve.c layout)
// ---------------------------------------------------------------------------

#define CHUNK_REQ_SIZE 92

static void chunk_req_encode(uint8_t buf[CHUNK_REQ_SIZE], const char *volume_id,
                              int32_t level, int64_t iz, int64_t iy, int64_t ix) {
  memset(buf, 0, CHUNK_REQ_SIZE);
  strncpy((char *)buf, volume_id, 63);

  uint32_t lv = htonl((uint32_t)level);
  memcpy(buf + 64, &lv, 4);

  uint64_t viz = htobe64((uint64_t)iz);
  uint64_t viy = htobe64((uint64_t)iy);
  uint64_t vix = htobe64((uint64_t)ix);
  memcpy(buf + 68, &viz, 8);
  memcpy(buf + 76, &viy, 8);
  memcpy(buf + 84, &vix, 8);
}

// ---------------------------------------------------------------------------
// TCP connect helper
// ---------------------------------------------------------------------------

static int tcp_connect(const char *host, int port) {
  struct addrinfo hints = { .ai_family = AF_INET, .ai_socktype = SOCK_STREAM };
  char port_str[16];
  snprintf(port_str, sizeof(port_str), "%d", port);

  struct addrinfo *res = NULL;
  if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) {
    fprintf(stderr, "connect: could not resolve %s\n", host);
    return -1;
  }

  int fd = socket(res->ai_family, res->ai_socktype, 0);
  if (fd < 0) { freeaddrinfo(res); return -1; }

  if (connect(fd, res->ai_addr, res->ai_addrlen) != 0) {
    fprintf(stderr, "connect: %s:%d: %s\n", host, port, strerror(errno));
    freeaddrinfo(res); close(fd); return -1;
  }

  freeaddrinfo(res);
  return fd;
}

// ---------------------------------------------------------------------------
// cmd_connect
// ---------------------------------------------------------------------------

int cmd_connect(int argc, char **argv) {
  if (argc < 1) {
    fputs("usage: volatile connect <host:port> [--volume <id>]\n", stderr);
    return 1;
  }

  // Parse host:port.
  char host[256] = "127.0.0.1";
  int  port      = 9876;
  const char *volume_id = NULL;

  char hostport[300];
  strncpy(hostport, argv[0], sizeof(hostport) - 1);
  char *colon = strrchr(hostport, ':');
  if (colon) {
    *colon = '\0';
    port = atoi(colon + 1);
    strncpy(host, hostport, sizeof(host) - 1);
  } else {
    strncpy(host, hostport, sizeof(host) - 1);
  }

  for (int i = 1; i < argc - 1; i++) {
    if (strcmp(argv[i], "--volume") == 0) volume_id = argv[i + 1];
  }

  printf("connecting to %s:%d...\n", host, port);
  int fd = tcp_connect(host, port);
  if (fd < 0) return 1;
  printf("connected\n\n");

  // --- PING / PONG ---
  printf("PING... ");
  fflush(stdout);
  if (protocol_send(fd, MSG_PING, NULL, 0) != PROTO_OK) {
    fputs("send failed\n", stderr); close(fd); return 1;
  }

  protocol_header_t hdr = {0};
  void *payload = NULL;
  int rc = protocol_recv(fd, &hdr, &payload, 3000);
  free(payload); payload = NULL;
  if (rc != PROTO_OK || hdr.msg_type != MSG_PONG) {
    fprintf(stderr, "no PONG (rc=%d)\n", rc); close(fd); return 1;
  }
  puts("PONG");

  // --- Chunk request ---
  if (volume_id) {
    printf("\nrequesting chunk [0,0,0] level 0 from volume '%s'...\n", volume_id);
    uint8_t req_buf[CHUNK_REQ_SIZE];
    chunk_req_encode(req_buf, volume_id, 0, 0, 0, 0);

    if (protocol_send(fd, MSG_CHUNK_REQ, req_buf, CHUNK_REQ_SIZE) != PROTO_OK) {
      fputs("chunk request send failed\n", stderr);
    } else {
      rc = protocol_recv(fd, &hdr, &payload, 5000);
      if (rc == PROTO_OK && hdr.msg_type == MSG_CHUNK_RESP) {
        printf("received chunk: %u bytes\n", hdr.payload_len);
      } else if (rc == PROTO_OK && hdr.msg_type == MSG_ERROR) {
        char err[256] = {0};
        if (payload && hdr.payload_len < sizeof(err))
          memcpy(err, payload, hdr.payload_len);
        fprintf(stderr, "server error: %s\n", err);
      } else {
        fprintf(stderr, "unexpected response (rc=%d type=%d)\n", rc, hdr.msg_type);
      }
      free(payload); payload = NULL;
    }
  }

  close(fd);
  puts("\ndisconnected");
  return 0;
}
