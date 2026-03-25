#include "server/protocol.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/select.h>
#include <sys/time.h>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

// Write exactly len bytes to fd; returns 0 on success, PROTO_ERR_IO on failure.
static int write_all(int fd, const void *buf, size_t len) {
  const uint8_t *p = (const uint8_t *)buf;
  while (len > 0) {
    ssize_t n = write(fd, p, len);
    if (n <= 0) return PROTO_ERR_IO;
    p += n;
    len -= (size_t)n;
  }
  return PROTO_OK;
}

// Read exactly len bytes from fd; returns 0 on success, PROTO_ERR_IO on failure.
static int read_all(int fd, void *buf, size_t len) {
  uint8_t *p = (uint8_t *)buf;
  while (len > 0) {
    ssize_t n = read(fd, p, len);
    if (n <= 0) return PROTO_ERR_IO;
    p += n;
    len -= (size_t)n;
  }
  return PROTO_OK;
}

// Wait up to timeout_ms for fd to become readable.
// Returns PROTO_OK, PROTO_ERR_TIMEOUT, or PROTO_ERR_IO.
static int wait_readable(int fd, int timeout_ms) {
  if (timeout_ms <= 0) return PROTO_OK;  // caller will block in read_all

  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd, &fds);

  struct timeval tv = {
    .tv_sec  = timeout_ms / 1000,
    .tv_usec = (timeout_ms % 1000) * 1000,
  };

  int rc = select(fd + 1, &fds, NULL, NULL, &tv);
  if (rc < 0) return PROTO_ERR_IO;
  if (rc == 0) return PROTO_ERR_TIMEOUT;
  return PROTO_OK;
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

void protocol_encode_header(const protocol_header_t *h, uint8_t buf[PROTOCOL_HEADER_SZ]) {
  uint32_t magic       = htonl(h->magic);
  uint16_t msg_type    = htons(h->msg_type);
  uint16_t flags       = htons(h->flags);
  uint32_t payload_len = htonl(h->payload_len);

  memcpy(buf + 0, &magic,       4);
  memcpy(buf + 4, &msg_type,    2);
  memcpy(buf + 6, &flags,       2);
  memcpy(buf + 8, &payload_len, 4);
}

int protocol_decode_header(const uint8_t buf[PROTOCOL_HEADER_SZ], protocol_header_t *h) {
  uint32_t magic;
  uint16_t msg_type, flags;
  uint32_t payload_len;

  memcpy(&magic,       buf + 0, 4);
  memcpy(&msg_type,    buf + 4, 2);
  memcpy(&flags,       buf + 6, 2);
  memcpy(&payload_len, buf + 8, 4);

  h->magic       = ntohl(magic);
  h->msg_type    = ntohs(msg_type);
  h->flags       = ntohs(flags);
  h->payload_len = ntohl(payload_len);

  if (h->magic != PROTOCOL_MAGIC) return PROTO_ERR_MAGIC;
  return PROTO_OK;
}

// ---------------------------------------------------------------------------
// Socket I/O
// ---------------------------------------------------------------------------

int protocol_send(int fd, msg_type_t msg_type, const void *payload, uint32_t len) {
  protocol_header_t h = {
    .magic       = PROTOCOL_MAGIC,
    .msg_type    = (uint16_t)msg_type,
    .flags       = 0,
    .payload_len = len,
  };

  uint8_t buf[PROTOCOL_HEADER_SZ];
  protocol_encode_header(&h, buf);

  int rc = write_all(fd, buf, PROTOCOL_HEADER_SZ);
  if (rc != PROTO_OK) return rc;

  if (len > 0 && payload != NULL) {
    rc = write_all(fd, payload, len);
    if (rc != PROTO_OK) return rc;
  }

  return PROTO_OK;
}

int protocol_recv(int fd, protocol_header_t *header_out, void **payload_out, int timeout_ms) {
  int rc = wait_readable(fd, timeout_ms);
  if (rc != PROTO_OK) return rc;

  uint8_t buf[PROTOCOL_HEADER_SZ];
  rc = read_all(fd, buf, PROTOCOL_HEADER_SZ);
  if (rc != PROTO_OK) return rc;

  rc = protocol_decode_header(buf, header_out);
  if (rc != PROTO_OK) return rc;

  if (header_out->payload_len == 0) {
    *payload_out = NULL;
    return PROTO_OK;
  }

  void *payload = malloc(header_out->payload_len);
  if (!payload) return PROTO_ERR_ALLOC;

  rc = read_all(fd, payload, header_out->payload_len);
  if (rc != PROTO_OK) {
    free(payload);
    *payload_out = NULL;
    return rc;
  }

  *payload_out = payload;
  return PROTO_OK;
}
