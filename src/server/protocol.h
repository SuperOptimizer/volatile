#pragma once
#include <stdint.h>
#include <stddef.h>

// ---------------------------------------------------------------------------
// Wire format: 12-byte fixed header + variable payload
// All multi-byte fields in network byte order (big-endian).
// ---------------------------------------------------------------------------

#define PROTOCOL_MAGIC     UINT32_C(0x564F4C54)  // "VOLT"
#define PROTOCOL_HEADER_SZ 12

typedef enum {
  MSG_PING = 0,
  MSG_PONG,
  MSG_AUTH_REQ,
  MSG_AUTH_RESP,
  MSG_CHUNK_REQ,
  MSG_CHUNK_RESP,
  MSG_SEG_UPDATE,
  MSG_SEG_ACK,
  MSG_ANNOT_UPDATE,
  MSG_ANNOT_ACK,
  MSG_COMPUTE_REQ,
  MSG_COMPUTE_STATUS,
  MSG_ERROR,
} msg_type_t;

typedef struct {
  uint32_t magic;        // must equal PROTOCOL_MAGIC
  uint16_t msg_type;     // msg_type_t
  uint16_t flags;        // reserved, set to 0
  uint32_t payload_len;  // bytes following header
} protocol_header_t;

// ---------------------------------------------------------------------------
// Return codes
// ---------------------------------------------------------------------------
#define PROTO_OK            0
#define PROTO_ERR_MAGIC    -1
#define PROTO_ERR_IO       -2
#define PROTO_ERR_TIMEOUT  -3
#define PROTO_ERR_ALLOC    -4
#define PROTO_ERR_TRUNC    -5

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

// Encode header into buf (must be >= PROTOCOL_HEADER_SZ bytes).
void protocol_encode_header(const protocol_header_t *header, uint8_t buf[PROTOCOL_HEADER_SZ]);

// Decode header from buf. Returns PROTO_OK or PROTO_ERR_MAGIC.
int protocol_decode_header(const uint8_t buf[PROTOCOL_HEADER_SZ], protocol_header_t *header);

// ---------------------------------------------------------------------------
// Socket I/O
// ---------------------------------------------------------------------------

// Write header + payload to fd. payload may be NULL if len == 0.
// Returns PROTO_OK or PROTO_ERR_IO.
int protocol_send(int fd, msg_type_t msg_type, const void *payload, uint32_t len);

// Read one message from fd into header_out and *payload_out.
// On success, *payload_out is malloc'd (caller must free); NULL if payload_len==0.
// timeout_ms <= 0 means block indefinitely.
// Returns PROTO_OK, PROTO_ERR_IO, PROTO_ERR_TIMEOUT, PROTO_ERR_MAGIC, or PROTO_ERR_ALLOC.
int protocol_recv(int fd, protocol_header_t *header_out, void **payload_out, int timeout_ms);
