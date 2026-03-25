#include "greatest.h"
#include "server/protocol.h"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>

// ---------------------------------------------------------------------------
// Header encode/decode roundtrip
// ---------------------------------------------------------------------------

TEST test_encode_decode_roundtrip(void) {
  protocol_header_t orig = {
    .magic       = PROTOCOL_MAGIC,
    .msg_type    = MSG_CHUNK_REQ,
    .flags       = 0,
    .payload_len = 1024,
  };

  uint8_t buf[PROTOCOL_HEADER_SZ];
  protocol_encode_header(&orig, buf);

  protocol_header_t decoded;
  int rc = protocol_decode_header(buf, &decoded);

  ASSERT_EQ(PROTO_OK, rc);
  ASSERT_EQ(PROTOCOL_MAGIC, decoded.magic);
  ASSERT_EQ(orig.msg_type,    decoded.msg_type);
  ASSERT_EQ(orig.flags,       decoded.flags);
  ASSERT_EQ(orig.payload_len, decoded.payload_len);
  PASS();
}

TEST test_magic_bytes_correct(void) {
  protocol_header_t h = {
    .magic = PROTOCOL_MAGIC, .msg_type = MSG_PING, .flags = 0, .payload_len = 0,
  };
  uint8_t buf[PROTOCOL_HEADER_SZ];
  protocol_encode_header(&h, buf);

  // First four bytes must be 0x56 0x4F 0x4C 0x54 ("VOLT") in network order.
  ASSERT_EQ(0x56, buf[0]);
  ASSERT_EQ(0x4F, buf[1]);
  ASSERT_EQ(0x4C, buf[2]);
  ASSERT_EQ(0x54, buf[3]);
  PASS();
}

TEST test_bad_magic_rejected(void) {
  uint8_t buf[PROTOCOL_HEADER_SZ] = {0};
  buf[0] = 0xDE; buf[1] = 0xAD; buf[2] = 0xBE; buf[3] = 0xEF;

  protocol_header_t h;
  int rc = protocol_decode_header(buf, &h);
  ASSERT_EQ(PROTO_ERR_MAGIC, rc);
  PASS();
}

// Encode/decode for every defined message type.
TEST test_all_msg_types(void) {
  msg_type_t types[] = {
    MSG_PING, MSG_PONG,
    MSG_AUTH_REQ, MSG_AUTH_RESP,
    MSG_CHUNK_REQ, MSG_CHUNK_RESP,
    MSG_SEG_UPDATE, MSG_SEG_ACK,
    MSG_ANNOT_UPDATE, MSG_ANNOT_ACK,
    MSG_COMPUTE_REQ, MSG_COMPUTE_STATUS,
    MSG_ERROR,
  };
  size_t n = sizeof(types) / sizeof(types[0]);

  for (size_t i = 0; i < n; i++) {
    protocol_header_t orig = {
      .magic = PROTOCOL_MAGIC, .msg_type = types[i], .flags = 0, .payload_len = 0,
    };
    uint8_t buf[PROTOCOL_HEADER_SZ];
    protocol_encode_header(&orig, buf);

    protocol_header_t decoded;
    ASSERT_EQ(PROTO_OK, protocol_decode_header(buf, &decoded));
    ASSERT_EQ(types[i], decoded.msg_type);
  }
  PASS();
}

TEST test_payload_len_roundtrip(void) {
  uint32_t lens[] = {0, 1, 255, 256, 65535, 1048576};
  for (size_t i = 0; i < sizeof(lens) / sizeof(lens[0]); i++) {
    protocol_header_t orig = {
      .magic = PROTOCOL_MAGIC, .msg_type = MSG_CHUNK_RESP, .flags = 0, .payload_len = lens[i],
    };
    uint8_t buf[PROTOCOL_HEADER_SZ];
    protocol_encode_header(&orig, buf);

    protocol_header_t decoded;
    ASSERT_EQ(PROTO_OK, protocol_decode_header(buf, &decoded));
    ASSERT_EQ(lens[i], decoded.payload_len);
  }
  PASS();
}

// ---------------------------------------------------------------------------
// Socket send/recv roundtrip via socketpair
// ---------------------------------------------------------------------------

TEST test_send_recv_no_payload(void) {
  int fds[2];
  ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));

  int rc = protocol_send(fds[0], MSG_PING, NULL, 0);
  ASSERT_EQ(PROTO_OK, rc);

  protocol_header_t h;
  void *payload = NULL;
  rc = protocol_recv(fds[1], &h, &payload, 1000);
  ASSERT_EQ(PROTO_OK, rc);
  ASSERT_EQ(PROTOCOL_MAGIC, h.magic);
  ASSERT_EQ((uint16_t)MSG_PING, h.msg_type);
  ASSERT_EQ(0u, h.payload_len);
  ASSERT_EQ(NULL, payload);

  close(fds[0]);
  close(fds[1]);
  PASS();
}

TEST test_send_recv_with_payload(void) {
  int fds[2];
  ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));

  const char *data = "hello volatile";
  uint32_t len = (uint32_t)strlen(data);

  int rc = protocol_send(fds[0], MSG_AUTH_REQ, data, len);
  ASSERT_EQ(PROTO_OK, rc);

  protocol_header_t h;
  void *payload = NULL;
  rc = protocol_recv(fds[1], &h, &payload, 1000);
  ASSERT_EQ(PROTO_OK, rc);
  ASSERT_EQ(len, h.payload_len);
  ASSERT(payload != NULL);
  ASSERT_EQ(0, memcmp(data, payload, len));

  free(payload);
  close(fds[0]);
  close(fds[1]);
  PASS();
}

#include <sys/socket.h>

TEST test_recv_timeout(void) {
  int fds[2];
  ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds));

  protocol_header_t h;
  void *payload = NULL;
  // Nothing sent — should time out quickly.
  int rc = protocol_recv(fds[1], &h, &payload, 50);
  ASSERT_EQ(PROTO_ERR_TIMEOUT, rc);

  close(fds[0]);
  close(fds[1]);
  PASS();
}

// ---------------------------------------------------------------------------
// Suite + main
// ---------------------------------------------------------------------------

SUITE(protocol_suite) {
  RUN_TEST(test_encode_decode_roundtrip);
  RUN_TEST(test_magic_bytes_correct);
  RUN_TEST(test_bad_magic_rejected);
  RUN_TEST(test_all_msg_types);
  RUN_TEST(test_payload_len_roundtrip);
  RUN_TEST(test_send_recv_no_payload);
  RUN_TEST(test_send_recv_with_payload);
  RUN_TEST(test_recv_timeout);
}

GREATEST_MAIN_DEFS();

int main(int argc, char **argv) {
  GREATEST_MAIN_BEGIN();
  RUN_SUITE(protocol_suite);
  GREATEST_MAIN_END();
}
