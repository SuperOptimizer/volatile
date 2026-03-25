#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "core/vol.h"
#include "server/protocol.h"

// ---------------------------------------------------------------------------
// Progressive chunk streaming — coarsest level first
//
// The streamer iterates over all OME-Zarr pyramid levels from the highest
// (most downsampled) to level 0 (full resolution).  For each level it
// enumerates every chunk that overlaps the requested region and emits one
// stream_packet per chunk.
//
// Wire messages:
//   MSG_CHUNK_STREAM_START — sent once before the first DATA packet.
//   MSG_CHUNK_STREAM_DATA  — one per chunk, carries level + compressed bytes.
//   MSG_CHUNK_STREAM_END   — sent after the last chunk of all levels.
//
// The client can stop consuming at any time; the server side has no
// persistent state beyond the chunk_streamer object.
// ---------------------------------------------------------------------------

// New message types (appended to the protocol enum)
#define MSG_CHUNK_STREAM_START  ((msg_type_t)13)
#define MSG_CHUNK_STREAM_DATA   ((msg_type_t)14)
#define MSG_CHUNK_STREAM_END    ((msg_type_t)15)

// Payload for MSG_CHUNK_STREAM_START (all fields big-endian on wire, but
// we handle endian in the caller; here we use host order for simplicity).
typedef struct {
  int64_t z0, y0, x0;   // requested region start (voxels at level 0)
  int64_t z1, y1, x1;   // requested region end   (exclusive)
  int32_t num_levels;    // how many levels will be streamed
} stream_start_payload;

// One packet produced by chunk_streamer_next.
typedef struct {
  int      level;             // pyramid level (high = coarse, 0 = full-res)
  uint8_t *data;              // malloc'd compressed chunk data — caller frees
  size_t   size;              // bytes in data
  bool     is_last_for_level; // true on the last chunk of this level
} stream_packet;

typedef struct chunk_streamer chunk_streamer;

// Create a streamer for the given region (z0,y0,x0)-(z1,y1,x1) at level 0 voxels.
// Iterations begin at the highest available pyramid level and descend to 0.
chunk_streamer *chunk_streamer_new(volume *vol,
                                   int64_t z0, int64_t y0, int64_t x0,
                                   int64_t z1, int64_t y1, int64_t x1);
void chunk_streamer_free(chunk_streamer *s);

// Fill *out with the next packet.  out->data is malloc'd; caller must free it.
// Returns true while packets remain, false when streaming is complete.
bool chunk_streamer_next(chunk_streamer *s, stream_packet *out);

// How many levels will be streamed.
int chunk_streamer_num_levels(const chunk_streamer *s);

// Current level being iterated (only valid while chunk_streamer_next returns true).
int chunk_streamer_current_level(const chunk_streamer *s);
