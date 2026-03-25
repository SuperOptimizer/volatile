/*
 * fuzz_compress4d.c — AFL++ / libFuzzer harness for the volatile ANS codec.
 *
 * Fuzz strategy: treat the raw input bytes as data to compress, then verify
 * that decode(encode(data)) == data.  Also fuzz the decode path directly by
 * feeding raw bytes as a "compressed stream" to catch any parser bugs.
 *
 * AFL++ usage:
 *   mkdir -p corpus/compress && dd if=/dev/urandom bs=256 count=1 > corpus/compress/seed
 *   afl-fuzz -i corpus/compress -o findings/compress -- ./fuzz_compress4d
 *
 * libFuzzer usage:
 *   ./fuzz_compress4d corpus/compress/
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/core/compress4d.h"

/* -------------------------------------------------------------------------
 * Shared fuzz logic.
 *
 * The input is split at a 1-byte header:
 *   data[0] bit 0 == 0  ->  roundtrip test  (encode then decode)
 *   data[0] bit 0 == 1  ->  raw decode test (feed data[1..] directly)
 * -------------------------------------------------------------------------*/
static void do_fuzz(const uint8_t *data, size_t size)
{
    if (size < 2)
        return;

    const int mode = data[0] & 0x01;
    const uint8_t *payload = data + 1;
    const size_t   payload_len = size - 1;

    if (mode == 0) {
        /* -----------------------------------------------------------------
         * Roundtrip: build a table from the payload, encode it, decode it,
         * and verify the output matches the original.
         * ----------------------------------------------------------------- */
        ans_table *tbl = ans_table_build(payload, payload_len);
        if (!tbl)
            return;

        size_t   enc_len = 0;
        uint8_t *enc = ans_encode(tbl, payload, payload_len, &enc_len);
        if (!enc) {
            ans_table_free(tbl);
            return;
        }

        uint8_t *dec = ans_decode(tbl, enc, enc_len, payload_len);
        if (dec) {
            /* Correctness check: decoded bytes must match original. */
            if (memcmp(dec, payload, payload_len) != 0) {
                /* If we ever reach here the codec has a bug. */
                __builtin_trap();
            }
            free(dec);
        }

        free(enc);
        ans_table_free(tbl);

    } else {
        /* -----------------------------------------------------------------
         * Raw decode: treat the payload as a compressed stream and attempt
         * to decode it into an arbitrarily-sized output buffer.  The goal
         * is to find crashes or reads out of bounds inside ans_decode().
         *
         * We derive orig_len from the first two bytes of the payload so the
         * fuzzer can explore the full range without being artificially capped.
         * ----------------------------------------------------------------- */
        if (payload_len < 3)
            return;

        /* Clamp orig_len to a sane value to avoid huge allocations. */
        const size_t orig_len =
            (size_t)((payload[0] << 8) | payload[1]) + 1;
        const size_t capped = orig_len < 4096 ? orig_len : 4096;

        /* Build a dummy uniform table (all symbols equally likely). */
        uint32_t counts[256];
        for (int i = 0; i < 256; i++)
            counts[i] = 1;
        ans_table *tbl = ans_table_from_counts(counts);
        if (!tbl)
            return;

        uint8_t *out = ans_decode(tbl, payload + 2, payload_len - 2, capped);
        free(out); /* NULL is fine */
        ans_table_free(tbl);
    }
}

/* -------------------------------------------------------------------------
 * libFuzzer entry point.
 * -------------------------------------------------------------------------*/
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    do_fuzz(data, size);
    return 0;
}

/* -------------------------------------------------------------------------
 * AFL++ stdin entry point.
 * -------------------------------------------------------------------------*/
#if defined(AFL_MAIN)
#include <stdio.h>

int main(void)
{
    uint8_t buf[1 << 20];
    size_t  n = fread(buf, 1, sizeof(buf), stdin);
    do_fuzz(buf, n);
    return 0;
}
#endif /* AFL_MAIN */
