/*
 * fuzz_json.c — AFL++ / libFuzzer harness for the volatile JSON parser.
 *
 * AFL++ usage:
 *   mkdir -p corpus/json && echo '{}' > corpus/json/seed.json
 *   afl-fuzz -i corpus/json -o findings/json -- ./fuzz_json
 *
 * libFuzzer usage (build with -fsanitize=fuzzer,address):
 *   ./fuzz_json corpus/json/
 */

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "../../src/core/json.h"

/* -------------------------------------------------------------------------
 * Shared fuzz logic: parse the input as JSON, then walk the resulting tree
 * to exercise accessor code paths.  Always frees before returning.
 * -------------------------------------------------------------------------*/
static void do_fuzz(const uint8_t *data, size_t size)
{
    /* json_parse expects a null-terminated string. */
    char *buf = malloc(size + 1);
    if (!buf)
        return;
    memcpy(buf, data, size);
    buf[size] = '\0';

    json_value *root = json_parse(buf);
    free(buf);

    if (!root)
        return;

    /* Exercise type-check and scalar accessor paths. */
    json_type t = json_typeof(root);
    switch (t) {
    case JSON_BOOL:
        json_get_bool(root, false);
        break;
    case JSON_NUMBER:
        json_get_number(root, 0.0);
        json_get_int(root, 0);
        break;
    case JSON_STRING:
        json_get_str(root);
        break;
    case JSON_ARRAY: {
        size_t len = json_array_len(root);
        for (size_t i = 0; i < len; i++) {
            const json_value *elem = json_array_get(root, i);
            if (elem)
                json_get_number(elem, 0.0); /* harmless on non-numbers */
        }
        break;
    }
    case JSON_OBJECT: {
        size_t len = json_object_len(root);
        (void)len;
        /* Probe a few common key names to hit json_object_get hash paths. */
        static const char *const probe_keys[] = {
            "id", "name", "value", "type", "data", "x", "y", "z",
        };
        for (size_t k = 0; k < sizeof(probe_keys)/sizeof(probe_keys[0]); k++)
            json_object_get(root, probe_keys[k]);
        break;
    }
    default:
        break;
    }

    json_free(root);
}

/* -------------------------------------------------------------------------
 * libFuzzer entry point (also compiled when using afl-cc with LLVMFuzzerTestOneInput
 * defined — AFL++ supports this interface natively via afl-cc -fsanitize=fuzzer).
 * -------------------------------------------------------------------------*/
int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    do_fuzz(data, size);
    return 0;
}

/* -------------------------------------------------------------------------
 * AFL++ stdin entry point — used when building without libFuzzer.
 * Only compiled when VOLATILE_FUZZER=afl++ and we are NOT using the
 * LLVMFuzzerTestOneInput shim provided by afl-cc.
 * -------------------------------------------------------------------------*/
#if defined(AFL_MAIN)
#include <stdio.h>

int main(void)
{
    uint8_t buf[1 << 20]; /* 1 MiB max input */
    size_t  n = fread(buf, 1, sizeof(buf), stdin);
    do_fuzz(buf, n);
    return 0;
}
#endif /* AFL_MAIN */
