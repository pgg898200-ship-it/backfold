/* EventFold native stream filter. Build with make. */
#include "../eventfold.h"
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#define ef_read _read
#else
#include <unistd.h>
#define ef_read read
#endif

#define BLOCK_SIZE 32768u
#define DEFAULT_LIMIT (16u * 1024u * 1024u)
static const unsigned char magic[] = {'E','F','L','1'};

typedef struct {
    unsigned char *data;
    size_t len, cap;
} Buffer;

static void fail(const char *message) {
    fprintf(stderr, "eventfold: %s\n", message);
    exit(1);
}

static void reserve(Buffer *b, size_t need) {
    if (need <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < need) {
        if (cap > SIZE_MAX / 2) { cap = need; break; }
        cap *= 2;
    }
    unsigned char *p = (unsigned char *)realloc(b->data, cap);
    if (!p) fail("allocation failed");
    b->data = p;
    b->cap = cap;
}

static void send_bytes(const unsigned char *p, size_t n) {
    if ((n && fwrite(p, 1, n, stdout) != n) || fflush(stdout)) fail("output error");
}

static size_t read_block(unsigned char *block) {
    for (;;) {
        long n = (long)ef_read(0, block, BLOCK_SIZE);
        if (n >= 0) return (size_t)n;
        if (errno != EINTR) fail("input error");
    }
}

static void advance(Buffer *previous, Buffer *current) {
    Buffer swap = *previous;
    *previous = *current;
    *current = swap;
    current->len = 0;
}

static void emit_record(Buffer *prev, Buffer *cur, Buffer *encoded) {
    size_t bound = ef_bound(cur->len);
    if (!bound) fail("frame too large");
    reserve(encoded, bound);
    size_t n = ef_encode(prev->data, prev->len, cur->data, cur->len,
                         encoded->data, encoded->cap);
    if (!n) fail("cannot encode frame");
    send_bytes(encoded->data, n);
    advance(prev, cur);
}

/* A CR may be the first byte of CRLF. Preserve the byte-level delimiter. */
static void pack_stream(size_t limit) {
    Buffer prev = {0}, cur = {0}, encoded = {0};
    unsigned char block[BLOCK_SIZE];
    int line_has_data = 0, pending_cr = 0, cr_ends_frame = 0;
    size_t n;
    send_bytes(magic, sizeof magic);
    while ((n = read_block(block)) != 0) {
        for (size_t i = 0; i < n; ++i) {
            unsigned char c = block[i];
            if (pending_cr) {
                pending_cr = 0;
                if (c == '\n') {
                    if (cur.len == limit) fail("SSE frame exceeds --max-frame");
                    reserve(&cur, cur.len + 1);
                    cur.data[cur.len++] = c;
                    if (cr_ends_frame) emit_record(&prev, &cur, &encoded);
                    continue;
                }
                if (cr_ends_frame) emit_record(&prev, &cur, &encoded);
            }
            if (cur.len == limit) fail("SSE frame exceeds --max-frame");
            reserve(&cur, cur.len + 1);
            cur.data[cur.len++] = c;
            if (c == '\r') {
                pending_cr = 1;
                cr_ends_frame = !line_has_data;
                line_has_data = 0;
            } else if (c == '\n') {
                if (!line_has_data) emit_record(&prev, &cur, &encoded);
                line_has_data = 0;
            } else {
                line_has_data = 1;
            }
        }
    }
    if (cur.len) emit_record(&prev, &cur, &encoded);
    free(prev.data); free(cur.data); free(encoded.data);
}

static void unpack_stream(size_t limit) {
    Buffer wire = {0}, prev = {0}, cur = {0};
    unsigned char block[BLOCK_SIZE];
    size_t n, hdr = 0;
    /* Decoders also accept non-optimal patches (tag plus six size_t varints). */
    const size_t wire_limit = limit + 1 + 6 * ((sizeof(size_t) * CHAR_BIT + 6) / 7);
    while ((n = read_block(block)) != 0) {
        size_t start = 0;
        while (hdr < sizeof magic && start < n)
            if (block[start++] != magic[hdr++]) fail("invalid EFL1 header");
        if (start == n) continue;
        reserve(&wire, wire.len + n - start);
        memcpy(wire.data + wire.len, block + start, n - start);
        wire.len += n - start;
        size_t pos = 0;
        while (pos < wire.len) {
            size_t out_len = 0, consumed = 0;
            size_t available = cur.cap < limit ? cur.cap : limit;
            int rc = ef_decode(prev.data, prev.len, wire.data + pos, wire.len - pos,
                               cur.data, available, &out_len, &consumed);
            if (rc == EF_MORE) break;
            if (out_len > limit) fail("decoded frame exceeds --max-frame");
            if (rc == EF_NOSPACE) {
                reserve(&cur, out_len);
                continue;
            }
            if (rc != EF_OK || !consumed) fail("invalid EventFold record");
            cur.len = out_len;
            send_bytes(cur.data, cur.len);
            advance(&prev, &cur);
            pos += consumed;
        }
        wire.len -= pos;
        if (wire.len > wire_limit) fail("encoded frame exceeds --max-frame");
        if (pos && wire.len) memmove(wire.data, wire.data + pos, wire.len);
    }
    if (hdr != sizeof magic || wire.len) fail("truncated EventFold stream");
    free(prev.data); free(cur.data); free(wire.data);
}

int main(int argc, char **argv) {
    size_t limit = DEFAULT_LIMIT;
    if (argc == 4 && strcmp(argv[2], "--max-frame") == 0) {
        char *end;
        errno = 0;
        unsigned long long n = strtoull(argv[3], &end, 10);
        if (argv[3][0] < '0' || argv[3][0] > '9' || *end || errno ||
            n > SIZE_MAX - BLOCK_SIZE - 64 || !ef_bound((size_t)n))
            fail("invalid --max-frame size");
        limit = (size_t)n;
    } else if (argc != 2) {
        fprintf(stderr, "usage: eventfold {pack|unpack} [--max-frame BYTES] < input > output\n");
        return 2;
    }
#ifdef _WIN32
    _setmode(0, _O_BINARY);
    _setmode(1, _O_BINARY);
#endif
    if (strcmp(argv[1], "pack") == 0) pack_stream(limit);
    else if (strcmp(argv[1], "unpack") == 0) unpack_stream(limit);
    else fail("expected pack or unpack");
    return 0;
}
