/*
 * SPDX-License-Identifier: MIT
 * MIT License
 *
 * Copyright (c) 2026 PGG Engineering
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
/* EventFold v1. Include this header in any C99 or C++ translation unit. */
#ifndef EVENTFOLD_H
#define EVENTFOLD_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#if CHAR_BIT != 8
#error EventFold requires 8-bit bytes
#endif

enum ef_status { EF_OK = 0, EF_MORE = 1, EF_NOSPACE = 2, EF_INVALID = 3 };

static inline size_t ef_var_size(size_t value) {
    size_t bytes = 1;
    while (value >= 128) {
        value >>= 7;
        ++bytes;
    }
    return bytes;
}
static inline size_t ef_head_size(size_t len) {
    return len < 63 ? 1 : 1 + ef_var_size(len - 63);
}

/* Maximum encoded length for a frame of n bytes, or 0 on overflow. */
static inline size_t ef_bound(size_t n) {
    size_t header = ef_head_size(n);
    return n > (size_t)-1 - header ? 0 : n + header;
}

static inline size_t ef_put_var(unsigned char *out, size_t value) {
    size_t pos = 0;
    while (value >= 128) {
        out[pos++] = (unsigned char)((value & 127) | 128);
        value >>= 7;
    }
    out[pos++] = (unsigned char)value;
    return pos;
}

static inline size_t ef_put_head(unsigned char *out, unsigned tag, size_t len) {
    if (len < 63) {
        out[0] = (unsigned char)(tag + len);
        return 1;
    }
    out[0] = (unsigned char)(tag + 63);
    return 1 + ef_put_var(out + 1, len - 63);
}

/* memcpy permits unaligned inputs and avoids aliasing assumptions. */
static inline int ef_words_equal(const unsigned char *a, const unsigned char *b) {
    uint64_t left, right;
    memcpy(&left, a, sizeof(left));
    memcpy(&right, b, sizeof(right));
    return left == right;
}

static inline size_t ef_common_front(const unsigned char *a, const unsigned char *b,
                                     size_t limit) {
    size_t pos = 0;
    while (limit - pos >= 8 && ef_words_equal(a + pos, b + pos)) pos += 8;
    while (pos < limit && a[pos] == b[pos]) ++pos;
    return pos;
}

static inline size_t ef_common_back(const unsigned char *a, size_t an,
                                    const unsigned char *b, size_t bn, size_t limit) {
    size_t count = 0;
    while (limit - count >= 8 && ef_words_equal(a + an - count - 8, b + bn - count - 8))
        count += 8;
    while (count < limit && a[an - count - 1] == b[bn - count - 1]) ++count;
    return count;
}

/* Search at most three small neighborhoods for an unchanged interior span.
 * Only compare full words in range. Extend the first useful anchor. */
static inline size_t ef_find_middle(const unsigned char *old, size_t on,
                                    const unsigned char *cur, size_t cn,
                                    size_t *old_skip, size_t *new_skip) {
    size_t last = cn - 8;
    size_t anchors[3] = {last / 2, last / 4, last - last / 4};
    size_t k;
    for (k = 0; k < 3; ++k) {
        size_t anchor = anchors[k];
        size_t lo = anchor > 32 ? anchor - 32 : 0;
        size_t hi = on - 8;
        const unsigned char *candidate;
        if (lo > hi) continue;
        if (hi > anchor && hi - anchor > 32) hi = anchor + 32;
        while (lo <= hi) {
            size_t at, front, back, limit;
            candidate = (const unsigned char *)memchr(old + lo, cur[anchor], hi - lo + 1);
            if (!candidate) break;
            at = lo + (size_t)(candidate - (old + lo));
            if (!ef_words_equal(candidate, cur + anchor)) {
                lo = at + 1;
                continue;
            }
            limit = at < anchor ? at : anchor;
            back = ef_common_back(old, at, cur, anchor, limit);
            limit = on - at < cn - anchor ? on - at : cn - anchor;
            front = 8 + ef_common_front(candidate + 8, cur + anchor + 8, limit - 8);
            if (back + front >= 16) {
                *old_skip = at - back;
                *new_skip = anchor - back;
                return back + front;
            }
            break;
        }
    }
    return 0;
}

/* Returns the number of encoded bytes, or 0 on invalid arguments, insufficient
 * capacity, or unrepresentable literal bound. out must not overlap prev/cur.
 * Initial previous frame is (NULL,0). Empty frames are supported. */
static inline size_t ef_encode(const unsigned char *prev, size_t pn,
                               const unsigned char *cur, size_t cn,
                               unsigned char *out, size_t cap) {
    size_t limit, prefix, suffix, inserted, overhead, literal, best, pos;
    if ((!prev && pn) || (!cur && cn) || !out || !cap) return 0;
    literal = ef_bound(cn);
    if (!literal) return 0;
    limit = pn < cn ? pn : cn;
    prefix = ef_common_front(prev, cur, limit);
    if (prefix == cn && prefix == pn) {
        out[0] = 0x80;
        return 1;
    }
    suffix = ef_common_back(prev, pn, cur, cn, limit - prefix);
    inserted = cn - prefix - suffix;
    overhead = ef_head_size(inserted) + ef_var_size(prefix) + ef_var_size(suffix);
    best = literal - inserted > overhead ? inserted + overhead : literal;
    if (prefix + suffix >= 8 && inserted >= 24 && pn - prefix - suffix >= 24) {
        size_t old_skip = 0, left = 0;
        size_t middle = ef_find_middle(prev + prefix, pn - prefix - suffix,
                                       cur + prefix, inserted, &old_skip, &left);
        if (middle) {
            size_t right = inserted - left - middle;
            size_t payload = inserted - middle;
            size_t header = 1 + ef_var_size(prefix) + ef_var_size(suffix)
                + ef_var_size(old_skip) + ef_var_size(middle)
                + ef_var_size(left) + ef_var_size(right);
            if (best - payload > header) {
                if (cap < header + payload) return 0;
                out[0] = 0x81;
                pos = 1;
                pos += ef_put_var(out + pos, prefix);
                pos += ef_put_var(out + pos, suffix);
                pos += ef_put_var(out + pos, old_skip);
                pos += ef_put_var(out + pos, middle);
                pos += ef_put_var(out + pos, left);
                pos += ef_put_var(out + pos, right);
                if (left) memcpy(out + pos, cur + prefix, left);
                if (right) memcpy(out + pos + left, cur + prefix + left + middle, right);
                return pos + payload;
            }
        }
    }
    if (best < literal) {
        if (cap < inserted + overhead) return 0;
        pos = ef_put_head(out, 0x40, inserted);
        pos += ef_put_var(out + pos, prefix);
        pos += ef_put_var(out + pos, suffix);
        if (inserted) memcpy(out + pos, cur + prefix, inserted);
        return pos + inserted;
    }
    if (cap < literal) return 0;
    pos = ef_put_head(out, 0, cn);
    if (cn) memcpy(out + pos, cur, cn);
    return literal;
}

/* Accept only minimal unsigned LEB128 that fits in size_t. */
static inline int ef_get_var(const unsigned char *wire, size_t wn,
                             size_t *pos, size_t *value) {
    size_t result = 0;
    unsigned shift = 0;
    while (*pos < wn) {
        unsigned byte = wire[(*pos)++];
        size_t group = byte & 127;
        if (group > ((size_t)-1 >> shift)) return EF_INVALID;
        result |= group << shift;
        if (!(byte & 128)) {
            if (shift && !group) return EF_INVALID;
            *value = result;
            return EF_OK;
        }
        if (sizeof(size_t) * CHAR_BIT - shift <= 7) return EF_INVALID;
        shift += 7;
    }
    return EF_MORE;
}

/* Decode a two-splice record. Its three references into prev must be ordered
 * and disjoint. Validate the entire record before copying any bytes. */
static inline int ef_decode_middle(const unsigned char *prev, size_t pn,
                                   const unsigned char *wire, size_t wn,
                                   unsigned char *out, size_t cap,
                                   size_t *out_len, size_t *consumed) {
    size_t field[6], i, pos = 1;
    size_t prefix, suffix, skip, middle, left, right, kept, payload, decoded;
    for (i = 0; i < 6; ++i) {
        int status = ef_get_var(wire, wn, &pos, &field[i]);
        if (status != EF_OK) return status;
    }
    prefix = field[0]; suffix = field[1]; skip = field[2]; middle = field[3];
    left = field[4]; right = field[5];
    if (prefix > pn || suffix > pn - prefix || skip > pn - prefix - suffix
        || middle > pn - prefix - suffix - skip) return EF_INVALID;
    kept = prefix + suffix + middle;
    if (left > (size_t)-1 - right) return EF_INVALID;
    payload = left + right;
    if (payload > (size_t)-1 - kept || payload > (size_t)-1 - pos) return EF_INVALID;
    if (payload > wn - pos) return EF_MORE;
    decoded = kept + payload;
    *out_len = decoded;
    *consumed = pos + payload;
    if (cap < decoded) return EF_NOSPACE;
    if (prefix) memcpy(out, prev, prefix);
    if (left) memcpy(out + prefix, wire + pos, left);
    if (middle) memcpy(out + prefix + left, prev + prefix + skip, middle);
    if (right) memcpy(out + prefix + left + middle, wire + pos + left, right);
    if (suffix) memcpy(out + decoded - suffix, prev + (pn - suffix), suffix);
    return EF_OK;
}

/* Decode one record, allowing trailing records. On success returns EF_OK and
 * sets both lengths. EF_NOSPACE on a complete record also sets both lengths,
 * enabling a sizing pass with (out=NULL,cap=0). EF_MORE requires more wire
 * bytes. EF_INVALID means malformed input or arguments. On either of those
 * two statuses both lengths are zero. Output is written only on EF_OK.
 * out must not overlap wire/prev; out_len and consumed must be distinct. */
static inline int ef_decode(const unsigned char *prev, size_t pn,
                            const unsigned char *wire, size_t wn,
                            unsigned char *out, size_t cap,
                            size_t *out_len, size_t *consumed) {
    size_t pos = 1, inserted, prefix = 0, suffix = 0, decoded;
    unsigned tag;
    int status;
    if (!out_len || !consumed || out_len == consumed) return EF_INVALID;
    *out_len = *consumed = 0;
    if ((!prev && pn) || (!wire && wn) || (!out && cap)) return EF_INVALID;
    if (wn == 0) return EF_MORE;
    tag = wire[0];
    if (tag == 0x81)
        return ef_decode_middle(prev, pn, wire, wn, out, cap, out_len, consumed);
    if (tag > 0x81) return EF_INVALID;
    if (tag == 0x80) {
        inserted = 0;
        prefix = pn;
    } else {
        inserted = tag & 63;
        if (inserted == 63) {
            size_t extra = 0;
            status = ef_get_var(wire, wn, &pos, &extra);
            if (status != EF_OK) return status;
            if (extra > (size_t)-1 - 63) return EF_INVALID;
            inserted += extra;
        }
        if (tag & 0x40) {
            status = ef_get_var(wire, wn, &pos, &prefix);
            if (status != EF_OK) return status;
            if (prefix > pn) return EF_INVALID;
            status = ef_get_var(wire, wn, &pos, &suffix);
            if (status != EF_OK) return status;
            if (suffix > pn - prefix) return EF_INVALID;
        }
    }
    if (inserted > (size_t)-1 - (prefix + suffix)) return EF_INVALID;
    if (inserted > (size_t)-1 - pos) return EF_INVALID;
    if (inserted > wn - pos) return EF_MORE;
    decoded = prefix + suffix + inserted;
    *out_len = decoded;
    *consumed = pos + inserted;
    if (cap < decoded) return EF_NOSPACE;
    if (prefix) memcpy(out, prev, prefix);
    if (inserted) memcpy(out + prefix, wire + pos, inserted);
    if (suffix) memcpy(out + prefix + inserted, prev + (pn - suffix), suffix);
    return EF_OK;
}

#endif /* EVENTFOLD_H */
