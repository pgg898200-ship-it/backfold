#pragma once
#include "../eventfold.h"
#include <zlib.h>
#include <zstd.h>
#if __has_include(<lz4frame.h>) && __has_include(<lz4.h>)
#include <lz4frame.h>
#include <lz4.h>
#else
#include "lz4frame_abi.h"
#endif
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using Bytes = std::vector<unsigned char>;
inline void require(bool ok, const std::string &why) {
    if (!ok) throw std::runtime_error(why);
}
inline void check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}

// Buffers are prepared once per batch and reused for the pass. All encode
// calls return bytes that must immediately decode into exactly one input event.
struct Codec {
    virtual ~Codec() = default;
    virtual size_t encode(const Bytes&, Bytes&) = 0;
    virtual size_t decode(const Bytes&, Bytes&) = 0;
    virtual int64_t enc_state() const { return -1; }
    virtual int64_t dec_state() const { return -1; }
};

struct Plain final : Codec {
    bool framed;
    explicit Plain(bool f): framed(f) {}
    size_t encode(const Bytes &a, Bytes &out) override {
        size_t header = framed ? ef_put_var(out.data(), a.size()) : 0;
        std::memcpy(out.data() + header, a.data(), a.size());
        return a.size() + header;
    }
    size_t decode(const Bytes &wire, Bytes &out) override {
        size_t pos = 0, n = wire.size();
        if (framed) check(ef_get_var(wire.data(), wire.size(), &pos, &n) == EF_OK,
                          "invalid raw record length");
        check(n <= out.size() && n == wire.size() - pos, "invalid raw input");
        std::memcpy(out.data(), wire.data() + pos, n);
        return n;
    }
    int64_t enc_state() const override { return 0; }
    int64_t dec_state() const override { return 0; }
};

struct Fold final : Codec {
    Bytes last_enc, last_dec;
    bool first_enc = true, first_dec = true;
    size_t encode(const Bytes &a, Bytes &out) override {
        size_t p = 0;
        if (first_enc) {
            std::memcpy(out.data(), "EFL1", 4);
            p = 4;
            first_enc = false;
        }
        size_t n = ef_encode(last_enc.data(), last_enc.size(), a.data(), a.size(),
                             out.data()+p, out.size()-p);
        check(n != 0, "EventFold encode failure");
        last_enc.assign(a.begin(), a.end());
        return p+n;
    }
    size_t decode(const Bytes &wire, Bytes &out) override {
        size_t p = 0;
        if (first_dec) {
            check(wire.size() >= 4 && !std::memcmp(wire.data(), "EFL1", 4), "bad EFL1");
            p = 4;
            first_dec = false;
        }
        size_t n = 0, used = 0;
        check(ef_decode(last_dec.data(), last_dec.size(), wire.data()+p, wire.size()-p,
                        out.data(), out.size(), &n, &used) == EF_OK && used == wire.size()-p,
              "EventFold decode failure");
        last_dec.assign(out.begin(), out.begin()+n);
        return n;
    }
    int64_t enc_state() const override { return last_enc.capacity(); }
    int64_t dec_state() const override { return last_dec.capacity(); }
};

inline void zstd_ok(size_t rc) {
    if (ZSTD_isError(rc)) throw std::runtime_error(ZSTD_getErrorName(rc));
}
struct Zstd final : Codec {
    ZSTD_CCtx *c = ZSTD_createCCtx();
    ZSTD_DCtx *d = ZSTD_createDCtx();
    Zstd(int level, bool small) {
        check(c && d, "zstd allocation");
        zstd_ok(ZSTD_CCtx_setParameter(c, ZSTD_c_compressionLevel, level));
        if (small) {
            zstd_ok(ZSTD_CCtx_setParameter(c, ZSTD_c_windowLog, 11));
            zstd_ok(ZSTD_CCtx_setParameter(c, ZSTD_c_hashLog, 10));
            zstd_ok(ZSTD_CCtx_setParameter(c, ZSTD_c_chainLog, 10));
        }
    }
    ~Zstd() override { ZSTD_freeCCtx(c); ZSTD_freeDCtx(d); }
    size_t encode(const Bytes &a, Bytes &out) override {
        ZSTD_inBuffer src{a.data(), a.size(), 0};
        ZSTD_outBuffer dst{out.data(), out.size(), 0};
        size_t pending = ZSTD_compressStream2(c, &dst, &src, ZSTD_e_flush);
        zstd_ok(pending);
        check(pending == 0 && src.pos == a.size(), "zstd flush did not complete");
        return dst.pos;
    }
    size_t decode(const Bytes &wire, Bytes &out) override {
        ZSTD_inBuffer src{wire.data(), wire.size(), 0};
        ZSTD_outBuffer dst{out.data(), out.size(), 0};
        while (src.pos < src.size) {
            size_t before = src.pos;
            zstd_ok(ZSTD_decompressStream(d, &dst, &src));
            check(src.pos > before, "zstd decode stalled");
        }
        return dst.pos;
    }
    int64_t enc_state() const override { return ZSTD_sizeof_CCtx(c); }
    int64_t dec_state() const override { return ZSTD_sizeof_DCtx(d); }
};

struct ZlibHeap {
    size_t live = 0, peak = 0;
    struct alignas(std::max_align_t) Header { size_t n; };
    static voidpf alloc(voidpf opaque, uInt count, uInt size) {
        size_t n = size_t(count) * size;
        if ((size && n / size != count) || n > SIZE_MAX-sizeof(Header)) return nullptr;
        auto *h = static_cast<Header*>(std::calloc(1, sizeof(Header)+n));
        if (!h) return nullptr;
        auto &self = *static_cast<ZlibHeap*>(opaque);
        self.live += n;
        self.peak = std::max(self.live, self.peak);
        h->n = n;
        return h+1;
    }
    static void release(voidpf opaque, voidpf p) {
        if (!p) return;
        auto *h = static_cast<Header*>(p)-1;
        static_cast<ZlibHeap*>(opaque)->live -= h->n;
        std::free(h);
    }
    void attach(z_stream &s) { s.zalloc = alloc; s.zfree = release; s.opaque = this; }
};
struct Gzip final : Codec {
    ZlibHeap ch, dh;
    z_stream c{}, d{};
    bool websocket;
    Gzip(int level, int window, int mem, bool ws = false): websocket(ws) {
        int bits = ws ? -window : 16+window;
        ch.attach(c); dh.attach(d);
        check(deflateInit2(&c, level, Z_DEFLATED, bits, mem, Z_DEFAULT_STRATEGY) == Z_OK,
              "gzip initialization");
        // Administrator-configured receiver can use the same window for the small profile.
        check(inflateInit2(&d, bits) == Z_OK, "gunzip initialization");
    }
    ~Gzip() override { deflateEnd(&c); inflateEnd(&d); }
    size_t encode(const Bytes &a, Bytes &out) override {
        c.next_in = const_cast<Bytef*>(a.data()); c.avail_in = static_cast<uInt>(a.size());
        c.next_out = out.data(); c.avail_out = static_cast<uInt>(out.size());
        check(deflate(&c, Z_SYNC_FLUSH) == Z_OK && c.avail_in == 0 && c.avail_out,
              "gzip flush did not complete");
        size_t emitted = out.size()-c.avail_out;
        if (websocket) {
            check(emitted >= 4 && !std::memcmp(out.data()+emitted-4, "\x00\x00\xff\xff", 4),
                  "missing RFC 7692 marker");
            emitted -= 4;
        }
        return emitted;
    }
    size_t decode(const Bytes &wire, Bytes &out) override {
        d.next_in = const_cast<Bytef*>(wire.data()); d.avail_in = static_cast<uInt>(wire.size());
        d.next_out = out.data(); d.avail_out = static_cast<uInt>(out.size());
        check(inflate(&d, Z_SYNC_FLUSH) == Z_OK && d.avail_in == 0 && d.avail_out,
              "gunzip decode failed");
        if (websocket) {
            Bytef marker[4] = {0, 0, 255, 255};
            d.next_in = marker; d.avail_in = 4;
            check(inflate(&d, Z_SYNC_FLUSH) == Z_OK && d.avail_in == 0 && d.avail_out,
                  "permessage-deflate marker failed");
        }
        return out.size()-d.avail_out;
    }
    int64_t enc_state() const override { return ch.peak + sizeof(c); }
    int64_t dec_state() const override { return dh.peak + sizeof(d); }
};

inline void lz_ok(size_t rc) {
    if (LZ4F_isError(rc)) throw std::runtime_error(LZ4F_getErrorName(rc));
}
struct Lz4 final : Codec {
    LZ4F_cctx_s *c = nullptr;
    LZ4F_dctx_s *d = nullptr;
    bool first = true;
    Lz4() {
        lz_ok(LZ4F_createCompressionContext(&c, LZ4F_getVersion()));
        lz_ok(LZ4F_createDecompressionContext(&d, LZ4F_getVersion()));
    }
    ~Lz4() override { LZ4F_freeCompressionContext(c); LZ4F_freeDecompressionContext(d); }
    size_t encode(const Bytes &a, Bytes &out) override {
        size_t p = 0;
        if (first) {
            p = LZ4F_compressBegin(c, out.data(), out.size(), nullptr);
            lz_ok(p);
            first = false;
        }
        size_t n = LZ4F_compressUpdate(c, out.data()+p, out.size()-p, a.data(), a.size(), nullptr);
        lz_ok(n); p += n;
        n = LZ4F_flush(c, out.data()+p, out.size()-p, nullptr);
        lz_ok(n);
        return p+n;
    }
    size_t decode(const Bytes &wire, Bytes &out) override {
        size_t in = 0, written = 0;
        while (in < wire.size()) {
            size_t n = wire.size()-in, m = out.size()-written;
            lz_ok(LZ4F_decompress(d, out.data()+written, &m, wire.data()+in, &n, nullptr));
            check(n > 0, "LZ4 decode stalled");
            in += n; written += m;
        }
        return written;
    }
};

inline std::unique_ptr<Codec> create_codec(const std::string &id) {
    if (id == "raw-sse") return std::make_unique<Plain>(false);
    if (id == "raw-varint") return std::make_unique<Plain>(true);
    if (id == "eventfold") return std::make_unique<Fold>();
    if (id == "zstd-1") return std::make_unique<Zstd>(1, false);
    if (id == "zstd-3") return std::make_unique<Zstd>(3, false);
    if (id == "zstd-1-small") return std::make_unique<Zstd>(1, true);
    if (id == "gzip-1") return std::make_unique<Gzip>(1, 15, 8);
    if (id == "ws-deflate-6") return std::make_unique<Gzip>(6, 15, 8, true);
    if (id == "gzip-6") return std::make_unique<Gzip>(6, 15, 8);
    if (id == "gzip-1-small") return std::make_unique<Gzip>(1, 11, 3);
    if (id == "lz4-linked") return std::make_unique<Lz4>();
    throw std::runtime_error("unknown codec: "+id);
}
