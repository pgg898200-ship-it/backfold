// Warm encoder cost for independently interleaved, immediately flushed streams.
// No queueing, HTTP, TLS, generator time, or decoder time in the timed region.
#include "../eventfold.h"
#include <zlib.h>
#include <algorithm>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" void consume_output(const unsigned char *data, size_t length);

static unsigned long long cpu_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) throw std::runtime_error("thread CPU clock");
    return static_cast<unsigned long long>(ts.tv_sec) * 1000000000ULL + ts.tv_nsec;
}
using Bytes = std::vector<unsigned char>;

struct Entry {
    Bytes previous;
};
struct Count {
    size_t requested = 0;
};
static voidpf tracked_alloc(voidpf opaque, uInt count, uInt size) {
    void *p = std::calloc(count, size);
    if (p) static_cast<Count *>(opaque)->requested += size_t(count) * size;
    return p;
}
static void tracked_free(voidpf, voidpf p) { std::free(p); }

static std::string event(size_t id, size_t tick) {
    static const char *words[] = {" north", " east", " west", " map", " the", ".", " a", " street",
                                  " northbound", " turn", " station", " right", " walk"};
    const char *word = words[(id * 13 + tick * 17 + id / 3) % 13];
    char s[384];
    int n = std::snprintf(s, sizeof(s),
        "data: {\"id\":\"chatcmpl-%08zu\",\"object\":\"chat.completion.chunk\","
        "\"created\":1770000000,\"model\":\"fleet-fixture\",\"choices\":[{\"index\":0,"
        "\"delta\":{\"content\":\"%s\"},\"finish_reason\":null}]}\n\n", id, word);
    return std::string(s, size_t(n));
}

int main(int argc, char **argv) {
    if (argc != 4) return 2;
    const std::string codec = argv[1];
    size_t streams = std::stoul(argv[2]), rounds = std::stoul(argv[3]);
    if ((codec != "eventfold" && codec != "gzip-w12-m3") ||
        !streams || streams > 16384 || rounds < 3 || rounds > 1000) return 2;
    const bool use_ef = codec == "eventfold";
    Count memory;
    std::vector<Entry> entries(use_ef ? streams : 0);
    std::vector<z_stream> z(use_ef ? 0 : streams);
    std::vector<std::string> inputs(streams);
    for (auto &s : z) {
        s.zalloc = tracked_alloc; s.zfree = tracked_free; s.opaque = &memory;
        if (deflateInit2(&s, 1, Z_DEFLATED, 16+12, 3, Z_DEFAULT_STRATEGY) != Z_OK) return 3;
    }
    unsigned char out[1024];
    unsigned long long wire = use_ef ? 4 * streams : 0, raw = 0;
    std::vector<double> ns;
    for (size_t tick = 0; tick < rounds; ++tick) {
        for (size_t i = 0; i < streams; ++i) inputs[i] = event(i, tick);
        auto start = cpu_ns();
        for (size_t i = 0; i < streams; ++i) {
            auto &s = inputs[i];
            auto *frame = reinterpret_cast<const unsigned char *>(s.data());
            size_t n;
            if (use_ef) {
                auto &prev = entries[i].previous;
                n = ef_encode(prev.data(), prev.size(), frame, s.size(), out, sizeof out);
                if (!n) return 3;
                prev.assign(frame, frame + s.size());
            } else {
                z[i].next_in = const_cast<Bytef *>(frame);
                z[i].avail_in = static_cast<uInt>(s.size());
                z[i].next_out = out; z[i].avail_out = sizeof out;
                if (deflate(&z[i], Z_SYNC_FLUSH) != Z_OK || z[i].avail_in || !z[i].avail_out) return 3;
                n = sizeof out - z[i].avail_out;
            }
            consume_output(out, n);
            wire += n; raw += s.size();
        }
        auto stop = cpu_ns();
        if (tick >= 2) ns.push_back(static_cast<double>(stop-start)/streams);
    }
    size_t state = entries.capacity() * sizeof(Entry) + z.capacity() * sizeof(z_stream) + memory.requested;
    for (auto &s : entries) state += s.previous.capacity();
    for (auto &s : z) {
        s.next_in = Z_NULL; s.avail_in = 0; s.next_out = out; s.avail_out = sizeof out;
        if (deflate(&s, Z_FINISH) != Z_STREAM_END) return 3;
        wire += sizeof out - s.avail_out;
        if (deflateEnd(&s) != Z_OK) return 3;
    }
    std::sort(ns.begin(), ns.end());
    std::printf("%s,%zu,%zu,%zu,%llu,%llu,%zu,%.2f\n", codec.c_str(), streams, rounds,
                streams * rounds, raw, wire, state, ns[ns.size()/2]);
}
