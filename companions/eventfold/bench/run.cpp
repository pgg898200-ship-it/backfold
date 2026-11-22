// Run from the repository root. See bench/README.md for measurement semantics.
#include "codecs.hpp"
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
using Stream = std::vector<Bytes>;
namespace fs = std::filesystem;

static Bytes bytes(const std::string &s) { return Bytes(s.begin(), s.end()); }
static std::vector<std::pair<std::string, Stream>> synthetics() {
    Stream token, alternating, random;
    std::mt19937 rng(1701u);
    const char *words[] = {" the", " exact", " answer", " or", " to", " decide", ".",
        " here", " false", " is", " be", " red", " compare", " value", " =",
        " next", " token", ",", " no", " a", " ->", " why", "{", " both"};
    const std::string alphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    for (unsigned i=0; i<2048; ++i) {
        std::string w = words[rng() % (sizeof(words)/sizeof(words[0]))];
        token.push_back(bytes("data: {\"id\":\"chatcmpl-fixed-session\",\"object\":"
            "\"chat.completion.chunk\",\"created\":1745500000,\"model\":\"example\","
            "\"choices\":[{\"index\":0,\"delta\":{\"content\":\""+w+
            "\"},\"finish_reason\":null}]}\n\n"));
        std::string part;
        const char *kind;
        switch (i % 4) {
            case 0: kind="text.delta"; part="\"text\":\""+w+"\""; break;
            case 1: kind="tool.input"; part="\"tool\":\"lookup\",\"arg\":"+std::to_string(rng()%1024); break;
            case 2: kind="worker.progress"; part="\"worker\":"+std::to_string(rng()%16)+
                ",\"elapsed_ms\":"+std::to_string(i*3); break;
            default: kind="tool.output"; part="\"result\":{\"count\":"+std::to_string(rng()%97)+"}";
        }
        alternating.push_back(bytes("event: "+std::string(kind)+"\ndata: {"+part+"}\n\n"));
        std::string s = "data: ";
        unsigned n = 48 + rng()%128;
        for (unsigned j=0; j<n; ++j) s += alphabet[rng() % alphabet.size()];
        random.push_back(bytes(s+"\n\n"));
    }
    return {{"syn-token",std::move(token)},{"syn-alternating",std::move(alternating)},
            {"syn-noisy-ascii",std::move(random)}};
}

static Stream read_events(const fs::path &path) {
    std::ifstream f(path,std::ios::binary);
    require(bool(f), "cannot open "+path.string());
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Stream out;
    size_t start=0, line=0, pos=0;
    while (pos<data.size()) {
        if (data[pos]!='\r' && data[pos]!='\n') { ++pos; continue; }
        bool blank = pos==line;
        char end = data[pos++];
        if (end=='\r' && pos<data.size() && data[pos]=='\n') ++pos;
        if (blank) { out.emplace_back(data.begin()+start,data.begin()+pos); start=pos; }
        line=pos;
    }
    if (start<data.size()) out.emplace_back(data.begin()+start, data.end());
    require(!out.empty(), "empty fixture "+path.string());
    return out;
}

static uint64_t cpu_ns() {
    timespec t{};
    check(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&t)==0, "CPU clock unavailable");
    return uint64_t(t.tv_sec)*1000000000ULL+t.tv_nsec;
}
static std::string samples(const std::vector<double>& v) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(2);
    for (size_t i=0; i<v.size(); ++i) { if (i) ss << "|"; ss << v[i]; }
    return ss.str();
}
static double median(std::vector<double> v) {
    std::sort(v.begin(),v.end());
    return (v[(v.size()-1)/2]+v[v.size()/2])/2;
}
static uint64_t sink;
struct Case {
    const std::string &id;
    const Stream &events;
    Stream wire;
    Bytes enc,dec;
    size_t raw=0,packed=0;
    int64_t enc_mem=0,dec_mem=0;
    Case(const std::string &name, const Stream &input): id(name),events(input) {
        size_t maximum=0;
        for (const auto &a:events) { maximum=std::max(maximum,a.size()); raw+=a.size(); }
        size_t bound = std::max({ef_bound(maximum)+4, ZSTD_compressBound(maximum),
                                LZ4F_compressBound(maximum,nullptr)})+131072;
        require(bound<=UINT_MAX, "event too large for zlib single-call API");
        enc.resize(bound);
        dec.resize(maximum+1);
        auto c=create_codec(id);
        for (const auto &a:events) {
            size_t n=c->encode(a,enc);
            require(n<=enc.size(),"encoder exceeded bound");
            wire.emplace_back(enc.begin(),enc.begin()+n);
            packed+=n;
            size_t m=c->decode(wire.back(),dec);
            require(m==a.size() && std::equal(a.begin(),a.end(),dec.begin()),
                    id+" mismatch after event "+std::to_string(wire.size()));
        }
        enc_mem=c->enc_state(); dec_mem=c->dec_state();
    }
    double measure(bool encode, size_t cycles) {
        uint64_t elapsed=0, sum=0;
        for (size_t cycle=0; cycle<cycles; ++cycle) {
            auto c=create_codec(id);
            uint64_t begin=cpu_ns();
            if (encode) {
                for (const auto &a:events) {
                    size_t n=c->encode(a,enc);
                    sum+=n+enc[n-1];
                }
            } else {
                for (const auto &w:wire) {
                    size_t n=c->decode(w,dec);
                    sum+=n+dec[n-1];
                }
            }
            elapsed+=cpu_ns()-begin;
        }
        sink+=sum;
        return double(elapsed)/(cycles*events.size());
    }
};
int main(int argc, char **argv) try {
    std::string directory = argc>=2 ? argv[1] : "fixtures";
    int reps=argc>=3 ? std::stoi(argv[2]) : 7;
    unsigned minimum=argc>=4 ? std::stoul(argv[3]) : 2048;
    require(reps>0 && reps<=1000 && minimum<=10000000,
            "usage: run [fixtures_dir] [repetitions=7] [minimum events/sample=2048]");
    auto inputs=synthetics();
    std::vector<fs::path> files;
    for (const auto &entry: fs::directory_iterator(directory))
        if (entry.path().extension()==".sse") files.push_back(entry.path());
    std::sort(files.begin(),files.end());
    require(!files.empty(),"fixture directory contains no SSE files");
    for (const auto &p:files) inputs.emplace_back("fixtures/"+p.stem().string(),read_events(p));
    auto holdout = fs::path(directory)/"holdout";
    if (fs::exists(holdout)) {
        files.clear();
        for (const auto &entry:fs::directory_iterator(holdout))
            if (entry.path().extension()==".sse") files.push_back(entry.path());
        std::sort(files.begin(),files.end());
        for (const auto &p:files) inputs.emplace_back("holdout/"+p.stem().string(),read_events(p));
    }
    std::cerr << "zstd=" << ZSTD_versionString() << " zlib=" << zlibVersion()
              << " lz4=" << LZ4_versionString() << " compiler=" << __VERSION__ << "\n";
    std::cout << "workload,codec,events,raw_bytes,wire_bytes,encode_ns_per_event,"
              << "decode_ns_per_event,encoder_state_bytes,decoder_state_bytes,"
              << "cycles_per_sample,encode_samples,decode_samples\n" << std::fixed << std::setprecision(2);
    for (const auto &[name, events]: inputs) {
        for (const std::string id:{"raw-sse","raw-varint","eventfold","lz4-linked","zstd-1",
                 "zstd-1-small","zstd-3","gzip-1","gzip-1-small","gzip-6","ws-deflate-6"}) {
            Case c(id,events);
            size_t cycles=std::max<size_t>(1,(minimum+events.size()-1)/events.size());
            c.measure(true,1); c.measure(false,1);
            std::vector<double> et,dt;
            for (int rep=0; rep<reps; ++rep) {
                if (rep%2) {
                    dt.push_back(c.measure(false,cycles)); et.push_back(c.measure(true,cycles));
                } else {
                    et.push_back(c.measure(true,cycles)); dt.push_back(c.measure(false,cycles));
                }
            }
            std::cout << name << ',' << id << ',' << events.size() << ',' << c.raw << ','
                      << c.packed << ',' << median(et) << ',' << median(dt) << ','
                      << c.enc_mem << ',' << c.dec_mem << ',' << cycles << ','
                      << samples(et) << ',' << samples(dt) << '\n';
        }
    }
    std::cerr << "readback checksum=" << sink << "\n";
    return 0;
} catch(const std::exception &e) {
    std::cerr << "benchmark failed: " << e.what() << "\n";
    return 1;
}
