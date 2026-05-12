// bench_art_baseline.cpp
// Conventional non-ML baselines for the DualIndex thesis: Adaptive Radix Tree
// (libart) and a sorted-array + binary-search "dumbest-baseline" floor.
//
// The same binary handles both — pick with --baseline {art,binsearch}.
// CSV schema matches what scripts/compare_art_vs_learned.py expects.
//
// Usage:
//   bench_art_baseline <data.txt> <N>
//       [--baseline {art|binsearch}]   default art
//       [--queries_point  100000]
//       [--queries_range  10000]
//       [--seed 42]
//       [--out_csv PATH]
//
// N=0 means "load all rows".
//
// Query types (mirror DualIndex):
//   point      — full 12-byte triple key  → ART art_search / binary search
//   single_hop — fix Source only          → ART art_iter_prefix (4B prefix)
//                                            binsearch lower_bound + linear scan
//   multi_hop  — fix (Source, Hop1)       → ART art_iter_prefix (8B prefix)
//                                            binsearch lower_bound + linear scan

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include "art.h"
}

using Triple = std::array<uint32_t, 3>;

// ── big-endian 4-byte / 12-byte key encoding ──────────────────────────────────
static inline void put_be32(uint32_t v, unsigned char* p) {
    p[0] = (unsigned char)(v >> 24);
    p[1] = (unsigned char)(v >> 16);
    p[2] = (unsigned char)(v >> 8);
    p[3] = (unsigned char)(v);
}
static inline void encode_key12(const Triple& t, unsigned char* out) {
    put_be32(t[0], out);
    put_be32(t[1], out + 4);
    put_be32(t[2], out + 8);
}

// ── data loader: 4-column space-separated text, take first 3 columns ──────────
static void load_text_triples(const std::string& fname, std::vector<Triple>& out, size_t N) {
    std::ifstream f(fname);
    if (!f) {
        std::cerr << "ERROR: cannot open " << fname << "\n";
        std::exit(1);
    }
    std::string line;
    size_t loaded = 0;
    while ((N == 0 || loaded < N) && std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Triple t;
        uint64_t v;
        bool ok = true;
        for (int d = 0; d < 3; ++d) {
            if (!(ss >> v)) { ok = false; break; }
            t[d] = static_cast<uint32_t>(v);
        }
        if (!ok) {
            std::cerr << "ERROR: short row at line " << (loaded + 1) << "\n";
            std::exit(1);
        }
        out.push_back(t);
        ++loaded;
    }
    if (N != 0 && loaded < N)
        std::cerr << "WARNING: requested " << N << " rows but file only has " << loaded << "\n";
}

// ── peak RSS via getrusage (Linux: ru_maxrss is in KB) ────────────────────────
static long long rss_kb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (long long)ru.ru_maxrss;
}

// ── latency stat helper ───────────────────────────────────────────────────────
struct Stats {
    double mean_us = 0.0;
    double p50_us  = 0.0;
    double p95_us  = 0.0;
    double p99_us  = 0.0;
    double qps     = 0.0;
};
static Stats compute_stats(std::vector<long long>& lat_ns) {
    Stats s;
    if (lat_ns.empty()) return s;
    std::sort(lat_ns.begin(), lat_ns.end());
    long long sum = 0;
    for (auto v : lat_ns) sum += v;
    s.mean_us = (double)sum / lat_ns.size() / 1000.0;
    auto pct = [&](double p) {
        size_t i = (size_t)(p * lat_ns.size());
        if (i >= lat_ns.size()) i = lat_ns.size() - 1;
        return lat_ns[i] / 1000.0;
    };
    s.p50_us = pct(0.50);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    s.qps    = (double)lat_ns.size() / ((double)sum / 1e9);
    return s;
}

// ── range query callback for ART: count how many leaves visited ───────────────
struct CountCtx { size_t n; };
static int count_cb(void* data, const unsigned char* /*key*/, uint32_t /*klen*/, void* /*val*/) {
    ((CountCtx*)data)->n++;
    return 0;
}

// ── binsearch baseline helpers ────────────────────────────────────────────────
static size_t bs_lower_bound_prefix(const std::vector<Triple>& sorted,
                                    uint32_t s, uint32_t h1, bool use_h1) {
    // Find the first index i such that data[i] >= (s, [h1?], 0).
    // We compare as 3-tuples; pad missing fields with 0.
    Triple target = { s, use_h1 ? h1 : 0u, 0u };
    auto it = std::lower_bound(sorted.begin(), sorted.end(), target,
        [](const Triple& a, const Triple& b) {
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        });
    return (size_t)(it - sorted.begin());
}

static size_t bs_prefix_count(const std::vector<Triple>& sorted,
                              uint32_t s, uint32_t h1, bool use_h1) {
    size_t i = bs_lower_bound_prefix(sorted, s, h1, use_h1);
    size_t cnt = 0;
    while (i < sorted.size() && sorted[i][0] == s &&
           (!use_h1 || sorted[i][1] == h1)) {
        ++cnt; ++i;
    }
    return cnt;
}

static bool bs_point_lookup(const std::vector<Triple>& sorted, const Triple& q) {
    auto it = std::lower_bound(sorted.begin(), sorted.end(), q,
        [](const Triple& a, const Triple& b) {
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        });
    return (it != sorted.end() && *it == q);
}

// ── shorten file path for CSV "dataset" field ─────────────────────────────────
static std::string dataset_short(const std::string& path) {
    auto slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    // strip "_triples.txt" if present
    const std::string suf = "_triples.txt";
    if (base.size() > suf.size() &&
        base.compare(base.size() - suf.size(), suf.size(), suf) == 0)
        base = base.substr(0, base.size() - suf.size());
    // strip plain ".txt"
    else if (base.size() > 4 && base.compare(base.size() - 4, 4, ".txt") == 0)
        base = base.substr(0, base.size() - 4);
    return base;
}

// ── main ──────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_art_baseline <data.txt> <N>\n"
                  << "  [--baseline {art|binsearch}]   default art\n"
                  << "  [--queries_point  Q]           default 100000\n"
                  << "  [--queries_range  Q]           default 10000\n"
                  << "  [--seed S]                     default 42\n"
                  << "  [--out_csv PATH]\n"
                  << "  N=0 means load all rows.\n";
        return 1;
    }

    std::string data_file   = argv[1];
    size_t      N           = std::stoul(argv[2]);
    std::string baseline    = "art";
    size_t      Qp          = 100000;
    size_t      Qr          = 10000;
    uint64_t    seed        = 42;
    std::string out_csv;

    for (int i = 3; i + 1 < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--baseline")      baseline = argv[i+1];
        else if (a == "--queries_point") Qp = std::stoull(argv[i+1]);
        else if (a == "--queries_range") Qr = std::stoull(argv[i+1]);
        else if (a == "--seed")          seed = std::stoull(argv[i+1]);
        else if (a == "--out_csv")       out_csv = argv[i+1];
    }
    if (baseline != "art" && baseline != "binsearch") {
        std::cerr << "ERROR: --baseline must be 'art' or 'binsearch'\n";
        return 1;
    }

    std::cout << "=== bench_art_baseline (" << baseline << ") ===\n";
    std::cout << "Loading: " << data_file << "  N=" << (N == 0 ? std::string("all") : std::to_string(N)) << "\n";
    std::vector<Triple> data;
    load_text_triples(data_file, data, N);
    size_t actual_N = data.size();
    std::cout << "Loaded " << actual_N << " triples.\n";

    // sort for binsearch (and as a sanity-stable order for sampling).
    // Data appears already sorted by (s,h1,h2) in the SNAP files, but we don't
    // rely on that — sort to be safe. (Cheap relative to the rest.)
    std::sort(data.begin(), data.end(),
        [](const Triple& a, const Triple& b) {
            if (a[0] != b[0]) return a[0] < b[0];
            if (a[1] != b[1]) return a[1] < b[1];
            return a[2] < b[2];
        });

    long long rss_before_kb = rss_kb();

    // ── build phase ───────────────────────────────────────────────────────────
    auto t_build0 = std::chrono::steady_clock::now();
    art_tree tree;
    if (baseline == "art") {
        if (art_tree_init(&tree) != 0) {
            std::cerr << "ERROR: art_tree_init failed\n";
            return 2;
        }
        unsigned char k[12];
        for (size_t i = 0; i < actual_N; ++i) {
            encode_key12(data[i], k);
            // Store i+1 so the null "not found" return is distinguishable from
            // a legitimate index-0 hit.
            art_insert(&tree, k, 12, (void*)(uintptr_t)(i + 1));
        }
    }
    // binsearch: no build work beyond the sort already done.
    auto t_build1 = std::chrono::steady_clock::now();
    double build_s = std::chrono::duration<double>(t_build1 - t_build0).count();
    long long rss_after_kb = rss_kb();
    double rss_mb = (rss_after_kb - rss_before_kb) / 1024.0;
    if (rss_mb < 0) rss_mb = 0;  // floor

    std::cout << "Build:   " << std::fixed << std::setprecision(3) << build_s << " s\n";
    std::cout << "RSS Δ:   " << std::fixed << std::setprecision(1) << rss_mb << " MB\n";
    if (baseline == "art")
        std::cout << "ART size (leaves): " << tree.size << "\n";

    // ── query workloads (seed-determined indices, shared across baselines) ────
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, actual_N - 1);

    // Sample 1000 warmup indices + Qp point indices + Qr single_hop indices + Qr multi_hop indices
    std::vector<size_t> warm_idx(1000);
    std::vector<size_t> point_idx(Qp);
    std::vector<size_t> single_idx(Qr);
    std::vector<size_t> multi_idx(Qr);
    for (auto& x : warm_idx)   x = dist(rng);
    for (auto& x : point_idx)  x = dist(rng);
    for (auto& x : single_idx) x = dist(rng);
    for (auto& x : multi_idx)  x = dist(rng);

    // ── helper closures for the two baselines ────────────────────────────────
    auto do_point_one = [&](size_t i) -> bool {
        if (baseline == "art") {
            unsigned char k[12];
            encode_key12(data[i], k);
            void* v = art_search(&tree, k, 12);
            return v == (void*)(uintptr_t)(i + 1);
        } else {
            return bs_point_lookup(data, data[i]);
        }
    };
    auto do_single_one = [&](size_t i) -> size_t {
        uint32_t s = data[i][0];
        if (baseline == "art") {
            unsigned char p[4]; put_be32(s, p);
            CountCtx ctx{0};
            art_iter_prefix(&tree, p, 4, count_cb, &ctx);
            return ctx.n;
        } else {
            return bs_prefix_count(data, s, 0, false);
        }
    };
    auto do_multi_one = [&](size_t i) -> size_t {
        uint32_t s = data[i][0], h1 = data[i][1];
        if (baseline == "art") {
            unsigned char p[8]; put_be32(s, p); put_be32(h1, p + 4);
            CountCtx ctx{0};
            art_iter_prefix(&tree, p, 8, count_cb, &ctx);
            return ctx.n;
        } else {
            return bs_prefix_count(data, s, h1, true);
        }
    };

    // ── warmup (untimed) ───────────────────────────────────────────────────────
    {
        volatile size_t sink = 0;
        for (size_t i : warm_idx) sink += (size_t)do_point_one(i);
        for (size_t i : warm_idx) sink += do_single_one(i);
        for (size_t i : warm_idx) sink += do_multi_one(i);
        (void)sink;
    }

    // ── timed phases ──────────────────────────────────────────────────────────
    auto run_point = [&]() {
        std::vector<long long> lat_ns; lat_ns.reserve(Qp);
        size_t correct = 0;
        for (size_t i : point_idx) {
            auto t0 = std::chrono::steady_clock::now();
            bool ok = do_point_one(i);
            auto t1 = std::chrono::steady_clock::now();
            lat_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            if (ok) ++correct;
        }
        return std::make_pair(std::move(lat_ns), correct);
    };
    auto run_range = [&](bool is_multi) {
        std::vector<long long> lat_ns; lat_ns.reserve(Qr);
        size_t total_results = 0;
        const auto& idxs = is_multi ? multi_idx : single_idx;
        for (size_t i : idxs) {
            auto t0 = std::chrono::steady_clock::now();
            size_t n = is_multi ? do_multi_one(i) : do_single_one(i);
            auto t1 = std::chrono::steady_clock::now();
            lat_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            total_results += n;
        }
        return std::make_tuple(std::move(lat_ns), total_results);
    };

    std::cout << "\nRunning queries (point=" << Qp << ", single_hop=" << Qr
              << ", multi_hop=" << Qr << ", warmup=1000 each)\n";

    auto [lat_point, point_correct] = run_point();
    auto stats_point = compute_stats(lat_point);
    double correctness_pct = 100.0 * point_correct / Qp;

    auto [lat_single, single_results] = run_range(false);
    auto stats_single = compute_stats(lat_single);
    double avg_single = (double)single_results / Qr;

    auto [lat_multi, multi_results] = run_range(true);
    auto stats_multi = compute_stats(lat_multi);
    double avg_multi = (double)multi_results / Qr;

    // ── report ────────────────────────────────────────────────────────────────
    auto print_row = [](const char* name, const Stats& s, double avg_results, double correctness) {
        std::cout << std::left << std::setw(12) << name
                  << std::right << std::fixed
                  << std::setprecision(3) << std::setw(10) << s.mean_us
                  << std::setprecision(3) << std::setw(10) << s.p50_us
                  << std::setprecision(3) << std::setw(10) << s.p95_us
                  << std::setprecision(3) << std::setw(10) << s.p99_us
                  << std::defaultfloat << std::setw(14) << (long long)s.qps
                  << std::fixed << std::setprecision(2) << std::setw(14) << avg_results
                  << std::setprecision(1) << std::setw(8) << correctness << "%\n";
    };
    std::cout << "\n" << std::left << std::setw(12) << "query_type"
              << std::right << std::setw(10) << "mean_us"
              << std::setw(10) << "p50_us"
              << std::setw(10) << "p95_us"
              << std::setw(10) << "p99_us"
              << std::setw(14) << "throughput"
              << std::setw(14) << "avg_results"
              << std::setw(9)  << "correct"
              << "\n";
    std::cout << std::string(87, '-') << "\n";
    print_row("point",      stats_point,  1.0,        correctness_pct);
    print_row("single_hop", stats_single, avg_single, 100.0);
    print_row("multi_hop",  stats_multi,  avg_multi,  100.0);

    // ── CSV ───────────────────────────────────────────────────────────────────
    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            std::string ds = dataset_short(data_file);
            std::string idxname = (baseline == "art") ? "ART" : "BinSearch";
            csv << "index,dataset,N,Q,query_type,build_s,index_rss_mb,"
                << "lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,"
                << "throughput_qps,avg_results_per_query,correctness_pct\n";
            csv << std::fixed;
            auto row = [&](const char* qt, size_t Q, const Stats& s,
                           double avg_results, double correct) {
                csv << idxname << "," << ds << "," << actual_N << "," << Q
                    << "," << qt
                    << "," << std::setprecision(6) << build_s
                    << "," << std::setprecision(3) << rss_mb
                    << "," << std::setprecision(4) << s.mean_us
                    << "," << std::setprecision(4) << s.p50_us
                    << "," << std::setprecision(4) << s.p95_us
                    << "," << std::setprecision(4) << s.p99_us
                    << "," << std::setprecision(0) << s.qps
                    << "," << std::setprecision(2) << avg_results
                    << "," << std::setprecision(2) << correct << "\n";
            };
            row("point",      Qp, stats_point,  1.0,        correctness_pct);
            row("single_hop", Qr, stats_single, avg_single, 100.0);
            row("multi_hop",  Qr, stats_multi,  avg_multi,  100.0);
            std::cout << "\nResults saved → " << out_csv << "\n";
        }
    }

    if (baseline == "art") art_tree_destroy(&tree);
    return 0;
}
