// bench_flood_k_sensitivity.cpp
// Sweeps Flood's compile-time grid parameter K to find the optimal grid
// resolution for GRAPH TRAVERSAL workloads (small result sets), as opposed to
// the OLAP analytical workloads Flood was originally calibrated for (K=20).
//
// K is a compile-time template parameter, so one executable is built per K via
// the FLOOD_K macro (see CMakeLists.txt foreach loop). Each run measures BOTH
// single-hop and multi-hop range query latency on one dataset and emits two
// CSV rows.
//
// Usage:
//   bench_flood_k_sensitivity <data.(txt|tpie)> <N>
//       [--queries Q]     queries per type   (default 10000)
//       [--seed S]        RNG seed           (default 42)
//       [--no_header]     suppress CSV header (for appending to a combined file)
//
// CSV schema (stdout):
//   dataset,K,query_type,mean_us,p50_us,p95_us,p99_us,build_time_ms,index_size_bytes,avg_results
//
// Query semantics (identical to bench_flood_wiki.cpp):
//   single_hop : pin SourceID (dim0), open Hop1 (dim1) + Hop2 (dim2)
//   multi_hop  : pin SourceID (dim0) + Hop1 (dim1), open Hop2 (dim2)

#include "../utils/type.hpp"
#include "../indexes/learned/flood.hpp"

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#ifndef FLOOD_K
#define FLOOD_K 20
#endif

static constexpr size_t DIM = 3;
static constexpr size_t K   = FLOOD_K;
static constexpr size_t EPS = 64;

using Point = point_t<DIM>;
using Box   = box_t<DIM>;
using FL    = bench::index::Flood<DIM, K, EPS>;

enum class QueryType { SINGLE_HOP, MULTI_HOP };

// ── data loading (identical dual-format logic as bench_flood_wiki.cpp) ────────
static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void load_tpie(const std::string& fname, std::vector<Point>& pts, size_t N) {
    pts.reserve(N);
    tpie::tpie_init();
    tpie::file_stream<double> in;
    in.open(fname);
    for (size_t i = 0; i < N; ++i) {
        Point p;
        for (size_t d = 0; d < DIM; ++d) p[d] = in.read();
        pts.push_back(p);
    }
    in.close();
    tpie::tpie_finish();
}

static void load_text(const std::string& fname, std::vector<Point>& pts, size_t N) {
    pts.reserve(N);
    std::ifstream f(fname);
    if (!f) { std::cerr << "ERROR: cannot open " << fname << "\n"; std::exit(1); }
    std::string line;
    size_t loaded = 0;
    while (loaded < N && std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Point p;
        double v;
        for (size_t d = 0; d < DIM; ++d) {
            if (!(ss >> v)) {
                std::cerr << "ERROR: short row at line " << (loaded + 1) << "\n";
                std::exit(1);
            }
            p[d] = v;
        }
        pts.push_back(p);
        ++loaded;
    }
    if (loaded < N)
        std::cerr << "WARNING: requested " << N << " rows but file only has " << loaded << "\n";
}

static void load_points(const std::string& fname, std::vector<Point>& pts, size_t N) {
    if (ends_with(fname, ".tpie")) load_tpie(fname, pts, N);
    else                           load_text(fname, pts, N);
}

static void compute_bounds(const std::vector<Point>& pts,
                           std::array<double, DIM>& dmin,
                           std::array<double, DIM>& dmax) {
    dmin.fill(std::numeric_limits<double>::max());
    dmax.fill(std::numeric_limits<double>::lowest());
    for (const auto& p : pts)
        for (size_t d = 0; d < DIM; ++d) {
            dmin[d] = std::min(dmin[d], p[d]);
            dmax[d] = std::max(dmax[d], p[d]);
        }
}

static Box build_box(const Point& q, QueryType qt,
                     const std::array<double, DIM>& dmin,
                     const std::array<double, DIM>& dmax) {
    Point lo, hi;
    if (qt == QueryType::SINGLE_HOP) {
        lo = {q[0], dmin[1], dmin[2]};
        hi = {q[0], dmax[1], dmax[2]};
    } else { // MULTI_HOP
        lo = {q[0], q[1], dmin[2]};
        hi = {q[0], q[1], dmax[2]};
    }
    return Box(lo, hi);
}

// Extract a short dataset label from the file path for the CSV "dataset" field.
static std::string dataset_short(const std::string& path) {
    auto slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const std::string suf = "_triples.txt";
    if (base.size() > suf.size() &&
        base.compare(base.size() - suf.size(), suf.size(), suf) == 0)
        return base.substr(0, base.size() - suf.size());
    if (base.size() > 4 && base.compare(base.size() - 4, 4, ".txt") == 0)
        return base.substr(0, base.size() - 4);
    return base;
}

struct Stats {
    double mean_us, p50_us, p95_us, p99_us, avg_results;
};

static Stats run_phase(FL& flood, const std::vector<Point>& qpts, QueryType qt,
                       const std::array<double, DIM>& dmin,
                       const std::array<double, DIM>& dmax) {
    std::vector<long long> lat_ns;
    lat_ns.reserve(qpts.size());
    size_t total_results = 0;

    for (const auto& qp : qpts) {
        Box box = build_box(qp, qt, dmin, dmax);
        auto t0 = std::chrono::steady_clock::now();
        auto res = flood.range_query(box);
        auto t1 = std::chrono::steady_clock::now();
        lat_ns.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        total_results += res.size();
    }

    Stats s;
    long long sum = 0;
    for (auto v : lat_ns) sum += v;
    s.mean_us = (double)sum / lat_ns.size() / 1000.0;
    std::sort(lat_ns.begin(), lat_ns.end());
    auto pct = [&](double p) {
        size_t i = (size_t)(p * lat_ns.size());
        if (i >= lat_ns.size()) i = lat_ns.size() - 1;
        return lat_ns[i] / 1000.0;
    };
    s.p50_us = pct(0.50);
    s.p95_us = pct(0.95);
    s.p99_us = pct(0.99);
    s.avg_results = (double)total_results / qpts.size();
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_flood_k_sensitivity <data.(txt|tpie)> <N>"
                  << " [--queries Q] [--seed S] [--no_header]\n";
        return 1;
    }

    std::string data_file = argv[1];
    size_t      N         = std::stoul(argv[2]);
    size_t      Q         = 10000;
    uint64_t    seed      = 42;
    bool        no_header = false;

    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--queries" && i + 1 < argc) Q = std::stoull(argv[++i]);
        else if (a == "--seed"    && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (a == "--no_header")               no_header = true;
    }

    std::vector<Point> points;
    load_points(data_file, points, N);
    if (points.empty()) { std::cerr << "ERROR: no points loaded\n"; return 2; }

    std::array<double, DIM> dmin, dmax;
    compute_bounds(points, dmin, dmax);

    // Sample query seed-points BEFORE building Flood (Flood sorts its input
    // in place). Sampling by value here makes the query sets fully independent
    // of K and of Flood's internal reordering, so all K executables see the
    // exact same workload for a given seed.
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, points.size() - 1);
    std::vector<Point> single_q(Q), multi_q(Q);
    for (auto& q : single_q) q = points[dist(rng)];
    for (auto& q : multi_q)  q = points[dist(rng)];

    // Build Flood (sorts `points` in place; query values already copied out).
    // Flood's constructor prints diagnostics to std::cout — silence them so the
    // CSV written to stdout stays clean. Redirect cout to a throwaway buffer for
    // the duration of construction, then restore.
    std::streambuf* cout_buf = std::cout.rdbuf();
    std::ostringstream sink;
    std::cout.rdbuf(sink.rdbuf());
    FL flood(points);
    std::cout.rdbuf(cout_buf);

    double   build_time_ms   = (double)flood.get_build_time();
    size_t   index_size_bytes = flood.index_size();

    Stats s_single = run_phase(flood, single_q, QueryType::SINGLE_HOP, dmin, dmax);
    Stats s_multi  = run_phase(flood, multi_q,  QueryType::MULTI_HOP,  dmin, dmax);

    std::string ds = dataset_short(data_file);

    std::ostream& out = std::cout;
    if (!no_header)
        out << "dataset,K,query_type,mean_us,p50_us,p95_us,p99_us,"
               "build_time_ms,index_size_bytes,avg_results\n";

    out << std::fixed;
    auto row = [&](const char* qt, const Stats& s) {
        out << ds << "," << K << "," << qt
            << "," << std::setprecision(4) << s.mean_us
            << "," << std::setprecision(4) << s.p50_us
            << "," << std::setprecision(4) << s.p95_us
            << "," << std::setprecision(4) << s.p99_us
            << "," << std::setprecision(3) << build_time_ms
            << "," << index_size_bytes
            << "," << std::setprecision(2) << s.avg_results << "\n";
    };
    row("single_hop", s_single);
    row("multi_hop",  s_multi);

    return 0;
}
