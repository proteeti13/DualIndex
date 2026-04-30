// bench_flood_wiki.cpp
// Flood range-query benchmark: single-hop and multi-hop graph traversal queries.
//
// ── What changed from the original (standalone repo) version ─────────────────
// REMOVED: degenerate point query approach ([x,x]×[y,y]×[z,z]).
//   The original harness encoded point lookups as zero-width range boxes because
//   Flood has no point-query API. This was a workaround that was only needed when
//   testing Flood against ZM-Index on the same point-query workload.
//   In the combined model, point queries are routed to ZM-Index; Flood is only
//   ever called for genuine range queries. The degenerate box code is gone.
//
// REMOVED: external binary query file (queries.bin).
//   The original required a pre-generated binary query file. Queries are now
//   sampled directly from the loaded dataset at runtime (same approach as
//   bench_zmindex_all), making the harness self-contained.
//
// ADDED: --query_type single_hop|multi_hop
//   Two real range query types that reflect actual graph traversal workloads:
//   single_hop  – Box([X,X] × [dmin1,dmax1] × [dmin2,dmax2])
//                 "Find ALL 2-hop paths from source node X"
//   multi_hop   – Box([X,X] × [Y,Y] × [dmin2,dmax2])
//                 "Find all destinations reachable from X through intermediate Y"
//
// ADDED: compute_bounds() — scans the loaded data to get global min/max for all
//   3 dimensions before building the index. Required to construct open-ended
//   range boxes. Flood internally only stores bounds for dims 0 and 1 (the
//   non-sort dims), so dim 2 bounds must be computed separately in the harness.
//
// ADDED: avg_results_per_query in output and CSV.
//   Range queries return variable-size result sets. This metric shows how "wide"
//   each query is — important for understanding latency differences between
//   single_hop (returns thousands) and multi_hop (returns tens to hundreds).
//
// UNCHANGED: flood.hpp algorithm, K=20, Eps=64, SortDim=2, .txt/.tpie loading.
// ─────────────────────────────────────────────────────────────────────────────
//
// Usage:
//   bench_flood_wiki <data.(txt|tpie)> <N>
//                    [--query_type single_hop|multi_hop]
//                    [--queries Q] [--seed S]
//                    [--out_csv FILE]
//
//   data.tpie : TPIE binary file — 3 doubles per point (fast)
//   data.txt  : space-separated: SourceID Hop1_ID Hop2_ID [extra cols ignored]
//   N         : number of data points to load

#include "../utils/type.hpp"
#include "../indexes/learned/flood.hpp"

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static constexpr size_t DIM = 3;
static constexpr size_t K   = 20;
static constexpr size_t EPS = 64;

using Point = point_t<DIM>;
using Box   = box_t<DIM>;
using Flood = bench::index::Flood<DIM, K, EPS>;

enum class QueryType { SINGLE_HOP, MULTI_HOP };

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
    if (!f) {
        std::cerr << "ERROR: cannot open " << fname << "\n";
        std::exit(1);
    }
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
    if (ends_with(fname, ".tpie")) {
        std::cout << "[load] TPIE binary: " << fname << "\n";
        load_tpie(fname, pts, N);
    } else {
        std::cout << "[load] text file:   " << fname << "\n";
        load_text(fname, pts, N);
    }
}

// Compute per-dimension min/max across all loaded points.
static void compute_bounds(const std::vector<Point>& pts,
                           std::array<double, DIM>& dmin,
                           std::array<double, DIM>& dmax) {
    dmin.fill(std::numeric_limits<double>::max());
    dmax.fill(std::numeric_limits<double>::lowest());
    for (const auto& p : pts) {
        for (size_t d = 0; d < DIM; ++d) {
            dmin[d] = std::min(dmin[d], p[d]);
            dmax[d] = std::max(dmax[d], p[d]);
        }
    }
}

// Sample Q random rows from data. The sampled row is the query "seed" —
// we extract the relevant dimensions per query type in build_box().
static std::vector<Point> sample_queries(const std::vector<Point>& pts,
                                         size_t Q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);
    std::vector<Point> qpts(Q);
    for (auto& q : qpts) q = pts[dist(rng)];
    return qpts;
}

// Construct the range box for the given query type.
//
//   single_hop: fix SourceID (dim 0), open range on dims 1 and 2.
//               "Return all 2-hop paths from source X."
//
//   multi_hop:  fix SourceID (dim 0) AND Hop1_ID (dim 1), open range on dim 2.
//               "Return all destinations reachable from X through intermediate Y."
static Box build_box(const Point& q, QueryType qt,
                     const std::array<double, DIM>& dmin,
                     const std::array<double, DIM>& dmax) {
    Point lo, hi;
    switch (qt) {
        case QueryType::SINGLE_HOP:
            lo = {q[0], dmin[1], dmin[2]};
            hi = {q[0], dmax[1], dmax[2]};
            break;
        case QueryType::MULTI_HOP:
            lo = {q[0], q[1], dmin[2]};
            hi = {q[0], q[1], dmax[2]};
            break;
    }
    return Box(lo, hi);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_flood_wiki <data.(txt|tpie)> <N>"
                  << " [--query_type single_hop|multi_hop]"
                  << " [--queries Q] [--seed S] [--out_csv FILE]\n";
        return 1;
    }

    std::string data_file = argv[1];
    size_t      N         = std::stoul(argv[2]);
    size_t      Q         = 100000;
    uint64_t    seed      = 42;
    QueryType   qt        = QueryType::SINGLE_HOP;
    std::string qt_name   = "single_hop";
    std::string out_csv;

    for (int i = 3; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--queries") {
            Q = std::stoull(argv[i + 1]);
        } else if (arg == "--seed") {
            seed = std::stoull(argv[i + 1]);
        } else if (arg == "--query_type") {
            qt_name = argv[i + 1];
            if (qt_name == "multi_hop") qt = QueryType::MULTI_HOP;
            else                        qt = QueryType::SINGLE_HOP;
        } else if (arg == "--out_csv") {
            out_csv = argv[i + 1];
        }
    }

    // ---- Load data ----
    std::cout << "Loading data: " << data_file << " (" << N << " points)\n";
    std::vector<Point> points;
    load_points(data_file, points, N);
    std::cout << "Loaded " << points.size() << " points.\n";

    // ---- Compute global bounds (needed to build open-ended range boxes) ----
    std::array<double, DIM> dmin, dmax;
    compute_bounds(points, dmin, dmax);
    std::cout << "Bounds: dim0=[" << dmin[0] << "," << dmax[0] << "]"
              << "  dim1=[" << dmin[1] << "," << dmax[1] << "]"
              << "  dim2=[" << dmin[2] << "," << dmax[2] << "]\n";

    // ---- Build Flood index ----
    Flood flood(points);

    // ---- Sample queries from dataset ----
    std::vector<Point> queries = sample_queries(points, Q, seed);
    Q = queries.size();
    std::cout << "Running " << Q << " " << qt_name << " queries (seed=" << seed << ")\n";

    // ---- Run range queries ----
    std::vector<long long> latencies_ns;
    latencies_ns.reserve(Q);
    size_t total_results = 0;

    for (const auto& qp : queries) {
        Box box = build_box(qp, qt, dmin, dmax);
        auto t0 = std::chrono::steady_clock::now();
        auto res = flood.range_query(box);
        auto t1 = std::chrono::steady_clock::now();
        latencies_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        total_results += res.size();
    }

    // ---- Compute metrics ----
    double sum_ns = 0.0;
    for (auto t : latencies_ns) sum_ns += static_cast<double>(t);
    double mean_us = (sum_ns / Q) / 1000.0;

    std::vector<long long> sorted_lat = latencies_ns;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    size_t p95_idx = static_cast<size_t>(0.95 * Q);
    if (p95_idx >= Q) p95_idx = Q - 1;
    double p95_us    = sorted_lat[p95_idx] / 1000.0;
    double total_s   = sum_ns / 1e9;
    double throughput = Q / total_s;
    double build_s   = flood.get_build_time() / 1000.0;
    double index_mb  = flood.index_size() / 1e6;
    double avg_results = static_cast<double>(total_results) / Q;

    // ---- Print results ----
    std::cout << "\n"
              << "-------------------------------------------------------\n"
              << "Flood Range Query Results\n"
              << "-------------------------------------------------------\n"
              << std::fixed << std::setprecision(4)
              << "Build Time (s):           " << build_s  << "\n"
              << "Index Size (MB):          " << index_mb << "\n\n"
              << "Query Type:               " << qt_name  << "\n"
              << "Mean Query Latency (us):  " << mean_us  << "\n"
              << "P95 Query Latency (us):   " << p95_us   << "\n"
              << std::defaultfloat
              << "Query Throughput (q/s):   " << static_cast<long long>(throughput) << "\n"
              << std::fixed << std::setprecision(2)
              << "Avg Results per Query:    " << avg_results << "\n"
              << "-------------------------------------------------------\n"
              << "Queries executed:         " << Q        << "\n"
              << "Flood Config:             Dim=" << DIM << " K=" << K << " Eps=" << EPS << "\n"
              << "Input file:               " << data_file << "\n"
              << "-------------------------------------------------------\n";

    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            csv << "index,dataset,N,Q,query_type,K,epsilon,"
                << "build_s,index_mb,"
                << "lat_mean_us,lat_p95_us,throughput_qps,avg_results_per_query\n";
            csv << std::fixed;
            csv << "flood," << data_file << "," << points.size() << "," << Q
                << "," << qt_name
                << "," << K << "," << EPS
                << "," << std::setprecision(6) << build_s
                << "," << std::setprecision(4) << index_mb
                << "," << std::setprecision(4) << mean_us
                << "," << std::setprecision(4) << p95_us
                << "," << std::setprecision(0) << throughput
                << "," << std::setprecision(2) << avg_results << "\n";
            std::cout << "Results saved → " << out_csv << "\n";
        }
    }

    return 0;
}
