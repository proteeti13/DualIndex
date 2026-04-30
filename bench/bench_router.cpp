// bench_router.cpp
// DualIndex Router benchmark — mixed point / range query workload.
//
// Builds both ZM-Index and Flood on the same dataset, then generates a
// configurable mix of three query types and routes each through the router:
//
//   point      — all 3 dims pinned   → ZM-Index (point_lookup)
//   single_hop — dim0 fixed, 1+2 open → Flood (range_query)
//   multi_hop  — dims 0+1 fixed, 2 open → Flood (range_query)
//
// The routing decision is made at query time by checking whether the 3D box
// is degenerate (min == max in every dimension).  No query-type metadata is
// passed to the router — it derives the routing purely from the box shape.
//
// Usage:
//   bench_router <data.(txt|tpie)> <N>
//               [--queries Q]            default 10000
//               [--point_frac F]         fraction of Q as point queries  default 0.333
//               [--single_frac F]        fraction of Q as single_hop     default 0.333
//                                        remainder becomes multi_hop
//               [--seed S]               default 42
//               [--verbose_count K]      print first K routing decisions  default 10
//               [--out_csv FILE]

#include "../indexes/router.hpp"
#include "../utils/type.hpp"

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

using Point  = point_t<DIM>;
using Box    = box_t<DIM>;
using Router = bench::index::DualIndexRouter<DIM, K, EPS>;

enum class QueryType { POINT, SINGLE_HOP, MULTI_HOP };

static const char* qt_str(QueryType qt) {
    switch (qt) {
        case QueryType::POINT:      return "point";
        case QueryType::SINGLE_HOP: return "single_hop";
        case QueryType::MULTI_HOP:  return "multi_hop";
    }
    return "unknown";
}

static const char* route_str(Router::Route r) {
    return r == Router::Route::ZM_INDEX ? "ZM-Index" : "Flood";
}

struct QuerySpec {
    QueryType type;
    Box       box;
};

struct QueryRecord {
    QueryType    type;
    Router::Route route;
    size_t       result_count;
    long long    latency_ns;
    bool         point_found;
};

// ── data loading (identical to bench_flood_wiki.cpp) ─────────────────────────

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
    if (ends_with(fname, ".tpie")) {
        std::cout << "[load] TPIE binary: " << fname << "\n";
        load_tpie(fname, pts, N);
    } else {
        std::cout << "[load] text file:   " << fname << "\n";
        load_text(fname, pts, N);
    }
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

// ── workload generation ───────────────────────────────────────────────────────

static std::vector<QuerySpec> gen_workload(
    const std::vector<Point>& data,
    const std::array<double, DIM>& dmin,
    const std::array<double, DIM>& dmax,
    size_t Q_point, size_t Q_single, size_t Q_multi,
    uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);

    std::vector<QuerySpec> queries;
    queries.reserve(Q_point + Q_single + Q_multi);

    for (size_t i = 0; i < Q_point; ++i) {
        const Point& p = data[dist(rng)];
        queries.push_back({QueryType::POINT, Box(p, p)});
    }
    for (size_t i = 0; i < Q_single; ++i) {
        const Point& q = data[dist(rng)];
        Point lo = {q[0], dmin[1], dmin[2]};
        Point hi = {q[0], dmax[1], dmax[2]};
        queries.push_back({QueryType::SINGLE_HOP, Box(lo, hi)});
    }
    for (size_t i = 0; i < Q_multi; ++i) {
        const Point& q = data[dist(rng)];
        Point lo = {q[0], q[1], dmin[2]};
        Point hi = {q[0], q[1], dmax[2]};
        queries.push_back({QueryType::MULTI_HOP, Box(lo, hi)});
    }

    // Shuffle to simulate a realistic interleaved access pattern
    std::shuffle(queries.begin(), queries.end(), rng);
    return queries;
}

// ── per-type statistics ────────────────────────────────────────────────────────

struct TypeStats {
    size_t count       = 0;
    double sum_ns      = 0.0;
    double p95_us      = 0.0;
    double throughput  = 0.0;
    double avg_results = 0.0;
    size_t total_results = 0;
    size_t hits        = 0; // point query hits
};

static TypeStats compute_stats(const std::vector<QueryRecord>& records, QueryType qt) {
    TypeStats s;
    std::vector<long long> lats;
    for (const auto& r : records) {
        if (r.type != qt) continue;
        s.count++;
        s.sum_ns += static_cast<double>(r.latency_ns);
        s.total_results += r.result_count;
        if (r.point_found) s.hits++;
        lats.push_back(r.latency_ns);
    }
    if (s.count == 0) return s;
    std::sort(lats.begin(), lats.end());
    size_t p95i = static_cast<size_t>(0.95 * s.count);
    if (p95i >= s.count) p95i = s.count - 1;
    s.p95_us      = lats[p95i] / 1000.0;
    double mean_s = s.sum_ns / s.count / 1e9;
    s.throughput  = 1.0 / mean_s;
    s.avg_results = static_cast<double>(s.total_results) / s.count;
    return s;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_router <data.(txt|tpie)> <N>\n"
                  << "  [--queries Q]          (default 10000)\n"
                  << "  [--point_frac F]       (default 0.333)\n"
                  << "  [--single_frac F]      (default 0.333)\n"
                  << "  [--seed S]             (default 42)\n"
                  << "  [--verbose_count K]    (default 10)\n"
                  << "  [--out_csv FILE]\n";
        return 1;
    }

    std::string data_file    = argv[1];
    size_t      N            = std::stoul(argv[2]);
    size_t      Q            = 10000;
    double      point_frac   = 1.0 / 3.0;
    double      single_frac  = 1.0 / 3.0;
    uint64_t    seed         = 42;
    size_t      verbose_cnt  = 10;
    std::string out_csv;

    for (int i = 3; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--queries")       Q           = std::stoull(argv[i+1]);
        else if (arg == "--point_frac")    point_frac  = std::stod(argv[i+1]);
        else if (arg == "--single_frac")   single_frac = std::stod(argv[i+1]);
        else if (arg == "--seed")          seed        = std::stoull(argv[i+1]);
        else if (arg == "--verbose_count") verbose_cnt = std::stoull(argv[i+1]);
        else if (arg == "--out_csv")       out_csv     = argv[i+1];
    }

    size_t Q_point  = static_cast<size_t>(Q * point_frac);
    size_t Q_single = static_cast<size_t>(Q * single_frac);
    size_t Q_multi  = Q - Q_point - Q_single;

    // ── load data ─────────────────────────────────────────────────────────────
    std::cout << "=== DualIndex Router Benchmark ===\n";
    std::cout << "Loading: " << data_file << "  N=" << N << "\n";
    std::vector<Point> data;
    load_points(data_file, data, N);
    N = data.size();
    std::cout << "Loaded " << N << " points.\n\n";

    // ── global bounds (needed to build open-ended range boxes) ────────────────
    std::array<double, DIM> dmin, dmax;
    compute_bounds(data, dmin, dmax);
    std::cout << "Bounds: dim0=[" << dmin[0] << "," << dmax[0] << "]"
              << "  dim1=[" << dmin[1] << "," << dmax[1] << "]"
              << "  dim2=[" << dmin[2] << "," << dmax[2] << "]\n\n";

    // ── build router (builds ZM-Index + Flood on owned data copies) ───────────
    std::cout << "Building DualIndexRouter...\n";
    Router router(data);
    std::cout << "\nZM-Index  — build: " << std::fixed << std::setprecision(3)
              << router.zm_build_s()    << " s   size: "
              << std::setprecision(2)   << router.zm_index_mb()    << " MB\n";
    std::cout << "Flood     — build: " << std::setprecision(3)
              << router.flood_build_s() << " s   size: "
              << std::setprecision(2)   << router.flood_index_mb() << " MB\n\n";

    // ── generate mixed workload ────────────────────────────────────────────────
    std::cout << "Query workload:  total=" << Q
              << "  point=" << Q_point
              << "  single_hop=" << Q_single
              << "  multi_hop=" << Q_multi
              << "  (shuffled, seed=" << seed << ")\n";
    std::cout << "  point      → ZM-Index  (point_lookup)\n";
    std::cout << "  single_hop → Flood     (range_query, 1 dim fixed)\n";
    std::cout << "  multi_hop  → Flood     (range_query, 2 dims fixed)\n\n";

    auto workload = gen_workload(data, dmin, dmax, Q_point, Q_single, Q_multi, seed);
    Q = workload.size();

    // ── run timed benchmark ────────────────────────────────────────────────────
    std::vector<QueryRecord> records;
    records.reserve(Q);

    for (auto& qs : workload) {
        auto result = router.query(qs.box);
        records.push_back({qs.type, result.route, result.result_count,
                           result.latency_ns, result.point_found});
    }

    // ── verbose routing trace ──────────────────────────────────────────────────
    size_t show = std::min(verbose_cnt, Q);
    if (show > 0) {
        std::cout << "Routing trace — first " << show << " queries:\n";
        std::cout << std::string(72, '-') << "\n";
        std::cout << std::left
                  << std::setw(6)  << "Query"
                  << std::setw(13) << "Type"
                  << std::setw(12) << "Routed to"
                  << std::setw(10) << "Results"
                  << std::setw(12) << "Lat (us)"
                  << "Note\n";
        std::cout << std::string(72, '-') << "\n";
        for (size_t i = 0; i < show; ++i) {
            const auto& r = records[i];
            std::cout << std::left
                      << std::setw(6)  << (i + 1)
                      << std::setw(13) << qt_str(r.type)
                      << std::setw(12) << route_str(r.route)
                      << std::setw(10) << r.result_count
                      << std::fixed << std::setprecision(2)
                      << std::setw(12) << (r.latency_ns / 1000.0);
            if (r.type == QueryType::POINT)
                std::cout << (r.point_found ? "found" : "not found");
            std::cout << "\n";
        }
        std::cout << std::string(72, '-') << "\n\n";
    }

    // ── per-type statistics ────────────────────────────────────────────────────
    auto s_point  = compute_stats(records, QueryType::POINT);
    auto s_single = compute_stats(records, QueryType::SINGLE_HOP);
    auto s_multi  = compute_stats(records, QueryType::MULTI_HOP);

    // overall
    double total_ns = s_point.sum_ns + s_single.sum_ns + s_multi.sum_ns;
    size_t total_Q  = s_point.count + s_single.count + s_multi.count;
    double overall_mean_us    = (total_ns / total_Q) / 1000.0;
    double overall_thruput    = total_Q / (total_ns / 1e9);
    double overall_avg_results = static_cast<double>(
        s_point.total_results + s_single.total_results + s_multi.total_results) / total_Q;

    auto print_row = [](const char* type, const char* index,
                        size_t cnt, const TypeStats& s, bool is_point_type) {
        std::cout << std::left
                  << std::setw(13) << type
                  << std::setw(11) << index
                  << std::right << std::setw(7) << cnt << "   ";
        std::cout << std::fixed << std::setprecision(2);
        double mean_us = (s.sum_ns / s.count) / 1000.0;
        std::cout << std::setw(10) << mean_us
                  << std::setw(10) << s.p95_us;
        std::cout << std::defaultfloat << std::setw(13)
                  << static_cast<long long>(s.throughput);
        if (is_point_type)
            std::cout << std::fixed << std::setprecision(1) << std::setw(11)
                      << (100.0 * s.hits / s.count) << "%";
        else
            std::cout << std::fixed << std::setprecision(1) << std::setw(12)
                      << s.avg_results;
        std::cout << "\n";
    };

    std::cout << "Per-type results:\n";
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::left
              << std::setw(13) << "Type"
              << std::setw(11) << "Index"
              << std::right << std::setw(7) << "Count" << "   "
              << std::setw(10) << "Mean(us)"
              << std::setw(10) << "P95(us)"
              << std::setw(13) << "QPS"
              << std::setw(12) << "AvgResults" << "\n";
    std::cout << std::string(72, '-') << "\n";
    if (s_point.count  > 0) print_row("point",      "ZM-Index", s_point.count,  s_point,  true);
    if (s_single.count > 0) print_row("single_hop", "Flood",    s_single.count, s_single, false);
    if (s_multi.count  > 0) print_row("multi_hop",  "Flood",    s_multi.count,  s_multi,  false);
    std::cout << std::string(72, '-') << "\n";
    std::cout << std::left << std::setw(24) << "OVERALL"
              << std::right << std::setw(7) << total_Q << "   "
              << std::fixed << std::setprecision(2)
              << std::setw(10) << overall_mean_us
              << std::setw(10) << "---"
              << std::defaultfloat
              << std::setw(13) << static_cast<long long>(overall_thruput)
              << std::fixed << std::setprecision(1)
              << std::setw(12) << overall_avg_results << "\n";
    std::cout << std::string(72, '-') << "\n\n";

    std::cout << "Config:   DualIndex K=" << K << " Eps=" << EPS << " Dim=" << DIM << "\n"
              << "Dataset:  " << data_file << "  N=" << N << "\n\n";

    // ── CSV output ─────────────────────────────────────────────────────────────
    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            csv << "index,dataset,N,Q,query_type,routed_to,K,epsilon,"
                << "build_s_zm,build_s_flood,index_mb_zm,index_mb_flood,"
                << "lat_mean_us,lat_p95_us,throughput_qps,avg_results_per_query\n";
            csv << std::fixed;

            auto write_row = [&](const char* qtype, const char* routed,
                                 size_t q_count, const TypeStats& s) {
                if (q_count == 0) return;
                double mean_us = (s.sum_ns / s.count) / 1000.0;
                csv << "dualindex," << data_file << "," << N << "," << q_count
                    << "," << qtype << "," << routed
                    << "," << K << "," << EPS
                    << "," << std::setprecision(6) << router.zm_build_s()
                    << "," << std::setprecision(6) << router.flood_build_s()
                    << "," << std::setprecision(4) << router.zm_index_mb()
                    << "," << std::setprecision(4) << router.flood_index_mb()
                    << "," << std::setprecision(4) << mean_us
                    << "," << std::setprecision(4) << s.p95_us
                    << "," << std::setprecision(0) << s.throughput
                    << "," << std::setprecision(2) << s.avg_results << "\n";
            };

            write_row("point",      "zmindex", s_point.count,  s_point);
            write_row("single_hop", "flood",   s_single.count, s_single);
            write_row("multi_hop",  "flood",   s_multi.count,  s_multi);

            // summary row across all types
            csv << "dualindex," << data_file << "," << N << "," << total_Q
                << ",all,mixed," << K << "," << EPS
                << "," << std::setprecision(6) << router.zm_build_s()
                << "," << std::setprecision(6) << router.flood_build_s()
                << "," << std::setprecision(4) << router.zm_index_mb()
                << "," << std::setprecision(4) << router.flood_index_mb()
                << "," << std::setprecision(4) << overall_mean_us
                << "," << std::setprecision(4) << 0.0  // p95 across mixed types is uninformative
                << "," << std::setprecision(0) << overall_thruput
                << "," << std::setprecision(2)
                << static_cast<double>(s_point.total_results + s_single.total_results + s_multi.total_results) / total_Q
                << "\n";

            std::cout << "Results saved → " << out_csv << "\n";
        }
    }

    return 0;
}
