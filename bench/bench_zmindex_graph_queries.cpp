/**
 * bench_zmindex_graph_queries.cpp
 *
 * Benchmarks ZM-Index with three graph-traversal query types:
 *
 *   point      — "Does the exact path source→hop1→hop2 exist?"
 *                e.g. "did user 3 vote for user 28, who then voted for user 6?"
 *                Internally: 3D point lookup.
 *
 *   single_hop — "Who are all the direct friends of node X?"
 *                e.g. "everyone user 3 voted for directly"
 *                Internally: 3D range box  [X,X] × [all] × [all]
 *
 *   multi_hop  — "Starting from X via intermediate Y, who can X reach in 2 hops?"
 *                e.g. "everyone reachable from user 3 through user 28"
 *                Internally: 3D range box  [X,X] × [Y,Y] × [all]
 *
 * Usage:
 *   bench_zmindex_graph_queries <file.(txt|tpie)> <N>
 *     [--query_type <point|single_hop|multi_hop>]
 *     [--queries Q]       (default 10000)
 *     [--seed S]          (default 42)
 *     [--dump_queries FILE]
 *     [--verbose]         (trace first 5 queries internally)
 *     [--verbose_count K] (trace first K queries, default 5)
 *     [--out_csv FILE]
 *
 * How ZM-Index works internally (brief):
 *   1. All 3D points are mapped to discrete grid cells (resolution ≈ N^(1/3)).
 *   2. Each (cell_x, cell_y, cell_z) triple is Morton-encoded into a single
 *      uint64 by interleaving the bits of the three cell IDs (Z-order curve).
 *   3. These uint64 keys are sorted and a 1D PGM learned index is built on them.
 *   4. For a point query: encode the query point → PGM predicts its position
 *      in the sorted array ± Epsilon → scan a window of 2*Epsilon+2 entries.
 *   5. For a range query: the 3D box is decomposed into Z-order curve segments,
 *      then each segment is searched with the 1D PGM.
 */

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

#include "../utils/type.hpp"
#include "../indexes/learned/zmindex.hpp"

// ── constants ─────────────────────────────────────────────────────────────
static constexpr size_t DIM  = 3;
static constexpr size_t EPSI = 64;

using Point = point_t<DIM>;
using Box   = box_t<DIM>;
using ZM    = bench::index::ZMIndex<DIM, EPSI>;

static const char* const DIM_LABEL[3] = {
    "SourceID (dim0)",
    "Hop1_ID  (dim1)",
    "Hop2_ID  (dim2)"
};

// ── query types ───────────────────────────────────────────────────────────
enum class QueryType { POINT, SINGLE_HOP, MULTI_HOP };

static QueryType parse_query_type(const std::string& s) {
    if (s == "point")      return QueryType::POINT;
    if (s == "single_hop") return QueryType::SINGLE_HOP;
    if (s == "multi_hop")  return QueryType::MULTI_HOP;
    std::cerr << "ERROR: unknown --query_type '" << s
              << "'.  Choose: point | single_hop | multi_hop\n";
    std::exit(1);
}

static const char* qt_name(QueryType qt) {
    switch (qt) {
        case QueryType::POINT:      return "point";
        case QueryType::SINGLE_HOP: return "single_hop";
        case QueryType::MULTI_HOP:  return "multi_hop";
    }
    return "unknown";
}

// ── data loading (same dual-format logic as bench_zmindex_wv.cpp) ─────────
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
        std::cerr << "WARNING: requested " << N << " rows but file has only " << loaded << "\n";
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

// ── query generation ──────────────────────────────────────────────────────
// Queries are always sampled from the data, so every query is guaranteed
// to have at least one matching triple in the index.
//
// The returned vector stores full Point triples.  For single_hop and
// multi_hop only the first 1 or 2 components are meaningful; the rest
// are zeroed.  build_box() uses only the relevant components.
static std::vector<Point> gen_queries(const std::vector<Point>& data,
                                       QueryType qt, size_t Q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, data.size() - 1);
    std::vector<Point> qs(Q);
    for (auto& q : qs) {
        const Point& src = data[dist(rng)];
        switch (qt) {
            case QueryType::POINT:
                q = src;
                break;
            case QueryType::SINGLE_HOP:
                q = {src[0], 0.0, 0.0};   // only SourceID matters
                break;
            case QueryType::MULTI_HOP:
                q = {src[0], src[1], 0.0}; // SourceID + Hop1_ID
                break;
        }
    }
    return qs;
}

// Build the 3D search box for range queries from a query point and global bounds.
static Box build_box(const Point& q, QueryType qt,
                     const std::array<double, DIM>& mins,
                     const std::array<double, DIM>& maxs) {
    Point lo, hi;
    switch (qt) {
        case QueryType::SINGLE_HOP:
            lo = {q[0], mins[1], mins[2]};
            hi = {q[0], maxs[1], maxs[2]};
            break;
        case QueryType::MULTI_HOP:
            lo = {q[0], q[1], mins[2]};
            hi = {q[0], q[1], maxs[2]};
            break;
        default:
            lo = hi = q; // degenerate box (point query fallback)
    }
    return Box(lo, hi);
}

// ── query dump ────────────────────────────────────────────────────────────
// Writes a human-readable file with a header explaining each column.
static void dump_queries(const std::vector<Point>& qs, QueryType qt,
                          const std::string& path,
                          const std::string& dataset, size_t N, uint64_t seed) {
    std::ofstream f(path);
    if (!f) { std::cerr << "WARNING: cannot write query file to " << path << "\n"; return; }

    f << "# ZM-Index graph query workload\n";
    f << "# query_type : " << qt_name(qt) << "\n";
    f << "# dataset    : " << dataset << "\n";
    f << "# N (loaded) : " << N << "\n";
    f << "# Q (queries): " << qs.size() << "\n";
    f << "# seed       : " << seed << "\n";

    switch (qt) {
        case QueryType::POINT:
            f << "# semantics  : exact match - does the path source -> hop1 -> hop2 exist?\n";
            f << "# graph note : each row is one 2-hop path in the Wiki-Vote network.\n";
            f << "#              source_id voted for hop1_id, who then voted for hop2_id.\n";
            f << "# columns    : source_id  hop1_id  hop2_id\n";
            f << "# -------------------------------------------------------\n";
            for (const auto& q : qs)
                f << std::fixed << std::setprecision(0)
                  << q[0] << " " << q[1] << " " << q[2] << "\n";
            break;

        case QueryType::SINGLE_HOP:
            f << "# semantics  : find all direct friends of source_id\n";
            f << "# graph note : returns all triples where the first column == source_id.\n";
            f << "#              unique hop1_id values in results = direct neighbors.\n";
            f << "# index op   : 3D range box [source_id, source_id] x [all] x [all]\n";
            f << "# columns    : source_id\n";
            f << "# -------------------------------------------------------\n";
            for (const auto& q : qs)
                f << std::fixed << std::setprecision(0) << q[0] << "\n";
            break;

        case QueryType::MULTI_HOP:
            f << "# semantics  : find all 2-hop neighbors reachable from source_id through hop1_id\n";
            f << "# graph note : fix source and intermediate node, enumerate all destinations.\n";
            f << "#              e.g. source=3 hop1=28 -> all Z where 3->28->Z exists\n";
            f << "# index op   : 3D range box [source_id, source_id] x [hop1_id, hop1_id] x [all]\n";
            f << "# columns    : source_id  hop1_id\n";
            f << "# -------------------------------------------------------\n";
            for (const auto& q : qs)
                f << std::fixed << std::setprecision(0) << q[0] << " " << q[1] << "\n";
            break;
    }
    std::cout << "Queries saved -> " << path << "  (" << qs.size() << " queries)\n";
}

// ── verbose internal trace ────────────────────────────────────────────────
// Shows what ZM-Index does step-by-step for one query.
static void trace_query(ZM& zm, const Point& q, QueryType qt, int idx,
                        const std::array<double, DIM>& mins,
                        const std::array<double, DIM>& maxs) {
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "\n  +-- Query #" << idx << "  [" << qt_name(qt) << "] ";

    switch (qt) {
        case QueryType::POINT:
            std::cout << "----------------------------------------------\n";
            std::cout << "  | Semantic: Does the exact path  "
                      << q[0] << " -> " << q[1] << " -> " << q[2] << "  exist?\n";
            std::cout << "  |\n";
            std::cout << "  | STEP 1 -- Normalize each value to a discrete grid cell:\n";
            std::cout << "  |   (The index divides each dimension into resolution="
                      << zm.get_resolution() << " equal-width buckets)\n";
            {
                auto cells = zm.point_to_cells(q);
                auto& ws   = zm.get_widths();
                for (size_t d = 0; d < DIM; ++d)
                    std::cout << "  |   " << DIM_LABEL[d] << "  val=" << std::setw(9) << q[d]
                              << "   range=[" << mins[d] << ", " << maxs[d] << "]"
                              << "   cell_width=" << ws[d]
                              << "   -> cell " << cells[d] << "\n";
                std::cout << "  |\n";
                std::cout << "  | STEP 2 -- Morton (Z-order) encode: interleave bits of ("
                          << cells[0] << ", " << cells[1] << ", " << cells[2] << ")\n";
                std::cout << "  |   Each cell ID contributes bits at every 3rd position in\n";
                std::cout << "  |   the final uint64. This is done in one CPU instruction (PDEP).\n";
                std::cout << "  |   Result: a single uint64 key that sorts nearby 3D cells together.\n";
            }
            std::cout << "  |\n";
            std::cout << "  | STEP 3 -- PGM learned-index lookup:\n";
            std::cout << "  |   The PGM predicts the key's rank in the sorted Morton array.\n";
            std::cout << "  |   Prediction error is bounded by Epsilon=" << EPSI
                      << ", so it scans a window\n";
            std::cout << "  |   of at most 2*" << EPSI << "+2 = " << 2*EPSI+2 << " entries.\n";
            {
                auto res = zm.point_lookup(const_cast<Point&>(q));
                std::cout << "  |\n";
                std::cout << "  | STEP 4 -- Result: "
                          << (res.found ? "FOUND  (Morton code present in window)" :
                                          "NOT FOUND  (code absent from window)")
                          << "\n";
                std::cout << "  |   PGM window scanned: " << res.pgm_window << " entries\n";
            }
            break;

        case QueryType::SINGLE_HOP: {
            std::cout << "----------------------------------------------\n";
            std::cout << "  | Semantic: Find all direct friends of node " << q[0] << "\n";
            std::cout << "  |   Graph query: MATCH (n {id:" << q[0]
                      << "})-[:EDGE]->(friend) RETURN friend\n";
            std::cout << "  |\n";
            Point lo = {q[0], mins[1], mins[2]};
            Point hi = {q[0], maxs[1], maxs[2]};
            Box box(lo, hi);
            auto lo_cells = zm.point_to_cells(lo);
            auto hi_cells = zm.point_to_cells(hi);
            std::cout << "  | STEP 1 -- Build 3D box (fix dim0=SourceID, let dim1+dim2 range freely):\n";
            std::cout << "  |   lo corner: (" << lo[0] << ", " << lo[1] << ", " << lo[2] << ")\n";
            std::cout << "  |   hi corner: (" << hi[0] << ", " << hi[1] << ", " << hi[2] << ")\n";
            std::cout << "  |\n";
            std::cout << "  | STEP 2 -- Convert box to cell coordinates:\n";
            for (size_t d = 0; d < DIM; ++d)
                std::cout << "  |   dim" << d << ": cells [" << lo_cells[d]
                          << ", " << hi_cells[d] << "]\n";
            std::cout << "  |\n";
            std::cout << "  | STEP 3 -- Z-order range scan:\n";
            std::cout << "  |   The 3D box is decomposed into Z-order curve intervals.\n";
            std::cout << "  |   Because Z-order doesn't perfectly align with axis-aligned boxes,\n";
            std::cout << "  |   a single box may become MANY non-contiguous Morton ranges.\n";
            std::cout << "  |   Each range is searched with the 1D PGM.\n";
            std::cout << "  |   (Wide box on dim1+dim2 means many intervals -> slower than multi_hop)\n";
            {
                auto t0 = std::chrono::steady_clock::now();
                auto res = zm.range_query(box);
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                std::set<double> unique_hop1;
                for (auto& r : res) unique_hop1.insert(r[1]);
                std::cout << "  |\n";
                std::cout << "  | STEP 4 -- Result: " << res.size()
                          << " matching triples in " << std::setprecision(2) << us << " us\n";
                std::cout << "  |   Unique direct neighbors (Hop1_ID values): "
                          << unique_hop1.size() << "\n";
            }
            break;
        }

        case QueryType::MULTI_HOP: {
            std::cout << "----------------------------------------------\n";
            std::cout << "  | Semantic: 2-hop neighbors of " << q[0]
                      << " reachable through " << q[1] << "\n";
            std::cout << "  |   Graph query: MATCH (n {id:" << q[0]
                      << "})-[:EDGE]->(m {id:" << q[1]
                      << "})-[:EDGE]->(dest) RETURN dest\n";
            std::cout << "  |\n";
            Point lo = {q[0], q[1], mins[2]};
            Point hi = {q[0], q[1], maxs[2]};
            Box box(lo, hi);
            auto lo_cells = zm.point_to_cells(lo);
            auto hi_cells = zm.point_to_cells(hi);
            std::cout << "  | STEP 1 -- Build 3D box (fix dim0+dim1, let dim2 range freely):\n";
            std::cout << "  |   lo corner: (" << lo[0] << ", " << lo[1] << ", " << lo[2] << ")\n";
            std::cout << "  |   hi corner: (" << hi[0] << ", " << hi[1] << ", " << hi[2] << ")\n";
            std::cout << "  |\n";
            std::cout << "  | STEP 2 -- Convert box to cell coordinates:\n";
            for (size_t d = 0; d < DIM; ++d)
                std::cout << "  |   dim" << d << ": cells [" << lo_cells[d]
                          << ", " << hi_cells[d] << "]\n";
            std::cout << "  |\n";
            std::cout << "  | STEP 3 -- Z-order range scan:\n";
            std::cout << "  |   Narrower box (dim0+dim1 fixed) -> fewer Z-order intervals.\n";
            std::cout << "  |   Tighter Morton-code range -> faster PGM search than single_hop.\n";
            {
                auto t0 = std::chrono::steady_clock::now();
                auto res = zm.range_query(box);
                auto t1 = std::chrono::steady_clock::now();
                double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
                std::cout << "  |\n";
                std::cout << "  | STEP 4 -- Result: " << res.size()
                          << " 2-hop paths found in "
                          << std::setprecision(2) << us << " us\n";
                std::cout << "  |   Reachable 2-hop destinations (Hop2_ID values): "
                          << res.size() << "\n";
            }
            break;
        }
    }
    std::cout << "  +-------------------------------------------------------\n";
}

// ── main ─────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_zmindex_graph_queries <file.(txt|tpie)> <N>\n"
                  << "  [--query_type <point|single_hop|multi_hop>]  (default: point)\n"
                  << "  [--queries Q]        (default: 10000)\n"
                  << "  [--seed S]           (default: 42)\n"
                  << "  [--dump_queries FILE]\n"
                  << "  [--verbose]          (trace first 5 queries step-by-step)\n"
                  << "  [--verbose_count K]  (how many to trace, default 5)\n"
                  << "  [--out_csv FILE]\n";
        return 1;
    }

    const std::string fname = argv[1];
    const size_t      N     = std::stoull(argv[2]);

    QueryType   qt            = QueryType::POINT;
    size_t      Q             = 10000;
    uint64_t    seed          = 42;
    std::string dump_file;
    std::string out_csv;
    bool        verbose       = false;
    size_t      verbose_count = 5;

    for (int i = 3; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verbose") { verbose = true; continue; }
        if (i + 1 >= argc) continue;
        if      (arg == "--query_type")    qt            = parse_query_type(argv[++i]);
        else if (arg == "--queries")       Q             = std::stoull(argv[++i]);
        else if (arg == "--seed")          seed          = std::stoull(argv[++i]);
        else if (arg == "--dump_queries")  dump_file     = argv[++i];
        else if (arg == "--out_csv")       out_csv       = argv[++i];
        else if (arg == "--verbose_count") verbose_count = std::stoull(argv[++i]);
    }

    // ── 1. load data ──────────────────────────────────────────────────────
    std::cout << "=== ZM-Index Graph Query Benchmark ===\n"
              << "Dataset : " << fname << "\n"
              << "N       : " << N     << "\n"
              << "Q       : " << Q     << "\n"
              << "Type    : " << qt_name(qt) << "\n"
              << "Seed    : " << seed  << "\n\n";

    std::vector<Point> data;
    load_points(fname, data, N);
    std::cout << "Loaded " << data.size() << " points.\n\n";

    // ── 2. generate query workload ─────────────────────────────────────────
    auto queries = gen_queries(data, qt, Q, seed);
    Q = queries.size();

    // ── 3. dump queries to file ────────────────────────────────────────────
    if (!dump_file.empty())
        dump_queries(queries, qt, dump_file, fname, N, seed);

    // ── 4. build index ─────────────────────────────────────────────────────
    std::cout << "Building ZMIndex<dim=" << DIM << ", epsilon=" << EPSI << "> ...\n";
    auto tb0 = std::chrono::steady_clock::now();
    ZM zm(data);
    auto tb1 = std::chrono::steady_clock::now();
    const double build_s = std::chrono::duration<double>(tb1 - tb0).count();
    const double idx_mb  = zm.index_size() / (1024.0 * 1024.0);

    const auto& mins = zm.get_mins();
    const auto& maxs = zm.get_maxs();

    // ── 5. verbose trace ───────────────────────────────────────────────────
    if (verbose) {
        size_t show = std::min(verbose_count, Q);
        std::cout << "\n====================================================\n";
        std::cout << "  INTERNAL TRACE — first " << show << " queries\n";
        std::cout << "====================================================\n";
        std::cout << "  Grid bounds per dimension:\n";
        for (size_t d = 0; d < DIM; ++d)
            std::cout << "    " << DIM_LABEL[d] << "  [" << mins[d] << ", " << maxs[d] << "]\n";
        for (size_t i = 0; i < show; ++i)
            trace_query(zm, queries[i], qt, (int)i + 1, mins, maxs);
        std::cout << "\n====================================================\n\n";
    }

    // ── 6. timed benchmark ─────────────────────────────────────────────────
    std::cout << "Running " << Q << " timed queries ...\n";
    std::vector<double> latencies(Q);
    size_t total_hits = 0;

    if (qt == QueryType::POINT) {
        for (size_t i = 0; i < Q; ++i) {
            auto t0 = std::chrono::steady_clock::now();
            auto r  = zm.point_lookup(const_cast<Point&>(queries[i]));
            auto t1 = std::chrono::steady_clock::now();
            latencies[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            if (r.found) ++total_hits;
        }
    } else {
        for (size_t i = 0; i < Q; ++i) {
            Box box = build_box(queries[i], qt, mins, maxs);
            auto t0 = std::chrono::steady_clock::now();
            auto r  = zm.range_query(box);
            auto t1 = std::chrono::steady_clock::now();
            latencies[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
            total_hits += r.size();
        }
    }

    // ── 7. statistics ──────────────────────────────────────────────────────
    double lat_mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / Q;
    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    double lat_p95   = sorted_lat[static_cast<size_t>(0.95 * Q)];
    double lat_p99   = sorted_lat[static_cast<size_t>(0.99 * Q)];
    double throughput = Q / (lat_mean * Q * 1e-6);
    double avg_results_per_query = static_cast<double>(total_hits) / Q;

    // ── 8. results ─────────────────────────────────────────────────────────
    std::cout << "\n"
              << "=======================================================\n"
              << "ZM-Index Graph Query Results\n"
              << "  query_type : " << qt_name(qt)    << "\n"
              << "  dataset    : " << fname           << "\n"
              << "  N          : " << data.size()     << "\n"
              << "  Q          : " << Q               << "\n"
              << "=======================================================\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Build Time (s)          : " << build_s  << "\n";
    std::cout << "Index Size (MB)         : " << idx_mb   << "\n\n";
    std::cout << "Mean Latency (us)       : " << lat_mean << "\n";
    std::cout << "P95  Latency (us)       : " << lat_p95  << "\n";
    std::cout << "P99  Latency (us)       : " << lat_p99  << "\n";
    std::cout << std::setprecision(0);
    std::cout << "Throughput (q/s)        : " << throughput << "\n";
    if (qt == QueryType::POINT) {
        std::cout << "Correctness (%)         : "
                  << std::setprecision(2) << 100.0 * total_hits / Q << "\n";
    } else {
        std::cout << "Avg results per query   : "
                  << std::setprecision(2) << avg_results_per_query << "\n";
        std::cout << "Total results returned  : " << total_hits << "\n";
    }
    std::cout << "=======================================================\n";

    // ── 9. CSV ─────────────────────────────────────────────────────────────
    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            csv << "index,query_type,dataset,N,Q,epsilon,"
                   "build_s,index_mb,lat_mean_us,lat_p95_us,lat_p99_us,"
                   "throughput_qps,avg_results_per_query\n";
            csv << std::fixed;
            csv << "zmindex," << qt_name(qt) << "," << fname << ","
                << data.size() << "," << Q << "," << EPSI << ","
                << std::setprecision(6) << build_s  << ","
                << std::setprecision(4) << idx_mb   << ","
                << std::setprecision(4) << lat_mean << ","
                << std::setprecision(4) << lat_p95  << ","
                << std::setprecision(4) << lat_p99  << ","
                << std::setprecision(0) << throughput << ","
                << std::setprecision(2) << avg_results_per_query << "\n";
            std::cout << "Results saved -> " << out_csv << "\n";
        }
    }

    return 0;
}
