/**
 * bench_zmindex_wv.cpp — ZM-Index point-lookup benchmark on Wiki-Vote dataset
 *
 * Thesis context
 * ──────────────
 * ZM-Index (Z-order / Morton curve index) linearises 3D keys
 * (SourceID, Hop1_ID, Hop2_ID) into a 1-D Morton-order sequence,
 * then applies a 1-D PGM learned index.  This is benchmarked as the
 * "flattened-1D" baseline, contrasted with true-3D learned indexes
 * (RSMI3D, LISA, Flood) tested on the same Wiki-Vote dataset.
 *
 * Data files expected
 *   datasets/wiki_vote_triples.txt   – space-separated SourceID Hop1_ID Hop2_ID [offset]
 *   Queries are sampled from the dataset (no separate query file needed).
 *
 * Offset column in wiki_vote_triples.txt
 * ──────────────────────────────────────
 * The offset encodes lexicographic rank (SourceID, Hop1_ID, Hop2_ID).
 * ZM-Index sorts data by Morton rank (different ordering), so it does NOT
 * predict lexicographic offsets.  Instead, it predicts position in
 * Morton-sorted order.  Correctness is therefore measured by exact Morton-
 * code retrieval: if the PGM search range contains the query's Morton code,
 * the lookup is correct.  This mirrors how RSMI3D correctness is defined
 * (predicted rank → binary scan finds the key).
 *
 * Usage
 * ─────
 *   bench_zmindex_wv <txt_file> <N>
 *                   [--queries Q] [--seed S]
 *                   [--out_csv FILE]
 *
 * Example
 *   build/bin/bench_zmindex_wv \
 *     datasets/wiki_vote_triples.txt \
 *     1000000 --queries 100000 \
 *     --out_csv build/results/zmindex_wikivote.csv
 */

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tpie/tpie.h>
#include <tpie/file_stream.h>

// ZMIndex<Dim, Epsilon> — provides point_lookup(Point&) → PointQueryResult
#include "../indexes/learned/zmindex.hpp"

static constexpr size_t DIM   = 3;
static constexpr size_t EPSI  = 64;

using Point3 = point_t<DIM>;
using ZM3    = bench::index::ZMIndex<DIM, EPSI>;

// ── helpers ─────────────────────────────────────────────────────────────────

static bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static void load_tpie(const std::string& fname,
                      std::vector<Point3>& pts, size_t N) {
    pts.reserve(N);
    tpie::tpie_init();
    tpie::file_stream<double> in;
    in.open(fname);
    for (size_t i = 0; i < N; ++i) {
        Point3 p;
        for (size_t d = 0; d < DIM; ++d) p[d] = in.read();
        pts.push_back(p);
    }
    in.close();
    tpie::tpie_finish();
}

static void load_text(const std::string& fname,
                      std::vector<Point3>& pts, size_t N) {
    pts.reserve(N);
    std::ifstream f(fname);
    if (!f) { std::cerr << "ERROR: cannot open " << fname << "\n"; std::exit(1); }
    std::string line;
    size_t loaded = 0;
    while (loaded < N && std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        Point3 p;
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

static void load_points(const std::string& fname,
                        std::vector<Point3>& pts, size_t N) {
    if (ends_with(fname, ".tpie")) {
        std::cout << "[load] TPIE binary: " << fname << "\n";
        load_tpie(fname, pts, N);
    } else {
        std::cout << "[load] text file:   " << fname << "\n";
        load_text(fname, pts, N);
    }
}

static std::vector<Point3> sample_queries(const std::vector<Point3>& pts,
                                          size_t Q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);
    std::vector<Point3> qpts(Q);
    for (auto& q : qpts) q = pts[dist(rng)];
    return qpts;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <file.(txt|tpie)> <N>"
              << " [--queries Q] [--seed S]"
              << " [--out_csv FILE]\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    const std::string txt_file = argv[1];  // accepts .txt or .tpie
    const size_t      N        = std::stoull(argv[2]);
    size_t            Q        = 100000;
    uint64_t          seed     = 42;
    std::string       out_csv;

    for (int i = 3; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--queries") Q       = std::stoull(argv[i + 1]);
        else if (arg == "--seed")    seed    = std::stoull(argv[i + 1]);
        else if (arg == "--out_csv") out_csv = argv[i + 1];
    }

    // ── banner ──────────────────────────────────────────────────────────────
    std::cout << "====================================================\n"
              << "  ZM-Index — Wiki-Vote 3D graph-key benchmark\n"
              << "====================================================\n"
              << "  Dataset  : " << txt_file << "\n"
              << "  N        : " << N << "\n"
              << "  Q        : " << Q << "\n"
              << "  Epsilon  : " << EPSI << "\n"
              << "  Approach : 3D keys linearised via Z-order (Morton)\n"
              << "             curve, then 1-D PGM learned index applied.\n"
              << "====================================================\n\n";

    // ── 1. load data ─────────────────────────────────────────────────────────
    std::vector<Point3> points;
    load_points(txt_file, points, N);
    std::cout << "[1] Loaded " << points.size() << " 3-D points.\n";

    // ── 2. sample query workload from dataset ─────────────────────────────────
    std::vector<Point3> queries = sample_queries(points, Q, seed);
    Q = queries.size();
    std::cout << "[2] Sampled " << Q << " query points (seed=" << seed << ").\n";

    // ── 3. build ZMIndex ──────────────────────────────────────────────────────
    std::cout << "[3] Building ZMIndex<" << DIM << ", " << EPSI << "> ...\n";
    const auto t_build_start = std::chrono::steady_clock::now();
    ZM3 zm(points);
    const auto t_build_end   = std::chrono::steady_clock::now();

    const double build_s =
        std::chrono::duration<double>(t_build_end - t_build_start).count();
    const double idx_mb = zm.index_size() / (1024.0 * 1024.0);

    std::cout << "    Grid resolution : " << zm.get_resolution() << "^"
              << DIM << "\n\n";

    // ── 4. run Q point lookups ────────────────────────────────────────────────
    std::cout << "[4] Running " << Q << " point lookups ...\n";

    std::vector<double> latencies(Q);   // per-query latency [us]
    size_t correct      = 0;
    size_t total_window = 0;

    for (size_t i = 0; i < Q; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        auto res = zm.point_lookup(queries[i]);
        const auto t1 = std::chrono::steady_clock::now();

        latencies[i]  = std::chrono::duration<double, std::micro>(t1 - t0).count();
        if (res.found)  ++correct;
        total_window += res.pgm_window;
    }

    // ── 5. statistics ─────────────────────────────────────────────────────────
    // Mean latency
    const double lat_mean =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / Q;

    // P95 latency (sort a copy to preserve original order)
    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    const double lat_p95  = sorted_lat[static_cast<size_t>(0.95 * Q)];

    // Throughput (q/s)
    const double total_s  = lat_mean * Q * 1e-6;
    const double throughput = Q / total_s;

    const double correct_pct  = 100.0 * correct / Q;
    const double avg_window   = static_cast<double>(total_window) / Q;

    // ── 6. thesis-format summary ──────────────────────────────────────────────
    std::cout << "\n"
              << "-------------------------------------------------------\n"
              << "ZM-Index Evaluation Results (wiki_vote_triples – "
              << N << " rows)\n"
              << "-------------------------------------------------------\n";
    std::cout << std::fixed;
    std::cout << "Build Time (s)           : "
              << std::setprecision(4) << build_s   << "\n";
    std::cout << "Index Size (MB)          : "
              << std::setprecision(4) << idx_mb    << "\n";
    std::cout << "Mean Lookup Latency (us) : "
              << std::setprecision(4) << lat_mean  << "\n";
    std::cout << "P95 Lookup Latency (us)  : "
              << std::setprecision(4) << lat_p95   << "\n";
    std::cout << "Query Throughput (q/s)   : "
              << std::setprecision(0) << throughput << "\n";
    std::cout << "Correctness (%)          : "
              << std::setprecision(2) << correct_pct << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "-- Secondary metrics (ZM-Index internals) --\n";
    std::cout << "Avg PGM Refine Window    : "
              << std::setprecision(2) << avg_window
              << "  (≈ 2×Epsilon+2 = " << 2*EPSI+2 << " expected)\n";
    std::cout << "Grid Resolution          : " << zm.get_resolution()
              << "^" << DIM << " cells\n";
    std::cout << "Morton Epsilon (PGM ε)   : " << EPSI << "\n";
    std::cout << "-------------------------------------------------------\n";
    std::cout << "Note: ZM-Index flattens 3D graph keys via Z-order curve.\n"
              << "      Offsets in wiki_vote_triples are LEX-rank; ZM-Index\n"
              << "      uses MORTON-rank internally (different ordering).\n"
              << "      Correctness = fraction where Morton code found in\n"
              << "      PGM search window (analogous to RSMI3D correctness).\n"
              << "-------------------------------------------------------\n";

    // ── 7. save CSV ───────────────────────────────────────────────────────────
    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            csv << "index,dataset,N,Q,epsilon,"
                << "build_s,index_mb,"
                << "lat_mean_us,lat_p95_us,throughput_qps,"
                << "correctness_pct,avg_pgm_window,grid_resolution\n";
            csv << std::fixed;
            csv << "zmindex,wiki_vote_1m," << N << "," << Q << "," << EPSI
                << "," << std::setprecision(6) << build_s
                << "," << std::setprecision(4) << idx_mb
                << "," << std::setprecision(4) << lat_mean
                << "," << std::setprecision(4) << lat_p95
                << "," << std::setprecision(0) << throughput
                << "," << std::setprecision(2) << correct_pct
                << "," << std::setprecision(2) << avg_window
                << "," << zm.get_resolution() << "\n";
            std::cout << "Results saved → " << out_csv << "\n";
        }
    }

    return 0;
}
