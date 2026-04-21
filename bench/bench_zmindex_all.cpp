/**
 * bench_zmindex_all.cpp — ZM-Index point-lookup benchmark for all thesis datasets
 *
 * Reads space-separated text files (SourceID Hop1_ID Hop2_ID [offset]) directly.
 * No TPIE required. Identical ZMIndex build/query path to bench_zmindex_wv.cpp.
 *
 * Usage:
 *   bench_zmindex_all <txt_file> <N> <dataset_name> <data_source>
 *                     <distribution> <graph_type> <full_size>
 *                     [--queries Q] [--seed S] [--out_csv FILE]
 *
 * Example:
 *   bench_zmindex_all /RSMI/datasets/uniform_sparse_60M.txt 1000000 \
 *     uniform_sparse Synthetic "uniform([1,10000000))" N/A 60000000 \
 *     --queries 100000 --seed 42 \
 *     --out_csv build/results/zm_uniform_sparse_1m.csv
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

#include "../indexes/learned/zmindex.hpp"

static constexpr size_t DIM  = 3;
static constexpr size_t EPSI = 64;

using Point3 = point_t<DIM>;
using ZM3    = bench::index::ZMIndex<DIM, EPSI>;

// ---------------------------------------------------------------------------
// Load first N rows from a space-separated text file.
// Format: col0 col1 col2 [col3 ...]  — first three cols → 3D point.
// ---------------------------------------------------------------------------
static void load_text(const std::string& fname,
                      std::vector<Point3>& pts,
                      size_t N)
{
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
        Point3 p;
        double v;
        for (size_t d = 0; d < DIM; ++d) {
            if (!(ss >> v)) {
                std::cerr << "ERROR: short row at line " << (loaded+1) << "\n";
                std::exit(1);
            }
            p[d] = v;
        }
        pts.push_back(p);
        ++loaded;
    }
    if (loaded < N) {
        std::cerr << "WARNING: requested " << N
                  << " rows but file only has " << loaded << "\n";
    }
}

// ---------------------------------------------------------------------------
// Sample Q random points from pts using the given seed.
// ---------------------------------------------------------------------------
static std::vector<Point3> sample_queries(const std::vector<Point3>& pts,
                                          size_t Q,
                                          uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);
    std::vector<Point3> qpts(Q);
    for (auto& q : qpts)
        q = pts[dist(rng)];
    return qpts;
}

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog
              << " <txt_file> <N> <dataset_name> <data_source>"
              << " <distribution> <graph_type> <full_size>"
              << " [--queries Q] [--seed S] [--out_csv FILE]\n";
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 8) { print_usage(argv[0]); return 1; }

    const std::string txt_file    = argv[1];
    const size_t      N           = std::stoull(argv[2]);
    const std::string dataset_name = argv[3];
    const std::string data_source  = argv[4];
    const std::string distribution = argv[5];
    const std::string graph_type   = argv[6];
    const size_t      full_size    = std::stoull(argv[7]);

    size_t      Q       = 100000;
    uint64_t    seed    = 42;
    std::string out_csv;

    for (int i = 8; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--queries") Q       = std::stoull(argv[++i]);
        else if (arg == "--seed")    seed    = std::stoull(argv[++i]);
        else if (arg == "--out_csv") out_csv = argv[++i];
    }

    std::cout << "====================================================\n"
              << "  ZM-Index Benchmark\n"
              << "====================================================\n"
              << "  Dataset  : " << dataset_name << "\n"
              << "  File     : " << txt_file << "\n"
              << "  N        : " << N << "\n"
              << "  Q        : " << Q << "\n"
              << "  Seed     : " << seed << "\n"
              << "  Epsilon  : " << EPSI << "\n"
              << "====================================================\n\n";

    // 1. Load data
    std::vector<Point3> points;
    load_text(txt_file, points, N);
    const size_t actual_N = points.size();
    std::cout << "[1] Loaded " << actual_N << " 3-D points.\n";

    // 2. Sample query workload
    if (Q > actual_N) Q = actual_N;
    auto queries = sample_queries(points, Q, seed);
    std::cout << "[2] Sampled " << Q << " query points (seed=" << seed << ").\n";

    // 3. Build ZMIndex (timer is inside constructor, but we also time externally)
    std::cout << "[3] Building ZMIndex<" << DIM << ", " << EPSI << "> ...\n";
    const auto t_build0 = std::chrono::steady_clock::now();
    ZM3 zm(points);
    const auto t_build1 = std::chrono::steady_clock::now();

    const double build_s =
        std::chrono::duration<double>(t_build1 - t_build0).count();
    const double idx_mb = zm.index_size() / (1024.0 * 1024.0);

    std::cout << "    Build time : " << std::fixed << std::setprecision(4)
              << build_s << " s\n"
              << "    Index size : " << std::setprecision(4)
              << idx_mb << " MB\n"
              << "    Resolution : " << zm.get_resolution() << "^" << DIM << "\n\n";

    // 4. Run Q point lookups
    std::cout << "[4] Running " << Q << " point lookups ...\n";

    std::vector<double> latencies(Q);
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

    // 5. Statistics
    const double lat_mean =
        std::accumulate(latencies.begin(), latencies.end(), 0.0) / Q;

    std::vector<double> sorted_lat = latencies;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    const double lat_p95 = sorted_lat[static_cast<size_t>(0.95 * Q)];

    const double total_s    = lat_mean * Q * 1e-6;
    const double throughput = Q / total_s;
    const double correct_pct  = 100.0 * correct / Q;
    const double avg_window   = static_cast<double>(total_window) / Q;

    // 6. Print summary
    std::cout << "\n"
              << "-------------------------------------------------------\n"
              << "ZM-Index Results: " << dataset_name << " N=" << actual_N << "\n"
              << "-------------------------------------------------------\n"
              << std::fixed
              << "Build Time (s)           : " << std::setprecision(4) << build_s   << "\n"
              << "Index Size (MB)          : " << std::setprecision(4) << idx_mb    << "\n"
              << "Mean Lookup Latency (us) : " << std::setprecision(4) << lat_mean  << "\n"
              << "P95 Lookup Latency (us)  : " << std::setprecision(4) << lat_p95   << "\n"
              << "Query Throughput (q/s)   : " << std::setprecision(0) << throughput << "\n"
              << "Correctness (%)          : " << std::setprecision(2) << correct_pct << "\n"
              << "Avg PGM Refine Window    : " << std::setprecision(2) << avg_window  << "\n"
              << "-------------------------------------------------------\n";

    if (correct_pct < 100.0) {
        std::cerr << "WARNING: correctness below 100%! ("
                  << correct << "/" << Q << " correct)\n"
                  << "  Possible data format or conversion issue.\n";
    }

    // 7. Save CSV
    if (!out_csv.empty()) {
        // Create directory if needed (best-effort via system call)
        const auto slash = out_csv.rfind('/');
        if (slash != std::string::npos) {
            const std::string dir = out_csv.substr(0, slash);
            std::system(("mkdir -p " + dir).c_str());
        }

        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            // Helper: quote a string field (handles commas in distribution names).
        auto q = [](const std::string& s) { return "\"" + s + "\""; };

        csv << "dataset_name,data_source,distribution,graph_type,"
                << "full_dataset_size,subset_size,"
                << "build_time_s,index_size_mb,"
                << "mean_latency_us,p95_latency_us,throughput_qps,"
                << "avg_pgm_refine_window,correctness_pct\n"
                << std::fixed;
            csv << q(dataset_name) << ","
                << q(data_source)  << ","
                << q(distribution) << ","
                << q(graph_type)   << ","
                << full_size    << ","
                << actual_N     << ","
                << std::setprecision(6) << build_s      << ","
                << std::setprecision(4) << idx_mb       << ","
                << std::setprecision(4) << lat_mean     << ","
                << std::setprecision(4) << lat_p95      << ","
                << std::setprecision(0) << throughput   << ","
                << std::setprecision(2) << avg_window   << ","
                << std::setprecision(2) << correct_pct  << "\n";
            std::cout << "Results saved → " << out_csv << "\n";
        }
    }

    return 0;
}
