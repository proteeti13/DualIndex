// bench_flood_wiki.cpp
// Flood range/point-query benchmark for wiki_vote_triples dataset.
// Accepts both space-separated .txt and TPIE binary .tpie input files.
// Each query point (x,y,z) is converted to a degenerate range box [x,x]×[y,y]×[z,z].
//
// Usage:
//   bench_flood_wiki <data.(txt|tpie)> <N> [--queries Q] [--seed S] [--label L] [--out_csv FILE]
//
//   data.tpie : TPIE binary file — 3 doubles per point (fast)
//   data.txt  : space-separated triples: SourceID Hop1_ID Hop2_ID [offset ...]
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

static std::vector<Point> sample_queries(const std::vector<Point>& pts, size_t Q, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, pts.size() - 1);
    std::vector<Point> qpts(Q);
    for (auto& q : qpts) q = pts[dist(rng)];
    return qpts;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: bench_flood_wiki <data.(txt|tpie)> <N>"
                  << " [--queries Q] [--seed S] [--label L] [--out_csv FILE]\n";
        return 1;
    }

    std::string data_file = argv[1];
    size_t N              = std::stoul(argv[2]);
    size_t Q              = 100000;
    uint64_t seed         = 42;
    std::string label;
    std::string out_csv;

    for (int i = 3; i + 1 < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--queries") Q       = std::stoull(argv[i + 1]);
        else if (arg == "--seed")    seed    = std::stoull(argv[i + 1]);
        else if (arg == "--label")   label   = argv[i + 1];
        else if (arg == "--out_csv") out_csv = argv[i + 1];
    }
    if (label.empty()) label = std::to_string(N);

    // ---- Load data ----
    std::cout << "Loading data: " << data_file << " (" << N << " points)" << std::endl;
    std::vector<Point> points;
    load_points(data_file, points, N);
    std::cout << "Loaded " << points.size() << " points." << std::endl;

    // ---- Build Flood index ----
    Flood flood(points);

    // ---- Sample queries from dataset ----
    std::vector<Point> queries = sample_queries(points, Q, seed);
    Q = queries.size();
    std::cout << "Running " << Q << " queries (seed=" << seed << ")" << std::endl;

    // ---- Run point queries as degenerate boxes ----
    std::vector<long long> latencies_ns;
    latencies_ns.reserve(Q);

    for (auto& qp : queries) {
        Box box(qp, qp);
        auto t0 = std::chrono::steady_clock::now();
        flood.range_query(box);
        auto t1 = std::chrono::steady_clock::now();
        latencies_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    }

    // ---- Compute metrics ----
    double sum_ns = 0.0;
    for (auto t : latencies_ns) sum_ns += t;
    double mean_us = (sum_ns / Q) / 1000.0;

    std::vector<long long> sorted_lat = latencies_ns;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    size_t p95_idx = static_cast<size_t>(0.95 * Q);
    if (p95_idx >= Q) p95_idx = Q - 1;
    double p95_us = sorted_lat[p95_idx] / 1000.0;

    double total_s   = sum_ns / 1e9;
    double throughput = Q / total_s;
    double build_s   = flood.get_build_time() / 1000.0;
    double index_mb  = flood.index_size() / 1e6;

    // ---- Print results ----
    std::cout << "\n"
              << "-------------------------------------------------------\n"
              << "Flood Evaluation Results (wiki_vote_triples - " << label << " triples)\n"
              << "-------------------------------------------------------\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Build Time (s):           " << build_s  << "\n";
    std::cout << "Index Size (MB):          " << index_mb << "\n\n";
    std::cout << "Mean Query Latency (us):  " << mean_us  << "\n";
    std::cout << "P95 Query Latency (us):   " << p95_us   << "\n";
    std::cout << std::defaultfloat;
    std::cout << "Query Throughput (q/s):   " << static_cast<long long>(throughput) << "\n"
              << "-------------------------------------------------------\n";
    std::cout << "Queries executed:         " << Q        << "\n";
    std::cout << "Query type:               degenerate point ([x,x]*[y,y]*[z,z])\n";
    std::cout << "Flood Config:             Dim=" << DIM << " K=" << K << " Eps=" << EPS << "\n";
    std::cout << "Input file:               " << data_file << "\n"
              << "-------------------------------------------------------\n";

    if (!out_csv.empty()) {
        std::ofstream csv(out_csv);
        if (!csv) {
            std::cerr << "WARNING: cannot write CSV to " << out_csv << "\n";
        } else {
            csv << "index,dataset,N,Q,K,epsilon,"
                << "build_s,index_mb,"
                << "lat_mean_us,lat_p95_us,throughput_qps\n";
            csv << std::fixed;
            csv << "flood," << label << "," << points.size() << "," << Q
                << "," << K << "," << EPS
                << "," << std::setprecision(6) << build_s
                << "," << std::setprecision(4) << index_mb
                << "," << std::setprecision(4) << mean_us
                << "," << std::setprecision(4) << p95_us
                << "," << std::setprecision(0) << throughput << "\n";
            std::cout << "Results saved → " << out_csv << "\n";
        }
    }

    return 0;
}
