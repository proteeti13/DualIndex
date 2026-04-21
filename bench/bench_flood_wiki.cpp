// bench_flood_wiki.cpp
// Dedicated Flood point-query benchmark for wiki_vote_triples dataset.
// Each query point (x,y,z) is converted to a degenerate range box [x,x]×[y,y]×[z,z].
//
// Usage:
//   bench_flood_wiki <data.tpie> <queries.bin> <N>
//
//   data.tpie   : TPIE file with N 3D points (x y z interleaved float64)
//   queries.bin : flat binary of 100K query points (3 float64 each: x y z)
//   N           : number of data points to load

#include "../utils/datautils.hpp"
#include "../utils/type.hpp"
#include "../indexes/learned/flood.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

static constexpr size_t DIM = 3;
static constexpr size_t K   = 20;
static constexpr size_t EPS = 64;

using Point = point_t<DIM>;
using Box   = box_t<DIM>;
using Flood = bench::index::Flood<DIM, K, EPS>;

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: bench_flood_wiki <data.tpie> <queries.bin> <N> [label]" << std::endl;
        return 1;
    }

    std::string data_file  = argv[1];
    std::string query_file = argv[2];
    size_t N               = std::stoul(argv[3]);
    std::string label      = (argc >= 5) ? argv[4] : std::to_string(N);

    // ---- Load index data ----
    std::cout << "Loading data: " << data_file << " (" << N << " points)" << std::endl;
    std::vector<Point> points;
    bench::utils::read_points<DIM>(points, data_file, N);

    // ---- Build Flood index ----
    Flood flood(points);

    // ---- Load query points ----
    std::cout << "Loading queries: " << query_file << std::endl;
    std::vector<Point> queries;
    {
        std::ifstream fin(query_file, std::ios::binary);
        if (!fin) {
            std::cerr << "ERROR: cannot open query file: " << query_file << std::endl;
            return 1;
        }
        double buf[DIM];
        while (fin.read(reinterpret_cast<char*>(buf), DIM * sizeof(double))) {
            Point p;
            for (size_t d = 0; d < DIM; ++d) p[d] = buf[d];
            queries.push_back(p);
        }
    }
    std::cout << "Loaded " << queries.size() << " query points" << std::endl;

    // ---- Run point queries as degenerate boxes ----
    // Use nanosecond-resolution timing directly; BaseIndex timer uses size_t microseconds
    // which truncates sub-microsecond queries to 0.
    std::vector<long long> latencies_ns;
    latencies_ns.reserve(queries.size());

    for (auto& qp : queries) {
        Box box(qp, qp);          // degenerate: min == max == query point
        auto t0 = std::chrono::steady_clock::now();
        flood.range_query(box);
        auto t1 = std::chrono::steady_clock::now();
        latencies_ns.push_back(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()
        );
    }

    // ---- Compute standardized metrics ----
    size_t Q         = latencies_ns.size();
    double sum_ns    = 0.0;
    for (auto t : latencies_ns) sum_ns += t;
    double mean_us   = (sum_ns / Q) / 1000.0;   // ns → us

    std::vector<long long> sorted_lat = latencies_ns;
    std::sort(sorted_lat.begin(), sorted_lat.end());
    size_t p95_idx = (size_t)(0.95 * Q);
    if (p95_idx >= Q) p95_idx = Q - 1;
    double p95_us = sorted_lat[p95_idx] / 1000.0;  // ns → us

    double total_s = sum_ns / 1e9;
    double throughput = Q / total_s;
    double build_s    = flood.get_build_time() / 1000.0;
    double index_mb   = flood.index_size() / 1e6;

    // ---- Print results ----
    std::cout << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Flood Evaluation Results (wiki_vote_triples - " << label << " triples)" << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Build Time (s):           " << build_s  << std::endl;
    std::cout << "Index Size (MB):          " << index_mb << std::endl;
    std::cout << std::endl;
    std::cout << "Mean Query Latency (us):  " << mean_us  << std::endl;
    std::cout << "P95 Query Latency (us):   " << p95_us   << std::endl;
    std::cout << std::defaultfloat;
    std::cout << "Query Throughput (q/s):   " << (long long)throughput << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Queries executed:         " << Q        << std::endl;
    std::cout << "Query type:               degenerate point ([x,x]*[y,y]*[z,z])" << std::endl;
    std::cout << "Flood Config:             Dim=" << DIM << " K=" << K << " Epsilon=" << EPS << " SortDim=" << DIM-1 << std::endl;
    std::cout << "Input file:               " << data_file  << std::endl;
    std::cout << "Query file:               " << query_file << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    return 0;
}
