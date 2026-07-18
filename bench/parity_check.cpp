// parity_check.cpp — correctness parity check for the final DualIndex config:
//   DualIndexRouter<3,4,64>  =  ZMIndex<3,64,BuildRange=false> + FloodSourceSort<3,4,64>
//
// Workload (seed 42): 10,000 positive point queries (sampled from the dataset),
// 10,000 negative point queries (guaranteed-absent triples), 10,000 single-hop
// and 10,000 multi-hop range queries. Every query is issued through
// DualIndexRouter::query(); every answer is verified against a per-query
// brute-force linear scan of the (lexicographically) sorted triple array.
//
// Reported: point false positives, point false negatives, and range-query
// result-set mismatches (full multiset comparison of returned triples via the
// router's range_results() test accessor). All must be 0.
//
// Usage: parity_check <triples.txt> <output.txt>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "../indexes/router.hpp"

using Point  = point_t<3>;
using Points = std::vector<Point>;
using Box    = box_t<3>;
using Router = bench::index::DualIndexRouter<3, 4, 64>;

static const unsigned SEED = 42;
static const size_t NQ = 10000;

static inline uint64_t pack(uint32_t s, uint32_t h1, uint32_t h2) {
    return (uint64_t(s) << 42) | (uint64_t(h1) << 21) | uint64_t(h2);
}

static bool triple_less(const Point& a, const Point& b) {
    if (a[0] != b[0]) return a[0] < b[0];
    if (a[1] != b[1]) return a[1] < b[1];
    return a[2] < b[2];
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <triples.txt> <output.txt>\n";
        return 2;
    }
    const std::string data_path = argv[1];
    const std::string out_path = argv[2];

    std::ostringstream rep; // report buffer (echoed to stdout and file)

    // ---------------- load ----------------
    Points pts;
    pts.reserve(5000000);
    {
        std::ifstream in(data_path);
        if (!in) { std::cerr << "cannot open " << data_path << "\n"; return 1; }
        double a, b, c; long idx;
        while (in >> a >> b >> c >> idx) pts.push_back({a, b, c});
    }
    rep << "dataset            : " << data_path << "\n";
    rep << "triples loaded     : " << pts.size() << "\n";

    // sorted triple array — the brute-force ground-truth structure
    Points sorted = pts;
    std::sort(sorted.begin(), sorted.end(), triple_less);
    const size_t N = sorted.size();

    double max_id = 0;
    for (auto& p : sorted)
        for (int d = 0; d < 3; ++d) max_id = std::max(max_id, p[d]);
    rep << "max coordinate     : " << (uint64_t)max_id << "\n";

    // membership set — used only to GENERATE guaranteed-absent negatives
    std::unordered_set<uint64_t> members;
    members.reserve(N * 2);
    for (auto& p : sorted)
        members.insert(pack((uint32_t)p[0], (uint32_t)p[1], (uint32_t)p[2]));

    // ---------------- build the final-config DualIndex ----------------
    Router router(pts);
    rep << "config             : ZMIndex<3,64,false> + FloodSourceSort<3,4,64>\n";
    rep << "zm build (s)       : " << router.zm_build_s() << "\n";
    rep << "flood build (s)    : " << router.flood_build_s() << "\n";
    rep << "zm size (MB)       : " << router.zm_index_mb() << "\n";
    rep << "flood size (MB)    : " << router.flood_index_mb() << "\n\n";

    // ---------------- workloads (seed 42) ----------------
    std::mt19937 rng(SEED);
    std::uniform_int_distribution<size_t> pick(0, N - 1);

    std::vector<Point> pos_q, neg_q;
    pos_q.reserve(NQ); neg_q.reserve(NQ);
    for (size_t i = 0; i < NQ; ++i) pos_q.push_back(sorted[pick(rng)]);
    {
        std::uniform_int_distribution<uint32_t> id(0, (uint32_t)max_id);
        while (neg_q.size() < NQ) {
            uint32_t s = id(rng), h1 = id(rng), h2 = id(rng);
            if (members.find(pack(s, h1, h2)) == members.end())
                neg_q.push_back({(double)s, (double)h1, (double)h2});
        }
    }
    // distinct sources for single-hop
    std::vector<double> sources;
    {
        std::unordered_set<uint32_t> seen;
        for (auto& p : sorted)
            if (seen.insert((uint32_t)p[0]).second) sources.push_back(p[0]);
    }
    std::vector<double> sh_q; sh_q.reserve(NQ);
    {
        std::uniform_int_distribution<size_t> ps(0, sources.size() - 1);
        for (size_t i = 0; i < NQ; ++i) sh_q.push_back(sources[ps(rng)]);
    }
    std::vector<std::pair<double, double>> mh_q; mh_q.reserve(NQ);
    for (size_t i = 0; i < NQ; ++i) {
        auto& t = sorted[pick(rng)];
        mh_q.emplace_back(t[0], t[1]);
    }
    rep << "workload           : " << NQ << " pos point, " << NQ << " neg point, "
        << NQ << " single-hop, " << NQ << " multi-hop (seed " << SEED << ")\n";
    rep << "distinct sources   : " << sources.size() << "\n\n";

    const double DMAX = std::numeric_limits<double>::max();
    size_t false_pos = 0, false_neg = 0, sh_mismatch = 0, mh_mismatch = 0;
    size_t routed_zm = 0, routed_flood = 0;

    // ---------------- point queries: positive ----------------
    for (auto& q : pos_q) {
        Box box(q, q);
        auto r = router.query(box);
        if (r.route == Router::Route::ZM_INDEX) routed_zm++;
        // brute-force linear scan of the sorted triple array
        bool truth = false;
        for (size_t i = 0; i < N; ++i) {
            if (sorted[i][0] == q[0] && sorted[i][1] == q[1] && sorted[i][2] == q[2]) {
                truth = true; break;
            }
        }
        if (r.point_found && !truth) false_pos++;
        if (!r.point_found && truth) false_neg++;
    }

    // ---------------- point queries: negative ----------------
    for (auto& q : neg_q) {
        Box box(q, q);
        auto r = router.query(box);
        if (r.route == Router::Route::ZM_INDEX) routed_zm++;
        bool truth = false;
        for (size_t i = 0; i < N; ++i) {
            if (sorted[i][0] == q[0] && sorted[i][1] == q[1] && sorted[i][2] == q[2]) {
                truth = true; break;
            }
        }
        if (r.point_found && !truth) false_pos++;
        if (!r.point_found && truth) false_neg++;
    }

    // ---------------- single-hop range: (s, *, *) ----------------
    for (double s : sh_q) {
        Point lo = {s, 0.0, 0.0}, hi = {s, DMAX, DMAX};
        Box box(lo, hi);
        auto r = router.query(box);
        if (r.route == Router::Route::FLOOD) routed_flood++;
        Points got = router.range_results(box);
        // brute-force scan
        Points exp;
        for (size_t i = 0; i < N; ++i)
            if (sorted[i][0] == s) exp.push_back(sorted[i]);
        std::sort(got.begin(), got.end(), triple_less);
        std::sort(exp.begin(), exp.end(), triple_less);
        if (got != exp || r.result_count != exp.size()) sh_mismatch++;
    }

    // ---------------- multi-hop range: (s, h1, *) ----------------
    for (auto& q : mh_q) {
        Point lo = {q.first, q.second, 0.0}, hi = {q.first, q.second, DMAX};
        Box box(lo, hi);
        auto r = router.query(box);
        if (r.route == Router::Route::FLOOD) routed_flood++;
        Points got = router.range_results(box);
        Points exp;
        for (size_t i = 0; i < N; ++i)
            if (sorted[i][0] == q.first && sorted[i][1] == q.second) exp.push_back(sorted[i]);
        std::sort(got.begin(), got.end(), triple_less);
        std::sort(exp.begin(), exp.end(), triple_less);
        if (got != exp || r.result_count != exp.size()) mh_mismatch++;
    }

    // ---------------- report ----------------
    rep << "routing check      : " << routed_zm << "/" << (2 * NQ) << " points -> ZM, "
        << routed_flood << "/" << (2 * NQ) << " ranges -> Flood\n\n";
    rep << "===== PARITY RESULTS (all must be 0) =====\n";
    rep << "point false positives        : " << false_pos << "\n";
    rep << "point false negatives        : " << false_neg << "\n";
    rep << "single-hop set mismatches    : " << sh_mismatch << "\n";
    rep << "multi-hop set mismatches     : " << mh_mismatch << "\n";
    bool pass = !false_pos && !false_neg && !sh_mismatch && !mh_mismatch
                && routed_zm == 2 * NQ && routed_flood == 2 * NQ;
    rep << "\nVERDICT: " << (pass ? "PASS" : "FAIL") << "\n";

    std::cout << rep.str();
    std::ofstream out(out_path);
    out << rep.str();
    std::cout << "\nreport written to " << out_path << std::endl;
    return pass ? 0 : 1;
}
