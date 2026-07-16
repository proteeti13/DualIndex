#pragma once

#include <cstddef>
#include <utility>
#include <cstdint>
#include <chrono>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>

#include "../base_index.hpp"
#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../pgm/pgm_index.hpp"
#include "../pgm/pgm_index_variants.hpp"
#include "../pgm/morton_nd.hpp"


namespace bench { namespace index {

// Epsilon:    the error bound of the underlying 1-D learned index
// BuildRange: when true (default, standalone ZM-Index) the constructor also
//             builds the MultidimensionalPGMIndex used by range_query / knn_query.
//             DualIndex routes all range/kNN queries to FloodSourceSort and only
//             ever calls point_lookup, so it instantiates with BuildRange=false to
//             skip that unused structure (saving build time and memory).
template<size_t Dim, size_t Epsilon=64, bool BuildRange=true>
class ZMIndex : public BaseIndex {

using Point = point_t<Dim>;
using Points = std::vector<Point>;
using Box = box_t<Dim>;
using Index = pgm::MultidimensionalPGMIndex<Dim, size_t, Epsilon>;
using morton = mortonnd::MortonNDBmi<Dim, uint64_t>;
using value_type = decltype(morton::Decode(0));

public:

ZMIndex(Points& points) : _data(points) {
    std::cout << "Construct ZM-Index " << "Epsilon=" << Epsilon << std::endl;

    auto start = std::chrono::steady_clock::now();

    // boundaries of each dimension
    std::fill(mins.begin(), mins.end(), std::numeric_limits<double>::max());
    std::fill(maxs.begin(), maxs.end(), std::numeric_limits<double>::min());

    for (size_t i=0; i<Dim; ++i) {
        for (auto& p : points) {
            mins[i] = std::min(p[i], mins[i]);
            maxs[i] = std::max(p[i], maxs[i]);
        }
    }

    // the grid resolution to calculate the Z-value is set to N^{1/d}
    this->resolution = static_cast<size_t>(pow(_data.size(), 1.0/Dim));
    
    // widths of each dimension
    for (size_t i=0; i<Dim; ++i) {
        widths[i] = (maxs[i] - mins[i]) / this->resolution;
    }

    // Multidimensional PGM (range / kNN). Skipped entirely when BuildRange=false
    // (DualIndex point-only path) -- this is the "unused index" removal.
    if constexpr (BuildRange) {
        std::vector<value_type> tuples;
        tuples.reserve(points.size());
        for (auto& p: points) {
            tuples.emplace_back(a2t(p));
        }
        pgm_idx = new Index(tuples.begin(), tuples.end());
    }

    // ---- v3 exact refinement: parallel-sort coordinates by Morton code ----
    // Build a true 1-D Morton-code PGM and keep the coordinate array sorted in
    // the SAME order. point_lookup predicts a position with the learned model,
    // then refines by scanning the local run of equal Morton codes comparing
    // exact 3-D coordinates -- the canonical "model predicts, local search
    // verifies" learned-index pattern. No auxiliary membership structure: the
    // sorted coordinate array IS the index payload. _data is reordered in place,
    // which is safe here -- range_query/knn use pgm_idx (built from an
    // independent copy above) and only count() reads _data.
    const size_t n = _data.size();
    std::vector<uint64_t> codes(n);
    for (size_t i = 0; i < n; ++i) codes[i] = morton_code(_data[i]);

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), size_t(0));
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b) { return codes[a] < codes[b]; });

    Points sorted_pts(n);
    std::vector<uint64_t> sorted_codes(n);
    for (size_t i = 0; i < n; ++i) {
        sorted_pts[i]   = _data[order[i]];
        sorted_codes[i] = codes[order[i]];
    }
    std::swap(_data, sorted_pts);                 // _data now in Morton order
    morton_pgm_ = new pgm::PGMIndex<uint64_t, Epsilon>(sorted_codes);
    // codes / order / sorted_pts / sorted_codes are freed at scope end; only the
    // tiny 1-D PGM and the (reordered, pre-existing) _data array remain.

    auto end = std::chrono::steady_clock::now();
    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
    std::cout << "Index Size: " << index_size() << " Bytes" << std::endl;
}

~ZMIndex() {
    delete this->pgm_idx;        // nullptr when BuildRange=false -- safe
    delete this->morton_pgm_;
}

// Result type for thesis-style point lookup.
// found      – true if the Morton code is in the PGM index (exact match)
// pgm_window – PGM search range size (≈ 2*Epsilon+2), the refinement window
struct PointQueryResult {
    bool   found;
    size_t pgm_window;
};

// Exact point lookup. 3D key -> grid cell IDs -> Morton code -> 1-D PGM predicts
// the position in the Morton-sorted coordinate array -> local scan over the run
// of equal Morton codes, comparing exact 3-D coordinates.
//
// Correctness: a real point's Morton code is present, so the PGM error bound
// locates its first occurrence within [lo, hi); we lower_bound there and then
// scan FORWARD over the whole equal-Morton run (which a dense grid cell can
// stretch beyond a fixed error window) until the code changes -- so no real
// match is ever missed (no false negatives). Distinct points that merely share
// a Morton cell fail the exact coordinate compare, so there are no false
// positives either. pgm_window reports the actual predicted error window.
PointQueryResult point_lookup(Point& q) {
    auto start = std::chrono::steady_clock::now();

    const uint64_t m = morton_code(q);
    const size_t n = _data.size();
    auto ap = morton_pgm_->search(m);
    size_t lo = ap.lo < n ? ap.lo : n;
    size_t hi = ap.hi < n ? ap.hi : n;
    if (lo > hi) lo = hi;

    // learned-model-guided locate: first entry with Morton code >= m in [lo, hi)
    auto cmp = [this](const Point& pt, uint64_t key) { return this->morton_code(pt) < key; };
    size_t i = std::lower_bound(_data.begin() + lo, _data.begin() + hi, m, cmp) - _data.begin();

    bool found = false;
    for (size_t pos = i; pos < n; ++pos) {
        if (morton_code(_data[pos]) != m) break;   // past the equal-Morton run
        bool eq = true;
        for (size_t d = 0; d < Dim; ++d) {
            if (_data[pos][d] != q[d]) { eq = false; break; }
        }
        if (eq) { found = true; break; }
    }

    auto end = std::chrono::steady_clock::now();
    point_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    point_count++;
    return {found, hi - lo};
}

Points range_query(Box& box) {
    static_assert(BuildRange,
        "range_query requires BuildRange=true; this ZM-Index was built point-only "
        "(DualIndex routes range queries to FloodSourceSort).");
    auto start = std::chrono::steady_clock::now();

    auto min_tup = a2t(box.min_corner());
    auto max_tup = a2t(box.max_corner());

    std::vector<std::array<size_t, Dim>> temp;
    for (auto it=this->pgm_idx->range(min_tup, max_tup); it!=this->pgm_idx->end(); ++it) {
        temp.emplace_back(get_array_from_tuple(*it));
    }

    auto end = std::chrono::steady_clock::now();
    range_count++;
    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    Points result;
    result.reserve(temp.size());
    for (auto& tp : temp) {
        Point p;
        for (size_t d=0; d<Dim; ++d) {
            p[d] = tp[d];
        }
        result.emplace_back(p);
    }
    
    return result;
}

// this is approx knn not exact knn
Points knn_query(Point& q, size_t k) {
    static_assert(BuildRange,
        "knn_query requires BuildRange=true; this ZM-Index was built point-only.");
    auto start = std::chrono::steady_clock::now();

    auto q_tup = a2t(q);
    auto results = this->pgm_idx->knn(q_tup, k);
    
    auto end = std::chrono::steady_clock::now();
    knn_count++;
    knn_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::vector<std::array<size_t, Dim>> temp;
    temp.reserve(k);
    Points result_points;
    result_points.reserve(k);

    for (auto& tup : results) {
        temp.emplace_back(get_array_from_tuple(tup));
    }
    for (auto& tp : temp) {
        Point p;
        for (size_t d=0; d<Dim; ++d) {
            p[d] = tp[d];
        }
        result_points.emplace_back(p);
    }
    return result_points;
}

inline size_t count() {
    return _data.size();
}

// index size in Bytes.
// v3 reports the honest footprint of the point-query path: the multidimensional
// PGM (used by range/knn), the 1-D Morton PGM, and the Morton-sorted coordinate
// array that serves as the refinement payload (replaces the v2 hash set, which
// was ~290 MB and uncounted). sizeof(Point) = Dim*8 B.
inline size_t index_size() {
    size_t bytes = (morton_pgm_ ? morton_pgm_->size_in_bytes() : 0)
                 + count() * sizeof(Point);
    // the multidimensional PGM contributes only when it was actually built
    if constexpr (BuildRange) bytes += pgm_idx->size_in_bytes();
    return bytes;
}

inline size_t get_resolution() {
    return this->resolution;
}

// Expose grid internals for external verbose/trace output
inline std::array<size_t, Dim> point_to_cells(const Point& p) const {
    std::array<size_t, Dim> cells;
    for (size_t i = 0; i < Dim; ++i) cells[i] = to_id_const(p[i], i);
    return cells;
}
inline const std::array<double, Dim>& get_mins()   const { return mins; }
inline const std::array<double, Dim>& get_maxs()   const { return maxs; }
inline const std::array<double, Dim>& get_widths() const { return widths; }

private:
// the grid resolution to compute the z address
// by default, it is set to N^{1/d}
size_t resolution;

std::array<double, Dim> mins;
std::array<double, Dim> maxs;
std::array<double, Dim> widths;

// internal data (reordered into Morton-code order during construction)
Points& _data;
// internal multidimensional pgm index (range / knn); nullptr when BuildRange=false
Index* pgm_idx = nullptr;
// 1-D PGM over the Morton codes of _data, in the same sorted order (point lookup)
pgm::PGMIndex<uint64_t, Epsilon>* morton_pgm_ = nullptr;

// encode a point into its single 64-bit Morton code via the per-dim grid cells
inline uint64_t morton_code(const Point& p) {
    return std::apply(
        [](auto... xs) { return morton::Encode(static_cast<uint64_t>(xs)...); },
        a2t(p));
}

// turn a double point to ints to compute the z-value
inline size_t to_id(double val, size_t I) {
    if (val <= this->mins[I]) return 0;
    if (val >= this->maxs[I]) return (this->maxs[I] - this->mins[I]) / this->widths[I];
    return static_cast<size_t>((val - this->mins[I])/this->widths[I]);
}
inline size_t to_id_const(double val, size_t I) const {
    if (val <= this->mins[I]) return 0;
    if (val >= this->maxs[I]) return (this->maxs[I] - this->mins[I]) / this->widths[I];
    return static_cast<size_t>((val - this->mins[I])/this->widths[I]);
}

template<typename Array, std::size_t... I>
inline auto a2t_impl(const Array& a, std::index_sequence<I...>) {
    return std::make_tuple(to_id(a[I], I)...);
}
 
template<typename T, std::size_t N, typename Indices = std::make_index_sequence<N>>
inline auto a2t(const std::array<T, N>& a) {
    return a2t_impl(a, Indices{});
}

template<typename tuple_t>
constexpr auto get_array_from_tuple(tuple_t&& tuple) {
    constexpr auto get_array = [](auto&& ... x){ return std::array{std::forward<decltype(x)>(x) ... }; };
    return std::apply(get_array, std::forward<tuple_t>(tuple));
}

};

}
}
