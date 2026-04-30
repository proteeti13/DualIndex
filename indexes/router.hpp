#pragma once

#include <chrono>
#include <cstddef>
#include <vector>

#include "learned/zmindex.hpp"
#include "learned/flood.hpp"
#include "../utils/type.hpp"

// DualIndexRouter — routes queries to the correct learned index.
//
//   Point query  (all dims pinned, min == max in every dim)  → ZMIndex::point_lookup
//   Range query  (at least one dim open, min < max)          → Flood::range_query
//
// Both indexes are built from independent copies of the input data so that
// Flood's in-place sort of its data copy does not affect ZM-Index's view.
// The owned copies are stored as member variables so references held by the
// indexes remain valid for the lifetime of the router.

namespace bench { namespace index {

template<size_t Dim, size_t K, size_t Eps = 64>
class DualIndexRouter {
public:
    using Point  = point_t<Dim>;
    using Points = std::vector<Point>;
    using Box    = box_t<Dim>;
    using ZM     = ZMIndex<Dim, Eps>;
    using FL     = Flood<Dim, K, Eps>;

    enum class Route { ZM_INDEX, FLOOD };

    struct Result {
        Route     route;
        size_t    result_count;
        long long latency_ns;
        bool      point_found; // meaningful only when route == ZM_INDEX
    };

    // Members are declared in initialization order:
    //   zm_data_ and flood_data_ first (owned storage),
    //   then zm_ and flood_ (which hold references into those vectors).
    // The constructor initializer list must follow this same order.

    explicit DualIndexRouter(const Points& pts)
        : zm_data_(pts), flood_data_(pts),
          zm_(zm_data_), flood_(flood_data_) {}

    Result query(Box& box) {
        Result r;
        auto t0 = std::chrono::steady_clock::now();
        if (is_point(box)) {
            r.route = Route::ZM_INDEX;
            Point p = box.min_corner();        // local copy avoids const& mismatch
            auto lr = zm_.point_lookup(p);
            r.result_count = lr.found ? 1 : 0;
            r.point_found  = lr.found;
        } else {
            r.route = Route::FLOOD;
            auto results   = flood_.range_query(box);
            r.result_count = results.size();
            r.point_found  = false;
        }
        auto t1 = std::chrono::steady_clock::now();
        r.latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        return r;
    }

    // Build-time and index-size accessors (non-const: BaseIndex methods are non-const)
    double zm_build_s()     { return zm_.get_build_time()    / 1000.0; }
    double flood_build_s()  { return flood_.get_build_time() / 1000.0; }
    double zm_index_mb()    { return zm_.index_size()    / 1e6; }
    double flood_index_mb() { return flood_.index_size() / 1e6; }

    static bool is_point(const Box& box) {
        for (size_t d = 0; d < Dim; ++d)
            if (box.min_corner()[d] != box.max_corner()[d]) return false;
        return true;
    }

private:
    Points zm_data_;    // owned copy — ZM-Index reads but never sorts this
    Points flood_data_; // owned copy — Flood sorts this in-place during build
    ZM     zm_;
    FL     flood_;
};

}} // namespace bench::index
