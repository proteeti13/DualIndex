#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <tuple>
#include <variant>
#include <boost/multi_array.hpp>
#include <vector>
#include <chrono>
#include <random>

#include "../base_index.hpp"
#include "../../utils/type.hpp"
#include "../../utils/common.hpp"
#include "../pgm/pgm_index.hpp"
#include "../pgm/pgm_index_variants.hpp"


namespace bench { namespace index {

// FloodSourceSort — a variant of Flood (flood.hpp) where dimension 0 (SourceID)
// is the SORT dimension and the remaining dimensions (1 .. Dim-1) form the grid.
//
// Why: in graph-traversal workloads every query filters SourceID, but the sort
// dimension is the only one that gets exact refinement (scan only matching
// points). The stock Flood sorts on the LAST dimension (Hop2), which our queries
// never filter — so SourceID can only prune to one grid column and Flood scans a
// whole column (≈ N/K points). Making SourceID the sort dimension turns single-
// and multi-hop range queries into near-exact scans.
//
// This is a focused copy of flood.hpp. The ONLY semantic change is the
// dimension assignment:
//     stock Flood : grid on dims 0..Dim-2 , sort on dim Dim-1
//     this class  : grid on dims 1..Dim-1 , sort on dim 0
// Data and queries stay in natural (Source, Hop1, Hop2) coordinates — no
// permutation needed. Internally, grid slot j (0..Dim-2) maps to data dim j+1.

template<size_t Dim, size_t K, size_t Eps=64>
class FloodSourceSort : public BaseIndex {

using Point = point_t<Dim>;
using Points = std::vector<Point>;
using Range = std::pair<size_t, size_t>;
using Box = box_t<Dim>;

using Index = pgm::PGMIndex<double, Eps>;

static constexpr size_t SortDim = 0;                       // SourceID is the sort dim
static inline size_t grid_dim(size_t j) { return j + 1; }  // grid slot j -> data dim j+1

public:

class Bucket {
    public:
    Points _local_points;
    // eps for each bucket is fixed to 16 based on a micro benchmark
    pgm::PGMIndex<double, 16>* _local_pgm;

    Bucket() : _local_pgm(nullptr) {} ;

    ~Bucket() {
        delete this->_local_pgm;
    }

    inline void insert(Point& p) {
        this->_local_points.emplace_back(p);
    }

    inline void build() {
        if (_local_points.size() == 0) {
            return;
        }
        // points are already in SourceID-sorted order (global sort by dim 0)
        std::vector<double> idx_data;
        idx_data.reserve(_local_points.size());
        for (const auto& p : _local_points) {
            idx_data.emplace_back(std::get<SortDim>(p));
        }

        _local_pgm = new pgm::PGMIndex<double, 16>(idx_data);
    }

    inline void search(Points& result, Box& box) {
        if (_local_pgm == nullptr) {
            return;
        }

        const size_t n = _local_points.size();
        auto min_key = std::get<SortDim>(box.min_corner());
        auto max_key = std::get<SortDim>(box.max_corner());

        // Points in this cell are sorted by the sort dimension (SourceID). Use the
        // PGM as a hint for the lower bound, then correct locally. Crucially, scan
        // forward until the sort-dim value EXCEEDS max_key — this captures the full
        // run of duplicate keys (a hub SourceID can have many rows in one cell),
        // which a single fixed PGM error-window would truncate.
        auto approx = this->_local_pgm->search(min_key);
        size_t i = approx.lo > n ? n : approx.lo;
        while (i < n && std::get<SortDim>(_local_points[i]) < min_key) ++i;
        while (i > 0 && std::get<SortDim>(_local_points[i - 1]) >= min_key) --i;

        for (; i < n && std::get<SortDim>(_local_points[i]) <= max_key; ++i) {
            if (bench::common::is_in_box(this->_local_points[i], box)) {
                result.emplace_back(this->_local_points[i]);
            }
        }
    }
};

FloodSourceSort(Points& points) : _data(points), bucket_size((points.size() + K - 1)/K) {
    std::cout << "Construct FloodSourceSort " << "K=" << K << " Epsilon=" << Eps
              << " SortDim=" << SortDim << " (grid on dims 1.." << (Dim-1) << ")" << std::endl;

    auto start = std::chrono::steady_clock::now();

    // dimension offsets when computing bucket ID (over the Dim-1 grid slots)
    for (size_t j=0; j<Dim-1; ++j) {
        this->dim_offset[j] = bench::common::ipow(K, j);
    }

    // sort points by SortDim (SourceID, dim 0)
    std::sort(_data.begin(), _data.end(), [](auto& p1, auto& p2) {
        return std::get<SortDim>(p1) < std::get<SortDim>(p2);
    });

    // boundaries of each grid dimension
    std::fill(mins.begin(), mins.end(), std::numeric_limits<double>::max());
    std::fill(maxs.begin(), maxs.end(), std::numeric_limits<double>::min());

    // train a CDF model on each grid dimension (data dims 1 .. Dim-1)
    std::vector<double> idx_data;
    idx_data.reserve(points.size());
    for (size_t j=0; j<Dim-1; ++j) {
        size_t d = grid_dim(j);
        for (const auto& p : _data) {
            mins[j] = std::min(p[d], mins[j]);
            maxs[j] = std::max(p[d], maxs[j]);

            idx_data.emplace_back(p[d]);
        }

        std::sort(idx_data.begin(), idx_data.end());
        this->indexes[j] = new Index(idx_data);

        idx_data.clear();
    }


    // note data are sorted by SortDim; bucketing preserves that order per cell
    for (auto& p : _data) {
        buckets[compute_id(p)].insert(p);
    }

    for (auto& b : buckets) {
        b.build();
    }

    auto end = std::chrono::steady_clock::now();
    build_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Build Time: " << get_build_time() << " [ms]" << std::endl;
    std::cout << "Index Size: " << index_size() << " Bytes" << std::endl;
}

Points range_query(Box& box) {
    auto start = std::chrono::steady_clock::now();

    // find all intersected cells
    std::vector<std::pair<size_t, size_t>> ranges;
    find_intersect_ranges(ranges, box);

    // search each cell using local models
    Points result;
    for (auto& range : ranges) {
        for (auto idx=range.first; idx<=range.second; ++idx) {
            this->buckets[idx].search(result, box);
        }
    }

    auto end = std::chrono::steady_clock::now();
    range_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    range_count ++;

    return result;
}

inline size_t count() {
    return _data.size();
}

inline size_t index_size() {
    // size of dimension-level learned index
    size_t cdf_size = 0;
    for (size_t j=0; j<Dim-1; ++j) {
        cdf_size += this->indexes[j]->size_in_bytes();
    }

    // size of models within each bucket
    size_t b_size = 0;
    for (size_t i=0; i<buckets.size(); ++i) {
        if (buckets[i]._local_pgm != nullptr) {
            b_size += buckets[i]._local_pgm->size_in_bytes();
        }
    }

    return cdf_size + b_size + count() * sizeof(size_t);
}


~FloodSourceSort() {
    for (size_t j=0; j<Dim-1; ++j) {
        delete this->indexes[j];
    }
}


private:
Points& _data;
std::array<Index*, Dim-1> indexes;
std::array<Bucket, bench::common::ipow(K, Dim-1)> buckets;
std::array<size_t, Dim-1> dim_offset;

std::array<double, Dim-1> mins;
std::array<double, Dim-1> maxs;

const size_t bucket_size;

inline void find_intersect_ranges(std::vector<std::pair<size_t, size_t>>& ranges, Box& qbox) {
    if (Dim == 2) {
        ranges.emplace_back(get_dim_idx(qbox.min_corner(), 0), get_dim_idx(qbox.max_corner(), 0));
    } else {
        // search range on the 1-st grid dimension
        ranges.emplace_back(get_dim_idx(qbox.min_corner(), 0), get_dim_idx(qbox.max_corner(), 0));

        // find all intersect ranges across the remaining grid dimensions
        for (size_t j=1; j<Dim-1; ++j) {
            auto start_idx = get_dim_idx(qbox.min_corner(), j);
            auto end_idx = get_dim_idx(qbox.max_corner(), j);

            std::vector<std::pair<size_t, size_t>> temp_ranges;
            for (auto idx=start_idx; idx<=end_idx; ++idx) {
                for (size_t k=0; k<ranges.size(); ++k) {
                    temp_ranges.emplace_back(ranges[k].first + idx*dim_offset[j], ranges[k].second + idx*dim_offset[j]);
                }
            }

            // update the range vector
            ranges = temp_ranges;
        }
    }
}

// locate the bucket on grid slot j (data dim j+1) using the learned CDF
inline size_t get_dim_idx(Point& p, size_t j) {
    size_t d = grid_dim(j);
    if (p[d] <= this->mins[j]) {
        return 0;
    }
    if (p[d] >= this->maxs[j]) {
        return K-1;
    }
    auto approx_pos = this->indexes[j]->search(p[d]).pos / this->bucket_size;
    return std::min(approx_pos, K-1);
}

inline size_t compute_id(Point& p) {
    size_t id = 0;

    for (size_t j=0; j<Dim-1; ++j) {
        auto current_idx = get_dim_idx(p, j);
        id += current_idx * dim_offset[j];
    }

    return id;
}


};
}
}
