# Experiments

This file documents the two main experiment tracks in this repository.
Both operate on the same source CSV dataset (`SourceID;Hop1_ID;Hop2_ID;Offset`).

---

## 1. LexFlattenBaseline (formerly "Path A")

### Goal
Establish a lower-bound baseline for learned point-lookup on the 3-D graph key
`(SourceID, Hop1_ID, Hop2_ID)` by **collapsing all three dimensions into a single
order-preserving scalar K**, then applying a 1-D PGM-Index.

**Key question answered:** How fast and accurate is a learned index if we flatten
the 3-D key into 1-D? This is the simplest possible learned index strategy.

**Thesis motivation:** The sensitivity experiment proves that the y-value of the
2-D encoding is irrelevant for PGM (which only sees x=K). This motivates the
True3DLearnedIndex, which keeps all three dimensions.

### Encoding
```
BASE2 = max(Hop2_ID) + 1
BASE1 = (max(Hop1_ID) + 1) × BASE2
K     = SourceID × BASE1 + Hop1_ID × BASE2 + Hop2_ID
```
Each point is stored as a degenerate 2-D point `(x=K, y=0)` so that the RSMI
pipeline (which expects 2-D input) can be reused without modification.

### Files
| File | Purpose |
|------|---------|
| `tools/datasets/lex_flatten_baseline/generate_dense_3d.py` | Generates the 10K CSV dataset |
| `tools/datasets/lex_flatten_baseline/csv_to_tpie_lfb.cpp` | CSV → TPIE converter (`csv_to_tpie_lfb` executable) |
| `bench/bench_lfb.cpp` | PGM-based point-lookup benchmark (`bench_lfb` executable) |
| `scripts/run_lex_flatten_baseline.sh` | End-to-end pipeline script |

### Dataset generation command
```bash
# From repo root:
python3 tools/datasets/lex_flatten_baseline/generate_dense_3d.py
# Output: tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv
```

### Benchmark commands (manual, step by step)
```bash
# Step 1: Build
cd build
cmake .. -DRSMI=OFF
make -j$(nproc) csv_to_tpie_lfb bench_lfb
cd ..

# Step 2: Convert CSV → TPIE (y_mode=const)
build/bin/csv_to_tpie_lfb \
  tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv \
  build/data/lex_flatten_baseline_10000_const.tpie \
  --y_mode=const --verify

# Step 3: Convert CSV → TPIE (y_mode=hop2)
build/bin/csv_to_tpie_lfb \
  tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv \
  build/data/lex_flatten_baseline_10000_hop2.tpie \
  --y_mode=hop2 --verify

# Step 4: Run benchmark (y_mode=const, epsilon=64)
build/bin/bench_lfb build/data/lex_flatten_baseline_10000_const.tpie 10000 \
  --queries=10000 --epsilon=64 --y_mode=const

# Step 5: Run sensitivity comparison (const vs hop2 side-by-side)
build/bin/bench_lfb build/data/lex_flatten_baseline_10000_const.tpie 10000 \
  --queries=10000 --epsilon=64 --y_mode=const --sensitivity
```

### One-shot command
```bash
bash scripts/run_lex_flatten_baseline.sh
# Optional overrides:
bash scripts/run_lex_flatten_baseline.sh --n=10000 --queries=10000 --epsilon=64
```

### Metrics produced
| Metric | Description |
|--------|-------------|
| Build time (ms) | Time to construct the PGM-Index from sorted keys |
| Index size (bytes) | Memory footprint of the learned index structure |
| Peak RSS (MB) | Peak resident memory including dataset load |
| Latency mean (ns) | Average per-query lookup time |
| Latency p50 / p95 / p99 (ns) | Percentile latencies |
| Prediction error mean | Mean \|predicted_rank - true_rank\| |
| Prediction error p95 / p99 | Tail prediction error |
| Refinement window mean | Mean scan width `hi - lo` after PGM prediction |
| Refinement window p95 / p99 | Tail scan width |
| Correctness (%) | Fraction of queries returning the exact correct key |

### Recorded results (10K dataset, ε=64, Q=10K, seed=42)
| Metric | const | hop2 |
|--------|-------|------|
| Build time (ms) | ~1.0 | ~0.9 |
| Index size (bytes) | 48 | 48 |
| Latency mean (ns) | ~46 | ~47 |
| Latency p99 (ns) | ~63 | ~63 |
| Prediction error mean | ~15.4 | ~15.4 |
| Prediction error p99 | 31 | 31 |
| Refinement window mean | ~130 | ~130 |
| Correctness | 100% | 100% |

**Observation:** const and hop2 produce identical results. This is expected:
PGM is a 1-D index on K and never sees the y-value.

---

## 2. True3DLearnedIndex (formerly "Path B")

### Goal
Replace the 1-D key collapse with **true 3-D spatial indexing** over
`(SourceID, Hop1_ID, Hop2_ID)`, using RSMI's 2-D spatial partitioning
extended to 3-D (or an alternative 3-D learned index approach).

**Key question to answer:** Does keeping all three dimensions improve prediction
error and refinement cost compared to the 1-D LexFlattenBaseline?

### Dataset generation command
```bash
# Step 1: Generate CSV (same dataset as LexFlattenBaseline)
python3 tools/datasets/lex_flatten_baseline/generate_dense_3d.py

# Step 2: Convert to 3-D TPIE format
cd build && make -j$(nproc) csv_to_tpie_true3d && cd ..

build/bin/csv_to_tpie_true3d \
  tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv \
  build/data/true3d_points.tpie \
  build/data/true3d_offsets.tpie \
  --verify
```

### Output format
- `build/data/true3d_points.tpie`: 3-D points; layout `[x, y, z, x, y, z, ...]`
  — compatible with `bench::utils::read_points<3>(pts, fname, N)`
- `build/data/true3d_offsets.tpie`: scalar offsets; layout `[off0, off1, ...]`
  — compatible with `bench::utils::read_points<1>(offs, fname, N)`
- Row `i` in points ↔ row `i` in offsets (same lexicographic sort order)

### Status
Scaffolding complete (data converter). For the runnable benchmark, see **Section 3: RSMI3D** below.

---

## 3. RSMI3D (True 3-D Learned Index — implemented)

### Goal
Run the full RSMI "predict position → refine" pipeline in **genuine 3-D** on
`(SourceID, Hop1_ID, Hop2_ID)` — no dimension collapse.

**Architecture differences from 2-D RSMI (indexes/rsmi/):**
| Aspect | RSMI (2-D) | RSMI3D (3-D) |
|--------|-----------|--------------|
| Leaf ordering | 2-D Hilbert curve | 3-D Hilbert curve (hilbert4 generic API) |
| Internal grid | `bit_num²` Z-order cells | `bit_num³` Z-order cells (3-D Morton) |
| Neural model input | 2 floats (x, y) | 3 floats (x, y, z) |
| MBR | x1,y1,x2,y2 | x1,y1,z1,x2,y2,z2 |
| Point equality | x == x' && y == y' | + z == z' |

All other logic (train model → cache model → predict index → bounded scan)
is identical. This is a **minimal-invasive 3-D fork**, not a new algorithm.

### Files
| File | Purpose |
|------|---------|
| `indexes/rsmi3d/` | Self-contained 3-D RSMI library |
| `indexes/learned/rsmi3d.hpp` | Benchmark wrapper (mirrors rsmi.hpp) |
| `tools/datasets/rsmi3d/csv_to_tpie_rsmi3d.cpp` | CSV → TPIE converter (same format as true3d) |
| `bench/bench_rsmi3d.cpp` | Point-lookup benchmark driver |

### Prerequisite: libtorch (CPU)
RSMI3D requires PyTorch/libtorch.  Download the Linux CPU build once:
```bash
cd ~
wget "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcpu.zip"
unzip libtorch-cxx11-abi-shared-with-deps-2.5.1+cpu.zip
# Result: ~/libtorch/  (CMakeLists.txt auto-detects this path)
```

### Build commands
```bash
cd build
cmake .. -DRSMI3D=ON -DRSMI=OFF
make -j$(nproc) csv_to_tpie_rsmi3d bench_rsmi3d
cd ..
```
To override the libtorch path:
```bash
cmake .. -DRSMI3D=ON -DTORCH3D_PATH=/custom/libtorch
```

### Dataset generation + conversion
```bash
# Step 1: Generate CSV (same file used by LexFlattenBaseline)
python3 tools/datasets/lex_flatten_baseline/generate_dense_3d.py
# Output: tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv

# Step 2: Convert to RSMI3D TPIE format
build/bin/csv_to_tpie_rsmi3d \
  tools/datasets/lex_flatten_baseline/data/dense_3d_with_offset.csv \
  build/data/rsmi3d_points.tpie \
  build/data/rsmi3d_offsets.tpie \
  --verify
```

### Run benchmark
```bash
# Create model cache directory
mkdir -p build/models/rsmi3d_10000

build/bin/bench_rsmi3d \
  build/data/rsmi3d_points.tpie \
  build/data/rsmi3d_offsets.tpie \
  10000 \
  build/models/rsmi3d_10000 \
  --queries=10000 \
  --seed=42
```
On the **first run** the neural model is trained and cached in `build/models/rsmi3d_10000/`.
Subsequent runs load the cached model and skip training.

### Scaling to 1 M points
Generate a larger CSV (e.g. 100×100×100 = 1 000 000 rows):
```bash
python3 -c "
from tools.datasets.lex_flatten_baseline.generate_dense_3d import generate_dense_3d
generate_dense_3d(n_src=100, n_hop1=100, n_hop2=100,
                  csv_filename='dense_3d_1M.csv')
"
# Then re-run converter + bench with N=1000000
build/bin/csv_to_tpie_rsmi3d \
  tools/datasets/lex_flatten_baseline/data/dense_3d_1M.csv \
  build/data/rsmi3d_points_1M.tpie \
  build/data/rsmi3d_offsets_1M.tpie

build/bin/bench_rsmi3d \
  build/data/rsmi3d_points_1M.tpie \
  build/data/rsmi3d_offsets_1M.tpie \
  1000000 \
  build/models/rsmi3d_1M \
  --queries=10000
```

### Metrics produced
| Metric | Description |
|--------|-------------|
| Build/train time (ms) | Time to construct + train the RSMI3D (first run trains; subsequent runs load cache) |
| Peak RSS (MB) | Peak resident memory |
| Latency mean (ns) | Average per-query lookup time |
| Latency p50 / p95 / p99 (ns) | Percentile latencies |
| Page access mean | Mean leaf nodes probed per query (refinement cost) |
| Page access p95 / p99 | Tail refinement cost |
| Correctness (%) | Fraction of queries where the exact 3-D key was found |

### Metrics to compare against LexFlattenBaseline
- Build time, peak RSS
- Latency (mean, p50, p95, p99 ns)
- Page access mean / p95 / p99 — **expected to narrow** with 3-D spatial structure
- Correctness (%)
