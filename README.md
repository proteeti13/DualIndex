<<<<<<< HEAD
# Evaluating Mtulit-dimensional Learned Indices
This is the source code repo for our emperical study on multi-dimensional learned indices.


## Compared Methods
### Learned Indices
We compare **six** recent multi-dimensional learned indices:
- **ZM-Index** [1]
- **ML-Index** [2] 
- **IF-Index** [3] 
- **RSMI** [4]  
- **LISA** [5] 
- **Flood** [6]

### Non-learned Baselines
- **FullScan**: sequential scan
- **R\*-tree** and **bulk-loading R-tree**: we use the implementation from `boost::geometry`
- **kdtree**: we use a header-only kdtree implementation `nanoflann`  https://github.com/jlblancoc/nanoflann
- **ANN**: another kntree viriant from `ANN` project http://www.cs.umd.edu/~mount/ANN/
- **Quad-tree**: we use the implementation from `GEOS`
- **Grid**: uniform grid (UG) and equal-depth grid (EDG)

## Compilation
### Step 1: Setup Dependencies
- `boost 1.79`: https://www.boost.org/users/history/version_1_79_0.html
- `TPIE`: https://github.com/thomasmoelhave/tpie
- `GEOS`: https://libgeos.org/
- `gperftools`: https://github.com/gperftools/gperftools
- `libtorch`: https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.4.0%2Bcpu.zip
- `numpy` and `matplotlib` for result visualization

### Step 2: Build RSMI and ANN
Most of the benchmark and indices (except `RSMI` and `ANN`) are implemented as header-only libraries. 

Compile `RSMI`:
```sh
cd indexes/rsmi
mkdir build && cd build
cmake ..
make
```

Compile `ANN`:
```sh
cd indexes/ann_1.1.2
make
```

### Step 3: Build Benchmark
Modify the following variables in `CMakeLists.txt`:
```
BOOST_ROOT, Boost_INCLUDE_DIR, Boost_LIBRARY_DIR: path to boost
TORCH_PATH: path to libtorch
EXECUTABLE_OUTPUT_PATH: path to compiled benchmark binaries
```

Compile RSMI benchmark:
```sh
mkdir build && cd build
cmake .. -DRSMI=ON
make
```

Compile RSMI benchmark with heap profiling enabled:
```sh
rm -rf * # clear cmake cache
cmake .. -DRSMI=ON -DPROFILE=ON
make
```

Compile benchmark for other indices:
```sh
rm -rf * # clear cmake cache
cmake ..
make
```

Compile benchmark for other indices with heap profiling enabled:
```sh
rm -rf * # clear cmake cache
cmake .. -DPROFILE=ON
make
```

## Run Experiments
We prepare a script to download the real datasets and prepare synthetic datasets:
```sh
cd scripts
bash prepare_data.sh
```

We prepare several scripts to run the experiments.

Run experiments on default settings: `bash run_exp.sh`

Run experiments by varying N: `bash run_exp_n.sh`

Run experiments by varying dim: `bash run_exp_dim.sh`

Run experiments by varying eps: `bash run_exp_eps.sh`

Run experiments of RSMI: `bash rsmi.sh`

The results are put in `/project_root/results`, and the figure drawing Jupyter notebooks are put in `/project_root/figures`.

## Reference
[1] Haixin Wang, Xiaoyi Fu, Jianliang Xu, and Hua Lu. 2019. Learned Index for Spatial Queries. In MDM. IEEE, 569–574.

[2] Angjela Davitkova, Evica Milchevski, and Sebastian Michel. 2020. The ML-Index: A Multidimensional, Learned Index for Point, Range, and Nearest-Neighbor Queries. In EDBT. OpenProceedings.org, 407–410.

[3] Ali Hadian, Ankit Kumar, and Thomas Heinis. 2020. Hands-off Model Integration in Spatial Index Structures. In AIDB@VLDB.

[4] Jianzhong Qi, Guanli Liu, Christian S. Jensen, and Lars Kulik. 2020. Effectively Learning Spatial Indices. Proc. VLDB Endow. 13, 11 (2020), 2341–2354.

[5] Pengfei Li, Hua Lu, Qian Zheng, Long Yang, and Gang Pan. 2020. LISA: A Learned Index Structure for Spatial Data. In SIGMOD Conference. ACM, 2119–2133.

[6] Vikram Nathan, Jialin Ding, Mohammad Alizadeh, and Tim Kraska. 2020. Learning Multi-Dimensional Indexes. In SIGMOD Conference. ACM, 985–1000.

=======
# DualIndex
Unified Indexing architecture; combines two specialized learned indexes under a unified query router. It routes point queries to [ZM-Index](https://ieeexplore.ieee.org/document/8788832) and range queries to [Flood](https://arxiv.org/abs/1912.01668).Still under construction - as a part of my Master Thesis.

## Data Model
`(SourceID, Hop1_ID, Hop2_ID) → Offset`

Each triple represents a 2-hop path in a graph: starting from a source node, 
traversing to a first-hop neighbor, then to a second-hop neighbor. For example, 
in a social network, the triple `(Alice, Bob, Charlie)` encodes the path 
"Alice is friends with Bob, who is friends with Charlie."

Triples are extracted from graph edge lists by enumerating all 2-hop paths 
`(u → v → w)`, lexicographically sorted by `(SourceID, Hop1_ID, Hop2_ID)`, 
and assigned sequential offsets representing their position in sorted order. 

The learned index approximates this mapping — given a 3D key, predict its 
offset — replacing traditional pointer-based graph traversal with direct 
position prediction.

## Datasets Used (So far)

**Real-world graphs (SNAP):**
- [**Wiki-Vote**](https://snap.stanford.edu/data/wiki-Vote.html) — directed voting network from Wikipedia adminship elections 
  (~4.5M triples)
- [**roadNet-CA**](https://snap.stanford.edu/data/roadNet-CA.html)— undirected California road network (~17.5M triples)
- [**web-Google**](https://snap.stanford.edu/data/web-Google.html) — directed web hyperlink graph (~60.7M triples)

**Synthetic distributions (60M triples each):**
- `uniform_sparse` — large coordinate space [1, 10M), globally sparse connectivity
- `uniform_dense` — small coordinate space [1, 500K), densely connected subgraphs
- `uniform_matched` — [1, 1M), density-matched baseline
- `normal` — hub-concentrated topology (μ=500K, σ=166K)
- `lognormal` — power-law skew typical of real social/web graphs

## Implementation Inspiration
The 2 models I have unified here are taken from the [learnedbench](https://github.com/qyliu-hkust/learnedbench) repo, which is based on their paper ["How good are multi-dimensional learned indexes? An experimental survey"](https://link.springer.com/article/10.1007/s00778-024-00893-6). I learnt and have been learning so much from this paper, as well as the legendary ["The Case for Learned Index Strcutures"](https://dl.acm.org/doi/abs/10.1145/3183713.3196909). 
>>>>>>> d912fff66259bc064b9c3b05f5c24ddc250ae4a1
