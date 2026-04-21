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
