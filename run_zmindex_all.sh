#!/usr/bin/env bash
# Regenerates all ZM-Index benchmark CSVs across every dataset + scalability factor.
# Output goes to build/results/zmindex_per_run/

set -e
cd "$(dirname "$0")"

BIN=build/bin/bench_zmindex_all
OUT=build/results/zmindex_per_run
Q=100000
SEED=42

run() {
    local file=$1 N=$2 name=$3 src=$4 dist=$5 gtype=$6 full=$7 tag=$8
    echo "  -> $tag  (N=$N)"
    $BIN "$file" "$N" "$name" "$src" "$dist" "$gtype" "$full" \
         --queries $Q --seed $SEED --out_csv "$OUT/${tag}.csv"
}

echo "=== ZM-Index full benchmark suite ==="
echo "Output dir: $OUT"
echo ""

# ── Wiki-Vote (SNAP directed, 4,542,805 rows) ─────────────────────────────
echo "[1/8] wiki_vote"
F=datasets/wiki_vote_triples.txt
run "$F" 1000000  wiki_vote SNAP real_graph_directed voting_network 4542805  zm_wiki_vote_1m
run "$F" 2500000  wiki_vote SNAP real_graph_directed voting_network 4542805  zm_wiki_vote_2500k
run "$F" 4542805  wiki_vote SNAP real_graph_directed voting_network 4542805  zm_wiki_vote_full

# ── RoadNet-CA (SNAP undirected, 17,523,394 rows) ─────────────────────────
echo "[2/8] roadnet_ca"
F=datasets/roadnet_ca_triples.txt
run "$F" 1000000  roadnet_ca SNAP real_graph_undirected road_network 17523394  zm_roadnet_ca_1m
run "$F" 2500000  roadnet_ca SNAP real_graph_undirected road_network 17523394  zm_roadnet_ca_2500k
run "$F" 17523394 roadnet_ca SNAP real_graph_undirected road_network 17523394  zm_roadnet_ca_full

# ── Web-Google (SNAP directed, 60,687,836 rows) ───────────────────────────
echo "[3/8] web_google"
F=datasets/web_google_triples.txt
run "$F" 1000000  web_google SNAP real_graph_directed web_graph 60687836  zm_web_google_1m
run "$F" 2500000  web_google SNAP real_graph_directed web_graph 60687836  zm_web_google_2500k
run "$F" 60687836 web_google SNAP real_graph_directed web_graph 60687836  zm_web_google_full

# ── Uniform Sparse (synthetic, 60M rows) ─────────────────────────────────
echo "[4/8] uniform_sparse"
F=datasets/uniform_sparse_60M.txt
run "$F" 1000000  uniform_sparse Synthetic uniform_sparse N/A 60000000  zm_uniform_sparse_1m
run "$F" 5000000  uniform_sparse Synthetic uniform_sparse N/A 60000000  zm_uniform_sparse_5m
run "$F" 10000000 uniform_sparse Synthetic uniform_sparse N/A 60000000  zm_uniform_sparse_10m

# ── Uniform Dense (synthetic, 60M rows) ──────────────────────────────────
echo "[5/8] uniform_dense"
F=datasets/uniform_dense_60M.txt
run "$F" 1000000  uniform_dense Synthetic uniform_dense N/A 60000000  zm_uniform_dense_1m
run "$F" 5000000  uniform_dense Synthetic uniform_dense N/A 60000000  zm_uniform_dense_5m
run "$F" 10000000 uniform_dense Synthetic uniform_dense N/A 60000000  zm_uniform_dense_10m

# ── Uniform Matched (synthetic, 60M rows) ────────────────────────────────
echo "[6/8] uniform_matched"
F=datasets/uniform_matched_60M.txt
run "$F" 1000000  uniform_matched Synthetic uniform_matched N/A 60000000  zm_uniform_matched_1m
run "$F" 5000000  uniform_matched Synthetic uniform_matched N/A 60000000  zm_uniform_matched_5m
run "$F" 10000000 uniform_matched Synthetic uniform_matched N/A 60000000  zm_uniform_matched_10m

# ── Normal (synthetic, 60M rows) ─────────────────────────────────────────
echo "[7/8] normal"
F=datasets/normal_60M.txt
run "$F" 1000000  normal Synthetic normal N/A 60000000  zm_normal_1m
run "$F" 5000000  normal Synthetic normal N/A 60000000  zm_normal_5m
run "$F" 10000000 normal Synthetic normal N/A 60000000  zm_normal_10m

# ── Lognormal (synthetic, 60M rows) ──────────────────────────────────────
echo "[8/8] lognormal"
F=datasets/lognormal_60M.txt
run "$F" 1000000  lognormal Synthetic lognormal N/A 60000000  zm_lognormal_1m
run "$F" 5000000  lognormal Synthetic lognormal N/A 60000000  zm_lognormal_5m
run "$F" 10000000 lognormal Synthetic lognormal N/A 60000000  zm_lognormal_10m

echo ""
echo "=== All done. CSVs written to $OUT/ ==="
ls -1 "$OUT/"
