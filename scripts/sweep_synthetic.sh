#!/usr/bin/env bash
# Synthetic K-sweep: stock Flood vs FloodSourceSort, K in {2,4,8,16,20},
# on the 5 synthetic 60M datasets, both query types. 2000 queries/type
# (reduced from 10000 so stock Flood's low-K runs on 60M stay tractable).
# Runs sortsource first (fast), then stock (slow). 30-min cap per run.

cd /home/proteeti/DualIndex || exit 1
OUT=results/k_sensitivity/synthetic_all.csv
LOG=results/k_sensitivity/synthetic_sweep.log
TMP=/tmp/synth_run_$$.txt
N=60000000
Q=2000
KS="2 4 8 16 20"
DSS="uniform_sparse uniform_dense uniform_matched normal lognormal"

echo "index,dataset,K,query_type,mean_us,p50_us,p95_us,p99_us,build_time_ms,index_size_bytes,avg_results" > "$OUT"
: > "$LOG"
echo "START $(date)  N=$N Q=$Q  K={$KS}" | tee -a "$LOG"

run_one() {   # $1=label  $2=exe_prefix  $3=dataset  $4=K
  label=$1; pref=$2; ds=$3; K=$4
  echo ">>> $label $ds K=$K  ($(date +%H:%M:%S))" | tee -a "$LOG"
  timeout 1800 build/bin/${pref}${K} datasets/${ds}_60M.txt $N --queries $Q --seed 42 --no_header 2>/dev/null > "$TMP"
  rc=$?
  if [ $rc -eq 0 ] && [ -s "$TMP" ]; then
    sed "s/^/${label},/" "$TMP" >> "$OUT"
    echo "    ok" | tee -a "$LOG"
  else
    echo "    TIMEOUT/FAIL (rc=$rc) — omitted" | tee -a "$LOG"
  fi
}

echo "=== PHASE 1: FloodSourceSort (fast) ===" | tee -a "$LOG"
for ds in $DSS; do for K in $KS; do run_one sortsource bench_flood_ss_k "$ds" "$K"; done; done

echo "=== PHASE 2: stock Flood (slow at low K) ===" | tee -a "$LOG"
for ds in $DSS; do for K in $KS; do run_one stock bench_flood_k "$ds" "$K"; done; done

rm -f "$TMP"
echo "=== DATA COLLECTION DONE ($(wc -l < "$OUT") lines) $(date) ===" | tee -a "$LOG"

echo "=== building pivot table ===" | tee -a "$LOG"
python3 scripts/pivot_synthetic.py 2>&1 | tee -a "$LOG"
echo "=== ALL DONE $(date) ===" | tee -a "$LOG"
