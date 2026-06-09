#!/usr/bin/env python3
"""
pivot_synthetic.py — turn the long synthetic sweep CSV into a readable table.

Reads:  results/k_sensitivity/synthetic_all.csv
          (index,dataset,K,query_type,mean_us,p50_us,p95_us,p99_us,build_time_ms,index_size_bytes,avg_results)
Writes: results/k_sensitivity/synthetic_pivot.csv   (mean latency, K as columns)
        results/k_sensitivity/synthetic_pivot.md     (same, markdown)

One row per (dataset, query_type, index); columns are the K values. Stock and
sortsource rows sit next to each other so the comparison is obvious.
"""
import csv, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "results", "k_sensitivity", "synthetic_all.csv")
OUT_CSV = os.path.join(ROOT, "results", "k_sensitivity", "synthetic_pivot.csv")
OUT_MD  = os.path.join(ROOT, "results", "k_sensitivity", "synthetic_pivot.md")

if not os.path.exists(SRC):
    print(f"ERROR: {SRC} not found", file=sys.stderr); sys.exit(1)

# val[(index,dataset,qt)][K] = mean_us ; also keep avg_results
val = defaultdict(dict)
avg = {}
ks_seen, ds_seen = set(), []
with open(SRC) as f:
    for row in csv.DictReader(f):
        key = (row["index"], row["dataset"], row["query_type"])
        K = int(row["K"])
        val[key][K] = float(row["mean_us"])
        avg[key] = float(row["avg_results"])
        ks_seen.add(K)
        if row["dataset"] not in ds_seen: ds_seen.append(row["dataset"])

Ks = sorted(ks_seen)
indexes = ["stock", "sortsource"]
qts = ["single_hop", "multi_hop"]

# ── CSV ───────────────────────────────────────────────────────────────────────
hdr = ["dataset", "query_type", "index", "avg_results"] + [f"K{k}_us" for k in Ks] + ["best_K", "best_us"]
rows = []
for ds in ds_seen:
    for qt in qts:
        for idx in indexes:
            key = (idx, ds, qt)
            if key not in val: continue
            series = val[key]
            cells = [series.get(k) for k in Ks]
            present = [(k, series[k]) for k in Ks if k in series]
            bestK, bestV = (min(present, key=lambda kv: kv[1]) if present else ("", ""))
            rows.append([ds, qt, idx, f"{avg.get(key,0):.1f}"]
                        + [("" if c is None else f"{c:.3f}") for c in cells]
                        + [bestK, ("" if bestV=="" else f"{bestV:.3f}")])

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr); w.writerows(rows)

# ── Markdown (grouped, stock vs sortsource adjacent) ──────────────────────────
def fmt(x):
    return "—" if x in (None, "") else (f"{float(x):,.1f}")

lines = ["# Synthetic K-sweep — mean range-query latency (µs)\n",
         f"Datasets: 60M rows each. Queries: 2,000/type, seed 42. K columns: {Ks}.",
         "Stock = Flood (sorts on Hop2). Sortsource = FloodSourceSort (sorts on SourceID).\n"]
for qt in qts:
    lines.append(f"\n## {qt}\n")
    lines.append("| dataset | index | avg res | " + " | ".join(f"K={k}" for k in Ks) + " | best K |")
    lines.append("|" + "---|"*(4+len(Ks)+1))
    for ds in ds_seen:
        for idx in indexes:
            key = (idx, ds, qt)
            if key not in val: continue
            series = val[key]
            present = [(k, series[k]) for k in Ks if k in series]
            bestK = min(present, key=lambda kv: kv[1])[0] if present else "—"
            cells = " | ".join(fmt(series.get(k)) for k in Ks)
            star_idx = "**"+idx+"**" if idx == "sortsource" else idx
            lines.append(f"| {ds} | {star_idx} | {avg.get(key,0):,.0f} | {cells} | **{bestK}** |")
with open(OUT_MD, "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote {os.path.relpath(OUT_CSV, ROOT)} and {os.path.relpath(OUT_MD, ROOT)}")
print(f"  datasets={len(ds_seen)}  K values={Ks}  rows={len(rows)}")
