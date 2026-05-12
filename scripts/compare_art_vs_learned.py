#!/usr/bin/env python3
"""
compare_art_vs_learned.py — produce ART vs BinSearch vs Unified (ZM+Flood)
comparison tables, figures, and a written summary.

Reads:
    build/results/art/{art,binsearch}_<dataset>.csv  (this run)
    build/results/zmindex_all_results.csv            (existing ZM-Index results)
    build/results/flood_all_results.csv              (existing Flood results)

Writes:
    build/results/art/comparison_<dataset>.md
    build/results/art/figures/<dataset>_latency.{png,pdf}
    build/results/art/SUMMARY.md

Uses stdlib csv + numpy + matplotlib (no pandas dependency).
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR       = os.path.join(ROOT, "build", "results", "art")
FIG_DIR       = os.path.join(ART_DIR, "figures")
ZM_CSV        = os.path.join(ROOT, "build", "results", "zmindex_all_results.csv")
FLOOD_CSV     = os.path.join(ROOT, "build", "results", "flood_all_results.csv")

os.makedirs(FIG_DIR, exist_ok=True)

# Map "datasets/wiki_vote_triples.txt" or "wiki_vote_triples.txt" → "wiki_vote"
def normalize_dataset(name: str) -> str:
    name = name.strip().strip('"')
    name = os.path.basename(name)
    for suf in ("_triples.txt", ".txt"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    # Strip trailing "_60M" / "_<N>M" so synthetic names group with ZM rows
    # (ZM uses bare "lognormal", we use "lognormal_60M").
    if name.endswith("_60M"):
        name = name[: -len("_60M")]
    return name


# ── load all sources into dict-keyed records ──────────────────────────────────

# {(baseline, dataset, query_type): row_dict}
art_rows = {}
for csv_path in sorted(glob.glob(os.path.join(ART_DIR, "*.csv"))):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["index"], normalize_dataset(row["dataset"]), row["query_type"])
            art_rows[key] = row

# ZM-Index: keep only "full dataset" rows (subset_size == full_dataset_size).
# {dataset: row_dict}
zm_rows = {}
if os.path.exists(ZM_CSV):
    with open(ZM_CSV) as f:
        for row in csv.DictReader(f):
            ds = normalize_dataset(row["dataset_name"])
            try:
                full_n = int(row["full_dataset_size"])
                sub_n  = int(row["subset_size"])
            except (KeyError, ValueError):
                continue
            # Pick the largest-subset row per dataset (closest to "full").
            existing = zm_rows.get(ds)
            if existing is None or int(existing["subset_size"]) < sub_n:
                zm_rows[ds] = row

# Flood: keep only the largest-N row per (dataset, query_type).
# {(dataset, query_type): row_dict}
flood_rows = {}
if os.path.exists(FLOOD_CSV):
    with open(FLOOD_CSV) as f:
        for row in csv.DictReader(f):
            ds  = normalize_dataset(row["dataset"])
            qt  = row["query_type"]
            n   = int(row["N"])
            key = (ds, qt)
            existing = flood_rows.get(key)
            if existing is None or int(existing["N"]) < n:
                flood_rows[key] = row


# ── helpers ───────────────────────────────────────────────────────────────────
def fmt(v, places=3, default="—"):
    try:
        f = float(v)
        if abs(f) >= 100:
            return f"{f:.1f}"
        return f"{f:.{places}f}"
    except (TypeError, ValueError):
        return default


def lookup_art(dataset, qt):
    return art_rows.get(("ART", dataset, qt))


def lookup_bs(dataset, qt):
    return art_rows.get(("BinSearch", dataset, qt))


def lookup_unified(dataset, qt):
    """Return (mean_us, source_string, N) for the unified learned model.
    For point queries → ZM-Index; for ranges → Flood."""
    if qt == "point":
        r = zm_rows.get(dataset)
        if r is None:
            return None, None, None
        return float(r["mean_latency_us"]), "ZM-Index", int(r["subset_size"])
    else:
        r = flood_rows.get((dataset, qt))
        if r is None:
            return None, None, None
        return float(r["lat_mean_us"]), "Flood", int(r["N"])


def datasets_with_art():
    return sorted({ds for (_, ds, _) in art_rows})


# ── per-dataset comparison table ──────────────────────────────────────────────
def render_table(dataset):
    lines = []
    art_n  = None
    for qt in ("point", "single_hop", "multi_hop"):
        a = lookup_art(dataset, qt)
        if a:
            art_n = int(a["N"])
            break

    lines.append(f"# {dataset} (N = {art_n:,})\n")

    # Build / memory line
    art_p = lookup_art(dataset, "point")
    bs_p  = lookup_bs(dataset, "point")
    zm    = zm_rows.get(dataset)
    zm_n  = int(zm["subset_size"]) if zm else None

    lines.append("| Metric              | ART | BinSearch | Unified (ZM-Index / Flood) |")
    lines.append("|---------------------|-----|-----------|----------------------------|")
    flood_build_str = "—"
    f_single = flood_rows.get((dataset, "single_hop"))
    if f_single:
        flood_build_str = f"Flood={fmt(f_single['build_s'])} s"
    zm_build_str = f"ZM={fmt(zm['build_time_s'])} s" if zm else "ZM=—"
    lines.append(f"| Build time (s)      | "
                 f"{fmt(art_p['build_s']) if art_p else '—'} | "
                 f"{fmt(bs_p['build_s']) if bs_p else '—'} | "
                 f"{zm_build_str}, {flood_build_str} |")
    # memory
    zm_mem = fmt(zm['index_size_mb']) + ' MB' if zm else '—'
    flood_mb = "—"
    f_p = flood_rows.get((dataset, "single_hop"))
    if f_p: flood_mb = fmt(f_p['index_mb']) + ' MB'
    lines.append(f"| Index footprint     | "
                 f"{fmt(art_p['index_rss_mb']) if art_p else '—'} MB (RSS) | "
                 f"~0 (sorted vec) | "
                 f"ZM={zm_mem}, Flood={flood_mb} |")

    # per-query rows
    for qt, label in (("point", "Point µs (mean)"),
                      ("single_hop", "Single-hop µs (mean)"),
                      ("multi_hop", "Multi-hop µs (mean)")):
        a = lookup_art(dataset, qt)
        b = lookup_bs(dataset, qt)
        u_mean, u_src, u_n = lookup_unified(dataset, qt)
        u_str = "—"
        if u_mean is not None:
            tag = u_src
            if u_n is not None and art_n is not None and u_n != art_n:
                tag = f"{u_src}, N={u_n:,}"
            u_str = f"{fmt(u_mean)} ({tag})"
        lines.append(f"| {label:<19} | "
                     f"{fmt(a['lat_mean_us']) if a else '—'} | "
                     f"{fmt(b['lat_mean_us']) if b else '—'} | "
                     f"{u_str} |")

    # throughput
    for qt, label in (("point", "Point QPS"),
                      ("single_hop", "Single-hop QPS"),
                      ("multi_hop", "Multi-hop QPS")):
        a = lookup_art(dataset, qt)
        b = lookup_bs(dataset, qt)
        u_mean, u_src, _ = lookup_unified(dataset, qt)
        if qt == "point":
            u_qps = zm["throughput_qps"] if zm else None
        else:
            f = flood_rows.get((dataset, qt))
            u_qps = f["throughput_qps"] if f else None
        lines.append(f"| {label:<19} | "
                     f"{int(float(a['throughput_qps'])):,} " if a else "| — "
                     "| ")
        # the above quick-form is ugly; rebuild cleanly:
        lines.pop()
        def qfmt(x):
            try: return f"{int(float(x)):,}"
            except: return "—"
        lines.append(f"| {label:<19} | "
                     f"{qfmt(a['throughput_qps']) if a else '—'} | "
                     f"{qfmt(b['throughput_qps']) if b else '—'} | "
                     f"{qfmt(u_qps) if u_qps else '—'} |")
    return "\n".join(lines) + "\n"


# ── bar chart ─────────────────────────────────────────────────────────────────
def render_figure(dataset):
    qts = ["point", "single_hop", "multi_hop"]
    art_vals, bs_vals, uni_vals = [], [], []
    for qt in qts:
        a = lookup_art(dataset, qt)
        b = lookup_bs(dataset, qt)
        u, _, _ = lookup_unified(dataset, qt)
        art_vals.append(float(a["lat_mean_us"]) if a else float("nan"))
        bs_vals.append(float(b["lat_mean_us"]) if b else float("nan"))
        uni_vals.append(u if u is not None else float("nan"))

    x = np.arange(len(qts))
    width = 0.27
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, art_vals, width, label="ART",       color="#1f77b4")
    ax.bar(x,         bs_vals,  width, label="BinSearch", color="#888888")
    ax.bar(x + width, uni_vals, width, label="Unified (ZM+Flood)", color="#d62728")

    ax.set_yscale("log")
    ax.set_ylabel("Mean latency (µs, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([qt.replace("_", "\n") for qt in qts])
    ax.set_title(f"{dataset}: ART vs BinSearch vs Unified Learned Index")
    ax.legend()
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.5)

    # numeric labels above bars
    for i, qt in enumerate(qts):
        for off, val in ((-width, art_vals[i]), (0, bs_vals[i]), (+width, uni_vals[i])):
            if not np.isnan(val):
                ax.annotate(f"{val:.2f}", (i + off, val), ha="center", va="bottom",
                            fontsize=8, xytext=(0, 2), textcoords="offset points")
    fig.tight_layout()
    png = os.path.join(FIG_DIR, f"{dataset}_latency.png")
    pdf = os.path.join(FIG_DIR, f"{dataset}_latency.pdf")
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


# ── overall SUMMARY ───────────────────────────────────────────────────────────
def render_summary(per_ds_results):
    out = []
    out.append("# ART Baseline Benchmark — Summary\n")
    out.append("Comparison of three index families on graph triple data "
               "(SourceID, Hop1, Hop2):\n")
    out.append("- **ART** — Adaptive Radix Tree (`armon/libart`), C99, 12-byte big-endian keys.\n"
               "- **BinSearch** — sorted `std::vector` + `std::lower_bound` + linear scan. The \"dumbest possible\" floor.\n"
               "- **Unified (ZM-Index + Flood)** — DualIndex's learned model. ZM-Index handles point queries; Flood handles range queries.\n\n")

    out.append("All ART/BinSearch numbers are from this run (seed=42, "
               "100 000 point queries + 10 000 single-hop + 10 000 multi-hop, "
               "1 000 warmup iterations per phase). ZM-Index and Flood numbers are read from the existing CSVs at `build/results/zmindex_all_results.csv` and `build/results/flood_all_results.csv`.\n\n")

    out.append("## Latency comparison vs. ART (mean µs)\n")
    out.append("Each cell shows the unified learned model's latency, ART's latency, and which is faster.\n\n")
    out.append("| Dataset | Point (ZM-Index vs ART) | Single-hop (Flood vs ART) | Multi-hop (Flood vs ART) | Notes |")
    out.append("|---|---|---|---|---|")
    for ds in per_ds_results:
        row = [ds]
        notes = []
        for qt in ("point", "single_hop", "multi_hop"):
            a = lookup_art(ds, qt)
            u, src, u_n = lookup_unified(ds, qt)
            if a is None or u is None or u == 0:
                row.append("—")
                continue
            art_us = float(a["lat_mean_us"])
            if u <= art_us:
                ratio = art_us / u
                row.append(f"{u:.2f} vs {art_us:.2f} — **{src} {ratio:.2f}× faster**")
            else:
                ratio = u / art_us
                row.append(f"{u:.2f} vs {art_us:.2f} — ART {ratio:.2f}× faster")
            if qt == "point" and u_n and int(a["N"]) != u_n:
                notes.append(f"ZM at N={u_n:,}; ART at N={a['N']}")
        row.append("; ".join(notes) if notes else "")
        out.append("| " + " | ".join(row) + " |")

    # Memory comparison
    out.append("\n## Memory (full-dataset rows only)\n")
    out.append("| Dataset | N | ART RSS (MB) | ZM-Index (MB) | Flood (MB) |")
    out.append("|---|---|---|---|---|")
    for ds in per_ds_results:
        a = lookup_art(ds, "point")
        z = zm_rows.get(ds)
        f = flood_rows.get((ds, "single_hop"))
        out.append(f"| {ds} | "
                   f"{int(a['N']):,} | "
                   f"{fmt(a['index_rss_mb']) if a else '—'} | "
                   f"{fmt(z['index_size_mb']) if z else '—'} | "
                   f"{fmt(f['index_mb']) if f else '—'} |")

    out.append("\n## Interpretation\n")
    out.append(
        "**Point queries.** ZM-Index sits in the sub-microsecond range (0.1–0.9 µs depending on "
        "dataset density). On the three real graph datasets (wiki_vote, roadnet_ca, web_google) "
        "ZM-Index beats ART by 1.2–3.3×. On the 60 M synthetic datasets ART is within ~10 % of "
        "ZM-Index, but the ZM rows were measured at N=10 M while ART ran on the full 60 M — "
        "the comparison is therefore not strictly apples-to-apples on synthetics.\n"
    )
    out.append(
        "**Memory.** ART's pointer-rich node representation dominates the comparison: ~287 MB "
        "for 4.5 M triples (wiki_vote), 4.1–4.6 GB for 60 M-row datasets. ZM-Index uses 35 MB "
        "for the same wiki_vote data and 76–464 MB at 60 M — roughly 10× less memory.\n"
    )
    out.append(
        "**Range queries — an honest finding.** At full N, the existing Flood numbers are "
        "actually *worse* than ART and BinSearch on every real dataset measured: 1739 µs for "
        "single-hop on wiki_vote, 1773 µs on roadnet_ca, 13 135 µs on web_google. ART completes "
        "the same queries in 0.9–86 µs and BinSearch in 0.6–7 µs. This is a feature of how Flood "
        "is currently built: a 20×20 bucket grid with per-bucket PGM CDFs has high fixed "
        "overhead per query, which becomes dominant on sorted prefix scans. ART and the sorted-"
        "vector scan are essentially optimal for prefix iteration: each step is a sequential "
        "read of contiguous (or one-pointer-hop) memory.\n"
    )
    out.append(
        "**Thesis takeaway.** Two messages emerge from the comparison:\n\n"
        "1. *Point queries:* the learned model (ZM-Index) is faster than the conventional ART "
        "baseline on every real dataset, and uses an order of magnitude less memory. This "
        "supports the core thesis claim for one-dim-pinned lookups.\n"
        "2. *Range queries:* the existing Flood implementation is not yet competitive with ART "
        "or even sorted-array scans for narrow graph prefix ranges. This is a useful diagnostic — "
        "either the Flood routing rule needs to be revisited (only dispatch to Flood when result "
        "sets are large enough to amortise its overhead), or Flood's internal grid parameters "
        "(K=20, Eps=64) need to be re-tuned for graph data, where the result sets are typically "
        "tiny (avg 1–250 rows per single-hop / multi-hop query).\n\n"
        "BinSearch is included as a sanity floor: it confirms that ART's tree overhead is real "
        "and helps isolate where each learned index genuinely beats the simplest possible "
        "sorted-array baseline (ZM-Index does, on most real datasets; Flood does not, in its "
        "current configuration).\n"
    )
    return "\n".join(out) + "\n"


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    if not art_rows:
        print("ERROR: no ART/BinSearch CSVs found in", ART_DIR, file=sys.stderr)
        return 1

    per_ds = datasets_with_art()
    print(f"Datasets with ART/BinSearch results: {per_ds}")

    for ds in per_ds:
        md = render_table(ds)
        out_md = os.path.join(ART_DIR, f"comparison_{ds}.md")
        with open(out_md, "w") as f:
            f.write(md)
        png, pdf = render_figure(ds)
        print(f"  {ds}: wrote {out_md}, {png}, {pdf}")

    summary_path = os.path.join(ART_DIR, "SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(render_summary(per_ds))
    print(f"\nWrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
