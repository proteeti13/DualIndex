#!/usr/bin/env python3
"""
plot_k_sensitivity.py — thesis figures for the Flood grid-size (K) sweep.

Reads:
    results/k_sensitivity/k_sensitivity.csv   (this experiment)
    build/results/art/art_<dataset>.csv       (ART baseline, optional)

Writes:
    figures/k_sensitivity/fig1_k_vs_latency.{pdf,png}
    figures/k_sensitivity/fig2_k_cost_breakdown.{pdf,png}
    figures/k_sensitivity/fig3_summary_table.{pdf,png}
    results/k_sensitivity/summary.md

stdlib csv + numpy + matplotlib only (no pandas).
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi":      300,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
    "savefig.pad_inches": 0.1,
})

# colorblind-safe (Wong / tab10-ish)
C_SINGLE = "#0072B2"   # blue
C_MULTI  = "#D55E00"   # vermillion
C_ART    = "#009E73"   # green
C_K20    = "#999999"   # gray

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH  = os.path.join(ROOT, "results", "k_sensitivity", "hop2_sort_dim", "k_sensitivity.csv")
ART_DIR   = os.path.join(ROOT, "build", "results", "art")
FIG_DIR   = os.path.join(ROOT, "figures", "k_sensitivity")
OUT_MD    = os.path.join(ROOT, "results", "k_sensitivity", "summary.md")
os.makedirs(FIG_DIR, exist_ok=True)

DATASETS = ["wiki_vote", "web_google", "roadnet_ca"]
DS_LABEL = {"wiki_vote": "Wiki-Vote", "web_google": "WebGoogle", "roadnet_ca": "RoadNet-CA"}
K_ORDER  = [2, 3, 4, 6, 10, 15, 20]


# ── load sweep CSV ────────────────────────────────────────────────────────────
# data[(dataset, qt)][K] = row dict
data = defaultdict(dict)
N_of = {}
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found", file=sys.stderr)
    sys.exit(1)

with open(CSV_PATH) as f:
    for row in csv.DictReader(f):
        ds = row["dataset"]
        qt = row["query_type"]
        K  = int(row["K"])
        data[(ds, qt)][K] = {
            "mean":  float(row["mean_us"]),
            "p50":   float(row["p50_us"]),
            "p95":   float(row["p95_us"]),
            "p99":   float(row["p99_us"]),
            "build": float(row["build_time_ms"]),
            "size":  int(row["index_size_bytes"]),
            "avg":   float(row["avg_results"]),
        }

present_datasets = [ds for ds in DATASETS if (ds, "single_hop") in data]
print(f"Datasets present in sweep: {present_datasets}")


# ── ART baseline (optional) ───────────────────────────────────────────────────
# art[(dataset, qt)] = mean_us
art = {}
for ds in present_datasets:
    p = os.path.join(ART_DIR, f"art_{ds}.csv")
    if os.path.exists(p):
        with open(p) as f:
            for row in csv.DictReader(f):
                if row["query_type"] in ("single_hop", "multi_hop"):
                    art[(ds, row["query_type"])] = float(row["lat_mean_us"])


def ks_present(ds, qt):
    return sorted(data[(ds, qt)].keys())


def series(ds, qt, field):
    ks = ks_present(ds, qt)
    return ks, [data[(ds, qt)][k][field] for k in ks]


def optimal_k(ds, qt, field="mean"):
    ks = ks_present(ds, qt)
    if not ks:
        return None, None
    best = min(ks, key=lambda k: data[(ds, qt)][k][field])
    return best, data[(ds, qt)][best][field]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — K vs latency, one subplot per dataset
# ─────────────────────────────────────────────────────────────────────────────
def fig1():
    n = len(present_datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.8), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, present_datasets):
        for qt, color, style, marker, lbl in [
            ("single_hop", C_SINGLE, "-",  "o", "single-hop"),
            ("multi_hop",  C_MULTI,  "--", "s", "multi-hop"),
        ]:
            ks, ys = series(ds, qt, "mean")
            if not ks:
                continue
            ax.plot(ks, ys, color=color, linestyle=style, marker=marker,
                    markersize=5, linewidth=1.6, label=lbl)
            bk, bv = optimal_k(ds, qt)
            ax.plot([bk], [bv], marker="*", markersize=16, color=color,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=5)

        # ART reference lines
        for qt, color in [("single_hop", C_SINGLE), ("multi_hop", C_MULTI)]:
            if (ds, qt) in art:
                ax.axhline(art[(ds, qt)], color=color, linestyle=":", linewidth=1.0, alpha=0.7)

        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xticks(K_ORDER)
        ax.set_xticklabels([str(k) for k in K_ORDER])
        ax.minorticks_off()
        ax.set_xlabel("grid resolution $K$")
        ax.set_title(DS_LABEL[ds])
        ax.grid(True, which="major", color="0.85", linewidth=0.5)

    axes[0].set_ylabel("mean range-query latency (µs, log)")
    handles = [
        plt.Line2D([], [], color=C_SINGLE, marker="o", linestyle="-", label="single-hop"),
        plt.Line2D([], [], color=C_MULTI, marker="s", linestyle="--", label="multi-hop"),
        plt.Line2D([], [], color="black", marker="*", linestyle="none", markersize=12, label="optimal $K$"),
        plt.Line2D([], [], color="0.4", linestyle=":", label="ART baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Effect of grid resolution $K$ on range query latency", y=1.03)
    fig.tight_layout()
    _save(fig, "fig1_k_vs_latency")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — cost breakdown: grouped bars (latency per K) + avg-results annotation
# ─────────────────────────────────────────────────────────────────────────────
def fig2():
    n = len(present_datasets)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 4.0), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, present_datasets):
        ks = ks_present(ds, "single_hop")
        x = np.arange(len(ks))
        w = 0.38
        s_vals = [data[(ds, "single_hop")][k]["mean"] for k in ks]
        m_vals = [data[(ds, "multi_hop")][k]["mean"]  for k in ks]

        ax.bar(x - w/2, s_vals, w, color=C_SINGLE, label="single-hop",
               edgecolor="black", linewidth=0.4)
        ax.bar(x + w/2, m_vals, w, color=C_MULTI, label="multi-hop",
               edgecolor="black", linewidth=0.4, hatch="//")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("grid resolution $K$")
        ax.set_title(DS_LABEL[ds])
        ax.grid(axis="y", which="major", color="0.85", linewidth=0.5)

        # annotate avg results per query (constant across K — show once near top)
        s_avg = data[(ds, "single_hop")][ks[0]]["avg"]
        m_avg = data[(ds, "multi_hop")][ks[0]]["avg"]
        ax.text(0.5, 0.97,
                f"avg results/query:\nsingle≈{s_avg:,.0f}, multi≈{m_avg:,.0f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
                color="#333", bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="0.7", lw=0.5))

    axes[0].set_ylabel("mean latency (µs, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Range-query cost vs grid resolution $K$ (result-set size annotated)", y=1.02)
    fig.tight_layout()
    _save(fig, "fig2_k_cost_breakdown")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — summary table rendered as a figure
# ─────────────────────────────────────────────────────────────────────────────
def fig3(graph_stats):
    headers = ["Dataset", "N (triples)", "avg deg.",
               "avg res.\nsingle", "avg res.\nmulti",
               "opt. K\n(single)", "opt. K\n(multi)",
               "lat@K=20\nsingle (µs)", "lat@opt\nsingle (µs)",
               "speedup\nsingle"]
    rows = []
    for ds in present_datasets:
        bk_s, bv_s = optimal_k(ds, "single_hop")
        bk_m, bv_m = optimal_k(ds, "multi_hop")
        lat20_s = data[(ds, "single_hop")].get(20, {}).get("mean", float("nan"))
        s_avg = data[(ds, "single_hop")][ks_present(ds, "single_hop")[0]]["avg"]
        m_avg = data[(ds, "multi_hop")][ks_present(ds, "multi_hop")[0]]["avg"]
        n = graph_stats.get(ds, {}).get("N", "?")
        deg = graph_stats.get(ds, {}).get("avg_deg", None)
        speedup_s = (lat20_s / bv_s) if (bv_s and bv_s > 0) else float("nan")
        rows.append([
            DS_LABEL[ds],
            f"{n:,}" if isinstance(n, int) else str(n),
            f"{deg:.1f}" if deg is not None else "n/a",
            f"{s_avg:,.0f}",
            f"{m_avg:,.0f}",
            str(bk_s),
            str(bk_m),
            f"{lat20_s:,.1f}",
            f"{bv_s:,.1f}",
            f"{speedup_s:.2f}×",
        ])

    fig, ax = plt.subplots(figsize=(min(2.0 + 1.05 * len(headers), 13), 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.7)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("0.8")
        if r == 0:
            cell.set_facecolor("#2171B5")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f2f2")
    ax.set_title("Optimal grid resolution per dataset vs graph characteristics",
                 fontsize=12, pad=14)
    fig.tight_layout()
    _save(fig, "fig3_summary_table")


def _save(fig, name):
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(pdf, ROOT)} / .png")


# ── graph degree stats (streamed from the dataset files) ──────────────────────
def compute_graph_stats():
    """avg direct out-degree = #distinct (src,hop1) edges / #distinct src.
    Streamed so memory stays bounded-ish; uses sets (we have plenty of RAM)."""
    stats = {}
    for ds in present_datasets:
        path = os.path.join(ROOT, "datasets", f"{ds}_triples.txt")
        if not os.path.exists(path):
            continue
        srcs = set()
        edges = set()
        n = 0
        try:
            with open(path) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    s = parts[0]; h1 = parts[1]
                    srcs.add(s)
                    edges.add((s, h1))
                    n += 1
            avg_deg = len(edges) / len(srcs) if srcs else None
            stats[ds] = {"N": n, "avg_deg": avg_deg}
            print(f"  {ds}: N={n:,}  distinct_src={len(srcs):,}  edges={len(edges):,}  avg_deg={avg_deg:.2f}")
        except Exception as e:
            print(f"  {ds}: graph-stat computation failed: {e}", file=sys.stderr)
            stats[ds] = {"N": n, "avg_deg": None}
    return stats


# ── markdown summary ──────────────────────────────────────────────────────────
def write_summary(graph_stats):
    lines = ["# Flood K-Sensitivity — Summary\n"]
    lines.append("Sweep of Flood's compile-time grid resolution K ∈ {2,3,4,6,10,15,20} "
                 "on three real SNAP graph datasets, 10 000 single-hop + 10 000 multi-hop "
                 "range queries each, seed=42, single trial.\n")
    lines.append("\n## Optimal K per dataset (by mean latency)\n")
    lines.append("| Dataset | query | optimal K | lat@opt (µs) | lat@K=20 (µs) | speedup | ART (µs) | vs ART |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in present_datasets:
        for qt in ("single_hop", "multi_hop"):
            bk, bv = optimal_k(ds, qt)
            lat20 = data[(ds, qt)].get(20, {}).get("mean", float("nan"))
            sp = (lat20 / bv) if bv else float("nan")
            a = art.get((ds, qt))
            a_str = f"{a:,.1f}" if a else "—"
            vs = f"{a/bv:.2f}× {'(Flood wins)' if a and bv < a else '(ART wins)'}" if a and bv else "—"
            lines.append(f"| {DS_LABEL[ds]} | {qt} | **{bk}** | {bv:,.1f} | {lat20:,.1f} | {sp:.2f}× | {a_str} | {vs} |")

    lines.append("\n## Graph characteristics\n")
    lines.append("| Dataset | N (triples) | avg out-degree | avg results single | avg results multi |")
    lines.append("|---|---|---|---|---|")
    for ds in present_datasets:
        gs = graph_stats.get(ds, {})
        n = gs.get("N", "?")
        deg = gs.get("avg_deg")
        s_avg = data[(ds, "single_hop")][ks_present(ds, "single_hop")[0]]["avg"]
        m_avg = data[(ds, "multi_hop")][ks_present(ds, "multi_hop")[0]]["avg"]
        lines.append(f"| {DS_LABEL[ds]} | {n:,} | {deg:.2f} | {s_avg:,.0f} | {m_avg:,.0f} |"
                     if isinstance(n, int) and deg else
                     f"| {DS_LABEL[ds]} | {n} | n/a | {s_avg:,.0f} | {m_avg:,.0f} |")

    with open(OUT_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote {os.path.relpath(OUT_MD, ROOT)}")


def main():
    print("Computing graph degree stats (streaming dataset files)…")
    graph_stats = compute_graph_stats()
    print("\nGenerating figures…")
    fig1()
    fig2()
    fig3(graph_stats)
    write_summary(graph_stats)
    return 0


if __name__ == "__main__":
    sys.exit(main())
