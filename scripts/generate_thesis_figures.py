#!/usr/bin/env python3
"""
generate_thesis_figures.py — publication-quality figures for the thesis
Experimentation chapter.

Reads:
    build/results/art/{ART,BinSearch}_*.csv  (new ART + BinSearch results)
    build/results/zmindex_all_results.csv    (existing ZM-Index point results)
    build/results/flood_all_results.csv      (existing Flood range results)

Writes:
    build/results/art/thesis_figures/fig_*.{pdf,png}

Uses stdlib csv + numpy + matplotlib only (no pandas).
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

# ── style ─────────────────────────────────────────────────────────────────────
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

# Colorblind-friendly palette
C_BS   = "#969696"   # gray   — BinSearch (sanity floor)
C_ART  = "#E6550D"   # orange — ART
C_ZM   = "#2171B5"   # blue   — ZM-Index / unified
C_FL   = "#31A354"   # green  — Flood
HATCH  = {"BS": "//", "ART": "\\\\", "ZM": "", "FL": "xx", "UNI": ""}

ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART_DIR   = os.path.join(ROOT, "build", "results", "art")
FIG_DIR   = os.path.join(ART_DIR, "thesis_figures")
ZM_CSV    = os.path.join(ROOT, "build", "results", "zmindex_all_results.csv")
FLOOD_CSV = os.path.join(ROOT, "build", "results", "flood_all_results.csv")
os.makedirs(FIG_DIR, exist_ok=True)

REAL_DS    = ["wiki_vote", "roadnet_ca", "web_google"]
SYNTH_DS   = ["uniform_sparse", "uniform_dense", "uniform_matched",
              "normal", "lognormal"]
ALL_DS     = REAL_DS + SYNTH_DS


# ── load CSVs ─────────────────────────────────────────────────────────────────
def normalize_ds(name):
    name = name.strip().strip('"')
    name = os.path.basename(name)
    for suf in ("_triples.txt", ".txt"):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    if name.endswith("_60M"):
        name = name[: -len("_60M")]
    return name


# {(baseline, dataset, query_type): row}
art_rows = {}
for path in glob.glob(os.path.join(ART_DIR, "*.csv")):
    if os.path.basename(path).startswith("comparison_"):
        continue
    with open(path) as f:
        for row in csv.DictReader(f):
            key = (row["index"], normalize_ds(row["dataset"]), row["query_type"])
            art_rows[key] = row

# ZM-Index: keep largest-subset row per dataset
zm_rows = {}
if os.path.exists(ZM_CSV):
    with open(ZM_CSV) as f:
        for row in csv.DictReader(f):
            ds = normalize_ds(row["dataset_name"])
            try:
                sub = int(row["subset_size"])
            except (KeyError, ValueError):
                continue
            existing = zm_rows.get(ds)
            if existing is None or int(existing["subset_size"]) < sub:
                zm_rows[ds] = row

# Flood: keep largest-N row per (dataset, query_type)
flood_rows = {}
if os.path.exists(FLOOD_CSV):
    with open(FLOOD_CSV) as f:
        for row in csv.DictReader(f):
            ds = normalize_ds(row["dataset"])
            qt = row["query_type"]
            n  = int(row["N"])
            key = (ds, qt)
            existing = flood_rows.get(key)
            if existing is None or int(existing["N"]) < n:
                flood_rows[key] = row


# ── helpers ───────────────────────────────────────────────────────────────────
def art_lat(ds, qt):
    r = art_rows.get(("ART", ds, qt));        return float(r["lat_mean_us"]) if r else None
def bs_lat(ds, qt):
    r = art_rows.get(("BinSearch", ds, qt));  return float(r["lat_mean_us"]) if r else None
def zm_lat(ds):
    r = zm_rows.get(ds);                      return float(r["mean_latency_us"]) if r else None
def flood_lat(ds, qt):
    r = flood_rows.get((ds, qt));             return float(r["lat_mean_us"]) if r else None

def art_qps(ds, qt):
    r = art_rows.get(("ART", ds, qt));        return float(r["throughput_qps"]) if r else None
def bs_qps(ds, qt):
    r = art_rows.get(("BinSearch", ds, qt));  return float(r["throughput_qps"]) if r else None
def zm_qps(ds):
    r = zm_rows.get(ds);                      return float(r["throughput_qps"]) if r else None
def flood_qps(ds, qt):
    r = flood_rows.get((ds, qt));             return float(r["throughput_qps"]) if r else None

def art_avg(ds, qt):
    r = art_rows.get(("ART", ds, qt));        return float(r["avg_results_per_query"]) if r else None

def art_n(ds):
    r = art_rows.get(("ART", ds, "point"));   return int(r["N"]) if r else None
def zm_n(ds):
    r = zm_rows.get(ds);                      return int(r["subset_size"]) if r else None


def stamp_speedup(ax, x, top, slow, fast, label_suffix=""):
    """Annotate a speedup ratio above a bar group."""
    if slow is None or fast is None or fast == 0:
        return
    ratio = slow / fast
    txt = f"{ratio:.1f}×" + label_suffix
    ax.annotate(txt, (x, top), ha="center", va="bottom",
                fontsize=8, xytext=(0, 4), textcoords="offset points",
                fontweight="bold", color="#444")


def save(fig, name):
    pdf = os.path.join(FIG_DIR, name + ".pdf")
    png = os.path.join(FIG_DIR, name + ".png")
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


def short_label(ds):
    # Tidy x-axis labels
    return {
        "wiki_vote":       "wiki_vote",
        "roadnet_ca":      "roadnet_ca",
        "web_google":      "web_google",
        "uniform_sparse":  "unif_sparse",
        "uniform_dense":   "unif_dense",
        "uniform_matched": "unif_matched",
        "normal":          "normal",
        "lognormal":       "lognormal",
    }.get(ds, ds)


def annotate_bars(ax, xs, vals, dy_pts=2, fmt="{:.2f}", fontsize=7):
    for x, v in zip(xs, vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        ax.annotate(fmt.format(v), (x, v), ha="center", va="bottom",
                    fontsize=fontsize, xytext=(0, dy_pts), textcoords="offset points")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Point Query Latency
# ─────────────────────────────────────────────────────────────────────────────
def fig1_point():
    datasets = [ds for ds in ALL_DS if art_lat(ds, "point") is not None]
    x = np.arange(len(datasets))
    w = 0.27

    bs  = [bs_lat(ds, "point")  for ds in datasets]
    art = [art_lat(ds, "point") for ds in datasets]
    zm  = [zm_lat(ds)           for ds in datasets]

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.bar(x - w, bs,  w, label="Sorted-Array (BinSearch)", color=C_BS,  hatch=HATCH["BS"],  edgecolor="black", linewidth=0.5)
    ax.bar(x,     art, w, label="ART",                       color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w, zm,  w, label="ZM-Index (learned)",        color=C_ZM,  hatch=HATCH["ZM"],  edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Mean point-query latency (µs, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(d) for d in datasets], rotation=25, ha="right")
    ax.set_title("Point Query Latency: Learned Index vs. Conventional Baselines",
                 pad=22)
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=False, ncol=3,
              bbox_to_anchor=(0.5, -0.22))

    # Headroom so 2-line speedup labels don't clip the title.
    all_vals = [v for v in bs + art + zm if v is not None and v > 0]
    if all_vals:
        ax.set_ylim(min(all_vals) * 0.5, max(all_vals) * 4.0)

    # Speedup annotation: ZM vs ART. Place at a fixed y above the highest bar.
    # Use top of axes (transformed) so labels don't fight the title.
    for i, ds in enumerate(datasets):
        a = art_lat(ds, "point"); z = zm_lat(ds)
        if a is None or z is None: continue
        top = max(a, z, bs[i] if bs[i] else 0)
        if z < a:
            txt = f"ZM\n{a/z:.1f}×"
            color = C_ZM
        else:
            txt = f"ART\n{z/a:.1f}×"
            color = C_ART
        ax.annotate(txt, (x[i], top), ha="center", va="bottom",
                    fontsize=7, xytext=(0, 4), textcoords="offset points",
                    fontweight="bold", color=color)

    # Note about N mismatch: place below the chart, above the legend.
    synth_with_mismatch = [ds for ds in datasets
                           if ds in SYNTH_DS and zm_n(ds) and art_n(ds)
                              and zm_n(ds) != art_n(ds)]
    if synth_with_mismatch:
        fig.text(0.5, 0.005,
                 "synthetic datasets: ZM-Index measured at N = 10 M, ART at N = 60 M",
                 ha="center", va="bottom",
                 fontsize=8, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save(fig, "fig1_point_query_latency_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Range Query Latency (single + multi hop subplots)
# ─────────────────────────────────────────────────────────────────────────────
def fig2_range():
    # Only datasets with Flood data → real datasets
    datasets = [ds for ds in REAL_DS if flood_lat(ds, "single_hop") is not None]
    x = np.arange(len(datasets))
    w = 0.27

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.0), sharey=True)
    for ax, qt, title in zip(axes, ("single_hop", "multi_hop"),
                             ("Single-Hop  (Source pinned)",
                              "Multi-Hop  (Source + Hop1 pinned)")):
        bs   = [bs_lat(ds, qt)    for ds in datasets]
        art  = [art_lat(ds, qt)   for ds in datasets]
        flood= [flood_lat(ds, qt) for ds in datasets]

        ax.bar(x - w, bs,    w, label="Sorted-Array",   color=C_BS,  hatch=HATCH["BS"],  edgecolor="black", linewidth=0.5)
        ax.bar(x,     art,   w, label="ART",            color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, flood, w, label="Flood (learned)",color=C_FL,  hatch=HATCH["FL"],  edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(d) for d in datasets], rotation=20, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)

        # avg-results annotation under each label
        for i, ds in enumerate(datasets):
            avg = art_avg(ds, qt)
            if avg is not None:
                ax.annotate(f"avg {avg:.0f}", (x[i], ax.get_ylim()[0]),
                            ha="center", va="bottom", fontsize=7, color="#666",
                            xytext=(0, 2), textcoords="offset points")

    axes[0].set_ylabel("Mean range-query latency (µs, log scale)")
    fig.suptitle("Range Query Latency: Single-Hop and Multi-Hop Traversal", y=1.02)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    return save(fig, "fig2_range_query_latency_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Memory Footprint
# ─────────────────────────────────────────────────────────────────────────────
def fig3_memory():
    datasets = [ds for ds in ALL_DS if art_rows.get(("ART", ds, "point"))]
    x = np.arange(len(datasets))
    w = 0.27

    art = [float(art_rows[("ART", ds, "point")]["index_rss_mb"]) for ds in datasets]
    zm  = [float(zm_rows[ds]["index_size_mb"]) if ds in zm_rows else np.nan
           for ds in datasets]
    fl  = [float(flood_rows[(ds, "single_hop")]["index_mb"]) if (ds, "single_hop") in flood_rows else np.nan
           for ds in datasets]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w, art, w, label="ART (RSS)",       color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
    ax.bar(x,     zm,  w, label="ZM-Index",         color=C_ZM,  hatch=HATCH["ZM"],  edgecolor="black", linewidth=0.5)
    ax.bar(x + w, fl,  w, label="Flood",            color=C_FL,  hatch=HATCH["FL"],  edgecolor="black", linewidth=0.5)

    # Raw data reference line: 12 bytes per triple, in MB
    raw_mb = [art_rows[("ART", ds, "point")] and (int(art_rows[("ART", ds, "point")]["N"]) * 12 / 1e6)
              for ds in datasets]
    ax.plot(x, raw_mb, "k--", linewidth=1.0, label="Raw data (12 B / triple)")

    ax.set_yscale("log")
    ax.set_ylabel("Index memory footprint (MB, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(d) for d in datasets], rotation=25, ha="right")
    ax.set_title("Index Memory Consumption")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=False, ncol=4,
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return save(fig, "fig3_memory_footprint_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Build Time
# ─────────────────────────────────────────────────────────────────────────────
def fig4_build():
    datasets = [ds for ds in ALL_DS if art_rows.get(("ART", ds, "point"))]
    x = np.arange(len(datasets))
    w = 0.21

    art = [float(art_rows[("ART", ds, "point")]["build_s"]) for ds in datasets]
    bs  = [max(float(art_rows[("BinSearch", ds, "point")]["build_s"]), 1e-3)
           if art_rows.get(("BinSearch", ds, "point")) else np.nan
           for ds in datasets]
    zm  = [float(zm_rows[ds]["build_time_s"]) if ds in zm_rows else np.nan
           for ds in datasets]
    fl  = [float(flood_rows[(ds, "single_hop")]["build_s"]) if (ds, "single_hop") in flood_rows else np.nan
           for ds in datasets]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - 1.5*w, bs,  w, label="Sorted-Array (sort)", color=C_BS,  hatch=HATCH["BS"],  edgecolor="black", linewidth=0.5)
    ax.bar(x - 0.5*w, art, w, label="ART",                  color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
    ax.bar(x + 0.5*w, zm,  w, label="ZM-Index",             color=C_ZM,  hatch=HATCH["ZM"],  edgecolor="black", linewidth=0.5)
    ax.bar(x + 1.5*w, fl,  w, label="Flood",                color=C_FL,  hatch=HATCH["FL"],  edgecolor="black", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_ylabel("Build time (seconds, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(d) for d in datasets], rotation=25, ha="right")
    ax.set_title("Index Construction Time")
    ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=False, ncol=4,
              bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout()
    return save(fig, "fig4_build_time_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Throughput across query types (1×3 subplots for real datasets)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_throughput():
    qts = ["point", "single_hop", "multi_hop"]
    qt_labels = ["Point", "Single-Hop", "Multi-Hop"]

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.6), sharey=True)
    x = np.arange(len(qts))
    w = 0.27

    for ax, ds in zip(axes, REAL_DS):
        bs   = [bs_qps(ds, qt)    for qt in qts]
        art  = [art_qps(ds, qt)   for qt in qts]
        uni  = [zm_qps(ds) if qt == "point" else flood_qps(ds, qt)
                for qt in qts]

        ax.bar(x - w, bs,  w, label="Sorted-Array", color=C_BS,  hatch=HATCH["BS"],  edgecolor="black", linewidth=0.5)
        ax.bar(x,     art, w, label="ART",          color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w, uni, w, label="Unified",      color=C_ZM,  hatch=HATCH["UNI"], edgecolor="black", linewidth=0.5)

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(qt_labels, rotation=15, ha="right")
        ax.set_title(short_label(ds))
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Throughput (queries / s, log scale)")
    fig.suptitle("Query Throughput Across Query Types and Datasets", y=1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return save(fig, "fig5_throughput_by_query_type")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Specialist routing advantage
# ─────────────────────────────────────────────────────────────────────────────
def fig6_routing():
    """For each real dataset, show ART (generalist) vs Unified (specialist
    routing — ZM for point, Flood for range)."""
    qts = ["point", "single_hop", "multi_hop"]
    qt_labels = ["Point", "Single-Hop", "Multi-Hop"]
    datasets = REAL_DS
    x = np.arange(len(qts))
    w = 0.35

    # Pre-compute global y-range across all panels (we use sharey).
    all_vals = []
    for ds in datasets:
        for qt in qts:
            for v in (art_lat(ds, qt),
                      (zm_lat(ds) if qt == "point" else flood_lat(ds, qt))):
                if v is not None and v > 0:
                    all_vals.append(v)
    y_lo = min(all_vals) * 0.5
    y_hi = max(all_vals) * 8.0    # 1 decade of headroom for annotations

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 4.0), sharey=True)
    for ax, ds in zip(axes, datasets):
        art = [art_lat(ds, qt) for qt in qts]
        uni = [zm_lat(ds) if qt == "point" else flood_lat(ds, qt)
               for qt in qts]
        ax.bar(x - w/2, art, w, label="ART (one structure)",   color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
        ax.bar(x + w/2, uni, w, label="Unified (routed)",      color=C_ZM,  hatch=HATCH["UNI"], edgecolor="black", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks(x)
        ax.set_xticklabels(qt_labels, rotation=15, ha="right")
        ax.set_title(short_label(ds))
        ax.grid(axis="y", which="both", linestyle=":", alpha=0.4)
        # ratio labels
        for i, qt in enumerate(qts):
            if art[i] and uni[i]:
                ratio = max(art[i], uni[i]) / min(art[i], uni[i])
                winner = "U" if uni[i] < art[i] else "A"
                color  = C_ZM if winner == "U" else C_ART
                ax.annotate(f"{winner}: {ratio:.1f}×",
                            (x[i], max(art[i], uni[i])),
                            ha="center", va="bottom", fontsize=8,
                            xytext=(0, 3), textcoords="offset points",
                            color=color, fontweight="bold")

    axes[0].set_ylabel("Mean latency (µs, log scale)")
    fig.suptitle("Specialist Routing vs. Generalist Index: The Unified Model Advantage",
                 y=1.04)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    return save(fig, "fig6_specialist_routing_advantage")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Distribution sensitivity (synthetic data)
# ─────────────────────────────────────────────────────────────────────────────
def fig7_distribution():
    # All 5 synthetic distributions
    datasets = [ds for ds in SYNTH_DS if art_lat(ds, "point") is not None
                                       and zm_lat(ds) is not None]
    x = np.arange(len(datasets))
    w = 0.35

    art = [art_lat(ds, "point") for ds in datasets]
    zm  = [zm_lat(ds)           for ds in datasets]

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(x - w/2, art, w, label="ART (distribution-agnostic)", color=C_ART, hatch=HATCH["ART"], edgecolor="black", linewidth=0.5)
    ax.bar(x + w/2, zm,  w, label="ZM-Index (learned)",          color=C_ZM,  hatch=HATCH["ZM"],  edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Mean point-query latency (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(d) for d in datasets], rotation=20, ha="right")
    ax.set_title("Distribution Sensitivity: Learned Index vs. ART (synthetic data)",
                 pad=12)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="upper center", frameon=False, ncol=2,
              bbox_to_anchor=(0.5, -0.22))

    # Headroom for value annotations
    cur_lo, cur_hi = ax.get_ylim()
    ax.set_ylim(cur_lo, cur_hi * 1.18)

    # Annotate values
    annotate_bars(ax, x - w/2, art, fmt="{:.2f}")
    annotate_bars(ax, x + w/2, zm,  fmt="{:.2f}")

    # Note about N mismatch — below the chart, not inside it.
    fig.text(0.5, 0.005,
             "Note: ZM-Index measured at N = 10 M, ART at N = 60 M",
             ha="center", va="bottom",
             fontsize=8, color="#555", style="italic")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return save(fig, "fig7_distribution_sensitivity")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(f"Output dir: {FIG_DIR}\n")
    figs = [
        ("fig1", "Point query latency: Sorted-Array, ART, ZM-Index across all datasets.", fig1_point),
        ("fig2", "Range query latency (single-hop + multi-hop): Sorted-Array, ART, Flood on the 3 real datasets.", fig2_range),
        ("fig3", "Index memory footprint with raw-data reference line.", fig3_memory),
        ("fig4", "Index construction time: all four index types.", fig4_build),
        ("fig5", "Throughput across query types, one panel per real dataset.", fig5_throughput),
        ("fig6", "Generalist ART vs. specialist routed Unified model, latency per query type.", fig6_routing),
        ("fig7", "Synthetic-distribution sensitivity: ART vs. ZM-Index across uniform/normal/lognormal.", fig7_distribution),
    ]

    print("# Thesis figures generated\n")
    for label, desc, fn in figs:
        try:
            pdf, png = fn()
            print(f"- **{label}** — {desc}")
            print(f"    PDF: {os.path.relpath(pdf, ROOT)}")
            print(f"    PNG: {os.path.relpath(png, ROOT)}")
        except Exception as e:
            print(f"- **{label}** — FAILED: {e}", file=sys.stderr)
            raise


if __name__ == "__main__":
    sys.exit(main())
