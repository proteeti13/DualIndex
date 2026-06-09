#!/usr/bin/env python3
"""
Thesis figure: stock Flood mean latency vs grid resolution K, with ART baselines.
Two panels (single_hop, multi_hop), 3 SNAP datasets, log-log axes.
Reads results/k_sensitivity/hop2_sort_dim/k_sensitivity.csv.
Saves PDF + SVG to figures/k_sensitivity/.
"""
import csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    10,
    "svg.fonttype": "none",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "results", "k_sensitivity", "hop2_sort_dim", "k_sensitivity.csv")
FIGDIR = os.path.join(ROOT, "figures", "k_sensitivity")
os.makedirs(FIGDIR, exist_ok=True)

# dataset -> (display name, color, marker)
DATASETS = {
    "wiki_vote":  ("wiki_vote",  "black",    "o"),
    "roadnet_ca": ("roadnet_ca", "#2E75B6",  "s"),
    "web_google": ("web_google", "#888888",  "^"),
}
ART = {
    "wiki_vote":  {"single_hop": 86.81, "multi_hop": 4.07},
    "roadnet_ca": {"single_hop": 0.89,  "multi_hop": 0.70},
    "web_google": {"single_hop": 4.17,  "multi_hop": 1.25},
}

# data[query_type][dataset] = list of (K, mean_us), sorted by K
data = defaultdict(lambda: defaultdict(list))
with open(SRC) as f:
    for row in csv.DictReader(f):
        data[row["query_type"]][row["dataset"]].append(
            (int(row["K"]), float(row["mean_us"])))
for qt in data:
    for ds in data[qt]:
        data[qt][ds].sort()

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
panels = [("single_hop", "(a)"), ("multi_hop", "(b)")]

for ax, (qt, label) in zip(axes, panels):
    for ds, (name, color, marker) in DATASETS.items():
        pts = data[qt].get(ds, [])
        if pts:
            ks = [k for k, _ in pts]
            ys = [y for _, y in pts]
            ax.plot(ks, ys, color=color, marker=marker, markersize=6,
                    markerfacecolor=color, linewidth=1.3, label=name, zorder=3)
        # ART baseline: horizontal dashed, same color
        ax.axhline(ART[ds][qt], color=color, linestyle="--", linewidth=1.1,
                   alpha=0.9, zorder=2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid resolution $K$")
    ax.set_xticks([2, 3, 4, 6, 10, 15, 20])
    ax.set_xticklabels(["2", "3", "4", "6", "10", "15", "20"])
    ax.minorticks_off()
    ax.grid(True, which="both", color="0.5", alpha=0.2, linewidth=0.6)
    # panel label inside, top-left
    ax.text(0.04, 0.95, label, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold")

axes[0].set_ylabel("mean latency (µs)")

# single legend: dataset entries (solid+marker) + one "ART baseline" entry (dashed)
handles = [Line2D([], [], color=c, marker=m, linestyle="-", markersize=6, label=n)
           for n, c, m in DATASETS.values()]
handles.append(Line2D([], [], color="black", linestyle="--", label="ART baseline"))
axes[0].legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)

fig.tight_layout()
pdf = os.path.join(FIGDIR, "fig_stock_k_vs_art.pdf")
svg = os.path.join(FIGDIR, "fig_stock_k_vs_art.svg")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(os.path.join(FIGDIR, "fig_stock_k_vs_art.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote:")
print(" ", os.path.relpath(pdf, ROOT))
print(" ", os.path.relpath(svg, ROOT))
