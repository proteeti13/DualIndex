#!/usr/bin/env python3
"""
Thesis figure: Stock Flood (best K) vs FloodSourceSort (best K) on the 5 synthetic
60M datasets. Two panels (single_hop, multi_hop), grouped bars, log-y, speedup
annotated per pair. Source: results/k_sensitivity/synthetic_all.csv.
Saves PDF + SVG (+ PNG preview) to figures/k_sensitivity/.
"""
import csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":  "serif",
    "font.serif":   ["Times New Roman", "DejaVu Serif"],
    "font.size":    10,
    "svg.fonttype": "none",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC  = os.path.join(ROOT, "results", "k_sensitivity", "synthetic_all.csv")
FIGDIR = os.path.join(ROOT, "figures", "k_sensitivity")
os.makedirs(FIGDIR, exist_ok=True)

DATASETS = [
    ("uniform_sparse_60M",  "Uniform\nSparse"),
    ("uniform_dense_60M",   "Uniform\nDense"),
    ("uniform_matched_60M", "Uniform\nMatch"),
    ("normal_60M",          "Normal"),
    ("lognormal_60M",       "Lognormal"),
]

# best[(index, dataset, qt)] = min mean_us over K
best = defaultdict(lambda: float("inf"))
with open(SRC) as f:
    for r in csv.DictReader(f):
        k = (r["index"], r["dataset"], r["query_type"])
        m = float(r["mean_us"])
        if m < best[k]:
            best[k] = m

def val(index, ds, qt):
    v = best[(index, ds, qt)]
    return None if v == float("inf") else v

C_STOCK = "#999999"; H_STOCK = ".."
C_FSS   = "#2E75B6"; H_FSS   = ""

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
panels = [("single_hop", "(a)  single-hop"), ("multi_hop", "(b)  multi-hop")]
x = np.arange(len(DATASETS))
w = 0.36

for ax, (qt, label) in zip(axes, panels):
    for i, (ds, _) in enumerate(DATASETS):
        s   = val("stock", ds, qt)
        fss = val("sortsource", ds, qt)
        if s is not None:
            ax.bar(i - w/2, s,   w, color=C_STOCK, hatch=H_STOCK, edgecolor="black",
                   linewidth=0.5, zorder=3)
        if fss is not None:
            ax.bar(i + w/2, fss, w, color=C_FSS, hatch=H_FSS, edgecolor="black",
                   linewidth=0.5, zorder=3)
        # speedup ratio above the pair
        if s is not None and fss not in (None, 0):
            top = max(s, fss)
            ax.annotate(f"{s/fss:,.0f}×", (i, top), ha="center", va="bottom",
                        fontsize=8, xytext=(0, 3), textcoords="offset points",
                        color=C_FSS, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("mean latency (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DATASETS], fontsize=8)
    ax.grid(True, axis="y", which="both", color="0.5", alpha=0.2, linewidth=0.6)
    ax.text(0.03, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 4)   # headroom for the speedup labels

legend = [
    Patch(facecolor=C_STOCK, hatch=H_STOCK, edgecolor="black", label="Stock Flood (best K)"),
    Patch(facecolor=C_FSS,   hatch=H_FSS,   edgecolor="black", label="FloodSourceSort (best K)"),
]
axes[0].legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.95)

fig.tight_layout()
pdf = os.path.join(FIGDIR, "fig_synthetic_bars.pdf")
svg = os.path.join(FIGDIR, "fig_synthetic_bars.svg")
png = os.path.join(FIGDIR, "fig_synthetic_bars.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote:", os.path.relpath(pdf, ROOT), os.path.relpath(svg, ROOT))
