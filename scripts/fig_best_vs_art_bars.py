#!/usr/bin/env python3
"""
Thesis figure: grouped bar chart, Stock Flood (best K) vs ART vs FloodSourceSort
(best K) across all 8 datasets (3 SNAP + 5 synthetic), two panels (single/multi hop).
Sources: hop2_sort_dim/k_sensitivity.csv, source_sort_dim/sortsource.csv,
synthetic_all.csv. Saves PDF + SVG (+ PNG preview) to figures/k_sensitivity/.
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
KD   = os.path.join(ROOT, "results", "k_sensitivity")
STOCK_SNAP = os.path.join(KD, "hop2_sort_dim", "k_sensitivity.csv")
FSS_SNAP   = os.path.join(KD, "source_sort_dim", "sortsource.csv")
SYNTH      = os.path.join(KD, "synthetic_all.csv")
FIGDIR = os.path.join(ROOT, "figures", "k_sensitivity")
os.makedirs(FIGDIR, exist_ok=True)

# dataset order + short labels; first 3 are SNAP (have ART), rest synthetic
DATASETS = [
    ("wiki_vote",           "wiki"),
    ("roadnet_ca",          "road"),
    ("web_google",          "web"),
    ("uniform_sparse_60M",  "uni_sp"),
    ("uniform_dense_60M",   "uni_dn"),
    ("uniform_matched_60M", "uni_mt"),
    ("normal_60M",          "norm"),
    ("lognormal_60M",       "logn"),
]
N_SNAP = 3
ART = {
    "wiki_vote":  {"single_hop": 86.81, "multi_hop": 4.07},
    "roadnet_ca": {"single_hop": 0.89,  "multi_hop": 0.70},
    "web_google": {"single_hop": 4.17,  "multi_hop": 1.25},
}

# best[(index, dataset, qt)] = min mean_us over K
best = defaultdict(lambda: float("inf"))
def add(index, ds, qt, mean):
    k = (index, ds, qt)
    if mean < best[k]:
        best[k] = mean

with open(STOCK_SNAP) as f:
    for r in csv.DictReader(f):
        add("stock", r["dataset"], r["query_type"], float(r["mean_us"]))
with open(FSS_SNAP) as f:
    for r in csv.DictReader(f):
        add("sortsource", r["dataset"], r["query_type"], float(r["mean_us"]))
with open(SYNTH) as f:
    for r in csv.DictReader(f):
        add(r["index"], r["dataset"], r["query_type"], float(r["mean_us"]))

def val(index, ds, qt):
    v = best[(index, ds, qt)]
    return None if v == float("inf") else v

# colors / hatches (grayscale-safe + B&W hatching)
C_STOCK = "#888888"; H_STOCK = ""
C_ART   = "#2E75B6"; H_ART   = "///"
C_FSS   = "#2E7D32"; H_FSS   = ""

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
panels = [("single_hop", "(a)  single-hop"), ("multi_hop", "(b)  multi-hop")]
x = np.arange(len(DATASETS))
w = 0.26

for ax, (qt, label) in zip(axes, panels):
    for i, (ds, _) in enumerate(DATASETS):
        s   = val("stock", ds, qt)
        a   = ART.get(ds, {}).get(qt)         # None for synthetic
        fss = val("sortsource", ds, qt)
        if s   is not None:
            ax.bar(i - w, s,   w, color=C_STOCK, hatch=H_STOCK, edgecolor="black", linewidth=0.5, zorder=3)
        if a   is not None:
            ax.bar(i,     a,   w, color=C_ART,   hatch=H_ART,   edgecolor="black", linewidth=0.5, zorder=3)
        if fss is not None:
            ax.bar(i + w, fss, w, color=C_FSS,   hatch=H_FSS,   edgecolor="black", linewidth=0.5, zorder=3)
            # speedup vs stock, annotated above FSS bar
            if s is not None and fss > 0:
                ax.annotate(f"{s/fss:.1f}×", (i + w, fss), ha="center", va="bottom",
                            fontsize=7.5, rotation=90, xytext=(0, 2),
                            textcoords="offset points", color=C_FSS, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylabel("mean latency (µs)")
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in DATASETS])
    ax.grid(True, axis="y", which="both", color="0.5", alpha=0.2, linewidth=0.6)
    # separator between SNAP and synthetic
    ax.axvline(N_SNAP - 0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.6, zorder=1)
    ax.text((N_SNAP - 1) / 2, 0.98, "SNAP (real)", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8, color="0.35")
    ax.text((N_SNAP + len(DATASETS) - 1) / 2, 0.98, "synthetic (60M)",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="0.35")
    ax.text(0.012, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    # headroom for rotated annotations
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 3)

legend = [
    Patch(facecolor=C_STOCK, hatch=H_STOCK, edgecolor="black", label="Stock Flood (best K)"),
    Patch(facecolor=C_ART,   hatch=H_ART,   edgecolor="black", label="ART (SNAP only)"),
    Patch(facecolor=C_FSS,   hatch=H_FSS,   edgecolor="black", label="FloodSourceSort (best K)"),
]
axes[0].legend(handles=legend, fontsize=8, loc="upper right", framealpha=0.95, ncol=1)

fig.tight_layout()
pdf = os.path.join(FIGDIR, "fig_best_vs_art_bars.pdf")
svg = os.path.join(FIGDIR, "fig_best_vs_art_bars.svg")
png = os.path.join(FIGDIR, "fig_best_vs_art_bars.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote:", os.path.relpath(pdf, ROOT), os.path.relpath(svg, ROOT))
