#!/usr/bin/env python3
"""
Thesis figure: the K-sensitivity inversion. Stock Flood latency falls as K rises;
FloodSourceSort latency rises as K rises — the curves cross. Two panels:
(a) web_google (SNAP), (b) uniform_sparse_60M (synthetic). Log-log, 4 lines each.
Sources: hop2_sort_dim/k_sensitivity.csv, source_sort_dim/sortsource.csv, synthetic_all.csv.
Saves PDF + SVG (+ PNG preview) to figures/k_sensitivity/.
"""
import csv, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# series[(index, dataset, qt)] = {K: mean_us}
series = defaultdict(dict)
def add(index, ds, qt, K, mean):
    series[(index, ds, qt)][K] = mean

with open(STOCK_SNAP) as f:
    for r in csv.DictReader(f):
        add("stock", r["dataset"], r["query_type"], int(r["K"]), float(r["mean_us"]))
with open(FSS_SNAP) as f:
    for r in csv.DictReader(f):
        add("sortsource", r["dataset"], r["query_type"], int(r["K"]), float(r["mean_us"]))
with open(SYNTH) as f:
    for r in csv.DictReader(f):
        add(r["index"], r["dataset"], r["query_type"], int(r["K"]), float(r["mean_us"]))

def xy(index, ds, qt):
    d = series.get((index, ds, qt), {})
    ks = sorted(d)
    return ks, [d[k] for k in ks]

# colors: stock = gray tones, sortsource = blue/dark tones
C_STOCK_S = "#999999"   # stock single-hop
C_STOCK_M = "#555555"   # stock multi-hop
C_FSS_S   = "#2E75B6"   # sortsource single-hop
C_FSS_M   = "#0B3D66"   # sortsource multi-hop

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
panels = [("web_google", "(a)  web_google"),
          ("uniform_sparse_60M", "(b)  uniform_sparse_60M")]

for ax, (ds, label) in zip(axes, panels):
    # stock = dashed, sortsource = solid; single=circle, multi=square
    spec = [
        ("stock",      "single_hop", C_STOCK_S, "--", "o", "Stock single-hop"),
        ("stock",      "multi_hop",  C_STOCK_M, "--", "s", "Stock multi-hop"),
        ("sortsource", "single_hop", C_FSS_S,   "-",  "o", "SourceSort single-hop"),
        ("sortsource", "multi_hop",  C_FSS_M,   "-",  "s", "SourceSort multi-hop"),
    ]
    for index, qt, color, ls, mk, lab in spec:
        ks, ys = xy(index, ds, qt)
        if ks:
            ax.plot(ks, ys, color=color, linestyle=ls, marker=mk, markersize=6,
                    markerfacecolor=color, linewidth=1.3, label=lab, zorder=3)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("grid resolution $K$")
    ax.set_xticks([2, 3, 4, 6, 8, 10, 15, 20])
    ax.set_xticklabels(["2", "3", "4", "6", "8", "10", "15", "20"])
    ax.minorticks_off()
    ax.grid(True, which="both", color="0.5", alpha=0.2, linewidth=0.6)
    ax.text(0.04, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")

axes[0].set_ylabel("mean latency (µs)")
axes[0].legend(fontsize=7.5, loc="center left", framealpha=0.95)

fig.tight_layout()
pdf = os.path.join(FIGDIR, "fig_k_inversion.pdf")
svg = os.path.join(FIGDIR, "fig_k_inversion.svg")
png = os.path.join(FIGDIR, "fig_k_inversion.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(svg, bbox_inches="tight")
fig.savefig(png, dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote:", os.path.relpath(pdf, ROOT), os.path.relpath(svg, ROOT))
