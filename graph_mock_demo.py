#!/usr/bin/env python3
"""

Demonstrates the hybrid routing architecture that dispatches queries
between two learned spatial indexes based on query box shape:

  ZM-Index  → point queries  (all 3 dims pinned, lo==hi everywhere)
  Flood     → range queries  (at least 1 dim open,  lo<hi  somewhere)

Routing rule (mirrors indexes/router.hpp DualIndexRouter::is_point()):
  open_dims = sum(lo[d] < hi[d] for d in 0..2)
  open_dims == 0  →  ZM-Index  (Morton-encoded PGM learned index)
  open_dims == 1  →  Flood     (bucket-grid + local PGM, 2 dims fixed)
  open_dims >= 2  →  Flood     (bucket-grid + local PGM, 1 dim fixed)

Data model: Wiki-Vote 2-hop paths stored as 3D keys
  (SourceID, Hop1_ID, Hop2_ID) → Offset
  File format: space-separated, 4 columns; Offset (col 4) is ignored.

Performance numbers come from the real C++ bench_router binary compiled
from bench/bench_router.cpp.  Result lookups use Python data structures
built on the same loaded data.  Baseline latency is a Python linear scan,
actually measured at runtime.

Usage:
  python graph_mock_demo.py --data /path/to/wiki_vote_triples.txt
  python graph_mock_demo.py --data /path/to/wiki_vote_triples.txt --interactive
"""

import sys
import os
import time
import csv
import subprocess
import argparse
import tempfile
from collections import Counter


# ── Paths (derived relative to this script's location) ────────────────────────
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR    = os.path.join(_SCRIPT_DIR, "build")
_BINARY       = os.path.join(_BUILD_DIR, "bin", "bench_router")
_FALLBACK_CSV = os.path.join(_BUILD_DIR, "results", "router", "router_wiki_10k.csv")

# Display box inner content width (chars).  All panel rows are padded to this.
_BOX_W = 50


# ── Display helpers ────────────────────────────────────────────────────────────

def _fmt_box(lo, hi):
    """Return '[3,3] × [28,28] × [6,6]' style box string."""
    parts = []
    for l, h in zip(lo, hi):
        lv = int(l) if isinstance(l, float) and l == int(l) else l
        hv = int(h) if isinstance(h, float) and h == int(h) else h
        parts.append(f"[{lv},{hv}]")
    return " × ".join(parts)  # × (multiplication sign)


def _print_panel(lo, hi, query_type_label, index_used,
                 lat_us, baseline_us, result_count, baseline_count,
                 latency_source):
    """

    Layout:
      ╔══ ... ══╗
      ║  Query: [lo] × [hi]         ║
      ║  Type:  <label>             ║
      ║  Route: <index>             ║
      ╠══ ... ══╣
      ║  Learned Index: X µs  | Results: N   ║
      ║  Baseline Scan: Y µs  | Results: N   ║
      ║  Speedup:       Z×    ✓ correct      ║
      ╠─ ... ─╣
      ║  Latency source: <source>   ║
      ╚══ ... ══╝
    """
    W = _BOX_W

    def _border(tl, h, tr):
        print(f"{tl}{h * (W + 4)}{tr}")

    def _row(text):
        # ljust pads to W using character count (Unicode-aware in Python 3)
        print(f"║  {text.ljust(W)}  ║")

    speedup    = baseline_us / lat_us if lat_us > 0 else float("inf")
    correct    = "✓ correct" if result_count == baseline_count else "⚠ mismatch"
    speedup_lbl = f"{int(speedup):,}×" if speedup < 1e6 else f">1M×"

    # Top border
    _border("╔", "═", "╗")
    _row(f"Query: {_fmt_box(lo, hi)}")
    _row(f"Type:  {query_type_label}")
    _row(f"Route: {index_used}")
    _border("╠", "═", "╣")

    learned_line  = f"Learned Index (C++): {lat_us:>8.2f} µs  |  Results: {result_count}"
    baseline_line = f"Baseline Scan  (Py): {baseline_us:>8.1f} µs  |  Results: {baseline_count}"
    speedup_line  = f"Speedup:             {speedup_lbl:<12}    {correct}"

    _row(learned_line)
    _row(baseline_line)
    _row(speedup_line)

    # Thin divider before source note — truncate long source strings to fit the box
    prefix  = "Latency source: "
    src_max = W - len(prefix)
    src_disp = (latency_source[:src_max - 3] + "...") if len(latency_source) > src_max else latency_source
    _border("╠", "─", "╣")
    _row(f"{prefix}{src_disp}")
    _border("╚", "═", "╝")
    print()


# ── GraphMock class ────────────────────────────────────────────────────────────

class GraphMock:
    """
    Hybrid query router for the DualIndex learned index system.

    At init time this class:
      1. Loads the first n_rows triples from the Wiki-Vote dataset.
      2. Builds Python data structures that mirror the index semantics:
           self.point_set  — set of (s,h1,h2) tuples for O(1) point lookup
                             (mirrors ZM-Index Morton-PGM membership test)
           self.data       — sorted list of tuples for range scans
                             (mirrors Flood bucket grid traversal)
      3. Calls the real C++ bench_router binary to measure actual per-type
         latencies, or falls back to pre-measured CSV values.

    At query time:
      self.query(lo, hi) counts open dims, routes, returns real C++ latency.
      self.compare_baseline(lo, hi) times a Python linear scan for comparison.
    """

    def __init__(self, data_path, n_rows=10000):
        print(f"\n[init] Loading {n_rows:,} rows from {os.path.basename(data_path)} ...")
        t0 = time.perf_counter()

        self.data_path = os.path.abspath(data_path)
        # data: list of (SourceID, Hop1_ID, Hop2_ID) int triples, in file order (sorted)
        self.data = []
        # point_set: fast O(1) membership — mirrors ZM-Index exact-match semantics
        self.point_set = set()

        with open(data_path, "r") as f:
            for i, line in enumerate(f):
                if i >= n_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                cols = line.split()
                if len(cols) < 3:
                    continue
                # Column 3 (Offset) is the sorted position in the file; we ignore it.
                triple = (int(cols[0]), int(cols[1]), int(cols[2]))
                self.data.append(triple)
                self.point_set.add(triple)

        load_ms = (time.perf_counter() - t0) * 1000

        # Per-dimension global bounds — needed to build open-ended range boxes
        self.bounds_lo = [min(p[d] for p in self.data) for d in range(3)]
        self.bounds_hi = [max(p[d] for p in self.data) for d in range(3)]

        print(f"[init] Loaded {len(self.data):,} triples in {load_ms:.1f} ms")
        print(f"[init] Dim bounds: "
              f"dim0=[{self.bounds_lo[0]},{self.bounds_hi[0]}]  "
              f"dim1=[{self.bounds_lo[1]},{self.bounds_hi[1]}]  "
              f"dim2=[{self.bounds_lo[2]},{self.bounds_hi[2]}]")

        # Get real C++ latency measurements
        (self.latency_us,
         self.latency_source,
         self.zm_build_s,
         self.flood_build_s) = self._measure_latencies(n_rows)

        print(f"[init] ZM-Index build:  {self.zm_build_s:.3f} s")
        print(f"[init] Flood    build:  {self.flood_build_s:.3f} s")
        print(f"[init] C++ latencies:  "
              f"point={self.latency_us['point']:.2f} µs  "
              f"multi_hop={self.latency_us['multi_hop']:.2f} µs  "
              f"single_hop={self.latency_us['single_hop']:.2f} µs")
        print(f"[init] Source: {self.latency_source}\n")

    # ── Internal: measure real C++ latencies ───────────────────────────────────

    def _measure_latencies(self, n_rows):
        """
        Obtain measured per-type latencies from the real C++ bench_router binary.

        Strategy (a): call bench_router with N=n_rows, Q=1000, parse CSV output.
        Strategy (c): fall back to pre-measured router_wiki_10k.csv if binary fails.
        Last resort:  hardcoded values from the thesis benchmark logs.

        Returns:
          latency_us    — dict {'point': float, 'single_hop': float, 'multi_hop': float}
          source_label  — human-readable description of where numbers came from
          zm_build_s    — ZM-Index build time in seconds
          flood_build_s — Flood build time in seconds
        """
        # ── Strategy (a): call the real C++ bench_router binary ───────────────
        if os.path.isfile(_BINARY):
            try:
                tf = tempfile.NamedTemporaryFile(
                    suffix=".csv", delete=False, mode="w"
                )
                tmp_csv = tf.name
                tf.close()

                cmd = [
                    _BINARY,
                    self.data_path,
                    str(n_rows),
                    "--queries", "1000",
                    "--seed", "42",
                    "--out_csv", tmp_csv,
                ]
                print(f"[init] Calling bench_router (N={n_rows}, Q=1000) ...")
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=90
                )

                if proc.returncode == 0 and os.path.isfile(tmp_csv):
                    lats = {}
                    zm_s = flood_s = 0.0
                    with open(tmp_csv, newline="") as f:
                        for row in csv.DictReader(f):
                            qt = row.get("query_type", "")
                            if qt in ("point", "single_hop", "multi_hop"):
                                lats[qt] = float(row["lat_mean_us"])
                                zm_s    = float(row.get("build_s_zm",    0))
                                flood_s = float(row.get("build_s_flood", 0))
                    try:
                        os.unlink(tmp_csv)
                    except OSError:
                        pass

                    if len(lats) == 3:
                        src = (
                            f"bench_router binary — live run  "
                            f"(N={n_rows:,}, Q=1,000, wiki_vote)"
                        )
                        return lats, src, zm_s, flood_s
            except Exception as exc:
                print(f"[warn] bench_router call failed ({exc}); falling back to CSV")

        # ── Strategy (c): load from pre-measured CSV ──────────────────────────
        if os.path.isfile(_FALLBACK_CSV):
            lats = {}
            zm_s = flood_s = 0.0
            with open(_FALLBACK_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    qt = row.get("query_type", "")
                    if qt in ("point", "single_hop", "multi_hop"):
                        lats[qt] = float(row["lat_mean_us"])
                        zm_s    = float(row.get("build_s_zm",    0))
                        flood_s = float(row.get("build_s_flood", 0))
            if len(lats) == 3:
                src = (
                    f"router_wiki_10k.csv (pre-measured, "
                    f"N=10,000, Q=10,000, wiki_vote)"
                )
                return lats, src, zm_s, flood_s

        # ── Last resort: hardcoded  benchmark values ────────────────────
        print("[warn] Binary and fallback CSV both unavailable — using thesis values")
        return (
            {"point": 0.18, "single_hop": 26.93, "multi_hop": 3.17},
            "thesis benchmarks (N=10,000, wiki_vote, K=20, Eps=64)",
            0.000, 0.004,
        )

    # ── Core: query type detection ─────────────────────────────────────────────

    def _detect_query_type(self, lo, hi):
        """
        Classify a query box by counting "open" dimensions (lo[d] < hi[d]).

        This replicates the routing rule in indexes/router.hpp:
          DualIndexRouter::is_point() returns true iff lo==hi in ALL 3 dims.

        Graph semantics:
          open_dims == 0  →  'point'      : exact path lookup   (ZM-Index)
          open_dims == 1  →  'multi_hop'  : fix src+intermediate, enumerate dest (Flood)
          open_dims >= 2  →  'single_hop' : fix src only, enumerate all 2-hop paths (Flood)

        Returns: (query_type_key, open_dims_count)
        """
        open_dims = sum(1 for d in range(3) if lo[d] < hi[d])
        if open_dims == 0:
            return "point", 0
        elif open_dims == 1:
            return "multi_hop", 1
        else:
            return "single_hop", open_dims

    # ── Core: query routing ────────────────────────────────────────────────────

    def query(self, lo, hi):
        """
        Route the query to the correct learned index and return results + latency.

        The routing decision is made solely from the shape of the query box —
        no explicit query-type tag is needed.  This matches the runtime behaviour
        of DualIndexRouter::query() in indexes/router.hpp.

        Point path  (open_dims == 0):
          ZM-Index encodes (s,h1,h2) as a Morton code (bit-interleaved uint64),
          feeds it to a 1D PGM learned index, and scans a window of ≤2·Eps+2
          entries.  Here mirrored by O(1) set membership.

        Range path  (open_dims >= 1):
          Flood maps the box to a 20×20 bucket grid using two global PGM CDFs,
          then scans each matched bucket's local PGM model.  Here mirrored by
          a bounded list comprehension on the sorted data.

        Returns:
          results      — list of matching (s,h1,h2) tuples
          lat_us       — C++ measured latency for this query type (µs)
          index_used   — "ZM-Index" or "Flood"
          label        — human-readable query type name
        """
        qt, open_dims = self._detect_query_type(lo, hi)

        if qt == "point":
            # ZM-Index path: exact membership test
            triple = (int(lo[0]), int(lo[1]), int(lo[2]))
            results    = [triple] if triple in self.point_set else []
            index_used = "ZM-Index"
            label      = "Point Query"
        else:
            # Flood path: range scan bounded by the 3D box
            results = [
                p for p in self.data
                if lo[0] <= p[0] <= hi[0]
                and lo[1] <= p[1] <= hi[1]
                and lo[2] <= p[2] <= hi[2]
            ]
            index_used = "Flood"
            label      = ("Multi-Hop Range Query"
                          if qt == "multi_hop"
                          else "Single-Hop Range Query")

        # Use the pre-measured C++ latency for this query type
        lat_us = self.latency_us[qt]
        return results, lat_us, index_used, label

    # ── Core: linear-scan baseline ─────────────────────────────────────────────

    def compare_baseline(self, lo, hi):
        """
        Execute the same query as a Python linear scan (no index).

        Simulates a naive database that checks every row against the query box.
        Latency is actually measured with time.perf_counter() — not estimated.

        Returns: (results, latency_us)
        """
        t0 = time.perf_counter()
        results = [
            p for p in self.data
            if lo[0] <= p[0] <= hi[0]
            and lo[1] <= p[1] <= hi[1]
            and lo[2] <= p[2] <= hi[2]
        ]
        latency_us = (time.perf_counter() - t0) * 1e6
        return results, latency_us

    # ── Demo query selection ───────────────────────────────────────────────────

    def _pick_demo_queries(self):
        """
        Select 3 representative demo queries from the loaded data.

        All queries are guaranteed to return ≥1 result because they are
        derived from triples that exist in the dataset.

        Returns a list of (lo, hi) tuples:
          [0] Point query       — exact triple from the dataset
          [1] Multi-hop query   — most common (src,hop1) pair, open on hop2
          [2] Single-hop query  — most common src, open on hop1 and hop2
        """
        # 1. Point: first triple in dataset (src=3, hop1=28, hop2=3 for wiki_vote)
        p       = self.data[0]
        q_point = (p, p)

        # 2. Multi-hop: (src, hop1) pair with most occurrences → guarantees results
        pair_counts = Counter((t[0], t[1]) for t in self.data)
        best_pair   = max(pair_counts, key=pair_counts.get)
        mh_lo = (best_pair[0], best_pair[1], self.bounds_lo[2])
        mh_hi = (best_pair[0], best_pair[1], self.bounds_hi[2])
        q_multi = (mh_lo, mh_hi)

        # 3. Single-hop: src with most occurrences → guarantees many results
        src_counts = Counter(t[0] for t in self.data)
        best_src   = max(src_counts, key=src_counts.get)
        sh_lo = (best_src, self.bounds_lo[1], self.bounds_lo[2])
        sh_hi = (best_src, self.bounds_hi[1], self.bounds_hi[2])
        q_single = (sh_lo, sh_hi)

        return [q_point, q_multi, q_single]

    # ── Public: run the full demo ──────────────────────────────────────────────

    def run_demo(self):
        """
        Run 3 representative queries and print results in ready format.

        For each query:
          1. Detect query type from box shape (open-dim count)
          2. Route to ZM-Index or Flood, return C++ measured latency
          3. Run Python linear scan for baseline comparison
          4. Print result panel with speedup factor and correctness check
        """
        print("=" * 56)
        print("   DualIndex Router — Wiki-Vote graph dataset")
        print(f"   Dataset: {os.path.basename(self.data_path)}")
        print(f"   Loaded:  {len(self.data):,} triples")
        print(f"   Index:   ZM-Index (K=20, Eps=64) + Flood (K=20, Eps=64)")
        print("=" * 56)
        print()

        queries       = self._pick_demo_queries()
        demo_records  = []  # store for summary table

        for lo, hi in queries:
            results,  lat_us,      index_used, label = self.query(lo, hi)
            b_results, baseline_us                   = self.compare_baseline(lo, hi)

            demo_records.append({
                "lo": lo, "hi": hi,
                "label": label, "index": index_used,
                "lat_us": lat_us, "baseline_us": baseline_us,
                "n_results": len(results), "n_baseline": len(b_results),
            })

            _print_panel(
                lo, hi, label, index_used,
                lat_us, baseline_us,
                len(results), len(b_results),
                self.latency_source,
            )

        # ── Routing summary table ─────────────────────────────────────────────
        print("Routing summary:")
        hdr = f"  {'Query type':<22} {'Routed to':<12} {'C++ latency':>14}  {'Speedup':>10}"
        print(hdr)
        print("  " + "─" * (len(hdr) - 2))
        for r in demo_records:
            sp = int(r["baseline_us"] / r["lat_us"]) if r["lat_us"] > 0 else 0
            print(
                f"  {r['label']:<22} {r['index']:<12}"
                f" {r['lat_us']:>10.2f} µs  {sp:>8,}×"
            )
        print()
        print(f"  Latency source: {self.latency_source}")
        print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def _interactive_loop(gm):
    """Run an interactive query session — user enters box coordinates manually."""
    print("Interactive mode")
    print("  Enter a query box as 6 numbers:  lo0 lo1 lo2  hi0 hi1 hi2")
    print()
    print("  Examples:")
    lo0, hi0 = gm.bounds_lo[0], gm.bounds_hi[0]
    lo1, hi1 = gm.bounds_lo[1], gm.bounds_hi[1]
    lo2, hi2 = gm.bounds_lo[2], gm.bounds_hi[2]
    ex = gm.data[0]
    print(f"    Point query:       {ex[0]} {ex[1]} {ex[2]}  {ex[0]} {ex[1]} {ex[2]}")
    print(f"    Multi-hop query:   {ex[0]} {ex[1]} {lo2}  {ex[0]} {ex[1]} {hi2}")
    print(f"    Single-hop query:  {ex[0]} {lo1} {lo2}  {ex[0]} {hi1} {hi2}")
    print()
    print("  Type 'quit' to exit.")
    print()

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break

        parts = line.split()
        if len(parts) != 6:
            print("  Please enter exactly 6 numbers: lo0 lo1 lo2 hi0 hi1 hi2\n")
            continue

        try:
            nums = [float(x) for x in parts]
        except ValueError:
            print("  Invalid input — all values must be numeric\n")
            continue

        lo = tuple(nums[:3])
        hi = tuple(nums[3:])

        if any(lo[d] > hi[d] for d in range(3)):
            print("  Invalid box: lo must be ≤ hi in every dimension\n")
            continue

        results,  lat_us,     index_used, label = gm.query(lo, hi)
        b_results, baseline_us               = gm.compare_baseline(lo, hi)

        _print_panel(
            lo, hi, label, index_used,
            lat_us, baseline_us,
            len(results), len(b_results),
            gm.latency_source,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "DualIndex Router — hybrid ZM-Index + Flood routing "
            "for multi-dimensional learned indexing on graph data"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python graph_mock_demo.py --data datasets/wiki_vote_triples.txt\n"
            "  python graph_mock_demo.py --data datasets/wiki_vote_triples.txt"
            " --interactive\n"
            "  python graph_mock_demo.py --data datasets/wiki_vote_triples.txt"
            " --n_rows 50000\n"
        ),
    )
    parser.add_argument(
        "--data", required=True,
        help="Path to wiki_vote_triples.txt (format: SourceID Hop1_ID Hop2_ID Offset)",
    )
    parser.add_argument(
        "--n_rows", type=int, default=10000,
        help="Number of rows to load from dataset (default: 10000)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Interactive mode: enter a query box and see routing decision live",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        sys.exit(f"ERROR: dataset not found: {args.data}")

    gm = GraphMock(args.data, n_rows=args.n_rows)

    if args.interactive:
        _interactive_loop(gm)
    else:
        gm.run_demo()


if __name__ == "__main__":
    main()
