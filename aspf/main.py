"""Demo entry point for ASPF.

Run as:
    python -m aspf.main
"""

from __future__ import annotations

import math
import random

from aspf.adapters import AdjListAdapter
from aspf.analyzer import GraphAnalyzer
from aspf.benchmark import BenchmarkRunner
from aspf.selector import StrategySelector


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------


def _build_unweighted_graph(n: int = 20) -> AdjListAdapter:
    """Sparse unweighted graph with metadata flag."""
    rng = random.Random(0)
    adj: dict = {i: [] for i in range(n)}
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.15:
                adj[u].append((v, 1))
                adj[v].append((u, 1))
    return AdjListAdapter(adj, meta={"unweighted": True})


def _build_dense_weighted_graph(n: int = 50) -> AdjListAdapter:
    """Dense graph with all-pairs random weights."""
    rng = random.Random(1)
    adj: dict = {i: [] for i in range(n)}
    for u in range(n):
        for v in range(n):
            if u != v:
                adj[u].append((v, rng.uniform(1.0, 100.0)))
    return AdjListAdapter(adj)


def _build_sparse_weighted_graph(n: int = 200) -> AdjListAdapter:
    """Sparse graph with large (non-integer) weights."""
    rng = random.Random(2)
    adj: dict = {i: [] for i in range(n)}
    for u in range(n):
        for _ in range(3):
            v = rng.randint(0, n - 1)
            if v != u:
                adj[u].append((v, rng.uniform(1.0, 1_000.0)))
    return AdjListAdapter(adj)


def _build_small_integer_graph(n: int = 100) -> AdjListAdapter:
    """Graph with small integer weights (good candidate for DIAL)."""
    rng = random.Random(3)
    adj: dict = {i: [] for i in range(n)}
    for u in range(n):
        for _ in range(5):
            v = rng.randint(0, n - 1)
            if v != u:
                adj[u].append((v, rng.randint(1, 10)))
    return AdjListAdapter(adj)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    graphs = [
        ("unweighted (BFS candidate)", _build_unweighted_graph(), 0),
        ("dense weighted (ArrayDijkstra candidate)", _build_dense_weighted_graph(), 0),
        ("sparse weighted (HeapDijkstra candidate)", _build_sparse_weighted_graph(), 0),
        ("small-integer weighted (DIAL candidate)", _build_small_integer_graph(), 0),
    ]

    runner = BenchmarkRunner(
        total_runs=10,
        warmup_runs=2,
        analyzer_seed=42,
    )

    print("=" * 70)
    print("Adaptive Shortest-Path Framework — Demo")
    print("=" * 70)

    for label, graph, source in graphs:
        print(f"\n[Graph] {label}")
        result = runner.run(graph, source)
        strategy = result["strategy"]
        features = result["features"]
        distances = result["distances"]
        median_ns = result["median_ns"]
        ci_lo, ci_hi = result["ci_95"]

        print(f"  Vertices      : {features['vertex_count']}")
        print(f"  Edges         : {features['edge_count']}")
        print(f"  Density       : {features['density']:.4f}")
        print(f"  Strategy      : {strategy}")
        print(f"  Median timing : {median_ns/1e6:.4f} ms")
        print(f"  95% CI (ms)   : [{ci_lo/1e6:.4f}, {ci_hi/1e6:.4f}]")

        # Show first 5 distances (sorted by vertex id).
        sample = sorted(
            ((v, d) for v, d in distances.items() if d < math.inf),
            key=lambda x: x[0],
        )[:5]
        print("  Distances (first 5 reachable):", sample)

    # Comparison mode on the sparse graph.
    print("\n" + "=" * 70)
    print("Comparison mode (sparse weighted graph)")
    print("=" * 70)
    _, sparse_graph, source = graphs[2]
    comp_result = runner.run(sparse_graph, source, compare_all=True)
    print(f"  Adaptive strategy : {comp_result['strategy']}")
    print(f"  All distances match: {comp_result['all_distances_match']}")
    if comp_result["mismatched_strategies"]:
        print(f"  Mismatched: {comp_result['mismatched_strategies']}")
    for strat, info in comp_result["comparison"].items():
        print(
            f"  {strat:20s}  median={info['median_ns']/1e6:.4f} ms  "
            f"match={info['matches_reference']}"
        )


if __name__ == "__main__":
    main()
