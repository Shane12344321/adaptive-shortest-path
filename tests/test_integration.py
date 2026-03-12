"""Integration tests for the full ASPF pipeline."""

from __future__ import annotations

import math

import pytest

from aspf.adapters import AdjListAdapter
from aspf.analyzer import BFS, DIAL, HEAP_DIJKSTRA, GraphAnalyzer
from aspf.benchmark import BenchmarkRunner
from aspf.selector import StrategySelector
from aspf.solvers import HeapDijkstraSolver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_pipeline(graph, source, config_path=None, seed=0):
    features = GraphAnalyzer(seed=seed).analyse(graph)
    strategy = StrategySelector(config_path).select(features)
    return strategy, features


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_unweighted_graph_selects_bfs(self):
        adj = {i: [(i + 1, 1)] for i in range(9)}
        adj[9] = []
        g = AdjListAdapter(adj, meta={"unweighted": True})
        strategy, features = _run_pipeline(g, 0)
        assert strategy == BFS
        assert features["is_unit_weight"] is True

    def test_unweighted_graph_correct_distances(self):
        adj = {i: [(i + 1, 1)] for i in range(9)}
        adj[9] = []
        g = AdjListAdapter(adj, meta={"unweighted": True})
        features = GraphAnalyzer(seed=0).analyse(g)
        from aspf.benchmark import _solver_for_strategy
        solver = _solver_for_strategy(BFS, features)
        dist = solver.solve(g, 0)
        assert dist[0] == 0
        assert dist[5] == 5
        assert dist[9] == 9

    def test_small_integer_graph_selects_dial(self):
        import random
        rng = random.Random(99)
        n = 50
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(4):
                v = rng.randint(0, n - 1)
                if v != u:
                    adj[u].append((v, rng.randint(1, 10)))
        g = AdjListAdapter(adj)
        strategy, _ = _run_pipeline(g, 0, seed=0)
        assert strategy == DIAL

    def test_small_integer_graph_dial_matches_heap_dijkstra(self):
        import random
        rng = random.Random(99)
        n = 50
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(4):
                v = rng.randint(0, n - 1)
                if v != u:
                    adj[u].append((v, rng.randint(1, 10)))
        g = AdjListAdapter(adj)
        features = GraphAnalyzer(seed=0).analyse(g)
        from aspf.benchmark import _solver_for_strategy, _dists_equal
        dial_dist = _solver_for_strategy(DIAL, features).solve(g, 0)
        heap_dist = HeapDijkstraSolver().solve(g, 0)
        assert _dists_equal(dial_dist, heap_dist)

    def test_sparse_large_weight_graph_selects_heap(self):
        import random
        rng = random.Random(55)
        n = 100
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(3):
                v = rng.randint(0, n - 1)
                if v != u:
                    adj[u].append((v, rng.uniform(1000.0, 100_000.0)))
        g = AdjListAdapter(adj)
        strategy, _ = _run_pipeline(g, 0, seed=0)
        assert strategy == HEAP_DIJKSTRA

    def test_sparse_large_weight_correct_distances(self):
        import random
        rng = random.Random(55)
        n = 100
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for _ in range(3):
                v = rng.randint(0, n - 1)
                if v != u:
                    adj[u].append((v, rng.uniform(1000.0, 100_000.0)))
        g = AdjListAdapter(adj)
        dist = HeapDijkstraSolver().solve(g, 0)
        assert dist[0] == 0.0
        for v, d in dist.items():
            assert d >= 0 or d == math.inf

    def test_disconnected_graph_unreachable_vertices_are_inf(self):
        adj = {
            0: [(1, 1)],
            1: [],
            2: [(3, 1)],  # component disconnected from 0
            3: [],
        }
        g = AdjListAdapter(adj)
        dist = HeapDijkstraSolver().solve(g, 0)
        assert dist[0] == 0.0
        assert dist[1] == 1.0
        assert dist[2] == math.inf
        assert dist[3] == math.inf

    def test_comparison_mode_all_distances_match_unweighted(self):
        adj = {i: [(i + 1, 1)] for i in range(9)}
        adj[9] = []
        g = AdjListAdapter(adj, meta={"unweighted": True})
        runner = BenchmarkRunner(total_runs=4, warmup_runs=1, analyzer_seed=0)
        result = runner.run(g, 0, compare_all=True)
        assert result["all_distances_match"] is True


class TestNetworkXIntegration:
    def test_networkx_pipeline(self):
        nx = pytest.importorskip("networkx")
        from aspf.adapters import NetworkXAdapter

        G = nx.DiGraph()
        for i in range(5):
            G.add_edge(i, (i + 1) % 5, weight=2)
        g = NetworkXAdapter(G)

        features = GraphAnalyzer(seed=0).analyse(g)
        strategy = StrategySelector().select(features)
        # Should be DIAL (small integer weights: max_weight=2, is_integer=True)
        from aspf.benchmark import _solver_for_strategy
        solver = _solver_for_strategy(strategy, features)
        dist = solver.solve(g, 0)
        assert dist[0] == 0.0
        # All vertices reachable in a cycle.
        for v in range(5):
            assert dist[v] < math.inf
