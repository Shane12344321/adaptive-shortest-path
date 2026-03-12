"""Tests for aspf/solvers.py."""

from __future__ import annotations

import math
import random

import pytest

from aspf.adapters import AdjListAdapter
from aspf.solvers import (
    ArrayDijkstraSolver,
    BFSSolver,
    DialSolver,
    HeapDijkstraSolver,
)

# ---------------------------------------------------------------------------
# Reference test graph
#
# 0 --4--> 1
# 0 --1--> 2
# 1 --1--> 3
# 2 --2--> 1
# 2 --5--> 3
# 3 --3--> 4
#
# Expected distances from 0: {0:0, 1:3, 2:1, 3:4, 4:7}
# ---------------------------------------------------------------------------

_ADJ = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: [(4, 3)],
    4: [],
}
_EXPECTED = {0: 0, 1: 3, 2: 1, 3: 4, 4: 7}

_UNIT_ADJ = {
    0: [(1, 1), (2, 1)],
    1: [(3, 1)],
    2: [(1, 1), (3, 1)],
    3: [(4, 1)],
    4: [],
}
_UNIT_EXPECTED = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def _graph(adj, meta=None):
    return AdjListAdapter(adj, meta=meta)


def _close(a, b, tol=1e-9):
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# BFSSolver
# ---------------------------------------------------------------------------


class TestBFSSolver:
    def test_basic_unweighted(self):
        g = _graph(_UNIT_ADJ, meta={"unweighted": True})
        dist = BFSSolver().solve(g, 0)
        for v, d in _UNIT_EXPECTED.items():
            assert dist[v] == d

    def test_unreachable_vertices(self):
        adj = {0: [(1, 1)], 1: [], 2: [], 3: []}
        g = _graph(adj)
        dist = BFSSolver().solve(g, 0)
        assert dist[2] == math.inf
        assert dist[3] == math.inf

    def test_source_distance_zero(self):
        g = _graph(_UNIT_ADJ)
        dist = BFSSolver().solve(g, 0)
        assert dist[0] == 0


# ---------------------------------------------------------------------------
# ArrayDijkstraSolver
# ---------------------------------------------------------------------------


class TestArrayDijkstraSolver:
    def test_basic_weighted(self):
        g = _graph(_ADJ)
        dist = ArrayDijkstraSolver().solve(g, 0)
        for v, d in _EXPECTED.items():
            assert _close(dist[v], d), f"v={v}: {dist[v]} != {d}"

    def test_unreachable_vertices(self):
        adj = {0: [(1, 2.0)], 1: [], 2: [], 3: []}
        g = _graph(adj)
        dist = ArrayDijkstraSolver().solve(g, 0)
        assert dist[2] == math.inf
        assert dist[3] == math.inf

    def test_source_distance_zero(self):
        g = _graph(_ADJ)
        dist = ArrayDijkstraSolver().solve(g, 0)
        assert dist[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# HeapDijkstraSolver
# ---------------------------------------------------------------------------


class TestHeapDijkstraSolver:
    def test_basic_weighted(self):
        g = _graph(_ADJ)
        dist = HeapDijkstraSolver().solve(g, 0)
        for v, d in _EXPECTED.items():
            assert _close(dist[v], d), f"v={v}: {dist[v]} != {d}"

    def test_unreachable_vertices(self):
        adj = {0: [(1, 2.0)], 1: [], 2: [], 3: []}
        g = _graph(adj)
        dist = HeapDijkstraSolver().solve(g, 0)
        assert dist[2] == math.inf
        assert dist[3] == math.inf

    def test_single_vertex(self):
        g = _graph({0: []})
        dist = HeapDijkstraSolver().solve(g, 0)
        assert dist[0] == pytest.approx(0.0)

    def test_source_distance_zero(self):
        g = _graph(_ADJ)
        dist = HeapDijkstraSolver().solve(g, 0)
        assert dist[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# DialSolver
# ---------------------------------------------------------------------------


class TestDialSolver:
    def test_basic_weighted(self):
        g = _graph(_ADJ)
        dist = DialSolver(max_weight=5).solve(g, 0)
        for v, d in _EXPECTED.items():
            assert _close(dist[v], d), f"v={v}: {dist[v]} != {d}"

    def test_unreachable_vertices(self):
        adj = {0: [(1, 1)], 1: [], 2: [], 3: []}
        g = _graph(adj)
        dist = DialSolver(max_weight=1).solve(g, 0)
        assert dist[2] == math.inf
        assert dist[3] == math.inf

    def test_max_weight_zero_raises_value_error(self):
        with pytest.raises(ValueError):
            DialSolver(max_weight=0)

    def test_max_weight_negative_raises_value_error(self):
        with pytest.raises(ValueError):
            DialSolver(max_weight=-1)

    def test_source_distance_zero(self):
        g = _graph(_ADJ)
        dist = DialSolver(max_weight=5).solve(g, 0)
        assert dist[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Cross-solver consistency
# ---------------------------------------------------------------------------


class TestCrossSolverConsistency:
    def test_all_dijkstra_variants_agree_on_reference_graph(self):
        g = _graph(_ADJ)
        ref = HeapDijkstraSolver().solve(g, 0)
        arr = ArrayDijkstraSolver().solve(g, 0)
        dial = DialSolver(max_weight=5).solve(g, 0)

        for v in ref:
            assert _close(ref[v], arr[v]), f"heap vs array at {v}"
            assert _close(ref[v], dial[v]), f"heap vs dial at {v}"

    def test_random_graph_all_solvers_agree(self):
        """All three Dijkstra variants agree on a random weighted graph."""
        rng = random.Random(777)
        n = 100
        adj = {i: [] for i in range(n)}
        for u in range(n):
            for v in range(n):
                if u != v and rng.random() < 0.3:
                    w = rng.randint(1, 20)
                    adj[u].append((v, w))

        g = _graph(adj)
        source = 0
        ref = HeapDijkstraSolver().solve(g, source)
        arr = ArrayDijkstraSolver().solve(g, source)
        dial = DialSolver(max_weight=20).solve(g, source)

        for v in ref:
            assert _close(ref[v], arr[v]), f"heap vs array at {v}: {ref[v]} vs {arr[v]}"
            assert _close(ref[v], dial[v]), f"heap vs dial at {v}: {ref[v]} vs {dial[v]}"
