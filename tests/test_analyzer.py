"""Tests for aspf/analyzer.py."""

from __future__ import annotations

import math

import pytest

from aspf.adapters import AdjListAdapter
from aspf.analyzer import (
    BFS,
    HEAP_DIJKSTRA,
    GraphAnalyzer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXPECTED_KEYS = {
    "density",
    "is_unit_weight",
    "is_unit_weight_sample",
    "max_weight_sample",
    "is_integer",
    "degree_variance_sample",
    "vertex_count",
    "edge_count",
    "recommended_strategy",
}


def _make_graph(adj, meta=None):
    return AdjListAdapter(adj, meta=meta)


def _unit_graph(n=5):
    """Small unit-weight graph (no metadata flag)."""
    adj = {i: [(i + 1, 1)] for i in range(n - 1)}
    adj[n - 1] = []
    return adj


def _weighted_graph():
    """Graph with varying weights."""
    return {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [(4, 3)],
        4: [],
    }


# ---------------------------------------------------------------------------
# TestTier1
# ---------------------------------------------------------------------------


class TestTier1:
    def test_unweighted_metadata_returns_bfs(self):
        g = _make_graph(_unit_graph(), meta={"unweighted": True})
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["is_unit_weight"] is True
        assert feat["recommended_strategy"] == BFS

    def test_empty_graph_returns_heap_default(self):
        g = _make_graph({0: [], 1: []})
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["recommended_strategy"] == HEAP_DIJKSTRA
        assert feat["edge_count"] == 0

    def test_single_vertex_density_zero(self):
        g = _make_graph({0: []})
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["density"] == 0.0

    def test_density_calculation(self):
        # 4 vertices, 4 directed edges → density = 4/(4*3) = 1/3
        adj = {
            0: [(1, 1), (2, 1)],
            1: [(2, 1)],
            2: [(3, 1)],
            3: [],
        }
        g = _make_graph(adj)
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["density"] == pytest.approx(4 / (4 * 3))

    def test_tier1_returns_all_expected_keys(self):
        g = _make_graph(_unit_graph(), meta={"unweighted": True})
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert _EXPECTED_KEYS.issubset(feat.keys())


# ---------------------------------------------------------------------------
# TestTier2
# ---------------------------------------------------------------------------


class TestTier2:
    def test_all_unit_weight_sample_does_not_select_bfs(self):
        """CRITICAL: sampling suggests unit weights but metadata is absent → NOT BFS."""
        g = _make_graph(_unit_graph())
        feat = GraphAnalyzer(seed=0).analyse(g)
        # Tier 2 may detect unit weights in sample…
        # …but recommended_strategy must NEVER be BFS without metadata flag.
        assert feat["recommended_strategy"] != BFS
        assert feat["is_unit_weight"] is False

    def test_integer_weights_detected(self):
        g = _make_graph(_weighted_graph())
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["is_integer"] is True

    def test_non_integer_weights_detected(self):
        adj = {
            0: [(1, 1.5), (2, 2.7)],
            1: [(2, 0.3)],
            2: [],
        }
        g = _make_graph(adj)
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["is_integer"] is False

    def test_max_weight_sample_correctness(self):
        adj = {
            0: [(1, 10), (2, 20)],
            1: [(2, 5)],
            2: [],
        }
        g = _make_graph(adj)
        feat = GraphAnalyzer(seed=0).analyse(g)
        # max_weight_sample should be <= 20 (max possible weight in graph).
        assert feat["max_weight_sample"] <= 20.0
        # And at least 1 (minimum weight in graph).
        assert feat["max_weight_sample"] >= 1.0

    def test_feature_dict_has_all_expected_keys(self):
        g = _make_graph(_weighted_graph())
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert _EXPECTED_KEYS.issubset(feat.keys())

    def test_is_unit_weight_sample_true_when_all_weights_are_one(self):
        g = _make_graph(_unit_graph())
        feat = GraphAnalyzer(seed=42).analyse(g)
        assert feat["is_unit_weight_sample"] is True

    def test_recommended_strategy_heap_dijkstra_for_general_graph(self):
        g = _make_graph(_weighted_graph())
        feat = GraphAnalyzer(seed=0).analyse(g)
        assert feat["recommended_strategy"] == HEAP_DIJKSTRA
