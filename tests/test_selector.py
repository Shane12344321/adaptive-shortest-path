"""Tests for aspf/selector.py."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from aspf.analyzer import BFS, DIAL, ARRAY_DIJKSTRA, HEAP_DIJKSTRA
from aspf.selector import StrategySelector, _DEFAULTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _features(**overrides):
    base = {
        "density": 0.1,
        "is_unit_weight": False,
        "is_unit_weight_sample": False,
        "max_weight_sample": 50.0,
        "is_integer": False,
        "degree_variance_sample": 0.0,
        "vertex_count": 100,
        "edge_count": 1000,
        "recommended_strategy": HEAP_DIJKSTRA,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestDecisionTree
# ---------------------------------------------------------------------------


class TestDecisionTree:
    def test_bfs_for_unit_weight(self):
        s = StrategySelector()
        assert s.select(_features(is_unit_weight=True)) == BFS

    def test_dial_for_small_integer(self):
        s = StrategySelector()
        feat = _features(is_integer=True, max_weight_sample=50.0)
        assert s.select(feat) == DIAL

    def test_dial_rejected_for_large_max_weight(self):
        s = StrategySelector()
        # bucket_threshold default is 100; 200 > 100 → not DIAL
        feat = _features(is_integer=True, max_weight_sample=200.0)
        result = s.select(feat)
        assert result != DIAL

    def test_dial_rejected_for_float_weights(self):
        s = StrategySelector()
        feat = _features(is_integer=False, max_weight_sample=5.0)
        assert s.select(feat) != DIAL

    def test_array_for_dense_small_graph(self):
        s = StrategySelector()
        # density=0.7 > 0.5+0.05=0.55, V=100 <= 10000
        feat = _features(density=0.7, vertex_count=100)
        assert s.select(feat) == ARRAY_DIJKSTRA

    def test_heap_default(self):
        s = StrategySelector()
        # sparse, non-integer, not unit-weight
        feat = _features(density=0.05)
        assert s.select(feat) == HEAP_DIJKSTRA

    def test_heap_for_large_v_even_if_dense(self):
        s = StrategySelector()
        # density > dense_threshold but V > size_limit
        feat = _features(density=0.9, vertex_count=20000)
        assert s.select(feat) == HEAP_DIJKSTRA


# ---------------------------------------------------------------------------
# TestConfidenceMargin
# ---------------------------------------------------------------------------


class TestConfidenceMargin:
    def test_within_margin_returns_heap(self):
        s = StrategySelector()
        # default dense_threshold=0.5, confidence_margin=0.05
        # density in (0.5, 0.55] → HEAP_DIJKSTRA
        feat = _features(density=0.52, vertex_count=100)
        assert s.select(feat) == HEAP_DIJKSTRA

    def test_at_boundary_returns_heap(self):
        s = StrategySelector()
        feat = _features(density=0.55, vertex_count=100)
        # density == dense_threshold + margin → still boundary → HEAP
        assert s.select(feat) == HEAP_DIJKSTRA

    def test_above_margin_returns_array(self):
        s = StrategySelector()
        # density=0.56 > 0.55 → ARRAY_DIJKSTRA
        feat = _features(density=0.56, vertex_count=100)
        assert s.select(feat) == ARRAY_DIJKSTRA

    def test_below_threshold_returns_heap(self):
        s = StrategySelector()
        feat = _features(density=0.3, vertex_count=100)
        assert s.select(feat) == HEAP_DIJKSTRA


# ---------------------------------------------------------------------------
# TestConfigLoading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_custom_config_loaded(self):
        config = {
            "bucket_threshold": 50,
            "dense_threshold": 0.3,
            "size_limit": 5000,
            "confidence_margin": 0.02,
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(config, f)
            path = f.name
        try:
            s = StrategySelector(config_path=path)
            t = s.thresholds
            assert t["bucket_threshold"] == 50
            assert t["dense_threshold"] == pytest.approx(0.3)
            assert t["size_limit"] == 5000
            assert t["confidence_margin"] == pytest.approx(0.02)
        finally:
            os.unlink(path)

    def test_missing_config_uses_defaults(self):
        s = StrategySelector(config_path="/nonexistent/path/config.json")
        t = s.thresholds
        for key, val in _DEFAULTS.items():
            assert t[key] == pytest.approx(val)

    def test_corrupt_config_uses_defaults(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("THIS IS NOT JSON {{{")
            path = f.name
        try:
            s = StrategySelector(config_path=path)
            t = s.thresholds
            for key, val in _DEFAULTS.items():
                assert t[key] == pytest.approx(val)
        finally:
            os.unlink(path)

    def test_thresholds_returns_copy(self):
        s = StrategySelector()
        t1 = s.thresholds
        t1["bucket_threshold"] = 999
        t2 = s.thresholds
        assert t2["bucket_threshold"] == _DEFAULTS["bucket_threshold"]
