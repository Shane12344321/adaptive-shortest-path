"""Tests for aspf/benchmark.py."""

from __future__ import annotations

import math

import pytest

from aspf.adapters import AdjListAdapter
from aspf.benchmark import BenchmarkRunner, _bootstrap_ci, _dists_equal, _median


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unweighted_graph():
    adj = {i: [(i + 1, 1)] for i in range(9)}
    adj[9] = []
    return AdjListAdapter(adj, meta={"unweighted": True})


def _small_graph():
    adj = {
        0: [(1, 4), (2, 1)],
        1: [(3, 1)],
        2: [(1, 2), (3, 5)],
        3: [(4, 3)],
        4: [],
    }
    return AdjListAdapter(adj)


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_median_odd(self):
        assert _median([3.0, 1.0, 2.0]) == pytest.approx(2.0)

    def test_median_even(self):
        assert _median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)

    def test_median_single(self):
        assert _median([7.0]) == pytest.approx(7.0)

    def test_median_empty(self):
        assert _median([]) == pytest.approx(0.0)

    def test_dists_equal_true(self):
        a = {0: 0.0, 1: 1.0, 2: math.inf}
        b = {0: 0.0, 1: 1.0, 2: math.inf}
        assert _dists_equal(a, b)

    def test_dists_equal_false(self):
        a = {0: 0.0, 1: 1.0}
        b = {0: 0.0, 1: 2.0}
        assert not _dists_equal(a, b)

    def test_bootstrap_ci_returns_tuple(self):
        data = [100.0, 200.0, 150.0, 120.0, 180.0]
        lo, hi = _bootstrap_ci(data, n_resamples=100, seed=0)
        assert lo <= hi

    def test_bootstrap_ci_empty(self):
        lo, hi = _bootstrap_ci([])
        assert lo == 0.0 and hi == 0.0


# ---------------------------------------------------------------------------
# BenchmarkRunner tests
# ---------------------------------------------------------------------------


class TestBenchmarkRunner:
    def test_basic_run_returns_expected_keys(self):
        runner = BenchmarkRunner(total_runs=4, warmup_runs=1, analyzer_seed=0)
        result = runner.run(_small_graph(), 0)
        for key in ("strategy", "features", "distances", "median_ns", "ci_95"):
            assert key in result, f"Missing key: {key}"

    def test_compare_all_returns_comparison_data(self):
        runner = BenchmarkRunner(total_runs=4, warmup_runs=1, analyzer_seed=0)
        result = runner.run(_unweighted_graph(), 0, compare_all=True)
        for key in ("comparison", "all_distances_match", "mismatched_strategies"):
            assert key in result, f"Missing key: {key}"

    def test_timing_is_positive(self):
        runner = BenchmarkRunner(total_runs=4, warmup_runs=1, analyzer_seed=0)
        result = runner.run(_small_graph(), 0)
        assert result["median_ns"] >= 0

    def test_ci_lo_le_hi(self):
        runner = BenchmarkRunner(total_runs=6, warmup_runs=2, analyzer_seed=0)
        result = runner.run(_small_graph(), 0)
        lo, hi = result["ci_95"]
        assert lo <= hi

    def test_unweighted_graph_compare_all_distances_match(self):
        runner = BenchmarkRunner(total_runs=4, warmup_runs=1, analyzer_seed=0)
        result = runner.run(_unweighted_graph(), 0, compare_all=True)
        assert result["all_distances_match"] is True
