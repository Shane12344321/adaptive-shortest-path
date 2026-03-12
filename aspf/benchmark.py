"""Benchmarking Module for ASPF.

Provides :class:`BenchmarkRunner` which times the adaptive pipeline and
optionally compares all four solvers on the same graph.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from aspf.adapters import GraphView
from aspf.analyzer import (
    BFS,
    DIAL,
    ARRAY_DIJKSTRA,
    HEAP_DIJKSTRA,
    GraphAnalyzer,
)
from aspf.selector import StrategySelector, _DEFAULTS
from aspf.solvers import (
    BFSSolver,
    ArrayDijkstraSolver,
    HeapDijkstraSolver,
    DialSolver,
    ShortestPathSolver,
)

# Default max_weight fallback for Dial solver (matches bucket_threshold default).
_DEFAULT_DIAL_MAX_WEIGHT: int = _DEFAULTS["bucket_threshold"]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _solver_for_strategy(strategy: str, features: Dict[str, Any]) -> ShortestPathSolver:
    """Instantiate the correct solver for *strategy*.

    For DIAL, uses ``max(1, int(features.get("max_weight_sample", 100)))``.
    """
    if strategy == BFS:
        return BFSSolver()
    if strategy == DIAL:
        mw = max(1, int(features.get("max_weight_sample", _DEFAULT_DIAL_MAX_WEIGHT)))
        return DialSolver(max_weight=mw)
    if strategy == ARRAY_DIJKSTRA:
        return ArrayDijkstraSolver()
    # Default / HEAP_DIJKSTRA
    return HeapDijkstraSolver()


def _dists_equal(a: Dict[Any, float], b: Dict[Any, float], tol: float = 1e-9) -> bool:
    """Return True if distance dicts *a* and *b* are equal within *tol*."""
    if set(a.keys()) != set(b.keys()):
        # Allow subset comparison: every key in *a* must match *b*.
        common = set(a.keys()) & set(b.keys())
        for v in common:
            if abs(a[v] - b[v]) > tol:
                return False
        return True
    for v, d_a in a.items():
        d_b = b.get(v, float("inf"))
        if abs(d_a - d_b) > tol:
            return False
    return True


def _median(xs: List[float]) -> float:
    """Return the median of list *xs*."""
    if not xs:
        return 0.0
    sorted_xs = sorted(xs)
    n = len(sorted_xs)
    mid = n // 2
    if n % 2 == 1:
        return sorted_xs[mid]
    return (sorted_xs[mid - 1] + sorted_xs[mid]) / 2.0


def _bootstrap_ci(
    data: List[float],
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap confidence interval for the median of *data*.

    Returns
    -------
    (lower, upper)
        The (1-confidence)/2 and (1+confidence)/2 quantiles of the bootstrap
        distribution of medians.
    """
    if not data:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(data)
    medians: List[float] = []
    for _ in range(n_resamples):
        resample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        medians.append(_median(resample))
    medians.sort()
    lo_idx = int((1.0 - confidence) / 2.0 * n_resamples)
    hi_idx = int((1.0 + confidence) / 2.0 * n_resamples) - 1
    lo_idx = max(0, min(lo_idx, n_resamples - 1))
    hi_idx = max(0, min(hi_idx, n_resamples - 1))
    return (medians[lo_idx], medians[hi_idx])


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------

_ALL_STRATEGIES = [BFS, DIAL, ARRAY_DIJKSTRA, HEAP_DIJKSTRA]


class BenchmarkRunner:
    """Run and time the adaptive shortest-path pipeline.

    Parameters
    ----------
    config_path:
        Path to a JSON thresholds config (passed through to
        :class:`~aspf.selector.StrategySelector`).
    total_runs:
        Total number of timed solver iterations (including warm-up).
    warmup_runs:
        Number of initial runs to discard as warm-up.
    analyzer_seed:
        Optional seed forwarded to :class:`~aspf.analyzer.GraphAnalyzer` for
        reproducible sampling.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        total_runs: int = 10,
        warmup_runs: int = 2,
        analyzer_seed: Optional[int] = None,
    ) -> None:
        self._config_path = config_path
        self._total_runs = total_runs
        self._warmup_runs = warmup_runs
        self._analyzer_seed = analyzer_seed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        graph: GraphView,
        source: Any,
        compare_all: bool = False,
    ) -> Dict[str, Any]:
        """Run the adaptive pipeline and return timing + result data.

        Parameters
        ----------
        graph:
            Input graph (any :class:`~aspf.adapters.GraphView`).
        source:
            Source vertex for SSSP.
        compare_all:
            If ``True``, also run all four solvers independently and
            cross-verify distances.

        Returns
        -------
        dict
            Always contains:
            ``strategy``, ``features``, ``distances``, ``median_ns``,
            ``ci_95``.

            When *compare_all* is ``True``, also contains:
            ``comparison``, ``all_distances_match``, ``mismatched_strategies``.
        """
        analyzer = GraphAnalyzer(seed=self._analyzer_seed)
        selector = StrategySelector(self._config_path)

        features = analyzer.analyse(graph)
        strategy = selector.select(features)
        solver = _solver_for_strategy(strategy, features)

        # Timed runs.
        timings_ns: List[int] = []
        distances: Dict[Any, float] = {}
        for i in range(self._total_runs):
            t0 = time.perf_counter_ns()
            result = solver.solve(graph, source)
            t1 = time.perf_counter_ns()
            if i >= self._warmup_runs:
                timings_ns.append(t1 - t0)
            distances = result

        median_ns = _median([float(t) for t in timings_ns])
        ci_95 = _bootstrap_ci([float(t) for t in timings_ns])

        output: Dict[str, Any] = {
            "strategy": strategy,
            "features": features,
            "distances": distances,
            "median_ns": median_ns,
            "ci_95": ci_95,
        }

        if compare_all:
            output.update(self._compare_all_solvers(graph, source, features, distances))

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compare_all_solvers(
        self,
        graph: GraphView,
        source: Any,
        features: Dict[str, Any],
        reference_distances: Dict[Any, float],
    ) -> Dict[str, Any]:
        """Run all four solvers and cross-verify against HeapDijkstra."""
        # Use HeapDijkstra as the authoritative reference.
        ref_solver = HeapDijkstraSolver()
        ref_distances = ref_solver.solve(graph, source)

        comparison: Dict[str, Any] = {}
        mismatched: List[str] = []

        for strat in _ALL_STRATEGIES:
            solver = _solver_for_strategy(strat, features)

            timings_ns: List[int] = []
            dists: Dict[Any, float] = {}
            for i in range(self._total_runs):
                t0 = time.perf_counter_ns()
                d = solver.solve(graph, source)
                t1 = time.perf_counter_ns()
                if i >= self._warmup_runs:
                    timings_ns.append(t1 - t0)
                dists = d

            med = _median([float(t) for t in timings_ns])
            ci = _bootstrap_ci([float(t) for t in timings_ns])
            match = _dists_equal(dists, ref_distances)

            comparison[strat] = {
                "distances": dists,
                "median_ns": med,
                "ci_95": ci,
                "matches_reference": match,
            }

            if not match:
                mismatched.append(strat)

        all_match = len(mismatched) == 0

        return {
            "comparison": comparison,
            "all_distances_match": all_match,
            "mismatched_strategies": mismatched,
        }
