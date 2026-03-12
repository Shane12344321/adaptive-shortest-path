"""Two-Tier Graph Analyzer for ASPF.

Extracts graph features with minimal overhead:
  - Tier 1  (O(1))      : metadata check + size/density from graph header.
  - Tier 2  (O(√E))     : random-sample edge scan (only when Tier 1 is inconclusive).

Strategy constants
------------------
BFS             -- breadth-first search (unit-weight only).
DIAL            -- Dial's bucket-queue algorithm (small integer weights).
ARRAY_DIJKSTRA  -- O(V²) array scan Dijkstra (best on dense graphs).
HEAP_DIJKSTRA   -- O((V+E) log V) heap Dijkstra (general-purpose default).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, Optional

from aspf.adapters import GraphView

# ---------------------------------------------------------------------------
# Strategy name constants
# ---------------------------------------------------------------------------

BFS = "BFS"
DIAL = "DIAL"
ARRAY_DIJKSTRA = "ARRAY_DIJKSTRA"
HEAP_DIJKSTRA = "HEAP_DIJKSTRA"

# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

_FEATURE_DEFAULTS: Dict[str, Any] = {
    "density": 0.0,
    "is_unit_weight": False,
    "is_unit_weight_sample": False,
    "max_weight_sample": 0.0,
    "is_integer": False,
    "degree_variance_sample": 0.0,
    "vertex_count": 0,
    "edge_count": 0,
    "recommended_strategy": HEAP_DIJKSTRA,
}


class GraphAnalyzer:
    """Analyses a :class:`~aspf.adapters.GraphView` and returns a feature dict.

    Parameters
    ----------
    seed:
        Optional integer seed for reproducible reservoir sampling in Tier 2.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, graph: GraphView) -> Dict[str, Any]:
        """Return a feature dict describing *graph*.

        The returned dict always contains exactly the keys listed in
        ``_FEATURE_DEFAULTS``.
        """
        features: Dict[str, Any] = dict(_FEATURE_DEFAULTS)

        V = graph.vertex_count()
        E = graph.edge_count()
        features["vertex_count"] = V
        features["edge_count"] = E

        # ------------------------------------------------------------------
        # Tier 1 – O(1) metadata + size/density checks
        # ------------------------------------------------------------------

        # Density for directed graphs: E / (V * (V-1))
        if V <= 1:
            density = 0.0
        else:
            density = E / (V * (V - 1))
        features["density"] = density

        meta = graph.metadata()

        # The ONLY safe path to BFS: explicit metadata flag.
        if meta.get("unweighted") is True:
            features["is_unit_weight"] = True
            features["recommended_strategy"] = BFS
            return features

        # Trivial graphs – nothing to analyse further.
        if E == 0 or V <= 1:
            features["recommended_strategy"] = HEAP_DIJKSTRA
            return features

        # ------------------------------------------------------------------
        # Tier 2 – O(√E) reservoir sampling
        # ------------------------------------------------------------------
        self._run_tier2(graph, E, features)
        return features

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_tier2(self, graph: GraphView, E: int, features: Dict[str, Any]) -> None:
        """Populate Tier-2 fields in *features* via reservoir sampling."""
        sample_size = max(1, int(math.isqrt(E)))

        # Reservoir-sample *sample_size* edges from the adjacency-list stream.
        reservoir: list = []  # each entry: (weight, src_vertex)
        seen = 0
        for v in graph.vertices():
            for _nbr, w in graph.neighbors(v):
                seen += 1
                if len(reservoir) < sample_size:
                    reservoir.append((w, v))
                else:
                    j = self._rng.randint(0, seen - 1)
                    if j < sample_size:
                        reservoir[j] = (w, v)

        if not reservoir:
            return

        weights = [w for w, _v in reservoir]
        src_vertices = [v for _w, v in reservoir]

        # Unit-weight sample flag (informational only – NOT used for BFS dispatch).
        is_unit_weight_sample = all(w == 1 for w in weights)
        features["is_unit_weight_sample"] = is_unit_weight_sample

        # Integer-weight detection.
        is_integer = all(isinstance(w, int) or (isinstance(w, float) and w.is_integer())
                         for w in weights) and all(w >= 0 for w in weights)
        features["is_integer"] = is_integer

        # Max weight from sample.
        features["max_weight_sample"] = float(max(weights))

        # Degree variance from sampled source vertices.
        degrees = []
        seen_verts: set = set()
        for v in src_vertices:
            if v not in seen_verts:
                seen_verts.add(v)
                deg = sum(1 for _ in graph.neighbors(v))
                degrees.append(deg)

        if len(degrees) > 1:
            mean_deg = sum(degrees) / len(degrees)
            variance = sum((d - mean_deg) ** 2 for d in degrees) / len(degrees)
        elif len(degrees) == 1:
            variance = 0.0
        else:
            variance = 0.0

        features["degree_variance_sample"] = variance

        # Tier-2 never recommends BFS – safe default is HEAP_DIJKSTRA.
        features["recommended_strategy"] = HEAP_DIJKSTRA
