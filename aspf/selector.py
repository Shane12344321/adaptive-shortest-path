"""Strategy Selector for ASPF.

Loads thresholds from a JSON config file (falls back to safe defaults if the
file is missing or corrupt) and maps a feature dict produced by
:class:`~aspf.analyzer.GraphAnalyzer` to a strategy string.

Decision tree
-------------
1. ``is_unit_weight`` (metadata flag only)        → BFS
2. ``is_integer`` and ``0 < max_weight <= bucket_threshold``  → DIAL
3. ``density > dense_threshold`` and ``V <= size_limit``
      BUT if ``dense_threshold < density <= dense_threshold + confidence_margin``
      → HEAP_DIJKSTRA (boundary case)
      ELSE → ARRAY_DIJKSTRA
4. else → HEAP_DIJKSTRA
"""

from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict, Optional

from aspf.analyzer import BFS, DIAL, ARRAY_DIJKSTRA, HEAP_DIJKSTRA

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULTS: Dict[str, Any] = {
    "bucket_threshold": 100,
    "dense_threshold": 0.5,
    "size_limit": 10000,
    "confidence_margin": 0.05,
}

# Path to the bundled default config shipped with the package.
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load thresholds from *path*, silently falling back to defaults."""
    if path is None:
        path = _DEFAULT_CONFIG_PATH
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return dict(_DEFAULTS)
        result = dict(_DEFAULTS)
        result.update({k: v for k, v in data.items() if k in _DEFAULTS})
        return result
    except Exception:
        return dict(_DEFAULTS)


# ---------------------------------------------------------------------------
# StrategySelector
# ---------------------------------------------------------------------------


class StrategySelector:
    """Maps a feature dict to the most suitable shortest-path strategy.

    Parameters
    ----------
    config_path:
        Path to a JSON file containing threshold overrides.  Pass ``None``
        (default) to use the bundled ``default_config.json``.  Missing or
        corrupt files are silently ignored and the built-in defaults apply.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        self._thresholds = _load_config(config_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def thresholds(self) -> Dict[str, Any]:
        """Return a copy of the active thresholds dict."""
        return copy.copy(self._thresholds)

    def select(self, features: Dict[str, Any]) -> str:
        """Return the recommended strategy string for the given feature dict.

        Parameters
        ----------
        features:
            Dict as returned by :py:meth:`GraphAnalyzer.analyse`.

        Returns
        -------
        str
            One of ``BFS``, ``DIAL``, ``ARRAY_DIJKSTRA``, ``HEAP_DIJKSTRA``.
        """
        t = self._thresholds

        bucket_threshold: float = t["bucket_threshold"]
        dense_threshold: float = t["dense_threshold"]
        size_limit: int = t["size_limit"]
        confidence_margin: float = t["confidence_margin"]

        # ------------------------------------------------------------------
        # Rule 1: unit-weight flag from metadata → BFS (only safe path).
        # ------------------------------------------------------------------
        if features.get("is_unit_weight", False):
            return BFS

        # ------------------------------------------------------------------
        # Rule 2: small integer weights → DIAL.
        # ------------------------------------------------------------------
        max_w = features.get("max_weight_sample", 0.0)
        if features.get("is_integer", False) and 0 < max_w <= bucket_threshold:
            return DIAL

        # ------------------------------------------------------------------
        # Rule 3: dense + small graph → ARRAY_DIJKSTRA (with boundary guard).
        # ------------------------------------------------------------------
        density: float = features.get("density", 0.0)
        V: int = features.get("vertex_count", 0)

        if density > dense_threshold and V <= size_limit:
            # Within the confidence margin of the boundary → safer default.
            if density <= dense_threshold + confidence_margin:
                return HEAP_DIJKSTRA
            return ARRAY_DIJKSTRA

        # ------------------------------------------------------------------
        # Rule 4: fallback.
        # ------------------------------------------------------------------
        return HEAP_DIJKSTRA
