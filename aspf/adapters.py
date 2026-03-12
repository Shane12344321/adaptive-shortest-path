"""Input Adapter Layer for ASPF.

Defines the GraphView abstract interface and two concrete adapters:
  - AdjListAdapter: wraps a plain adjacency-list dict (zero-copy).
  - NetworkXAdapter: wraps a NetworkX DiGraph / Graph (zero-copy).
"""

from __future__ import annotations

import abc
from typing import Any, Dict, Iterator, Tuple


class GraphView(abc.ABC):
    """Abstract interface for a graph viewed by the ASPF pipeline."""

    @abc.abstractmethod
    def neighbors(self, v: Any) -> Iterator[Tuple[Any, float]]:
        """Return an iterator of (neighbor, weight) tuples for vertex *v*."""

    @abc.abstractmethod
    def vertex_count(self) -> int:
        """Return the number of vertices."""

    @abc.abstractmethod
    def edge_count(self) -> int:
        """Return the number of directed edges (arcs)."""

    @abc.abstractmethod
    def vertices(self) -> Iterator[Any]:
        """Return an iterator over all vertex identifiers."""

    def metadata(self) -> Dict[str, Any]:
        """Return graph-level metadata dict (default: empty dict)."""
        return {}


class AdjListAdapter(GraphView):
    """Zero-copy wrapper around a ``dict[vertex, list[tuple[vertex, weight]]]``.

    Parameters
    ----------
    adj:
        Adjacency list mapping each vertex to a list of ``(neighbour, weight)``
        pairs.  The adapter holds a reference; no data is copied.
    meta:
        Optional metadata dict (e.g. ``{"unweighted": True}``).
    """

    def __init__(
        self,
        adj: Dict[Any, list],
        meta: Dict[str, Any] | None = None,
    ) -> None:
        self._adj = adj
        self._meta: Dict[str, Any] = meta if meta is not None else {}
        # Pre-compute edge count once at construction time.
        # Note: if the adjacency list is mutated after construction the cached
        # count will be stale.  vertex_count() and neighbors() always reflect
        # the live dict, but edge_count() returns the value fixed at init.
        self._edge_count: int = sum(len(neighbours) for neighbours in adj.values())

    # ------------------------------------------------------------------
    # GraphView interface
    # ------------------------------------------------------------------

    def neighbors(self, v: Any) -> Iterator[Tuple[Any, float]]:
        return iter(self._adj.get(v, []))

    def vertex_count(self) -> int:
        return len(self._adj)

    def edge_count(self) -> int:
        return self._edge_count

    def vertices(self) -> Iterator[Any]:
        return iter(self._adj)

    def metadata(self) -> Dict[str, Any]:
        return self._meta


class NetworkXAdapter(GraphView):
    """Zero-copy wrapper around a NetworkX graph object.

    Edge weights are read from the ``"weight"`` attribute (defaulting to 1 if
    absent).  Graph-level attributes are exposed via :py:meth:`metadata`.

    Parameters
    ----------
    G:
        A NetworkX ``Graph`` or ``DiGraph``.  The adapter holds a reference;
        no data is copied.
    """

    def __init__(self, G: Any) -> None:
        self._G = G

    # ------------------------------------------------------------------
    # GraphView interface
    # ------------------------------------------------------------------

    def neighbors(self, v: Any) -> Iterator[Tuple[Any, float]]:
        try:
            for nbr, data in self._G[v].items():
                yield nbr, data.get("weight", 1)
        except KeyError:
            return

    def vertex_count(self) -> int:
        return self._G.number_of_nodes()

    def edge_count(self) -> int:
        # NetworkX number_of_edges() counts undirected edges once;
        # for directed graphs it counts arcs.  We want directed arc count.
        try:
            # DiGraph / MultiDiGraph
            return self._G.number_of_edges()
        except Exception:
            return 0

    def vertices(self) -> Iterator[Any]:
        return iter(self._G.nodes())

    def metadata(self) -> dict:
        return dict(self._G.graph)
