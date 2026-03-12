"""Execution Engine for ASPF.

Implements four shortest-path solvers behind a common ABC:

BFSSolver           -- BFS (unit-weight graphs only).
ArrayDijkstraSolver -- O(V²) array scan Dijkstra.
HeapDijkstraSolver  -- O((V+E) log V) heap Dijkstra (general-purpose default).
DialSolver          -- Dial's circular bucket-queue algorithm.
"""

from __future__ import annotations

import abc
import collections
import heapq
import math
from typing import Any, Dict

from aspf.adapters import GraphView


class ShortestPathSolver(abc.ABC):
    """Abstract base class for shortest-path solvers."""

    @abc.abstractmethod
    def solve(self, graph: GraphView, source: Any) -> Dict[Any, float]:
        """Compute single-source shortest distances from *source*.

        Parameters
        ----------
        graph:
            The graph to operate on.
        source:
            The source vertex identifier.

        Returns
        -------
        dict
            Mapping ``vertex → distance``.  Unreachable vertices have distance
            ``math.inf``.
        """


# ---------------------------------------------------------------------------
# BFS Solver
# ---------------------------------------------------------------------------


class BFSSolver(ShortestPathSolver):
    """Standard BFS shortest-path solver.

    **Only valid for unit-weight graphs.**  Returns integer distances
    (0, 1, 2, …).
    """

    def solve(self, graph: GraphView, source: Any) -> Dict[Any, float]:
        dist: Dict[Any, float] = {v: math.inf for v in graph.vertices()}
        if source not in dist:
            dist[source] = math.inf
            return dist

        dist[source] = 0
        queue: collections.deque = collections.deque([source])

        while queue:
            u = queue.popleft()
            d_u = dist[u]
            for v, _w in graph.neighbors(u):
                if v not in dist:
                    dist[v] = math.inf
                if dist[v] == math.inf:
                    dist[v] = d_u + 1
                    queue.append(v)

        return dist


# ---------------------------------------------------------------------------
# Array Dijkstra Solver
# ---------------------------------------------------------------------------


class ArrayDijkstraSolver(ShortestPathSolver):
    """O(V²) Dijkstra with linear-scan for minimum-distance vertex.

    Efficient on dense graphs where E ≈ V².
    """

    def solve(self, graph: GraphView, source: Any) -> Dict[Any, float]:
        dist: Dict[Any, float] = {v: math.inf for v in graph.vertices()}
        if source not in dist:
            dist[source] = math.inf
            return dist

        dist[source] = 0.0
        visited: set = set()

        while True:
            # Linear scan for minimum-distance unvisited vertex.
            u = None
            u_dist = math.inf
            for v, d in dist.items():
                if v not in visited and d < u_dist:
                    u_dist = d
                    u = v

            if u is None:
                break

            visited.add(u)

            for v, w in graph.neighbors(u):
                if v not in dist:
                    dist[v] = math.inf
                new_d = u_dist + w
                if new_d < dist[v]:
                    dist[v] = new_d

        return dist


# ---------------------------------------------------------------------------
# Heap Dijkstra Solver
# ---------------------------------------------------------------------------


class HeapDijkstraSolver(ShortestPathSolver):
    """O((V + E) log V) Dijkstra with binary min-heap (lazy deletion).

    General-purpose default.
    """

    def solve(self, graph: GraphView, source: Any) -> Dict[Any, float]:
        dist: Dict[Any, float] = {v: math.inf for v in graph.vertices()}
        if source not in dist:
            dist[source] = math.inf
            return dist

        dist[source] = 0.0
        heap = [(0.0, source)]

        while heap:
            d_u, u = heapq.heappop(heap)
            # Lazy deletion: skip stale entries.
            if d_u > dist[u]:
                continue
            for v, w in graph.neighbors(u):
                if v not in dist:
                    dist[v] = math.inf
                new_d = d_u + w
                if new_d < dist[v]:
                    dist[v] = new_d
                    heapq.heappush(heap, (new_d, v))

        return dist


# ---------------------------------------------------------------------------
# Dial Solver
# ---------------------------------------------------------------------------


class DialSolver(ShortestPathSolver):
    """Dial's algorithm using a circular bucket array.

    Parameters
    ----------
    max_weight:
        Maximum integer edge weight (must be >= 1).

    Raises
    ------
    ValueError
        If *max_weight* < 1.
    """

    def __init__(self, max_weight: int) -> None:
        if max_weight < 1:
            raise ValueError(
                f"max_weight must be >= 1, got {max_weight!r}"
            )
        self._max_weight = int(max_weight)

    def solve(self, graph: GraphView, source: Any) -> Dict[Any, float]:
        num_buckets = self._max_weight + 1

        dist: Dict[Any, float] = {v: math.inf for v in graph.vertices()}
        if source not in dist:
            dist[source] = math.inf
            return dist

        dist[source] = 0.0

        # Circular bucket array – each bucket is a list of vertices.
        buckets: list = [[] for _ in range(num_buckets)]
        buckets[0].append(source)

        current_bucket = 0
        processed = 0
        total = len(dist)

        for _ in range(total * num_buckets + total):
            # Advance to the next non-empty bucket.
            while not buckets[current_bucket % num_buckets]:
                current_bucket += 1
                if current_bucket > total * num_buckets + total:
                    break

            bucket_idx = current_bucket % num_buckets
            if not buckets[bucket_idx]:
                break

            u = buckets[bucket_idx].pop()

            # Stale-entry check: vertex's current distance should map to
            # this bucket index.
            if dist[u] == math.inf:
                continue
            if int(dist[u]) % num_buckets != bucket_idx:
                continue

            processed += 1

            for v, w in graph.neighbors(u):
                if v not in dist:
                    dist[v] = math.inf
                new_d = dist[u] + w
                if new_d < dist[v]:
                    dist[v] = new_d
                    b = int(new_d) % num_buckets
                    buckets[b].append(v)

            if processed >= total:
                break

        return dist
