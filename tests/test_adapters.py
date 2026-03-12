"""Tests for aspf/adapters.py."""

from __future__ import annotations

import pytest

from aspf.adapters import AdjListAdapter, GraphView


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def simple_adj():
    return {
        0: [(1, 1.0), (2, 2.0)],
        1: [(2, 3.0)],
        2: [],
    }


# ---------------------------------------------------------------------------
# TestAdjListAdapter
# ---------------------------------------------------------------------------


class TestAdjListAdapter:
    def test_isinstance_graphview(self):
        a = AdjListAdapter(simple_adj())
        assert isinstance(a, GraphView)

    def test_vertex_count(self):
        a = AdjListAdapter(simple_adj())
        assert a.vertex_count() == 3

    def test_edge_count(self):
        a = AdjListAdapter(simple_adj())
        assert a.edge_count() == 3  # (0→1), (0→2), (1→2)

    def test_neighbors_normal(self):
        a = AdjListAdapter(simple_adj())
        nbrs = list(a.neighbors(0))
        assert (1, 1.0) in nbrs
        assert (2, 2.0) in nbrs
        assert len(nbrs) == 2

    def test_neighbors_empty_vertex(self):
        a = AdjListAdapter(simple_adj())
        assert list(a.neighbors(2)) == []

    def test_neighbors_missing_vertex(self):
        a = AdjListAdapter(simple_adj())
        assert list(a.neighbors(99)) == []

    def test_vertices(self):
        a = AdjListAdapter(simple_adj())
        assert set(a.vertices()) == {0, 1, 2}

    def test_metadata_default_empty(self):
        a = AdjListAdapter(simple_adj())
        assert a.metadata() == {}

    def test_metadata_custom(self):
        a = AdjListAdapter(simple_adj(), meta={"unweighted": True})
        assert a.metadata() == {"unweighted": True}

    def test_zero_copy(self):
        """Mutating the original dict must be reflected in the adapter."""
        adj = simple_adj()
        a = AdjListAdapter(adj)
        assert a.edge_count() == 3
        # Add a new vertex with an edge.
        adj[3] = [(0, 5.0)]
        # vertex_count and neighbors should see the change.
        assert a.vertex_count() == 4
        nbrs = list(a.neighbors(3))
        assert nbrs == [(0, 5.0)]


# ---------------------------------------------------------------------------
# TestNetworkXAdapter
# ---------------------------------------------------------------------------


class TestNetworkXAdapter:
    @pytest.fixture(autouse=True)
    def _import_nx(self):
        pytest.importorskip("networkx")

    def _make_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=2.5)
        G.add_edge(0, 2, weight=1.0)
        G.add_edge(1, 2, weight=0.5)
        return G

    def test_vertex_count(self):
        from aspf.adapters import NetworkXAdapter
        a = NetworkXAdapter(self._make_graph())
        assert a.vertex_count() == 3

    def test_edge_count(self):
        from aspf.adapters import NetworkXAdapter
        a = NetworkXAdapter(self._make_graph())
        assert a.edge_count() == 3

    def test_neighbors(self):
        from aspf.adapters import NetworkXAdapter
        a = NetworkXAdapter(self._make_graph())
        nbrs = dict(a.neighbors(0))
        assert nbrs[1] == pytest.approx(2.5)
        assert nbrs[2] == pytest.approx(1.0)

    def test_isolated_vertex_neighbors(self):
        import networkx as nx
        from aspf.adapters import NetworkXAdapter
        G = nx.DiGraph()
        G.add_node(99)
        a = NetworkXAdapter(G)
        assert list(a.neighbors(99)) == []

    def test_default_weight_1(self):
        import networkx as nx
        from aspf.adapters import NetworkXAdapter
        G = nx.DiGraph()
        G.add_edge(0, 1)  # no weight attr
        a = NetworkXAdapter(G)
        nbrs = list(a.neighbors(0))
        assert len(nbrs) == 1
        assert nbrs[0] == (1, 1)

    def test_metadata_from_graph_attrs(self):
        import networkx as nx
        from aspf.adapters import NetworkXAdapter
        G = nx.DiGraph(unweighted=True, name="test")
        a = NetworkXAdapter(G)
        m = a.metadata()
        assert m.get("unweighted") is True
        assert m.get("name") == "test"

    def test_zero_copy(self):
        import networkx as nx
        from aspf.adapters import NetworkXAdapter
        G = nx.DiGraph()
        G.add_edge(0, 1, weight=1.0)
        a = NetworkXAdapter(G)
        assert a.vertex_count() == 2
        # Mutate G – adapter should reflect the change.
        G.add_node(5)
        assert a.vertex_count() == 3
