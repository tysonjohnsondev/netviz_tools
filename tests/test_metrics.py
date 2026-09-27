from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import InsufficientDataError, UnknownMetricError


@pytest.fixture
def small(flows: pd.DataFrame) -> nx.DiGraph[Any]:
    g = nv.build_graph(flows[flows.year == 2020])
    assert isinstance(g, nx.DiGraph)
    return g


def test_centrality_default_columns_and_order(small: nx.DiGraph[Any]) -> None:
    df = nv.metrics.centrality(small)
    assert df.columns.tolist() == ["in_strength", "out_strength", "pagerank", "betweenness"]
    assert df.index.name == "node"
    assert df["in_strength"].is_monotonic_decreasing
    assert df.loc["A", "out_strength"] == 15.0
    assert df.loc["C", "in_strength"] == 7.0
    assert np.isclose(df["pagerank"].sum(), 1.0)


def test_all_kinds(small: nx.DiGraph[Any]) -> None:
    kinds = ["strength", "in_degree", "out_degree", "degree", "reverse_pagerank"]
    df = nv.metrics.centrality(small, kinds)  # type: ignore[arg-type]
    assert df.loc["A", "strength"] == 16.0
    assert df.loc["A", "out_degree"] == 2
    assert df.loc["A", "degree"] == 3
    # A sends the most, so it is the most important source.
    assert df["reverse_pagerank"].idxmax() == "A"


def test_betweenness_uses_inverse_weights() -> None:
    g: nx.DiGraph[str] = nx.DiGraph()
    # Two routes from s to t: via a (large flows, short) or via b (small flows, long).
    g.add_weighted_edges_from(
        [("s", "a", 100.0), ("a", "t", 100.0), ("s", "b", 1.0), ("b", "t", 1.0)]
    )
    df = nv.metrics.centrality(g, ["betweenness"], normalized=False)
    assert df.loc["a", "betweenness"] == 1.0
    assert df.loc["b", "betweenness"] == 0.0


def test_undirected_and_empty_graphs() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_weighted_edges_from([("a", "b", 2.0), ("b", "c", 1.0)])
    df = nv.metrics.centrality(g, ["in_strength", "out_degree", "pagerank", "reverse_pagerank"])
    assert df.loc["b", "in_strength"] == 3.0
    assert np.allclose(df["pagerank"], df["reverse_pagerank"])
    empty: nx.DiGraph[str] = nx.DiGraph()
    empty.add_nodes_from(["x", "y"])
    pr = nv.metrics.centrality(empty, ["pagerank"])
    assert pr["pagerank"].tolist() == [0.5, 0.5]
    assert nv.metrics.centrality(empty, []).shape == (2, 0)


def test_unknown_kind(small: nx.DiGraph[Any]) -> None:
    with pytest.raises(UnknownMetricError, match="closeness"):
        nv.metrics.centrality(small, ["closeness"])  # type: ignore[list-item]


def test_communities_are_seeded_and_labelled_by_size(wheat_graph: nx.DiGraph[Any]) -> None:
    a = nv.metrics.communities(wheat_graph, seed=1)
    b = nv.metrics.communities(wheat_graph, seed=1)
    pd.testing.assert_series_equal(a, b)
    assert a.name == "community"
    assert set(a.index) == set(wheat_graph.nodes)
    sizes = a.value_counts().sort_index()
    assert sizes.is_monotonic_decreasing
    assert 0.2 < nv.metrics.modularity(wheat_graph, a) < 1.0


def test_two_cliques_split() -> None:
    g: nx.DiGraph[str] = nx.DiGraph()
    for group in ("abc", "xyz"):
        for u in group:
            for v in group:
                if u != v:
                    g.add_edge(u, v, weight=10.0)
    g.add_edge("a", "x", weight=0.1)
    methods: tuple[nv.metrics.CommunityMethod, ...] = ("louvain", "greedy_modularity")
    for method in methods:
        part = nv.metrics.communities(g, method)
        assert part["a"] == part["b"] == part["c"]
        assert part["x"] == part["y"] == part["z"]
        assert part["a"] != part["x"]
    q = nv.metrics.community_graph(g, part)
    assert q.number_of_nodes() == 2
    assert q.nodes[0]["size"] == 3
    assert q[part["a"]][part["x"]]["weight"] == pytest.approx(0.1)
    assert q[0][0]["weight"] == pytest.approx(60.0)


def test_community_graph_undirected_and_empty() -> None:
    g: nx.Graph[str] = nx.Graph()
    g.add_nodes_from("ab")
    part = pd.Series({"a": 0, "b": 1})
    q = nv.metrics.community_graph(g, part)
    assert not q.is_directed()
    assert q.number_of_edges() == 0


def test_community_errors(small: nx.DiGraph[Any]) -> None:
    with pytest.raises(UnknownMetricError):
        nv.metrics.communities(small, "girvan_newman")  # type: ignore[arg-type]
    empty: nx.DiGraph[str] = nx.DiGraph()
    empty.add_node("a")
    with pytest.raises(InsufficientDataError):
        nv.metrics.communities(empty)
