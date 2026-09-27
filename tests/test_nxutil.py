from __future__ import annotations

from typing import Any

import networkx as nx

from netviz_tools._nxutil import as_simple, has_weights


def test_as_simple_passthrough_and_collapse() -> None:
    g: nx.Graph[Any] = nx.path_graph(3)
    assert as_simple(g) is g
    assert not has_weights(g)
    mg: nx.MultiDiGraph[Any] = nx.MultiDiGraph(name="m")
    mg.add_node("a", kind="x")
    mg.add_edge("a", "b", weight=2.0)
    mg.add_edge("a", "b", weight=3.0)
    mg.add_edge("a", "b")
    mg.add_edge("b", "a", weight=1.0)
    s = as_simple(mg)
    assert isinstance(s, nx.DiGraph)
    assert not s.is_multigraph()
    assert s["a"]["b"]["weight"] == 6.0
    assert s["b"]["a"]["weight"] == 1.0
    assert s.nodes["a"]["kind"] == "x"
    assert s.graph["collapsed_multigraph"] is True
    assert has_weights(s)
    ug = as_simple(nx.MultiGraph([(1, 2), (2, 1)]))
    assert type(ug) is nx.Graph
    assert ug[1][2]["weight"] == 2.0
