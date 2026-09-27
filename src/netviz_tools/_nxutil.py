"""Small helpers for accepting any NetworkX graph (private)."""

from __future__ import annotations

from typing import Any

import networkx as nx


def has_weights(g: nx.Graph[Any], weight: str = "weight") -> bool:
    """Return True if at least one edge carries the ``weight`` attribute."""
    return any(weight in d for *_, d in g.edges(data=True))


def as_simple(g: nx.Graph[Any], weight: str = "weight") -> nx.Graph[Any]:
    """Return ``g`` without parallel edges.

    Simple graphs are returned unchanged (not copied). A
    :class:`networkx.MultiGraph` or :class:`networkx.MultiDiGraph` is collapsed
    into a :class:`networkx.Graph` or :class:`networkx.DiGraph` whose edge
    ``weight`` is the sum over parallel edges (an edge without the attribute
    counts as 1). Node attributes and ``G.graph`` are kept, and
    ``G.graph["collapsed_multigraph"]`` is set to True.

    Parameters
    ----------
    g
        Any NetworkX graph.
    weight
        Edge attribute to sum.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
    """
    if not g.is_multigraph():
        return g
    out: nx.Graph[Any] = nx.DiGraph() if g.is_directed() else nx.Graph()
    out.graph.update(g.graph)
    out.graph["collapsed_multigraph"] = True
    out.add_nodes_from(g.nodes(data=True))
    for u, v, d in g.edges(data=True):
        w = float(d.get(weight, 1.0))
        if out.has_edge(u, v):
            out[u][v][weight] += w
        else:
            out.add_edge(u, v, **{weight: w})
    return out
