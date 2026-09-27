"""Node centrality and community detection for flow graphs."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Final, Literal, TypeAlias, get_args

import networkx as nx
import pandas as pd

from netviz_tools.errors import InsufficientDataError, UnknownMetricError

__all__ = [
    "CentralityKind",
    "CommunityMethod",
    "centrality",
    "communities",
    "community_graph",
    "modularity",
]

CentralityKind: TypeAlias = Literal[
    "in_strength",
    "out_strength",
    "strength",
    "in_degree",
    "out_degree",
    "degree",
    "pagerank",
    "reverse_pagerank",
    "betweenness",
]
"""Supported centrality measures.

* ``in_strength`` / ``out_strength`` / ``strength``: weighted in, out and
  total degree, in the graph's unit.
* ``in_degree`` / ``out_degree`` / ``degree``: number of partners.
* On a directed graph, ``degree`` and ``strength`` count incoming plus outgoing edges.
* ``pagerank``: weighted PageRank. A node ranks high when large flows arrive
  from nodes that themselves rank high (an important destination).
* ``reverse_pagerank``: PageRank on the reversed graph. A node ranks high when
  it sends large flows to nodes that themselves rank high (an important
  source). Equal to ``pagerank`` on undirected graphs.
* ``betweenness``: weighted betweenness, using ``1 / weight`` as the edge
  length so that large flows count as short, well-used routes.
"""

CommunityMethod: TypeAlias = Literal["louvain", "greedy_modularity"]

DEFAULT_KINDS: Final[tuple[CentralityKind, ...]] = (
    "in_strength",
    "out_strength",
    "pagerank",
    "betweenness",
)


def _strength(g: nx.Graph[Any], mode: str) -> dict[Any, float]:
    if g.is_directed() and mode == "in":
        return dict(g.in_degree(weight="weight"))  # type: ignore[attr-defined]
    if g.is_directed() and mode == "out":
        return dict(g.out_degree(weight="weight"))  # type: ignore[attr-defined]
    return dict(g.degree(weight="weight"))


def _degree(g: nx.Graph[Any], mode: str) -> dict[Any, float]:
    if g.is_directed() and mode == "in":
        return dict(g.in_degree())  # type: ignore[attr-defined]
    if g.is_directed() and mode == "out":
        return dict(g.out_degree())  # type: ignore[attr-defined]
    return dict(g.degree())


def _pagerank(g: nx.Graph[Any], alpha: float, *, reverse: bool) -> dict[Any, float]:
    if g.number_of_edges() == 0:
        return dict.fromkeys(g, 1.0 / max(len(g), 1))
    h = g.reverse(copy=False) if reverse and g.is_directed() else g  # type: ignore[attr-defined]
    return dict(nx.pagerank(h, alpha=alpha, weight="weight"))


def _betweenness(g: nx.Graph[Any], normalized: bool) -> dict[Any, float]:
    h: nx.Graph[Any] = g.__class__()
    h.add_nodes_from(g)
    h.add_weighted_edges_from(
        ((u, v, 1.0 / w) for u, v, w in g.edges(data="weight", default=1.0) if w > 0),
        weight="length",
    )
    return dict(nx.betweenness_centrality(h, weight="length", normalized=normalized))


def centrality(
    g: nx.Graph[Any],
    kinds: Sequence[CentralityKind] = DEFAULT_KINDS,
    *,
    normalized: bool = True,
    alpha: float = 0.85,
) -> pd.DataFrame:
    """Compute several centrality measures at once.

    Parameters
    ----------
    g
        A graph whose edges carry ``"weight"``, as built by
        :func:`netviz_tools.build_graph`.
    kinds
        Measures to compute, in column order. See :data:`CentralityKind`.
    normalized
        Normalize betweenness by the number of node pairs.
    alpha
        PageRank damping factor.

    Returns
    -------
    pandas.DataFrame
        One row per node, one column per measure, sorted by the first measure
        in descending order. The index is named ``"node"``.

    Raises
    ------
    UnknownMetricError
        If a name in ``kinds`` is not supported.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> g = nv.build_graph(nv.datasets.faostat.load_sample(items="Wheat", years=2021))
    >>> nv.metrics.centrality(g, ["out_strength", "pagerank"]).columns.tolist()
    ['out_strength', 'pagerank']
    """
    valid = get_args(CentralityKind)
    bad = [k for k in kinds if k not in valid]
    if bad:
        raise UnknownMetricError(f"unknown centrality kind(s) {bad}; choose from {list(valid)}")
    compute: dict[str, Callable[[], dict[Any, float]]] = {
        "in_strength": lambda: _strength(g, "in"),
        "out_strength": lambda: _strength(g, "out"),
        "strength": lambda: _strength(g, "all"),
        "in_degree": lambda: _degree(g, "in"),
        "out_degree": lambda: _degree(g, "out"),
        "degree": lambda: _degree(g, "all"),
        "pagerank": lambda: _pagerank(g, alpha, reverse=False),
        "reverse_pagerank": lambda: _pagerank(g, alpha, reverse=True),
        "betweenness": lambda: _betweenness(g, normalized),
    }
    nodes = list(g.nodes)
    data = {k: pd.Series(compute[k](), dtype=float).reindex(nodes) for k in kinds}
    df = pd.DataFrame(data, index=pd.Index(nodes, name="node"))
    if kinds:
        df = df.sort_values(kinds[0], ascending=False, kind="stable")
    return df


def communities(
    g: nx.Graph[Any],
    method: CommunityMethod = "louvain",
    *,
    seed: int = 0,
    resolution: float = 1.0,
) -> pd.Series:
    """Partition the nodes of a graph into communities.

    Parameters
    ----------
    g
        A graph whose edges carry ``"weight"``. Directed graphs use the
        directed form of modularity.
    method
        ``"louvain"`` (default) or ``"greedy_modularity"`` (Clauset, Newman
        and Moore), both from NetworkX.
    seed
        Random seed for Louvain, so results are reproducible.
    resolution
        Modularity resolution. Values above 1 give more, smaller communities.

    Returns
    -------
    pandas.Series
        Community id per node, named ``"community"``. Ids are ``0, 1, ...`` in
        order of decreasing community size, with ties broken by the smallest
        member label, so the labelling is stable across runs.

    Raises
    ------
    UnknownMetricError
        If ``method`` is not supported.
    InsufficientDataError
        If the graph has no edges.
    """
    if method not in get_args(CommunityMethod):
        raise UnknownMetricError(
            f"unknown community method {method!r}; choose from {list(get_args(CommunityMethod))}"
        )
    if g.number_of_edges() == 0:
        raise InsufficientDataError("community detection needs a graph with at least one edge")
    parts: list[set[Any]] | list[frozenset[Any]]
    if method == "louvain":
        parts = nx.community.louvain_communities(
            g, weight="weight", resolution=resolution, seed=seed
        )
    else:
        parts = nx.community.greedy_modularity_communities(
            g, weight="weight", resolution=resolution
        )
    ordered = sorted(parts, key=lambda c: (-len(c), min(map(str, c))))
    labels = {node: i for i, comm in enumerate(ordered) for node in comm}
    return pd.Series(labels, name="community", dtype="int64").rename_axis("node").sort_index()


def modularity(g: nx.Graph[Any], partition: pd.Series) -> float:
    """Return the weighted modularity of a partition.

    Parameters
    ----------
    g
        The graph that was partitioned.
    partition
        Community id per node, as returned by :func:`communities`.

    Returns
    -------
    float
        Modularity, between -0.5 and 1. Higher means denser within-community
        flows than expected by chance.
    """
    groups = [set(idx) for _, idx in partition.groupby(partition).groups.items()]
    return float(nx.community.modularity(g, groups, weight="weight"))


def community_graph(g: nx.Graph[Any], partition: pd.Series) -> nx.DiGraph[Any] | nx.Graph[Any]:
    """Collapse each community into a single node.

    Parameters
    ----------
    g
        The graph that was partitioned.
    partition
        Community id per node, as returned by :func:`communities`.

    Returns
    -------
    networkx.DiGraph or networkx.Graph
        One node per community with attributes ``size`` and ``members``. Edge
        ``weight`` is the total flow between communities; flows inside a
        community become a self-loop.
    """
    q: nx.DiGraph[Any] | nx.Graph[Any] = nx.DiGraph() if g.is_directed() else nx.Graph()
    members: dict[int, list[Any]] = {}
    for node, cid in partition.items():
        members.setdefault(int(cid), []).append(node)
    for cid, nodes in sorted(members.items()):
        q.add_node(cid, size=len(nodes), members=sorted(nodes, key=str))
    lookup = partition.to_dict()
    frame = pd.DataFrame(
        [(lookup[u], lookup[v], w) for u, v, w in g.edges(data="weight", default=1.0)],
        columns=["u", "v", "w"],
    )
    if not frame.empty:
        totals = frame.groupby(["u", "v"], as_index=False)["w"].sum()
        q.add_weighted_edges_from(
            zip(
                totals["u"].astype(int),
                totals["v"].astype(int),
                totals["w"].astype(float),
                strict=True,
            )
        )
    return q
