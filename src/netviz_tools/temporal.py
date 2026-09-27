"""Track graph-level and node-level metrics across a sequence of graphs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import Any, Final, Literal, TypeAlias, cast, get_args

import networkx as nx
import numpy as np
import pandas as pd

from netviz_tools.errors import UnknownMetricError
from netviz_tools.metrics import CentralityKind, centrality

__all__ = ["GraphMetric", "centrality_series", "graph_summary", "metric_series"]

GraphMetric: TypeAlias = Literal[
    "n_nodes",
    "n_edges",
    "density",
    "total_weight",
    "mean_strength",
    "reciprocity",
    "out_strength_hhi",
    "in_strength_hhi",
]
"""Supported graph-level metrics.

* ``total_weight``: sum of edge weights (for trade, world volume).
* ``mean_strength``: mean total strength per node.
* ``reciprocity``: share of edges whose reverse edge also exists.
* ``out_strength_hhi`` / ``in_strength_hhi``: Herfindahl-Hirschman index of
  out-strength (supplier concentration) or in-strength (buyer concentration)
  shares, between ``1 / n`` and 1.
"""

DEFAULT_METRICS: Final[tuple[GraphMetric, ...]] = (
    "n_nodes",
    "n_edges",
    "density",
    "total_weight",
    "out_strength_hhi",
)


def _hhi(values: Iterable[float]) -> float:
    arr = np.fromiter(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return float("nan")
    shares = arr / total
    return float((shares**2).sum())


def _graph_metric(g: nx.Graph[Any], name: str) -> float:
    n = g.number_of_nodes()
    if name == "n_nodes":
        return float(n)
    if name == "n_edges":
        return float(g.number_of_edges())
    if name == "density":
        return float(nx.density(g)) if n > 1 else 0.0
    if name == "total_weight":
        return float(sum(w for *_, w in g.edges(data="weight", default=1.0)))
    if name == "mean_strength":
        return float(np.mean([s for _, s in g.degree(weight="weight")])) if n else float("nan")
    if name == "reciprocity":
        if not g.is_directed() or g.number_of_edges() == 0:
            return float("nan")
        # nx.reciprocity returns a float when no nodes are given.
        return float(cast("float", nx.reciprocity(g)))
    if name == "out_strength_hhi":
        deg = g.out_degree(weight="weight") if g.is_directed() else g.degree(weight="weight")  # type: ignore[attr-defined]
        return _hhi(s for _, s in deg)
    # in_strength_hhi
    deg = g.in_degree(weight="weight") if g.is_directed() else g.degree(weight="weight")  # type: ignore[attr-defined]
    return _hhi(s for _, s in deg)


def graph_summary(
    g: nx.Graph[Any],
    metrics: Sequence[GraphMetric] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Compute graph-level metrics for a single graph.

    Parameters
    ----------
    g
        A graph whose edges carry ``"weight"``.
    metrics
        Metric names; see :data:`GraphMetric`.

    Returns
    -------
    dict
        Metric name to value.

    Raises
    ------
    UnknownMetricError
        If a metric name is not supported.
    """
    valid = get_args(GraphMetric)
    bad = [m for m in metrics if m not in valid]
    if bad:
        raise UnknownMetricError(f"unknown graph metric(s) {bad}; choose from {list(valid)}")
    return {m: _graph_metric(g, m) for m in metrics}


def _index(keys: list[Hashable], name: str | Sequence[str] | None) -> pd.Index:
    if keys and all(isinstance(k, tuple) for k in keys):
        names = [name] if isinstance(name, str) else name
        return pd.MultiIndex.from_tuples(keys, names=names)  # type: ignore[arg-type]
    return pd.Index(keys, name=name if isinstance(name, str) or name is None else name[0])


def metric_series(
    graphs: Mapping[Hashable, nx.Graph[Any]],
    metrics: Sequence[GraphMetric] = DEFAULT_METRICS,
    *,
    custom: Mapping[str, Callable[[nx.Graph[Any]], float]] | None = None,
    index_name: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute graph-level metrics for each graph in a mapping.

    Parameters
    ----------
    graphs
        Graphs keyed by period, for example the output of
        :func:`netviz_tools.graphs_by` with ``by="year"``.
    metrics
        Built-in metric names; see :data:`GraphMetric`.
    custom
        Extra metrics as ``{name: function}``. Each function takes a graph and
        returns a float.
    index_name
        Name for the index. Tuple keys produce a :class:`pandas.MultiIndex`.

    Returns
    -------
    pandas.DataFrame
        One row per graph (in the mapping's order), one column per metric.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(items="Wheat")
    >>> ts = nv.temporal.metric_series(nv.graphs_by(flows, by="year"), ["total_weight"])
    >>> ts.index.min() >= 2010
    True
    """
    keys = list(graphs)
    rows = []
    for key in keys:
        row: dict[str, float] = dict(graph_summary(graphs[key], metrics))
        for name, fn in (custom or {}).items():
            row[name] = float(fn(graphs[key]))
        rows.append(row)
    columns = [*metrics, *(custom or {})]
    return pd.DataFrame(rows, index=_index(keys, index_name), columns=columns)


def centrality_series(
    graphs: Mapping[Hashable, nx.Graph[Any]],
    kind: CentralityKind = "pagerank",
    *,
    nodes: Sequence[Hashable] | None = None,
    fill_value: float = 0.0,
    index_name: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """Track one centrality measure for each node across graphs.

    Parameters
    ----------
    graphs
        Graphs keyed by period.
    kind
        Centrality measure; see :data:`netviz_tools.metrics.CentralityKind`.
    nodes
        Nodes to keep as columns. Defaults to every node seen in any graph.
    fill_value
        Value for a node that is absent from a graph. The default of 0 means
        "no flows that period".
    index_name
        Name for the index.

    Returns
    -------
    pandas.DataFrame
        One row per graph, one column per node.
    """
    keys = list(graphs)
    series = {i: centrality(graphs[k], [kind])[kind] for i, k in enumerate(keys)}
    df = pd.DataFrame(series).T
    df.index = _index(keys, index_name)
    if nodes is not None:
        df = df.reindex(columns=list(nodes))
    df = df.fillna(fill_value)
    df.columns.name = "node"
    return df
