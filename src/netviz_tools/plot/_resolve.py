"""Turn ``color_by`` / ``size_by`` / ``edge_*_by`` arguments into values (private)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal, TypeAlias, cast, get_args

import networkx as nx
import numpy as np
import pandas as pd

from netviz_tools._nxutil import has_weights
from netviz_tools.errors import UnknownMetricError
from netviz_tools.metrics import CentralityKind, centrality, communities
from netviz_tools.plot._style import (
    CONTINENT_COLORS,
    MISSING_COLOR,
    MISSING_LABEL,
    OTHER_COLOR,
    OTHER_LABEL,
    PALETTE,
)

NodeMetric: TypeAlias = Literal[
    "degree",
    "in_degree",
    "out_degree",
    "strength",
    "in_strength",
    "out_strength",
    "pagerank",
    "reverse_pagerank",
    "betweenness",
    "community",
]
"""Metric names accepted by ``color_by``, ``size_by`` and ``sort_by``.

On a directed graph ``degree`` and ``strength`` count incoming plus outgoing
edges. ``community`` uses the ``partition=`` you pass, or Louvain communities
with a fixed seed.
"""

NodeSpec: TypeAlias = str | Mapping[Any, Any] | pd.Series | None
"""A node attribute name, a :data:`NodeMetric` name, ``"auto"``, a mapping or
Series from node to value, or ``None``."""

EdgeSpec: TypeAlias = str | Mapping[tuple[Any, Any], Any] | None
"""An edge attribute name (such as ``"weight"``), ``"auto"``, ``"source"`` or
``"target"`` (edges take the colour of that end node; ``edge_color_by`` only),
a mapping from ``(u, v)`` to value, or ``None``."""

METRICS: Final = get_args(NodeMetric)
_WEIGHT_METRICS: Final = {"strength", "in_strength", "out_strength"}


@dataclass(frozen=True, eq=False)
class NodeValues:
    """Resolved node values.

    ``metric`` is True when the values were computed from the graph (so they
    change from one period to the next); ``in_weight_units`` when they are
    measured in the unit of the edge weights (strengths).
    """

    series: pd.Series
    name: str
    label: str
    categorical: bool
    metric: bool = False
    in_weight_units: bool = False


@dataclass(frozen=True, eq=False)
class Categories:
    """Category label per item and colour per label, in legend order."""

    labels: pd.Series
    colors: dict[str, str]


def metric_label(name: str, labels: Mapping[str, str], *, directed: bool) -> str:
    """Return the display label of a metric or attribute name."""
    if name in labels:
        return labels[name]
    out, inn, weight = labels["out"], labels["in"], labels["weight"]
    derived = {
        "out_strength": out,
        "in_strength": inn,
        "strength": f"{out} + {inn.lower()}" if directed else f"Total {weight.lower()}",
        "degree": "Degree (in + out)" if directed else "Degree",
        "in_degree": "In-degree",
        "out_degree": "Out-degree",
        "pagerank": "PageRank",
        "reverse_pagerank": "Reverse PageRank",
        "betweenness": "Betweenness",
        "community": "Community",
    }
    return derived.get(name, name.replace("_", " ").capitalize())


def is_numeric(values: pd.Series) -> bool:
    """Return True when every present value is a real number (booleans are not)."""
    present = values.dropna()
    if present.empty:
        return False
    return all(isinstance(v, Real) and not isinstance(v, bool | np.bool_) for v in present)


def auto_color(g: nx.Graph[Any]) -> str | None:
    """``color_by="auto"``: communities when the graph has edges, else no colour."""
    return "community" if g.number_of_edges() else None


def auto_size(g: nx.Graph[Any]) -> str:
    """``size_by="auto"``: strength when edges have weights, else degree."""
    return "strength" if has_weights(g) else "degree"


def compute_metric(
    g: nx.Graph[Any], name: str, *, partition: pd.Series | None = None, seed: int = 0
) -> pd.Series:
    """Compute one :data:`NodeMetric` for every node of ``g``."""
    nodes = list(g.nodes)
    if name == "community":
        if partition is not None:
            part = partition.reindex(nodes)
        elif g.number_of_edges():
            part = communities(g, seed=seed).reindex(nodes)
        else:
            part = pd.Series(0, index=nodes)
        return part
    df = centrality(g, [cast(CentralityKind, name)])
    return df[name].reindex(nodes)


def node_values(
    g: nx.Graph[Any],
    spec: NodeSpec,
    *,
    arg: str,
    labels: Mapping[str, str],
    partition: pd.Series | None = None,
    seed: int = 0,
) -> NodeValues | None:
    """Resolve a node spec on ``g``.

    A string is looked up first as a node attribute (present on at least one
    node), then as a metric name. Numeric attribute values are continuous;
    anything else, and ``"community"``, is categorical.
    """
    if spec is None:
        return None
    nodes = list(g.nodes)
    directed = g.is_directed()
    if isinstance(spec, str):
        if spec == "auto":
            chosen = auto_color(g) if arg == "color_by" else auto_size(g)
            if chosen is None:
                return None
            spec = chosen
        if any(spec in d for _, d in g.nodes(data=True)):
            values = pd.Series([g.nodes[n].get(spec) for n in nodes], index=nodes, dtype=object)
            numeric = is_numeric(values)
            if numeric:
                values = values.astype(float)
            label = metric_label(spec, labels, directed=directed)
            return NodeValues(values, spec, label, categorical=not numeric)
        if spec in METRICS:
            values = compute_metric(g, spec, partition=partition, seed=seed)
            return NodeValues(
                values,
                spec,
                metric_label(spec, labels, directed=directed),
                categorical=spec == "community",
                metric=True,
                in_weight_units=spec in _WEIGHT_METRICS,
            )
        attrs = sorted({k for _, d in g.nodes(data=True) for k in d})
        raise UnknownMetricError(
            f"{arg}={spec!r} is neither a node attribute nor a metric. "
            f"Node attributes: {attrs or 'none'}; metrics: {list(METRICS)}"
        )
    series = spec if isinstance(spec, pd.Series) else pd.Series(dict(spec), dtype=object)
    values = series.reindex(nodes)
    numeric = is_numeric(values)
    if numeric:
        values = values.astype(float)
    name = arg if series.name is None or series.name == "" else str(series.name)
    return NodeValues(
        values, name, labels.get(name, name.replace("_", " ")), categorical=not numeric
    )


def categories(values: pd.Series, *, name: str = "") -> Categories:
    """Assign colours to categorical values.

    Community ids are ordered numerically and labelled ``"Community 0"`` and
    so on; continents use :data:`CONTINENT_COLORS`; booleans are ordered
    False, True; other values by decreasing frequency, ties by name. The
    first eight categories get :data:`PALETTE` colours, the rest are shown as
    ``"Other"``; missing values as ``"No value"``.
    """
    if name == "community":
        texts = [None if pd.isna(v) else f"Community {int(v)}" for v in values]
        ordered = [f"Community {int(v)}" for v in sorted(values.dropna().unique())]
    else:
        texts = [None if pd.isna(v) or v == "" else str(v) for v in values]
        uniq = {t for t in texts if t is not None}
        if uniq and uniq <= set(CONTINENT_COLORS):
            ordered = [c for c in CONTINENT_COLORS if c in uniq]
        elif uniq and uniq <= {"True", "False"}:
            ordered = [c for c in ("False", "True") if c in uniq]
        else:
            counts = Counter(t for t in texts if t is not None)
            ordered = sorted(counts, key=lambda c: (-counts[c], c))
    if ordered and set(ordered) <= set(CONTINENT_COLORS):
        colors = {c: CONTINENT_COLORS[c] for c in ordered}
    else:
        colors = dict(zip(ordered[: len(PALETTE)], PALETTE, strict=False))
    labels = [MISSING_LABEL if c is None else (c if c in colors else OTHER_LABEL) for c in texts]
    if OTHER_LABEL in labels:
        colors[OTHER_LABEL] = OTHER_COLOR
    if MISSING_LABEL in labels:
        colors[MISSING_LABEL] = MISSING_COLOR
    return Categories(pd.Series(labels, index=values.index, dtype=object), colors)


def edge_values(
    edges: Sequence[tuple[Hashable, Hashable, Mapping[str, Any]]],
    spec: EdgeSpec,
    *,
    arg: str,
    directed: bool,
) -> pd.Series | None:
    """Resolve an edge spec to one value per edge (positional index)."""
    if spec is None:
        return None
    if isinstance(spec, str):
        vals = [d.get(spec) for *_, d in edges]
        if edges and all(v is None for v in vals):
            present = sorted({k for *_, d in edges for k in d})
            raise UnknownMetricError(
                f"{arg}={spec!r} is not an edge attribute. Edge attributes: {present or 'none'}"
            )
        series = pd.Series(vals, dtype=object)
    else:
        lookup = dict(spec)

        def get(u: Hashable, v: Hashable) -> Any:
            if (u, v) in lookup:
                return lookup[(u, v)]
            return None if directed else lookup.get((v, u))

        series = pd.Series([get(u, v) for u, v, _ in edges], dtype=object)
    if is_numeric(series):
        return series.astype(float)
    return series


def select_top(values: pd.Series, n: int | None) -> list[Hashable]:
    """Order nodes by value, largest first (ties by label), keeping ``n``."""
    frame = pd.DataFrame(
        {
            "v": pd.to_numeric(values, errors="coerce").fillna(-np.inf).to_numpy(),
            "k": [str(x) for x in values.index],
        },
        index=range(len(values)),
    )
    order = frame.sort_values(["v", "k"], ascending=[False, True], kind="stable").index
    nodes = [values.index[i] for i in order]
    return nodes if n is None else nodes[:n]
