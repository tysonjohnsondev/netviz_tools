"""Accept a graph, a flow table, or a mapping of graphs as plot input (private)."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, TypeAlias

import networkx as nx
import pandas as pd

from netviz_tools._nxutil import as_simple, has_weights
from netviz_tools.errors import MixedSliceError
from netviz_tools.plot._style import graph_labels, merge_labels, unit_of

PlotData: TypeAlias = "nx.Graph[Any] | pd.DataFrame | Mapping[Any, nx.Graph[Any]]"
"""What plot functions accept: a NetworkX graph, a flow table (see
:data:`netviz_tools.FLOW_COLUMNS`), or a mapping from period to graph such as
the output of ``graphs_by(flows, by="time")``."""

Selector: TypeAlias = "Hashable | Iterable[Hashable] | None"
"""A single value, several values, or ``None`` for all."""


def is_flow_table(data: object) -> bool:
    """Return True for a DataFrame with ``source`` and ``target`` columns."""
    return isinstance(data, pd.DataFrame) and {"source", "target"} <= set(data.columns)


def as_list(sel: Selector) -> list[Hashable] | None:
    """Normalize a selector into a list (``None`` stays ``None``)."""
    if sel is None:
        return None
    if isinstance(sel, str | bytes) or not isinstance(sel, Iterable):
        return [sel]
    return list(sel)


def check_known(wanted: list[Hashable], available: Iterable[Hashable], arg: str) -> None:
    """Raise ValueError naming the wanted values that are not available."""
    avail = list(available)
    missing = [w for w in wanted if w not in avail]
    if missing:
        shown = ", ".join(map(str, avail[:12])) + (", ..." if len(avail) > 12 else "")
        raise ValueError(f"{arg}={missing!r} not found; available: {shown}")


@dataclass(frozen=True, eq=False)
class Frames:
    """One simple graph per period, plus display labels and the unit."""

    graphs: dict[Hashable, nx.Graph[Any]]
    labels: dict[str, str]
    unit: str

    @property
    def animated(self) -> bool:
        """True when there is more than one period."""
        return len(self.graphs) > 1

    @property
    def first(self) -> nx.Graph[Any]:
        """The first graph."""
        return next(iter(self.graphs.values()))

    @cached_property
    def union(self) -> nx.Graph[Any]:
        """All periods combined into one graph with summed edge weights.

        Edges carry ``weight`` only when at least one period has weights (an
        edge without one then counts as 1).
        """
        if not self.animated:
            return self.first
        first = self.first
        out: nx.Graph[Any] = nx.DiGraph() if first.is_directed() else nx.Graph()
        out.graph.update(first.graph)
        out.graph["time"] = list(self.graphs)
        out.graph["aggregate"] = "sum"
        weighted = any(has_weights(g) for g in self.graphs.values())
        for g in self.graphs.values():
            for n, d in g.nodes(data=True):
                if n in out:
                    out.nodes[n].update(d)
                else:
                    out.add_node(n, **d)
            for u, v, w in g.edges(data="weight", default=1.0):
                if not weighted:
                    out.add_edge(u, v)
                elif out.has_edge(u, v):
                    out[u][v]["weight"] += float(w)
                else:
                    out.add_edge(u, v, weight=float(w))
        return out

    def single(self, fn: str) -> nx.Graph[Any]:
        """Return the only graph, or explain how to pick one period."""
        if self.animated:
            keys = list(self.graphs)
            raise ValueError(
                f"{fn} draws one period but the data has {len(keys)} "
                f"({keys[0]} to {keys[-1]}); pass time=<one period>"
            )
        return self.first


def to_frames(
    data: Any,
    *,
    time: Selector,
    category: Selector,
    labels: Mapping[str, str] | None,
    node_attrs: pd.DataFrame | None = None,
    directed: bool = True,
) -> Frames:
    """Normalize plot input into :class:`Frames`.

    Raises
    ------
    ValueError
        If ``time``/``category`` select values that do not exist, or are
        given for input that has no such dimension.
    MixedSliceError
        If a flow table still holds several categories after filtering.
    """
    times = as_list(time)
    cats = as_list(category)
    if isinstance(data, nx.Graph):
        if times is not None or cats is not None:
            raise ValueError(
                "time= and category= select slices of a flow table or a mapping of graphs; "
                "a single graph has one slice. Pass the flow table instead, or drop them."
            )
        if node_attrs is not None:
            raise ValueError("node_attrs= is only used with a flow table; set attributes on g")
        g = as_simple(data)
        return Frames({None: g}, merge_labels(graph_labels(g), labels), unit_of(g))
    if is_flow_table(data):
        return _frames_from_flows(data, times, cats, labels, node_attrs, directed)
    if isinstance(data, Mapping):
        if cats is not None:
            raise ValueError("category= needs a flow table; filter the graphs yourself")
        keys = list(data)
        if times is not None:
            check_known(times, keys, "time")
            keys = [k for k in keys if k in times]
        if not keys:
            raise ValueError("no graphs to plot")
        graphs = {k: as_simple(data[k]) for k in keys}
        first = next(iter(graphs.values()))
        return Frames(graphs, merge_labels(graph_labels(first), labels), unit_of(first))
    raise TypeError(
        "expected a networkx graph, a flow table with source/target columns, or a mapping "
        f"of graphs, not {type(data).__name__}"
    )


def _frames_from_flows(
    flows: pd.DataFrame,
    times: list[Hashable] | None,
    cats: list[Hashable] | None,
    labels: Mapping[str, str] | None,
    node_attrs: pd.DataFrame | None,
    directed: bool,
) -> Frames:
    from netviz_tools.graph import build_graph

    df = flows
    if cats is not None:
        if "category" not in df.columns:
            raise ValueError("category= given but the flow table has no 'category' column")
        check_known(cats, sorted(df["category"].unique(), key=str), "category")
        df = df[df["category"].isin(cats)]
    if "category" in df.columns and df["category"].nunique() > 1:
        shown = ", ".join(map(str, sorted(df["category"].unique(), key=str)[:6]))
        raise MixedSliceError(
            f"the flows contain {df['category'].nunique()} categories ({shown}, ...); "
            "pick one with category=..."
        )
    periods: list[Hashable] = (
        sorted(df["time"].unique().tolist()) if "time" in df.columns else [None]
    )
    if times is not None:
        if "time" not in df.columns:
            raise ValueError("time= given but the flow table has no 'time' column")
        check_known(times, periods, "time")
        periods = [p for p in periods if p in times]
    if not periods or df.empty:
        raise ValueError("no flows left to plot after filtering")
    graphs: dict[Hashable, nx.Graph[Any]] = {}
    for p in periods:
        part = df if p is None else df[df["time"] == p]
        part.attrs = dict(flows.attrs)
        graphs[p] = build_graph(part, directed=directed, node_attrs=node_attrs)
    first = next(iter(graphs.values()))
    lab = flows.attrs.get("labels")
    return Frames(
        graphs, merge_labels(lab if isinstance(lab, Mapping) else None, labels), unit_of(first)
    )


def long_table(data: Any, labels: Mapping[str, str] | None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a flow table and merged display labels.

    Accepts a flow table, or a mapping from period to graph (each graph
    becomes the rows of its period, with the graph's ``category`` and
    ``unit``).
    """
    if is_flow_table(data):
        lab = data.attrs.get("labels")
        return data, merge_labels(lab if isinstance(lab, Mapping) else None, labels)
    if isinstance(data, Mapping):
        rows = []
        first_labels: Mapping[str, str] | None = None
        for k, g in data.items():
            first_labels = first_labels or graph_labels(g)
            cat, unit = g.graph.get("category", ""), unit_of(g)
            rows += [
                (u, v, k, cat, float(w), unit) for u, v, w in g.edges(data="weight", default=1.0)
            ]
        df = pd.DataFrame(rows, columns=["source", "target", "time", "category", "weight", "unit"])
        return df, merge_labels(first_labels, labels)
    raise TypeError(
        "expected a flow table with source/target columns or a mapping from period to graph, "
        f"not {type(data).__name__}"
    )
