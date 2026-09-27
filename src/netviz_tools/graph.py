"""Build NetworkX graphs from flow frames.

A graph represents exactly one *slice* of the data: one year and one item,
unless you ask for an explicit aggregation. Every graph stores the slice it came
from in ``G.graph``, and every edge carries its amount in the ``"weight"``
attribute, which is the default weight name used by NetworkX algorithms.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd

from netviz_tools._schema import FlowFrame
from netviz_tools.errors import MixedSliceError, MixedUnitError, SchemaError

__all__ = ["Aggregate", "build_graph", "graph_to_flows", "graphs_by"]

Aggregate: TypeAlias = Literal["sum", "mean"]
"""How to combine flows across slices: ``"sum"`` totals them; ``"mean"``
divides the total by the number of slices present, so a pair that is missing
from a slice counts as zero for that slice."""

_SLICE_COLUMNS = ("year", "item")


def _describe(values: pd.Series) -> str:
    uniq = sorted(values.dropna().unique().tolist(), key=str)
    shown = ", ".join(map(str, uniq[:4])) + (", ..." if len(uniq) > 4 else "")
    return f"{len(uniq)} values of {values.name!r} ({shown})"


def _require_columns(flows: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in flows.columns]
    if missing:
        raise SchemaError([f"missing columns {missing}"])


def build_graph(
    flows: FlowFrame,
    source: str = "exporter",
    target: str = "importer",
    weight: str = "quantity",
    *,
    aggregate: Aggregate | None = None,
    directed: bool = True,
    self_loops: bool = False,
    node_attrs: pd.DataFrame | None = None,
) -> nx.DiGraph[Any] | nx.Graph[Any]:
    """Build a weighted graph from one slice of a flow frame.

    Parameters
    ----------
    flows
        A flow frame, or any table with the ``source``, ``target`` and
        ``weight`` columns.
    source, target, weight
        Column names for the edge endpoints and the edge amount.
    aggregate
        Required when ``flows`` contains more than one year or item. ``"sum"``
        totals the flows; ``"mean"`` averages them over the slices present.
        Duplicate rows for the same pair inside one slice are always summed.
    directed
        Return a :class:`networkx.DiGraph` (default). If ``False``, return a
        :class:`networkx.Graph` in which both directions of a pair are summed.
    self_loops
        Keep flows whose source equals their target. Dropped by default.
    node_attrs
        Optional table indexed by node label. Its columns become node
        attributes for the nodes present in the graph, for example
        :func:`netviz_tools.datasets.faostat.countries`.

    Returns
    -------
    networkx.DiGraph or networkx.Graph
        Edge amounts are stored under ``"weight"``. ``G.graph`` records
        ``weight`` (the source column), ``unit``, ``year``, ``item`` and
        ``aggregate`` where they are known.

    Raises
    ------
    MixedSliceError
        If ``flows`` spans several years or items and ``aggregate`` is ``None``.
    MixedUnitError
        If ``flows`` has more than one ``unit``.
    SchemaError
        If a required column is missing or weights are negative or missing.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(items="Wheat", years=2021)
    >>> g = nv.build_graph(flows)
    >>> g.graph["year"], g.graph["unit"]
    (2021, 't')
    """
    _require_columns(flows, (source, target, weight))
    mixed = [c for c in _SLICE_COLUMNS if c in flows.columns and flows[c].nunique() > 1]
    if mixed and aggregate is None:
        detail = " and ".join(_describe(flows[c]) for c in mixed)
        raise MixedSliceError(
            f"flows contain {detail}. Build one graph per slice with graphs_by(), "
            "filter the flows first, or pass aggregate='sum' or aggregate='mean'."
        )
    if aggregate not in (None, "sum", "mean"):
        raise ValueError(f"aggregate must be 'sum', 'mean' or None, not {aggregate!r}")
    if "unit" in flows.columns and flows["unit"].nunique() > 1:
        raise MixedUnitError(
            f"flows contain {_describe(flows['unit'])}; filter to one unit before building a graph."
        )
    w = flows[weight]
    if not pd.api.types.is_numeric_dtype(w):
        raise SchemaError([f"weight column {weight!r} must be numeric, not {w.dtype}"])
    if w.isna().any() or (w < 0).any():
        raise SchemaError([f"weight column {weight!r} has missing or negative values"])

    src = flows[source]
    tgt = flows[target]
    keep = w.to_numpy() > 0
    if not self_loops:
        keep &= (src != tgt).to_numpy()
    if not directed:
        swap = (src.astype(str) > tgt.astype(str)).to_numpy()
        src, tgt = src.where(~swap, tgt), tgt.where(~swap, src)
    edges = (
        pd.DataFrame(
            {"u": src.to_numpy()[keep], "v": tgt.to_numpy()[keep], "w": w.to_numpy()[keep]}
        )
        .groupby(["u", "v"], sort=True)["w"]
        .sum()
    )
    present = [c for c in _SLICE_COLUMNS if c in flows.columns]
    n_slices = max(len(flows[present].drop_duplicates()), 1) if present else 1
    if aggregate == "mean":
        edges = edges / n_slices

    g: nx.DiGraph[Any] | nx.Graph[Any] = nx.DiGraph() if directed else nx.Graph()
    u = edges.index.get_level_values(0)
    v = edges.index.get_level_values(1)
    g.add_weighted_edges_from(zip(u, v, edges.to_numpy(dtype=float), strict=True))
    g.graph.update(_slice_meta(flows, weight, aggregate, n_slices))
    if node_attrs is not None:
        present_nodes = node_attrs.index.intersection(pd.Index(list(g.nodes)))
        attrs = node_attrs.loc[present_nodes]
        nx.set_node_attributes(g, attrs.to_dict(orient="index"))
    return g


def _slice_meta(
    flows: pd.DataFrame, weight: str, aggregate: Aggregate | None, n_slices: int
) -> dict[str, Any]:
    meta: dict[str, Any] = {"weight": weight, "aggregate": aggregate}
    for col in ("unit", *_SLICE_COLUMNS):
        if col not in flows.columns:
            continue
        uniq = flows[col].dropna().unique()
        if len(uniq) == 1:
            value = uniq[0]
            meta[col] = value.item() if isinstance(value, np.generic) else value
        elif len(uniq) > 1:
            meta[col] = sorted((x.item() if isinstance(x, np.generic) else x) for x in uniq)
    if aggregate is not None:
        meta["n_slices"] = n_slices
    return meta


def graphs_by(
    flows: FlowFrame,
    by: str | Sequence[str] = ("year", "item"),
    **kwargs: Any,
) -> dict[Hashable, nx.DiGraph[Any] | nx.Graph[Any]]:
    """Build one graph per group of a flow frame.

    Parameters
    ----------
    flows
        A flow frame.
    by
        Column or columns to group on. The default builds one graph per
        (year, item) slice.
    **kwargs
        Passed to :func:`build_graph`. If you group by ``"year"`` only and the
        flows contain several items, pass ``aggregate=`` here.

    Returns
    -------
    dict
        Keys are group values in sorted order: scalars when grouping on one
        column, tuples otherwise.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(items="Wheat", years=range(2019, 2023))
    >>> sorted(nv.graphs_by(flows, by="year"))
    [2019, 2020, 2021, 2022]
    """
    cols = [by] if isinstance(by, str) else list(by)
    _require_columns(flows, cols)
    out: dict[Hashable, nx.DiGraph[Any] | nx.Graph[Any]] = {}
    for key, group in flows.groupby(cols, sort=True, observed=True):
        k: Hashable = key[0] if len(cols) == 1 else key
        if isinstance(k, np.generic):
            k = k.item()
        elif isinstance(k, tuple):
            k = tuple(x.item() if isinstance(x, np.generic) else x for x in k)
        out[k] = build_graph(group, **kwargs)
    return out


def graph_to_flows(
    g: nx.Graph[Any],
    source: str = "exporter",
    target: str = "importer",
    weight: str = "quantity",
) -> pd.DataFrame:
    """Convert a graph back into a flow table.

    Scalar slice metadata stored by :func:`build_graph` (``year``, ``item``,
    ``unit``) is added as constant columns, so a graph built from one slice
    converts back into a valid flow frame.

    Parameters
    ----------
    g
        A graph whose edges carry ``"weight"``.
    source, target, weight
        Output column names.

    Returns
    -------
    pandas.DataFrame
        One row per edge, sorted by source and target.
    """
    rows = list(g.edges(data="weight", default=1.0))
    df = pd.DataFrame(rows, columns=[source, target, weight])
    df[weight] = df[weight].astype(float)
    for col in ("year", "item", "unit"):
        value = g.graph.get(col)
        if value is not None and not isinstance(value, list):
            df[col] = value
    if "year" in df.columns:
        df["year"] = df["year"].astype("int64")
    ordered = [c for c in (source, target, "year", "item", weight, "unit") if c in df.columns]
    return df[ordered].sort_values([source, target], ignore_index=True)
