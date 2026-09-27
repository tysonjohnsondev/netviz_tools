"""Build NetworkX graphs from flow frames.

A graph represents exactly one *slice* of the data: one ``time`` value and one
``category``, unless you ask for an explicit aggregation. Every graph stores the
slice it came from in ``G.graph``, and every edge carries its amount in the
``"weight"`` attribute, which is the default weight name used by NetworkX
algorithms.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd

from netviz_tools._schema import FLOW_COLUMNS, FlowFrame
from netviz_tools.analysis import suggest
from netviz_tools.errors import MixedSliceError, MixedUnitError, SchemaError

__all__ = ["Aggregate", "build_graph", "graph_to_flows", "graphs_by"]

Aggregate: TypeAlias = Literal["sum", "mean"]
"""How to combine flows across slices: ``"sum"`` totals them; ``"mean"``
divides the total by the number of slices present, so a pair that is missing
from a slice counts as zero for that slice."""

_SLICE_COLUMNS = ("time", "category")


def _sorted(values: Iterable[Any]) -> list[Any]:
    items = [x.item() if isinstance(x, np.generic) else x for x in values]
    try:
        return sorted(items)
    except TypeError:
        return sorted(items, key=str)


def _describe(values: pd.Series) -> str:
    uniq = _sorted(values.dropna().unique())
    shown = ", ".join(map(str, uniq[:4])) + (", ..." if len(uniq) > 4 else "")
    return f"{len(uniq)} values of {values.name!r} ({shown})"


def _require_columns(flows: pd.DataFrame, cols: Iterable[str]) -> None:
    missing = [c for c in cols if c not in flows.columns]
    if missing:
        raise SchemaError([f"missing columns {missing}"])


def _select(flows: pd.DataFrame, col: str, wanted: object, note: str = "") -> pd.DataFrame:
    """Keep the rows whose ``col`` is ``wanted`` (a value or an iterable of values)."""
    if wanted is None:
        return flows
    _require_columns(flows, [col])
    if isinstance(wanted, str | bytes) or not isinstance(wanted, Iterable):
        values: list[Any] = [wanted]
    else:
        values = list(wanted)
    if not values:
        raise ValueError(f"{col}= needs at least one value")
    available = _sorted(flows[col].dropna().unique())
    present = set(available)
    for value in values:
        if value in present:
            continue
        msg = f"no flows with {col}={value!r}{note}."
        if isinstance(value, str):
            close = suggest(value, [x for x in available if isinstance(x, str)], n=3)
            if close:
                msg += " Did you mean: " + ", ".join(map(repr, close)) + "?"
        shown = ", ".join(map(repr, available[:10])) + (", ..." if len(available) > 10 else "")
        raise ValueError(f"{msg} Available values ({len(available)}): {shown}")
    return flows[flows[col].isin(values)]


def _labels(flows: pd.DataFrame) -> dict[str, str] | None:
    labels = flows.attrs.get("labels")
    return dict(labels) if isinstance(labels, Mapping) else None


def _filter(flows: pd.DataFrame, time: object, category: object) -> pd.DataFrame:
    out = _select(flows, "time", time)
    note = " among the selected time values" if time is not None else ""
    return _select(out, "category", category, note)


def build_graph(
    flows: FlowFrame,
    *,
    time: int | Iterable[int] | None = None,
    category: Hashable | Iterable[Hashable] | None = None,
    aggregate: Aggregate | None = None,
    directed: bool = True,
    self_loops: bool = False,
    node_attrs: pd.DataFrame | None = None,
) -> nx.DiGraph[Any] | nx.Graph[Any]:
    """Build a weighted graph from one slice of a flow frame.

    Parameters
    ----------
    flows
        A flow frame, or any table with ``source``, ``target`` and ``weight``
        columns.
    time
        Keep only flows with this ``time`` value, or with any of these values
        (for example ``range(2015, 2020)``), before building.
    category
        Keep only flows with this ``category`` value, or with any of these
        values, before building.
    aggregate
        Required when the (filtered) flows contain more than one ``time`` or
        ``category`` value. ``"sum"`` totals the flows; ``"mean"`` averages
        them over the slices present. Duplicate rows for the same pair inside
        one slice are always summed.
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
        ``time``, ``category`` and ``unit`` where they are known (a list of
        values when several were aggregated), ``aggregate``, ``n_slices``
        when aggregating, and ``labels`` (a copy of ``flows.attrs["labels"]``)
        when the flows carry display labels.

    Raises
    ------
    MixedSliceError
        If the flows span several ``time`` or ``category`` values and
        ``aggregate`` is ``None``.
    MixedUnitError
        If the flows have more than one ``unit``.
    ValueError
        If a ``time`` or ``category`` value matches no flows. The message
        lists the available values.
    SchemaError
        If a required column is missing or weights are negative or missing.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample()
    >>> g = nv.build_graph(flows, time=2021, category="Wheat")
    >>> g.graph["time"], g.graph["category"], g.graph["unit"]
    (2021, 'Wheat', 't')
    >>> g.graph["labels"]["source"]
    'Exporter'
    """
    labels = _labels(flows)
    _require_columns(flows, ("source", "target", "weight"))
    flows = _filter(flows, time, category)
    mixed = [c for c in _SLICE_COLUMNS if c in flows.columns and flows[c].nunique() > 1]
    if mixed and aggregate is None:
        detail = " and ".join(_describe(flows[c]) for c in mixed)
        raise MixedSliceError(
            f"flows contain {detail}. Select one slice with time= or category=, build one "
            "graph per slice with graphs_by(), or pass aggregate='sum' or aggregate='mean'."
        )
    if aggregate not in (None, "sum", "mean"):
        raise ValueError(f"aggregate must be 'sum', 'mean' or None, not {aggregate!r}")
    if "unit" in flows.columns and flows["unit"].nunique() > 1:
        raise MixedUnitError(
            f"flows contain {_describe(flows['unit'])}; filter to one unit before building a graph."
        )
    w = flows["weight"]
    if not pd.api.types.is_numeric_dtype(w):
        raise SchemaError([f"column 'weight' must be numeric, not {w.dtype}"])
    if w.isna().any() or (w < 0).any():
        raise SchemaError(["column 'weight' has missing or negative values"])

    src = flows["source"]
    tgt = flows["target"]
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
    g.graph.update(_slice_meta(flows, aggregate, n_slices))
    if labels is not None:
        g.graph["labels"] = labels
    if node_attrs is not None:
        present_nodes = node_attrs.index.intersection(pd.Index(list(g.nodes)))
        attrs = node_attrs.loc[present_nodes]
        nx.set_node_attributes(g, attrs.to_dict(orient="index"))
    return g


def _slice_meta(flows: pd.DataFrame, aggregate: Aggregate | None, n_slices: int) -> dict[str, Any]:
    meta: dict[str, Any] = {"aggregate": aggregate}
    for col in ("unit", *_SLICE_COLUMNS):
        if col not in flows.columns:
            continue
        uniq = flows[col].dropna().unique()
        if len(uniq) == 1:
            value = uniq[0]
            meta[col] = value.item() if isinstance(value, np.generic) else value
        elif len(uniq) > 1:
            meta[col] = _sorted(uniq)
    if aggregate is not None:
        meta["n_slices"] = n_slices
    return meta


def graphs_by(
    flows: FlowFrame,
    by: str | Sequence[str] = ("time", "category"),
    **kwargs: Any,
) -> dict[Hashable, nx.DiGraph[Any] | nx.Graph[Any]]:
    """Build one graph per group of a flow frame.

    Parameters
    ----------
    flows
        A flow frame.
    by
        Column or columns to group on. The default builds one graph per
        (time, category) slice.
    **kwargs
        Passed to :func:`build_graph`. ``time=`` and ``category=`` filters are
        applied to the whole frame before grouping. If you group by ``"time"``
        only and the flows contain several categories, pass ``aggregate=`` or
        ``category=`` here.

    Returns
    -------
    dict
        Keys are group values in sorted order: scalars when grouping on one
        column, tuples otherwise. Every graph carries ``flows.attrs["labels"]``
        in ``G.graph["labels"]`` when present.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(years=range(2019, 2023))
    >>> sorted(nv.graphs_by(flows, by="time", category="Wheat"))
    [2019, 2020, 2021, 2022]
    """
    cols = [by] if isinstance(by, str) else list(by)
    labels = _labels(flows)
    flows = _filter(flows, kwargs.pop("time", None), kwargs.pop("category", None))
    _require_columns(flows, cols)
    out: dict[Hashable, nx.DiGraph[Any] | nx.Graph[Any]] = {}
    for key, group in flows.groupby(cols, sort=True, observed=True):
        k: Hashable = key[0] if len(cols) == 1 else key
        if isinstance(k, np.generic):
            k = k.item()
        elif isinstance(k, tuple):
            k = tuple(x.item() if isinstance(x, np.generic) else x for x in k)
        g = build_graph(group, **kwargs)
        # groupby does not reliably keep DataFrame.attrs, so pass labels on explicitly.
        if labels is not None:
            g.graph["labels"] = dict(labels)
        out[k] = g
    return out


def graph_to_flows(g: nx.Graph[Any]) -> pd.DataFrame:
    """Convert a graph back into a flow table.

    Scalar slice metadata stored by :func:`build_graph` (``time``,
    ``category``, ``unit``) is added as constant columns, so a graph built
    from one slice converts back into a valid flow frame. ``G.graph["labels"]``,
    when present, is copied into ``df.attrs["labels"]``.

    Parameters
    ----------
    g
        A graph whose edges carry ``"weight"`` (missing weights count as 1).

    Returns
    -------
    pandas.DataFrame
        One row per edge with the columns of :data:`netviz_tools.FLOW_COLUMNS`
        that are known, sorted by ``source`` and ``target``.
    """
    rows = list(g.edges(data="weight", default=1.0))
    df = pd.DataFrame(rows, columns=["source", "target", "weight"])
    df["weight"] = df["weight"].astype(float)
    for col in ("time", "category", "unit"):
        value = g.graph.get(col)
        if value is not None and not isinstance(value, list):
            df[col] = value
    if "time" in df.columns:
        df["time"] = df["time"].astype("int64")
    ordered = [c for c in FLOW_COLUMNS if c in df.columns]
    out = df[ordered].sort_values(["source", "target"], ignore_index=True)
    labels = g.graph.get("labels")
    if isinstance(labels, Mapping):
        out.attrs["labels"] = dict(labels)
    return out
