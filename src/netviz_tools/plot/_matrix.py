"""Adjacency-matrix heatmap, ordered so that groups show up as blocks."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._data import PlotData, Selector, to_frames
from netviz_tools.plot._render import resolve_focus
from netviz_tools.plot._resolve import NodeSpec, auto_size, node_values, select_top
from netviz_tools.plot._style import (
    EN_DASH,
    SEQUENTIAL,
    TEXT_COLOR,
    base_layout,
    default_title,
    fmt,
    nice_ticks,
    wants_log,
)

__all__ = ["adjacency"]


def adjacency(
    data: PlotData,
    *,
    sort_by: NodeSpec = "community",
    top_n: int | None = 50,
    focus: Hashable | Iterable[Hashable] | None = None,
    time: Selector = None,
    category: Selector = None,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 700,
    partition: pd.Series | None = None,
    seed: int = 42,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Draw the weighted adjacency matrix as a heatmap.

    Rows are sources and columns targets. Ordering the nodes by community
    (the default) makes dense groups show up as blocks along the diagonal;
    thin lines mark the group boundaries. Cells without an edge are blank, and
    colours are log-scaled when weights span more than two orders of
    magnitude. A matrix stays readable where a node-link diagram of a dense
    graph turns into a hairball.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, or a mapping from period to graph.
        One period only.
    sort_by
        Node order: ``"community"`` (default), any categorical node attribute
        or mapping (grouped, groups ordered by size), any numeric attribute or
        metric (largest first), or ``None`` (largest nodes first).
    top_n
        Keep the ``top_n`` nodes with the largest strength (degree when edges
        have no weight). ``None`` keeps all.
    focus
        Node or nodes that are always kept; their row and column labels are
        drawn in bold.
    time, category, labels, title, height, seed, node_attrs
        As in :func:`~netviz_tools.plot.network`.
    partition
        Community id per node for ``sort_by="community"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    g = frames.single("adjacency")
    lab = frames.labels
    focus_nodes = resolve_focus(g, focus)
    size = node_values(g, auto_size(g), arg="size_by", labels=lab)
    assert size is not None
    nodes = select_top(size.series, top_n)
    nodes += [f for f in focus_nodes if f not in set(nodes)]
    sizes = size.series.reindex(nodes).astype(float)
    order, groups = resolve_order(g, nodes, sizes, sort_by, lab, partition, seed)
    idx = {n: i for i, n in enumerate(order)}
    mat = np.full((len(order), len(order)), np.nan)
    directed = g.is_directed()
    for u, v, w in g.subgraph(order).edges(data="weight", default=1.0):
        mat[idx[u], idx[v]] = float(w)
        if not directed:
            mat[idx[v], idx[u]] = float(w)
    finite = mat[np.isfinite(mat)]
    log = wants_log(finite)
    z = np.log10(1.0 + mat) if log else mat
    names = [f"<b>{n}</b>" if n in focus_nodes else str(n) for n in order]
    unit = frames.unit
    if directed:
        row_title, col_title = lab["source"], lab["target"]

        def pair(i: int, j: int) -> str:
            return f"{row_title}: {order[i]}<br>{col_title}: {order[j]}"
    else:
        row_title = col_title = lab["node"]

        def pair(i: int, j: int) -> str:
            return f"{order[i]} {EN_DASH} {order[j]}"

    hover = [
        [
            f"{pair(i, j)}<br>{lab['weight']}: {fmt(mat[i, j], unit)}"
            if np.isfinite(mat[i, j])
            else ""
            for j in range(len(order))
        ]
        for i in range(len(order))
    ]
    colorbar: dict[str, Any] = {
        "title": {"text": f"{lab['weight']}" + (f" ({unit})" if unit else "")},
        "thickness": 14,
    }
    if log:
        values = nice_ticks(float(finite.min()), float(finite.max()), log=True)
        colorbar.update(
            tickvals=np.log10(1.0 + np.asarray(values)).tolist(),
            ticktext=[fmt(v) for v in values],
        )
    fig = go.Figure(
        go.Heatmap(
            z=z.tolist(),
            x=names,
            y=names,
            text=hover,
            hoverinfo="text",
            colorscale=[list(p) for p in SEQUENTIAL],
            colorbar=colorbar,
            hoverongaps=False,
            xgap=0,
            ygap=0,
        )
    )
    shapes: list[dict[str, Any]] = []
    edge = len(order) - 0.5
    style = {"type": "line", "line": {"color": TEXT_COLOR, "width": 0.8}, "layer": "above"}
    for b in np.cumsum([len(members) for members in groups[:-1]]).tolist():
        shapes.append({**style, "x0": b - 0.5, "x1": b - 0.5, "y0": -0.5, "y1": edge})
        shapes.append({**style, "y0": b - 0.5, "y1": b - 0.5, "x0": -0.5, "x1": edge})
    layout = base_layout(title or default_title(g, "adjacency matrix"), height)
    tick_font = {"size": 9 if len(order) > 40 else 11}
    layout.update(
        shapes=shapes,
        xaxis={
            "title": {"text": col_title},
            "type": "category",
            "side": "top",
            "tickangle": -60,
            "tickfont": tick_font,
            "showgrid": False,
        },
        yaxis={
            "title": {"text": row_title},
            "type": "category",
            "autorange": "reversed",
            "tickfont": tick_font,
            "showgrid": False,
            "scaleanchor": "x",
        },
        margin={"l": 10, "r": 10, "t": 140, "b": 10},
        plot_bgcolor="#ffffff",
    )
    fig.update_layout(**layout)
    return fig


def resolve_order(
    g: nx.Graph[Any],
    nodes: list[Hashable],
    sizes: pd.Series,
    sort_by: NodeSpec,
    lab: Mapping[str, str],
    partition: pd.Series | None,
    seed: int,
) -> tuple[list[Hashable], list[list[Hashable]]]:
    """Order nodes for the matrix.

    Returns the order and, when ``sort_by`` is categorical, the groups in
    that order (largest total size first, nodes largest first within each).
    """
    if sort_by is None:
        return nodes, []
    resolved = node_values(g, sort_by, arg="sort_by", labels=lab, partition=partition, seed=seed)
    assert resolved is not None  # sort_by is not None
    vals = resolved.series.reindex(nodes)
    if not resolved.categorical:
        return select_top(vals.astype(float), None), []
    frame = pd.DataFrame(
        {
            "k": ["" if pd.isna(v) else str(v) for v in vals],
            "s": sizes.reindex(nodes).fillna(0.0).to_numpy(),
            "n": [str(n) for n in nodes],
        },
        index=range(len(nodes)),
    )
    frame["g"] = frame.groupby("k")["s"].transform("sum")
    frame = frame.sort_values(
        ["g", "k", "s", "n"], ascending=[False, True, False, True], kind="stable"
    )
    groups = [[nodes[i] for i in part.index] for _, part in frame.groupby("k", sort=False)]
    return [n for members in groups for n in members], groups
