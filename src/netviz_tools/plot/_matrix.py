"""Adjacency-matrix heatmap, ordered so that groups show up as blocks."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._data import PlotData, Selector, to_frames
from netviz_tools.plot._render import resolve_focus
from netviz_tools.plot._resolve import NodeSpec, auto_size, node_values, select_top
from netviz_tools.plot._sankey import single_period
from netviz_tools.plot._style import (
    SEQUENTIAL,
    TEXT_COLOR,
    base_layout,
    default_title,
    fmt,
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
    g = single_period(frames, "adjacency")
    lab = frames.labels
    focus_nodes = resolve_focus(g, focus)
    size = node_values(g, auto_size(g), arg="size_by", labels=lab)
    assert size is not None
    nodes = select_top(size.values, top_n)
    nodes += [f for f in focus_nodes if f not in set(nodes)]
    sizes = size.values.reindex(nodes).astype(float)
    groups: list[tuple[str, list[Hashable]]] = []
    order = resolve_order(g, nodes, sizes, sort_by, lab, partition, seed, groups)
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
    hover = [
        [
            f"{lab['source']}: {order[i]}<br>{lab['target']}: {order[j]}<br>"
            f"{lab['weight']}: {fmt(mat[i, j], unit)}"
            if np.isfinite(mat[i, j])
            else ""
            for j in range(len(order))
        ]
        for i in range(len(order))
    ]
    colorbar: dict[str, object] = {
        "title": {"text": f"{lab['weight']}" + (f" ({unit})" if unit else "")}
    }
    if log and finite.size:
        ticks = np.linspace(float(np.nanmin(z)), float(np.nanmax(z)), 5)
        colorbar.update(tickvals=ticks.tolist(), ticktext=[fmt(10**t - 1) for t in ticks])
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
    shapes = []
    pos = 0
    for _, members in groups[:-1]:
        pos += len(members)
        for axis in ("x", "y"):
            line = {"type": "line", "line": {"color": TEXT_COLOR, "width": 0.8}, "layer": "above"}
            if axis == "x":
                line.update(x0=pos - 0.5, x1=pos - 0.5, y0=-0.5, y1=len(order) - 0.5)
            else:
                line.update(y0=pos - 0.5, y1=pos - 0.5, x0=-0.5, x1=len(order) - 0.5)
            shapes.append(line)
    layout = base_layout(title or default_title(g, "adjacency matrix"), height)
    tick_font = {"size": 9 if len(order) > 40 else 11}
    layout.update(
        shapes=shapes,
        xaxis={
            "title": {"text": lab["target"]},
            "side": "top",
            "tickangle": -60,
            "tickfont": tick_font,
            "showgrid": False,
        },
        yaxis={
            "title": {"text": lab["source"]},
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
    g: object,
    nodes: list[Hashable],
    sizes: pd.Series,
    sort_by: NodeSpec,
    lab: Mapping[str, str],
    partition: pd.Series | None,
    seed: int,
    groups: list[tuple[str, list[Hashable]]],
) -> list[Hashable]:
    """Order nodes for the matrix; fill ``groups`` when the order is grouped."""
    import networkx as nx

    assert isinstance(g, nx.Graph)
    if sort_by is None:
        return nodes
    resolved = node_values(g, sort_by, arg="sort_by", labels=lab, partition=partition, seed=seed)
    assert resolved is not None
    vals = resolved.values.reindex(nodes)
    if not resolved.categorical:
        return select_top(vals.astype(float), None)
    keys = vals.map(lambda v: "" if pd.isna(v) else str(v))
    frame = pd.DataFrame(
        {
            "k": keys.to_numpy(),
            "s": sizes.reindex(nodes).fillna(0).to_numpy(),
            "n": [str(n) for n in nodes],
        },
        index=range(len(nodes)),
    )
    group_size = frame.groupby("k")["s"].transform("sum")
    frame["g"] = group_size
    frame = frame.sort_values(
        ["g", "k", "s", "n"], ascending=[False, True, False, True], kind="stable"
    )
    order = [nodes[i] for i in frame.index]
    for k, part in frame.groupby("k", sort=False):
        groups.append((str(k), [nodes[i] for i in part.index]))
    return order
