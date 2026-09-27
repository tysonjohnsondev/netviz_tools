"""Interactive node-link diagram of a flow graph."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._style import (
    EDGE_COLOR,
    SURFACE,
    categories,
    default_title,
    scale,
    top_nodes,
    unit_of,
)

__all__ = ["Layout", "community_layout", "network"]

Layout: TypeAlias = Literal["spring", "kamada_kawai", "circular", "community"]

Position: TypeAlias = Mapping[Hashable, tuple[float, float]]


def community_layout(
    g: nx.Graph[Any], partition: pd.Series, *, seed: int = 42, spread: float = 4.0
) -> dict[Hashable, tuple[float, float]]:
    """Place each community in its own cluster.

    Community centres are laid out with Kamada-Kawai on a ring of the
    communities, then each community is laid out around its centre, with a
    radius that grows with the square root of its size.

    Parameters
    ----------
    g
        The graph.
    partition
        Community id per node, as returned by
        :func:`netviz_tools.metrics.communities`.
    seed
        Seed for the within-community spring layouts.
    spread
        Distance scale between community centres.

    Returns
    -------
    dict
        Node to ``(x, y)``.
    """
    groups: dict[int, list[Hashable]] = {}
    for node, cid in partition.items():
        if node in g:
            groups.setdefault(int(cid), []).append(node)
    ring = nx.cycle_graph(len(groups)) if len(groups) > 2 else nx.path_graph(len(groups))
    centres = nx.kamada_kawai_layout(ring, scale=spread) if len(groups) > 1 else {0: np.zeros(2)}
    pos: dict[Hashable, tuple[float, float]] = {}
    for (_, members), centre in zip(sorted(groups.items()), centres.values(), strict=True):
        sub = g.subgraph(members)
        radius = 0.35 * math.sqrt(len(members))
        inner = nx.spring_layout(sub, seed=seed, center=centre, scale=radius, weight=None)
        pos.update({n: (float(x), float(y)) for n, (x, y) in inner.items()})
    return pos


def _layout(
    g: nx.Graph[Any], layout: Layout, partition: pd.Series | None, seed: int
) -> dict[Hashable, tuple[float, float]]:
    if layout == "spring":
        raw = nx.spring_layout(g, seed=seed, weight=None)
    elif layout == "kamada_kawai":
        raw = nx.kamada_kawai_layout(g, weight=None)
    elif layout == "circular":
        raw = nx.circular_layout(g)
    elif layout == "community":
        if partition is None:
            raise ValueError("layout='community' needs partition= (see metrics.communities)")
        return community_layout(g, partition, seed=seed)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return {n: (float(x), float(y)) for n, (x, y) in raw.items()}


def network(
    g: nx.Graph[Any],
    *,
    top_n: int | None = 30,
    layout: Layout = "spring",
    color_by: str | None = "continent",
    partition: pd.Series | None = None,
    label_top: int = 12,
    min_weight_quantile: float | None = None,
    seed: int = 42,
    title: str | None = None,
    height: int = 650,
) -> go.Figure:
    """Draw a flow graph as an interactive node-link diagram.

    Node size and edge width are log-scaled so that the largest flows do not
    hide the rest. Nodes are grouped into legend entries by ``color_by``, so
    identity never depends on colour alone; hover text gives exact amounts.

    Parameters
    ----------
    g
        A graph from :func:`netviz_tools.build_graph`.
    top_n
        Keep the ``top_n`` nodes with the largest total strength. ``None``
        keeps all nodes.
    layout
        ``"spring"`` (default), ``"kamada_kawai"``, ``"circular"``, or
        ``"community"`` (needs ``partition``).
    color_by
        Node attribute used for colour and legend groups (default
        ``"continent"``), ``"community"`` (needs ``partition``), or ``None``.
        The first eight categories get their own colour; the rest are shown
        as "Other".
    partition
        Community ids from :func:`netviz_tools.metrics.communities`.
    label_top
        Show text labels for this many of the largest nodes.
    min_weight_quantile
        Drop edges below this weight quantile (between 0 and 1) to reduce
        clutter.
    seed
        Layout seed.
    title
        Figure title. Defaults to one built from the graph's item and year.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown; call ``fig.show()`` yourself.
    """
    nodes = top_nodes(g, top_n)
    sub = g.subgraph(nodes).copy()
    if min_weight_quantile is not None and sub.number_of_edges():
        weights = np.array([w for *_, w in sub.edges(data="weight", default=1.0)])
        cut = float(np.quantile(weights, min_weight_quantile))
        sub.remove_edges_from(
            [(u, v) for u, v, w in sub.edges(data="weight", default=1.0) if w < cut]
        )
    pos = _layout(sub, layout, partition, seed)
    unit = unit_of(g)
    fig = go.Figure()

    edges = list(sub.edges(data="weight", default=1.0))
    if edges:
        widths = scale([w for *_, w in edges], 0.4, 5.0)
        buckets = np.digitize(widths, np.linspace(0.4, 5.0, 5)[1:-1])
        for b in sorted(set(buckets.tolist())):
            xs: list[float | None] = []
            ys: list[float | None] = []
            for (u, v, _), bb in zip(edges, buckets, strict=True):
                if bb == b:
                    xs += [pos[u][0], pos[v][0], None]
                    ys += [pos[u][1], pos[v][1], None]
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"width": float(np.mean(widths[buckets == b])), "color": EDGE_COLOR},
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"edges {b}",
                )
            )
        arrow = "→" if sub.is_directed() else "—"
        fig.add_trace(
            go.Scatter(
                x=[(pos[u][0] + pos[v][0]) / 2 for u, v, _ in edges],
                y=[(pos[u][1] + pos[v][1]) / 2 for u, v, _ in edges],
                mode="markers",
                marker={"size": 8, "opacity": 0},
                hovertext=[f"{u} {arrow} {v}: {w:,.0f}{unit}" for u, v, w in edges],
                hoverinfo="text",
                showlegend=False,
                name="flows",
            )
        )

    strength = dict(g.degree(weight="weight"))
    sizes = dict(zip(nodes, scale([strength[n] for n in nodes], 8.0, 42.0), strict=True))
    labelled = set(nodes[:label_top])
    labels, colors = categories(sub, nodes, color_by, partition)
    for cat, color in colors.items():
        members = [n for n in nodes if labels[n] == cat]
        if not members:
            continue
        fig.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in members],
                y=[pos[n][1] for n in members],
                mode="markers+text",
                text=[str(n) if n in labelled else "" for n in members],
                textposition="top center",
                textfont={"size": 11, "color": "#3d3d3a"},
                marker={
                    "size": [sizes[n] for n in members],
                    "color": color,
                    "line": {"width": 2, "color": SURFACE},
                },
                hovertext=[
                    f"{n}<br>{cat + '<br>' if cat else ''}strength {strength[n]:,.0f}{unit}"
                    for n in members
                ],
                hoverinfo="text",
                name=cat or "nodes",
                showlegend=bool(cat),
            )
        )
    fig.update_layout(
        title={"text": title or default_title(g, "trade network")},
        template="plotly_white",
        height=height,
        hovermode="closest",
        legend={"title": {"text": (color_by or "").replace("_", " ").capitalize()}},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
        xaxis={"visible": False},
        yaxis={"visible": False, "scaleanchor": "x"},
    )
    return fig
