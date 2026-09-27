"""Sankey diagram of the largest exporters and importers."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._style import OTHER_COLOR, categories, default_title, rgba, unit_of

__all__ = ["sankey"]


def sankey(
    g: nx.Graph[Any],
    *,
    top_n: int = 10,
    other: bool = True,
    color_by: str | None = "continent",
    title: str | None = None,
    height: int = 600,
) -> go.Figure:
    """Draw flows from the largest exporters (left) to the largest importers (right).

    Each country can appear on both sides. Placing senders and receivers in
    separate columns keeps the diagram free of cycles, which a Sankey cannot
    draw.

    Parameters
    ----------
    g
        A directed graph from :func:`netviz_tools.build_graph`.
    top_n
        Number of exporters and of importers to show individually.
    other
        Group the remaining exporters and importers into "Other exporters"
        and "Other importers". If ``False``, their flows are left out.
    color_by
        Node attribute used for node colours (default ``"continent"``), or
        ``None``. Links take the colour of their exporter.
    title
        Figure title. Defaults to one built from the graph's item and year.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If the graph is undirected.
    """
    if not g.is_directed():
        raise ValueError("sankey needs a directed graph")
    dg: nx.DiGraph[Any] = g  # type: ignore[assignment]
    out_s = pd.Series(dict(dg.out_degree(weight="weight")), dtype=float)
    in_s = pd.Series(dict(dg.in_degree(weight="weight")), dtype=float)
    exporters = out_s[out_s > 0].sort_values(ascending=False, kind="stable").index[:top_n].tolist()
    importers = in_s[in_s > 0].sort_values(ascending=False, kind="stable").index[:top_n].tolist()
    exp_set, imp_set = set(exporters), set(importers)

    rows = []
    for u, v, w in dg.edges(data="weight", default=1.0):
        src = u if u in exp_set else ("Other exporters" if other else None)
        dst = v if v in imp_set else ("Other importers" if other else None)
        if src is not None and dst is not None:
            rows.append((src, dst, float(w)))
    links = (
        pd.DataFrame(rows, columns=["src", "dst", "w"])
        .groupby(["src", "dst"], sort=False)["w"]
        .sum()
    )

    left: list[Hashable] = [*exporters, *(["Other exporters"] if other else [])]
    right: list[Hashable] = [*importers, *(["Other importers"] if other else [])]
    left_idx = {n: i for i, n in enumerate(left)}
    right_idx = {n: len(left) + i for i, n in enumerate(right)}

    real = list(dict.fromkeys([*exporters, *importers]))
    labels, colors = categories(dg, real, color_by, None)

    def node_color(n: Hashable) -> str:
        return colors[labels[n]] if n in labels else OTHER_COLOR

    node_colors = [node_color(n) for n in left] + [node_color(n) for n in right]
    unit = unit_of(g)
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=",.0f",
            valuesuffix=unit,
            node={
                "label": [str(n) for n in left] + [str(n) for n in right],
                "color": node_colors,
                "pad": 12,
                "thickness": 16,
                "line": {"width": 0},
                "x": [0.001] * len(left) + [0.999] * len(right),
                "y": [(i + 0.5) / len(left) for i in range(len(left))]
                + [(i + 0.5) / len(right) for i in range(len(right))],
            },
            link={
                "source": [left_idx[s] for s, _ in links.index],
                "target": [right_idx[d] for _, d in links.index],
                "value": links.to_numpy().tolist(),
                "color": [rgba(node_color(s), 0.35) for s, _ in links.index],
            },
        )
    )
    fig.update_layout(
        title={"text": title or default_title(g, "flows, top exporters to top importers")},
        template="plotly_white",
        height=height,
        font={"size": 11},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig
