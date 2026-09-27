"""Sankey diagram of the largest sources and targets."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._data import PlotData, Selector, to_frames
from netviz_tools.plot._render import resolve_focus
from netviz_tools.plot._resolve import NodeSpec, categories, node_values
from netviz_tools.plot._style import OTHER_COLOR, base_layout, default_title, fmt, rgba

__all__ = ["sankey"]


def plural(word: str) -> str:
    """Return a naive English plural for column headings."""
    return word if word.endswith("s") else word + "s"


def sankey(
    data: PlotData,
    *,
    color_by: NodeSpec = "auto",
    top_n: int = 10,
    focus: Hashable | Iterable[Hashable] | None = None,
    time: Selector = None,
    category: Selector = None,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 600,
    seed: int = 42,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Draw flows from the largest sources (left) to the largest targets (right).

    Each node can appear on both sides. Placing sources and targets in
    separate columns keeps the diagram free of cycles, which a Sankey cannot
    draw.

    Parameters
    ----------
    data
        A directed NetworkX graph, a flow table, or a mapping from period to
        graph. One period only: pass ``time=`` when the data span several.
    color_by
        Categorical node colour: a node attribute, ``"community"``, a mapping,
        ``None``, or ``"auto"`` (communities). Links take the colour of their
        source.
    top_n
        Number of sources and of targets to show individually. The rest are
        grouped into "Other" on each side.
    focus
        Node or nodes to follow: only flows touching them are drawn.
    time, category, labels, title, height, seed, node_attrs
        As in :func:`~netviz_tools.plot.network`.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If the graph is undirected, the data span several periods, or
        ``color_by`` is numeric.
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    g = frames.single("sankey")
    if not g.is_directed():
        raise ValueError("sankey needs a directed graph")
    lab = frames.labels
    focus_nodes = set(resolve_focus(g, focus))
    edges = [
        (u, v, float(w))
        for u, v, w in g.edges(data="weight", default=1.0)
        if u != v and (not focus_nodes or u in focus_nodes or v in focus_nodes)
    ]
    frame = pd.DataFrame(edges, columns=["src", "dst", "w"])
    out_s = frame.groupby("src")["w"].sum().sort_values(ascending=False, kind="stable")
    in_s = frame.groupby("dst")["w"].sum().sort_values(ascending=False, kind="stable")
    sources = out_s.index[:top_n].tolist()
    targets = in_s.index[:top_n].tolist()
    other_src = f"Other {plural(lab['source'].lower())}"
    other_dst = f"Other {plural(lab['target'].lower())}"
    grouped = frame.assign(
        src=frame["src"].where(frame["src"].isin(sources), other_src),
        dst=frame["dst"].where(frame["dst"].isin(targets), other_dst),
    )
    links = grouped.groupby(["src", "dst"], sort=False, as_index=False)["w"].sum()

    left: list[Hashable] = [*sources, *([other_src] if len(out_s) > top_n else [])]
    right: list[Hashable] = [*targets, *([other_dst] if len(in_s) > top_n else [])]
    left_idx = {n: i for i, n in enumerate(left)}
    right_idx = {n: len(left) + i for i, n in enumerate(right)}

    colors: dict[Hashable, str] = {}
    resolved = node_values(g, color_by, arg="color_by", labels=lab, seed=seed)
    legend: dict[str, str] = {}
    if resolved is not None:
        if not resolved.categorical:
            raise ValueError("sankey colours need a categorical color_by (text or community)")
        real = list(dict.fromkeys([*sources, *targets]))
        cats = categories(resolved.series.reindex(real), name=resolved.name)
        colors = {n: cats.colors[str(cats.labels[n])] for n in real}
        legend = cats.colors

    def node_color(n: Hashable) -> str:
        return colors.get(n, OTHER_COLOR if resolved is not None else "#2a78d6")

    unit = frames.unit
    src, dst, values = links["src"].tolist(), links["dst"].tolist(), links["w"].tolist()
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            valueformat=",.0f",
            valuesuffix=f" {unit}" if unit else "",
            node={
                "label": [str(n) for n in left] + [str(n) for n in right],
                "color": [node_color(n) for n in left] + [node_color(n) for n in right],
                "pad": 12,
                "thickness": 16,
                "line": {"width": 0},
                "x": [0.001] * len(left) + [0.999] * len(right),
                "y": [(i + 0.5) / len(left) for i in range(len(left))]
                + [(i + 0.5) / len(right) for i in range(len(right))],
            },
            link={
                "source": [left_idx[s] for s in src],
                "target": [right_idx[d] for d in dst],
                "value": values,
                "color": [rgba(node_color(s), 0.35) for s in src],
                "customdata": [
                    f"{lab['source']}: {s}<br>{lab['target']}: {d}<br>"
                    f"{lab['weight']}: {fmt(w, unit)}"
                    for s, d, w in zip(src, dst, values, strict=True)
                ],
                "hovertemplate": "%{customdata}<extra></extra>",
            },
        )
    )
    if title is None:
        what = f"flows, top {plural(lab['source'].lower())} to top {plural(lab['target'].lower())}"
        title = default_title(g, what)
    layout = base_layout(title, height)
    layout["annotations"] = [
        {
            "text": f"<b>{lab['source']}</b>",
            "x": 0,
            "y": 1.04,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "left",
            "showarrow": False,
        },
        {
            "text": f"<b>{lab['target']}</b>",
            "x": 1,
            "y": 1.04,
            "xref": "paper",
            "yref": "paper",
            "xanchor": "right",
            "showarrow": False,
        },
    ]
    layout["margin"] = {"l": 10, "r": 10, "t": 80, "b": 10}
    layout["font"] = {**layout["font"], "size": 11}
    fig.update_layout(**layout)
    if legend:
        for name, color in legend.items():
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker={"size": 10, "color": color},
                    name=name,
                    showlegend=True,
                    hoverinfo="skip",
                )
            )
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            legend={"title": {"text": resolved.label if resolved else ""}},
        )
    return fig
