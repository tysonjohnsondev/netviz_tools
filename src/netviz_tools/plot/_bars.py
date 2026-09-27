"""Bar charts: rank nodes by a value, and compare one node across categories."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._data import PlotData, Selector, as_list, is_flow_table, to_frames
from netviz_tools.plot._render import resolve_focus
from netviz_tools.plot._resolve import NodeSpec, node_values, select_top
from netviz_tools.plot._style import (
    DECREASE_COLOR,
    INCREASE_COLOR,
    OTHER_COLOR,
    PALETTE,
    TEXT_COLOR,
    base_layout,
    fmt,
    fmt_pct,
    lower_label,
    merge_labels,
)

__all__ = ["Role", "compare", "ranking"]

Role: TypeAlias = Literal["out", "in", "both"]
"""Which side of a node's flows to show: what it sends (``"out"``), what it
receives (``"in"``), or both side by side."""


def _bar_height(n: int) -> int:
    return max(320, 110 + 28 * n)


def ranking(
    data: PlotData,
    *,
    size_by: NodeSpec = "auto",
    top_n: int = 15,
    focus: Hashable | Iterable[Hashable] | None = None,
    time: Selector = None,
    category: Selector = None,
    change: bool = False,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int | None = None,
    seed: int = 42,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Rank nodes by a value in a horizontal bar chart.

    With one period, bars show the ``top_n`` nodes. With several periods, bars
    show the last period and a grey tick marks each node's value in the
    period before, so rises and falls are visible at a glance; hover gives
    both values and the change. With ``change=True`` the bars show the change
    itself (last period minus the one before), sorted by size, increases in
    blue and decreases in orange.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, or a mapping from period to graph.
    size_by
        The value to rank by (bar length): a numeric node attribute, a metric
        such as ``"out_strength"`` (for trade data, exports), or a mapping.
        ``"auto"`` is strength, or degree when edges have no weight.
    top_n
        Number of bars.
    focus
        Node or nodes to highlight. They are always shown, even outside the
        top ``top_n``.
    time
        Periods to use; the last two are compared. ``None`` uses all.
    category, labels, title, height, seed, node_attrs
        As in :func:`~netviz_tools.plot.network`. ``height`` defaults to one
        that fits the bars.
    change
        Show the change between the last two periods instead of the level.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If ``change=True`` and there is only one period, or ``size_by`` is
        not numeric.
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    lab = frames.labels
    keys = list(frames.graphs)
    per: dict[Hashable, pd.Series] = {}
    label = ""
    in_units = False
    for k, g in frames.graphs.items():
        nv_ = node_values(g, size_by, arg="size_by", labels=lab, seed=seed)
        if nv_ is None or nv_.categorical:
            raise ValueError("ranking needs a numeric size_by (an attribute or metric)")
        per[k] = nv_.values.astype(float)
        label, in_units = nv_.label, nv_.in_weight_units
    unit = frames.unit if in_units else ""
    table = pd.DataFrame(per).fillna(0.0)
    last = table[keys[-1]]
    prev = table[keys[-2]] if len(keys) > 1 else None
    focus_nodes = resolve_focus(frames.union(), focus)
    if change:
        if prev is None:
            raise ValueError("change=True needs at least two periods")
        delta = last - prev
        order = select_top(delta.abs(), top_n)
    else:
        order = select_top(last, top_n)
    order += [f for f in focus_nodes if f not in set(order)]
    order = order[::-1]  # plotly draws the first bar at the bottom
    names = [str(n) for n in order]
    fig = go.Figure()
    period = keys[-1]
    if change:
        assert prev is not None
        d = (last - prev).reindex(order)
        pct = [
            (float(d[n]) / float(prev[n])) if float(prev[n]) > 0 else float("nan") for n in order
        ]
        colors = [INCREASE_COLOR if float(d[n]) >= 0 else DECREASE_COLOR for n in order]
        fig.add_trace(
            go.Bar(
                x=d.tolist(),
                y=names,
                orientation="h",
                marker={
                    "color": colors,
                    "line": {
                        "width": [2 if n in focus_nodes else 0 for n in order],
                        "color": TEXT_COLOR,
                    },
                },
                text=[fmt_pct(p) if np.isfinite(p) else "new" for p in pct],
                textposition="outside",
                cliponaxis=False,
                hovertext=[
                    f"<b>{n}</b><br>{keys[-2]}: {fmt(prev[n], unit)}<br>{period}: {fmt(last[n], unit)}"
                    f"<br>change: {fmt(float(d[n]), unit)} ({fmt_pct(p) if np.isfinite(p) else 'new'})"
                    for n, p in zip(order, pct, strict=True)
                ],
                hoverinfo="text",
                name="change",
                showlegend=False,
            )
        )
        default = f"Largest changes in {lower_label(label)}, {keys[-2]} to {period}"
        x_title = f"Change in {lower_label(label)}" + (f" ({unit})" if unit else "")
    else:
        colors = [PALETTE[0] if not focus_nodes or n in focus_nodes else OTHER_COLOR for n in order]
        hover = []
        for n in order:
            text = f"<b>{n}</b><br>{period if period is not None else label}: {fmt(last[n], unit)}"
            if prev is not None:
                p = float(prev[n])
                pct = (float(last[n]) - p) / p if p > 0 else float("nan")
                text += f"<br>{keys[-2]}: {fmt(p, unit)}<br>change: {fmt_pct(pct) if np.isfinite(pct) else 'new'}"
            hover.append(text)
        fig.add_trace(
            go.Bar(
                x=last.reindex(order).tolist(),
                y=names,
                orientation="h",
                marker={"color": colors},
                hovertext=hover,
                hoverinfo="text",
                name=str(period) if period is not None else label,
                showlegend=prev is not None,
            )
        )
        if prev is not None:
            fig.add_trace(
                go.Scatter(
                    x=prev.reindex(order).tolist(),
                    y=names,
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 18,
                        "line": {"width": 3, "color": TEXT_COLOR},
                    },
                    name=str(keys[-2]),
                    hoverinfo="skip",
                )
            )
        head = f"Top {min(top_n, len(order))} by {lower_label(label)}"
        default = f"{head}, {period}" if period is not None else head
        x_title = label + (f" ({unit})" if unit else "")
    cat = frames.first.graph.get("category")
    if title is None:
        title = f"{cat}: {default[:1].lower() + default[1:]}" if isinstance(cat, str) else default
    layout = base_layout(
        title, height or _bar_height(len(order)), margin={"l": 10, "r": 40, "t": 60, "b": 40}
    )
    layout.update(
        xaxis={"title": {"text": x_title}, "zeroline": True, "zerolinecolor": "#bdbcb6"},
        yaxis={"automargin": True},
        legend={"orientation": "h", "y": -0.12, "x": 0},
        bargap=0.3,
    )
    fig.update_layout(**layout)
    return fig


def _long_table(data: Any, labels: Mapping[str, str] | None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a flow table and merged labels from a flow table or a mapping of graphs."""
    if is_flow_table(data):
        lab = data.attrs.get("labels")
        return data, merge_labels(lab if isinstance(lab, Mapping) else None, labels)
    if isinstance(data, Mapping):
        rows = []
        first_labels = None
        unit = ""
        for k, g in data.items():
            first_labels = first_labels or g.graph.get("labels")
            unit = unit or (g.graph.get("unit") or "")
            cat = g.graph.get("category", "")
            for u, v, w in g.edges(data="weight", default=1.0):
                rows.append((u, v, k, cat, float(w), unit))
        df = pd.DataFrame(rows, columns=["source", "target", "time", "category", "weight", "unit"])
        return df, merge_labels(first_labels, labels)
    raise TypeError(
        "expected a flow table with source/target columns or a mapping from period to graph, "
        f"not {type(data).__name__}"
    )


def compare(
    data: pd.DataFrame,
    focus: Hashable,
    *,
    role: Role = "both",
    category: Selector = None,
    time: Selector = None,
    top_n: int | None = 15,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Compare what one node sends and receives across categories.

    Answers questions such as "which products does this country import, and
    which does it export?". One row per category; with ``role="both"`` the
    outgoing and incoming totals sit side by side. With several periods the
    last period is shown, and grey ticks mark the period before.

    Parameters
    ----------
    data
        A flow table with several categories (for example several FAOSTAT
        items).
    focus
        The node to describe.
    role
        ``"out"``, ``"in"`` or ``"both"`` (default).
    category
        Categories to include. ``None`` includes every category the node has
        flows in.
    time
        Period or periods to use; the last one is shown and the one before is
        marked. ``None`` uses all.
    top_n
        Keep the ``top_n`` categories with the largest total. ``None`` keeps all.
    labels, title
        As in :func:`~netviz_tools.plot.network`.
    height
        Figure height in pixels; defaults to one that fits the bars.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If the node has no flows, or the selected categories use more than
        one unit (they could not share an axis).
    """
    if not is_flow_table(data):
        raise TypeError("compare needs a flow table with several categories")
    df, lab = _long_table(data, labels)
    cats = as_list(category)
    if cats is not None:
        df = df[df["category"].isin(cats)]
    times = as_list(time)
    if times is not None:
        df = df[df["time"].isin(times)]
    rows = df[(df["source"] == focus) | (df["target"] == focus)]
    if rows.empty:
        from netviz_tools.analysis import suggest

        nodes = pd.unique(pd.concat([df["source"], df["target"]]).astype(str))
        hint = suggest(str(focus), nodes)
        extra = f". Did you mean: {', '.join(map(repr, hint))}?" if hint else ""
        raise ValueError(f"{focus!r} has no flows in the selection{extra}")
    units = rows["unit"].unique() if "unit" in rows.columns else np.array([""])
    if len(units) > 1:
        raise ValueError(
            f"the selected categories use {len(units)} units ({', '.join(map(str, units))}); "
            "pass category= to compare categories measured in the same unit"
        )
    unit = str(units[0]) if len(units) else ""
    periods = sorted(rows["time"].unique().tolist()) if "time" in rows.columns else [None]
    last_p = periods[-1]
    prev_p = periods[-2] if len(periods) > 1 else None

    def totals(p: Hashable) -> pd.DataFrame:
        part = rows if p is None else rows[rows["time"] == p]
        out = part[part["source"] == focus].groupby("category")["weight"].sum()
        inn = part[part["target"] == focus].groupby("category")["weight"].sum()
        return pd.DataFrame({"out": out, "in": inn}).fillna(0.0)

    cur = totals(last_p)
    before = totals(prev_p) if prev_p is not None else None
    sides = ["out", "in"] if role == "both" else [role]
    size = cur[sides].sum(axis=1)
    order = select_top(size, top_n)[::-1]
    names = [str(c) for c in order]
    fig = go.Figure()
    for side, color in zip(sides, (PALETTE[0], PALETTE[1]), strict=False):
        name = lab[side]
        values = cur[side].reindex(order).fillna(0.0)
        hover = []
        for c in order:
            text = f"<b>{c}</b><br>{name}, {last_p}: {fmt(values[c], unit)}"
            if before is not None:
                p = float(before[side].get(c, 0.0))
                pct = (float(values[c]) - p) / p if p > 0 else float("nan")
                text += f"<br>{name}, {prev_p}: {fmt(p, unit)}<br>change: {fmt_pct(pct) if np.isfinite(pct) else 'new'}"
            hover.append(text)
        fig.add_trace(
            go.Bar(
                x=values.tolist(),
                y=names,
                orientation="h",
                name=name,
                marker={"color": color},
                hovertext=hover,
                hoverinfo="text",
                offsetgroup=side,
            )
        )
        if before is not None:
            fig.add_trace(
                go.Scatter(
                    x=before[side].reindex(order).fillna(0.0).tolist(),
                    y=names,
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 14,
                        "line": {"width": 2.5, "color": TEXT_COLOR},
                    },
                    name=f"{name}, {prev_p}",
                    hoverinfo="skip",
                    showlegend=side == sides[0],
                    legendgroup="prev",
                    offsetgroup=side,
                )
            )
    if title is None:
        what = " and ".join(lower_label(lab[s]) for s in sides)
        title = f"{focus}: {what} by {lower_label(lab['category'])}" + (
            f", {last_p}" if last_p is not None else ""
        )
    layout = base_layout(
        title,
        height or _bar_height(len(order) * len(sides)),
        margin={"l": 10, "r": 20, "t": 60, "b": 40},
    )
    layout.update(
        barmode="group",
        xaxis={"title": {"text": lab["weight"] + (f" ({unit})" if unit else "")}},
        yaxis={"automargin": True},
        legend={"orientation": "h", "y": -0.12, "x": 0},
        bargap=0.25,
    )
    fig.update_layout(**layout)
    return fig
