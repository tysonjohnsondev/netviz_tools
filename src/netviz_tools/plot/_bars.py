"""Bar charts: rank nodes by a value, and compare one node across categories."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Final, Literal, TypeAlias

import pandas as pd
import plotly.graph_objects as go

from netviz_tools.analysis import suggest
from netviz_tools.plot._data import (
    PlotData,
    Selector,
    as_list,
    check_known,
    is_flow_table,
    long_table,
    to_frames,
)
from netviz_tools.plot._render import resolve_focus
from netviz_tools.plot._resolve import NodeSpec, node_series, node_values, select_top
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
)

__all__ = ["Role", "compare", "ranking"]

Role: TypeAlias = Literal["out", "in", "both"]
"""Which side of a node's flows to show: what it sends (``"out"``), what it
receives (``"in"``), or both side by side."""


_LEGEND_ON_TOP: Final = {
    "orientation": "h",
    "x": 0,
    "xanchor": "left",
    "y": 1.0,
    "yanchor": "bottom",
}


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
    keys = list(frames.graphs)
    per: dict[Hashable, pd.Series] = {}
    resolved = None
    for k, g in frames.graphs.items():
        resolved = node_values(g, size_by, arg="size_by", labels=frames.labels, seed=seed)
        if resolved is None or resolved.categorical:
            raise ValueError("ranking needs a numeric size_by (an attribute or metric)")
        per[k] = resolved.series.astype(float)
    assert resolved is not None  # there is at least one period
    label = resolved.label
    unit = frames.unit if resolved.in_weight_units else ""
    table = pd.DataFrame(per).fillna(0.0)
    last = table[keys[-1]].to_dict()
    prev = table[keys[-2]].to_dict() if len(keys) > 1 else None
    before = keys[-2] if len(keys) > 1 else None
    period = keys[-1] if keys[-1] is not None else frames.first.graph.get("time")
    focus_nodes = resolve_focus(frames.union, focus)
    if change and prev is None:
        raise ValueError("change=True needs at least two periods")
    delta = {n: last[n] - prev[n] for n in last} if prev is not None else {}
    rank_by = {n: abs(d) for n, d in delta.items()} if change else last
    order = select_top(node_series(rank_by, dtype=float), top_n)
    order += [f for f in focus_nodes if f not in set(order)]
    order = order[::-1]  # plotly draws the first bar at the bottom
    names = [str(n) for n in order]

    def change_text(n: Hashable) -> str:
        assert prev is not None
        p = prev[n]
        return fmt_pct((last[n] - p) / p) if p > 0 else "new"

    fig = go.Figure()
    if change:
        assert prev is not None
        fig.add_trace(
            go.Bar(
                x=[delta[n] for n in order],
                y=names,
                orientation="h",
                marker={
                    "color": [INCREASE_COLOR if delta[n] >= 0 else DECREASE_COLOR for n in order],
                    "line": {
                        "width": [2 if n in focus_nodes else 0 for n in order],
                        "color": TEXT_COLOR,
                    },
                },
                text=[change_text(n) for n in order],
                textposition="outside",
                cliponaxis=False,
                hovertext=[
                    f"<b>{n}</b><br>{before}: {fmt(prev[n], unit)}<br>{period}: "
                    f"{fmt(last[n], unit)}<br>change: {fmt(delta[n], unit)} ({change_text(n)})"
                    for n in order
                ],
                hoverinfo="text",
                name="change",
                showlegend=False,
            )
        )
        default = f"Largest changes in {lower_label(label)}, {before} to {period}"
        x_title = f"Change in {lower_label(label)}" + (f" ({unit})" if unit else "")
    else:
        hover = []
        for n in order:
            text = f"<b>{n}</b><br>{period if period is not None else label}: {fmt(last[n], unit)}"
            if prev is not None:
                text += f"<br>{before}: {fmt(prev[n], unit)}<br>change: {change_text(n)}"
            hover.append(text)
        colors = [PALETTE[0] if not focus_nodes or n in focus_nodes else OTHER_COLOR for n in order]
        fig.add_trace(
            go.Bar(
                x=[last[n] for n in order],
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
                    x=[prev[n] for n in order],
                    y=names,
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 18,
                        "line": {"width": 3, "color": TEXT_COLOR},
                    },
                    name=str(before),
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
        title, height or _bar_height(len(order)), margin={"l": 10, "r": 40, "t": 80, "b": 40}
    )
    xaxis: dict[str, Any] = {
        "title": {"text": x_title},
        "zeroline": True,
        "zerolinecolor": "#bdbcb6",
    }
    if change:  # leave room for the percentage labels outside the bars
        lo, hi = min(0.0, *delta.values()), max(0.0, *delta.values())
        pad = 0.14 * (hi - lo)
        xaxis["range"] = [lo - pad if lo < 0 else lo, hi + pad if hi > 0 else hi]
    layout.update(
        xaxis=xaxis,
        yaxis={"automargin": True, "ticksuffix": " "},
        legend=_LEGEND_ON_TOP,
        bargap=0.3,
    )
    fig.update_layout(**layout)
    return fig


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
    if role not in ("out", "in", "both"):
        raise ValueError(f"role must be 'out', 'in' or 'both', not {role!r}")
    df, lab = long_table(data, labels)
    cats = as_list(category)
    if cats is not None:
        check_known(cats, sorted(df["category"].unique(), key=str), "category")
        df = df[df["category"].isin(cats)]
    times = as_list(time)
    if times is not None:
        check_known(times, sorted(df["time"].unique()), "time")
        df = df[df["time"].isin(times)]
    rows = df[(df["source"] == focus) | (df["target"] == focus)]
    if rows.empty:
        nodes = pd.unique(pd.concat([df["source"], df["target"]]).astype(str))
        hint = suggest(str(focus), list(nodes))
        extra = f". Did you mean: {', '.join(map(repr, hint))}?" if hint else ""
        raise ValueError(f"{focus!r} has no flows in the selection{extra}")
    units = sorted(rows["unit"].unique().tolist()) if "unit" in rows.columns else [""]
    if len(units) > 1:
        raise ValueError(
            f"the selected categories use {len(units)} units ({', '.join(map(str, units))}); "
            "pass category= to compare categories measured in the same unit"
        )
    unit = str(units[0])
    periods = sorted(rows["time"].unique().tolist()) if "time" in rows.columns else [None]
    last_p = periods[-1]
    prev_p = periods[-2] if len(periods) > 1 else None
    sides: list[Literal["out", "in"]] = ["out", "in"] if role == "both" else [role]

    def totals(p: Hashable) -> dict[str, dict[Hashable, float]]:
        part = rows if p is None else rows[rows["time"] == p]
        return {
            side: part[part[col] == focus].groupby("category")["weight"].sum().to_dict()
            for side, col in (("out", "source"), ("in", "target"))
        }

    cur = totals(last_p)
    before = totals(prev_p) if prev_p is not None else None
    size = pd.Series(
        {c: sum(cur[side].get(c, 0.0) for side in sides) for c in rows["category"].unique()},
        dtype=float,
    )
    order = select_top(size, top_n)[::-1]
    names = [str(c) for c in order]
    colors = {"out": PALETTE[0], "in": PALETTE[1]}
    bars, ticks = [], []
    # Horizontal groups stack their traces bottom-up: add "in" first so that
    # "out" sits on top, and reverse the legend so it still reads out, in.
    for side in reversed(sides):
        name = lab[side]
        values = [cur[side].get(c, 0.0) for c in order]
        hover = []
        for c, v in zip(order, values, strict=True):
            text = f"<b>{c}</b><br>{name}, {last_p}: {fmt(v, unit)}"
            if before is not None:
                p = before[side].get(c, 0.0)
                change = fmt_pct((v - p) / p) if p > 0 else "new"
                text += f"<br>{name}, {prev_p}: {fmt(p, unit)}<br>change: {change}"
            hover.append(text)
        bars.append(
            go.Bar(
                x=values,
                y=names,
                orientation="h",
                name=name,
                marker={"color": colors[side]},
                hovertext=hover,
                hoverinfo="text",
                offsetgroup=side,
            )
        )
        if before is not None:
            ticks.append(
                go.Scatter(
                    x=[before[side].get(c, 0.0) for c in order],
                    y=names,
                    orientation="h",
                    mode="markers",
                    marker={
                        "symbol": "line-ns",
                        "size": 14,
                        "line": {"width": 2.5, "color": TEXT_COLOR},
                    },
                    name=str(prev_p),
                    hoverinfo="skip",
                    showlegend=side == sides[0],
                    legendgroup="prev",
                    offsetgroup=side,
                )
            )
    fig = go.Figure([*ticks, *bars])
    if title is None:
        what = " and ".join(lower_label(lab[s]) for s in sides)
        title = f"{focus}: {what} by {lower_label(lab['category'])}" + (
            f", {last_p}" if last_p is not None else ""
        )
    layout = base_layout(
        title,
        height or _bar_height(len(order) * len(sides)),
        margin={"l": 10, "r": 20, "t": 80, "b": 40},
    )
    layout.update(
        barmode="group",
        scattermode="group",
        xaxis={"title": {"text": lab["weight"] + (f" ({unit})" if unit else "")}},
        yaxis={"automargin": True, "ticksuffix": " "},
        legend={**_LEGEND_ON_TOP, "traceorder": "reversed"},
        bargap=0.25,
    )
    fig.update_layout(**layout)
    return fig
