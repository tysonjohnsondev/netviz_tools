"""Line charts for metric time series and degree-distribution fits."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from netviz_tools.analysis import suggest
from netviz_tools.plot._bars import Role
from netviz_tools.plot._data import Selector, as_list, check_known, is_flow_table, long_table
from netviz_tools.plot._style import PALETTE, base_layout, fmt, fmt_pct, lower_label, merge_labels
from netviz_tools.stats import Distribution, PowerLawFit

__all__ = ["degree_distribution", "time_series"]


def _series_from_flows(
    data: Any,
    *,
    focus: list[Hashable],
    role: Role,
    category: Selector,
    time: Selector,
    labels: Mapping[str, str] | None,
) -> tuple[pd.DataFrame, dict[str, str], str, str]:
    """Aggregate flows into a period-indexed table, one column per line."""
    df, lab = long_table(data, labels)
    if "time" not in df.columns:
        raise ValueError("time_series needs flows with a 'time' column")
    has_cats = "category" in df.columns
    cats = as_list(category)
    if cats is not None:
        if not has_cats:
            raise ValueError("category= given but the flows have no 'category' column")
        check_known(cats, sorted(df["category"].unique(), key=str), "category")
        df = df[df["category"].isin(cats)]
    times = as_list(time)
    if times is not None:
        check_known(times, sorted(df["time"].unique()), "time")
        df = df[df["time"].isin(times)]
    units = sorted(df["unit"].unique().tolist()) if "unit" in df.columns else [""]
    if len(units) > 1:
        raise ValueError(
            f"the selection uses {len(units)} units ({', '.join(map(str, units))}); "
            "pass category= to plot series measured in the same unit"
        )
    if df.empty:
        raise ValueError("no flows left to plot after filtering")
    unit = str(units[0])
    groups: list[tuple[str | None, pd.DataFrame]] = (
        [(str(c), part) for c, part in df.groupby("category", sort=True)]
        if has_cats
        else [(None, df)]
    )
    many_cats = len(groups) > 1
    columns: dict[str, pd.Series] = {}
    if not focus:
        for c, part in groups:
            columns[c if c is not None else lab["weight"]] = part.groupby("time")["weight"].sum()
        what = f"total {lower_label(lab['weight'])}"
    else:
        known_nodes = {str(n) for n in pd.concat([df["source"], df["target"]]).unique()}
        for n in focus:
            if str(n) not in known_nodes:
                hint = suggest(str(n), sorted(known_nodes))
                extra = f". Did you mean: {', '.join(map(repr, hint))}?" if hint else ""
                raise ValueError(f"focus node {n!r} has no flows in the selection{extra}")
        sides: list[Literal["out", "in"]] = ["out", "in"] if role == "both" else [role]
        for n in focus:
            for c, part in groups:
                for side in sides:
                    col = "source" if side == "out" else "target"
                    bits = [str(n)] if len(focus) > 1 else []
                    if many_cats and c is not None:
                        bits.append(c)
                    name = ", ".join(bits)
                    if len(sides) > 1 or not name:
                        name = f"{name}: {lab[side]}" if name else lab[side]
                    columns[name] = part[part[col] == n].groupby("time")["weight"].sum()
        what = " and ".join(lower_label(lab[s]) for s in sides)
        what = f"{', '.join(map(str, focus))}: {what}"
    periods = sorted(df["time"].unique().tolist())
    table = pd.DataFrame(columns).reindex(periods).fillna(0.0)
    table.index.name = "time"
    head = what[:1].upper() + what[1:]
    only = groups[0][0]
    if not many_cats and only is not None:
        head = f"{only}, {what}" if not focus else f"{head} ({only})"
    return table, lab, unit, head


def time_series(
    data: pd.DataFrame | Mapping[Hashable, Any],
    columns: Sequence[str] | None = None,
    *,
    focus: Hashable | Iterable[Hashable] | None = None,
    role: Role = "both",
    category: Selector = None,
    time: Selector = None,
    change: bool = False,
    facet: bool = False,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot values over time, one line per series.

    Accepts three kinds of input:

    * a table indexed by period, one column per series, such as the output
      of :func:`netviz_tools.temporal.metric_series`;
    * a flow table (or a mapping from period to graph): the total weight per
      period, one line per category;
    * the same with ``focus=``: what the focus node sends and receives per
      period, for example one country's exports and imports of one item.

    Hover shows each value and its change from the previous period.

    Parameters
    ----------
    data
        A period-indexed table, a flow table, or a mapping from period to graph.
    columns
        With a period-indexed table: columns to plot, in legend order.
        Defaults to all columns.
    focus
        With flows: node or nodes to follow.
    role
        With ``focus``: ``"out"``, ``"in"`` or ``"both"`` (default).
    category
        With flows: category or categories to plot, one line each.
        ``None`` uses all.
    time
        With flows: periods to keep. ``None`` keeps all.
    change
        Plot the change from the previous period, in percent, instead of the
        value.
    facet
        Draw each series in its own panel with its own y axis. Use this when
        the series have different units or scales; it avoids a misleading
        second y axis.
    labels
        Display labels, as in :func:`~netviz_tools.plot.network`.
    title
        Figure title.
    y_title
        Y-axis title (ignored when ``facet`` is true; each panel is titled by
        its series).
    height
        Figure height in pixels. Defaults to 450, or 220 per panel.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If more than eight series would share one panel. Select fewer, or use
        ``facet=True``.
    """
    focus_list = as_list(focus) or []
    unit = ""
    auto_title = ""
    lab = merge_labels(labels)
    if is_flow_table(data) or isinstance(data, Mapping):
        if columns is not None:
            raise ValueError("columns= applies to a period-indexed table; use category= or focus=")
        df, lab, unit, auto_title = _series_from_flows(
            data, focus=focus_list, role=role, category=category, time=time, labels=labels
        )
        cols = [str(c) for c in df.columns]
    else:
        if focus is not None or category is not None or time is not None:
            raise ValueError("focus=, category= and time= apply to flows, not to a metric table")
        df = data
        cols = list(columns) if columns is not None else [str(c) for c in df.columns]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"columns not in the table: {missing}")
    values = df[cols].astype(float)
    pct = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    shown = pct * 100 if change else values
    x = [str(i) if isinstance(i, tuple) else i for i in df.index]
    if change:
        y_title = "Change from previous period (%)"
        auto_title = f"{auto_title}, change from previous period" if auto_title else ""
    elif y_title is None and unit:
        y_title = f"{lab['weight']} ({unit})"

    def hover(col: str) -> list[str]:
        out = []
        for i, (idx, v, p) in enumerate(zip(df.index, values[col], pct[col], strict=True)):
            text = f"<b>{col}</b><br>{idx}: {fmt(v, unit)}"
            if i > 0:
                text += f"<br>change: {fmt_pct(p)}"
            out.append(text)
        return out

    def line(col: str, color: str, *, legend: bool) -> go.Scatter:
        return go.Scatter(
            x=x,
            y=shown[col].tolist(),
            mode="lines+markers",
            name=col,
            showlegend=legend,
            line={"width": 2, "color": color},
            marker={"size": 7},
            hovertext=hover(col),
            hoverinfo="text",
        )

    layout = base_layout(
        title if title is not None else auto_title,
        height or (220 * len(cols) if facet else 450),
        margin={"l": 60, "r": 20, "t": 70, "b": 40},
    )
    if facet:
        fig = make_subplots(
            rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols, vertical_spacing=0.08
        )
        for i, col in enumerate(cols, start=1):
            fig.add_trace(line(col, PALETTE[0], legend=False), row=i, col=1)
    else:
        if len(cols) > len(PALETTE):
            raise ValueError(
                f"{len(cols)} series in one panel is too many to tell apart; "
                f"select at most {len(PALETTE)} or pass facet=True"
            )
        fig = go.Figure(
            [line(c, color, legend=True) for c, color in zip(cols, PALETTE, strict=False)]
        )
        layout["yaxis"] = {"title": {"text": y_title}}
        if change:
            fig.add_hline(y=0, line={"color": "#bdbcb6", "width": 1})
    fig.update_layout(**layout)
    if auto_title:
        fig.update_xaxes(title={"text": lab["time"]}, row=len(cols) if facet else None)
    return fig


def degree_distribution(
    fit: PowerLawFit,
    *,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 500,
) -> go.Figure:
    """Plot the empirical CCDF with the fitted power-law, lognormal and exponential tails.

    Both axes are logarithmic. Fitted curves start at ``xmin`` and are scaled
    by the tail fraction so that they meet the empirical curve there.

    Parameters
    ----------
    fit
        Result of :func:`netviz_tools.stats.degree_distribution_fit`.
    labels
        Display labels; a label for the fitted quantity (for example
        ``{"out_strength": "Exports"}``) names the x axis.
    title
        Figure title. Defaults to a summary of the fit.
    height
        Figure height in pixels.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.
    """
    values = np.asarray(fit.values)
    uniq = np.unique(values)
    ccdf = np.array([(values >= u).mean() for u in uniq])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=uniq.tolist(),
            y=ccdf.tolist(),
            mode="markers",
            name="observed",
            marker={"size": 8, "color": "#52514e", "line": {"width": 1, "color": "#ffffff"}},
        )
    )
    grid = np.unique(np.geomspace(fit.xmin, values.max(), 200))
    if fit.discrete:
        grid = np.unique(np.round(grid))
    frac = fit.n_tail / fit.n
    curves: tuple[tuple[Distribution, str, str], ...] = (
        ("power_law", PALETTE[0], f"power law (alpha {fit.alpha:.2f})"),
        ("lognormal", PALETTE[1], "lognormal"),
        ("exponential", PALETTE[2], "exponential"),
    )
    for dist, color, label in curves:
        fig.add_trace(
            go.Scatter(
                x=grid.tolist(),
                y=(frac * fit.ccdf(grid, dist)).tolist(),
                mode="lines",
                name=label,
                line={"width": 2, "color": color},
            )
        )
    ln, ex = fit.vs_lognormal, fit.vs_exponential
    x_title = (labels or {}).get(fit.quantity, fit.quantity.replace("_", " "))
    default = (
        f"{x_title[:1].upper()}{x_title[1:]} distribution<br><sup>power law vs lognormal: "
        f"R={ln.loglikelihood_ratio:.1f} (p={ln.p_value:.2f}); vs exponential: "
        f"R={ex.loglikelihood_ratio:.1f} (p={ex.p_value:.2f})</sup>"
    )
    layout = base_layout(title or default, height, margin={"l": 60, "r": 20, "t": 80, "b": 50})
    layout.update(
        xaxis={"type": "log", "title": {"text": x_title}},
        yaxis={"type": "log", "title": {"text": "P(X ≥ x)"}},
    )
    fig.update_layout(**layout)
    return fig
