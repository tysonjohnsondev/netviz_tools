"""Line charts for metric time series and degree-distribution fits."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from netviz_tools.plot._style import PALETTE
from netviz_tools.stats import Distribution, PowerLawFit

__all__ = ["degree_distribution", "time_series"]


def time_series(
    df: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    facet: bool = False,
    title: str | None = None,
    y_title: str | None = None,
    height: int | None = None,
) -> go.Figure:
    """Plot one line per column of a table indexed by period.

    Use it with :func:`netviz_tools.temporal.metric_series` or
    :func:`netviz_tools.temporal.centrality_series`.

    Parameters
    ----------
    df
        Table indexed by period (for example year).
    columns
        Columns to plot, in legend order. Defaults to all columns.
    facet
        Draw each column in its own panel with its own y axis. Use this when
        the columns have different units or scales; it avoids a misleading
        second y axis.
    title
        Figure title.
    y_title
        Y-axis title (ignored when ``facet`` is true; each panel is titled by
        its column).
    height
        Figure height in pixels. Defaults to 450, or 220 per panel.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        If more than eight columns would share one panel. Select the columns
        to compare, or use ``facet=True``.
    """
    cols = list(columns) if columns is not None else [str(c) for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in df: {missing}")
    x = [str(i) if isinstance(i, tuple) else i for i in df.index]
    if facet:
        fig = make_subplots(
            rows=len(cols), cols=1, shared_xaxes=True, subplot_titles=cols, vertical_spacing=0.08
        )
        for i, col in enumerate(cols, start=1):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines+markers",
                    name=col,
                    showlegend=False,
                    line={"width": 2, "color": PALETTE[0]},
                    marker={"size": 8},
                ),
                row=i,
                col=1,
            )
        fig.update_layout(height=height or 220 * len(cols))
    else:
        if len(cols) > len(PALETTE):
            raise ValueError(
                f"{len(cols)} series in one panel is too many to tell apart; "
                f"select at most {len(PALETTE)} columns or pass facet=True"
            )
        fig = go.Figure()
        for col, color in zip(cols, PALETTE, strict=False):
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines+markers",
                    name=str(col),
                    line={"width": 2, "color": color},
                    marker={"size": 8},
                )
            )
        fig.update_layout(height=height or 450, yaxis_title=y_title, hovermode="x unified")
    fig.update_layout(
        title={"text": title or ""},
        template="plotly_white",
        margin={"l": 60, "r": 20, "t": 70, "b": 40},
    )
    return fig


def degree_distribution(
    fit: PowerLawFit, *, title: str | None = None, height: int = 500
) -> go.Figure:
    """Plot the empirical CCDF with the fitted power-law, lognormal and exponential tails.

    Both axes are logarithmic. Fitted curves start at ``xmin`` and are scaled
    by the tail fraction so that they meet the empirical curve there.

    Parameters
    ----------
    fit
        Result of :func:`netviz_tools.stats.degree_distribution_fit`.
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
            x=uniq,
            y=ccdf,
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
                x=grid,
                y=frac * fit.ccdf(grid, dist),
                mode="lines",
                name=label,
                line={"width": 2, "color": color},
            )
        )
    ln, ex = fit.vs_lognormal, fit.vs_exponential
    default = (
        f"{fit.quantity.replace('_', ' ')} distribution: power law vs lognormal "
        f"R={ln.loglikelihood_ratio:.1f} (p={ln.p_value:.2f}), vs exponential "
        f"R={ex.loglikelihood_ratio:.1f} (p={ex.p_value:.2f})"
    )
    fig.update_layout(
        title={"text": title or default, "font": {"size": 14}},
        template="plotly_white",
        height=height,
        xaxis={"type": "log", "title": fit.quantity.replace("_", " ")},
        yaxis={"type": "log", "title": "P(X ≥ x)"},
        margin={"l": 60, "r": 20, "t": 70, "b": 50},
    )
    return fig
