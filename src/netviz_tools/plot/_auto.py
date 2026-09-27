"""One entry point: name the chart, or let the data choose it."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Hashable, Mapping
from typing import Any, Final, Literal, TypeAlias

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._bars import compare, ranking
from netviz_tools.plot._charts import degree_distribution, time_series
from netviz_tools.plot._data import Selector, as_list, is_flow_table, to_frames
from netviz_tools.plot._geo import flow_map, has_coordinates
from netviz_tools.plot._matrix import adjacency
from netviz_tools.plot._network import ego, network
from netviz_tools.plot._sankey import sankey
from netviz_tools.stats import PowerLawFit

__all__ = ["KINDS", "Kind", "auto", "choose_kind"]

Kind: TypeAlias = Literal[
    "auto",
    "network",
    "flow_map",
    "sankey",
    "adjacency",
    "ego",
    "ranking",
    "compare",
    "time_series",
    "degree_distribution",
]
"""Chart names accepted by :func:`auto`. ``"map"`` is accepted as an alias of
``"flow_map"``."""

KINDS: Final[Mapping[str, Callable[..., go.Figure]]] = {
    "network": network,
    "flow_map": flow_map,
    "map": flow_map,
    "sankey": sankey,
    "adjacency": adjacency,
    "ego": ego,
    "ranking": ranking,
    "compare": compare,
    "time_series": time_series,
    "degree_distribution": degree_distribution,
}
"""Every chart function reachable through :func:`auto`, by name."""


def _single_focus(focus: Selector) -> Hashable | None:
    items = as_list(focus)
    return items[0] if items is not None and len(items) == 1 else None


def _n_categories(data: pd.DataFrame, category: Selector) -> int:
    if "category" not in data.columns:
        return 1
    cats = as_list(category)
    col = data["category"]
    return int(col[col.isin(cats)].nunique()) if cats is not None else int(col.nunique())


def _only_sends_or_receives(g: nx.Graph[Any]) -> bool:
    if not g.is_directed() or g.number_of_edges() == 0:
        return False
    dg: nx.DiGraph[Any] = g  # type: ignore[assignment]
    return all(dg.in_degree(n) == 0 or dg.out_degree(n) == 0 for n in dg)


def choose_kind(data: object, **kwargs: Any) -> tuple[str, str]:
    """Return the chart :func:`auto` would draw, and the reason, without drawing it.

    The rules are checked in this order:

    ====================================================  ======================
    condition                                             chart
    ====================================================  ======================
    ``data`` is a power-law fit                           ``degree_distribution``
    ``data`` is a table indexed by period (not flows)     ``time_series``
    one ``focus`` node, flows with several categories     ``compare``
    one ``focus`` node, several periods                   ``time_series``
    one ``focus`` node                                    ``ego``
    every node has coordinates                            ``flow_map``
    directed, and no node both sends and receives         ``sankey``
    anything else                                         ``network``
    ====================================================  ======================

    ``flow_map`` and ``network`` are animated over time when the data span
    several periods. Nodes have coordinates when they carry ``lon``/``lat``
    attributes, when ``coords=`` is given, or when they are FAOSTAT country
    names.

    Parameters
    ----------
    data
        What would be passed to :func:`auto`.
    **kwargs
        The keyword arguments that would be passed (``focus``, ``time``,
        ``category``, ``labels``, ``node_attrs``, ``coords`` are used).

    Returns
    -------
    tuple of str
        ``(kind, reason)``.
    """
    if isinstance(data, PowerLawFit):
        return "degree_distribution", "the data is a power-law fit"
    if isinstance(data, pd.DataFrame) and not is_flow_table(data):
        return "time_series", "the data is a table of values by period"
    focus = _single_focus(kwargs.get("focus"))
    if (
        focus is not None
        and isinstance(data, pd.DataFrame)
        and is_flow_table(data)
        and _n_categories(data, kwargs.get("category")) > 1
    ):
        return "compare", f"one focus node ({focus}) and several categories"
    frames = to_frames(
        data,
        time=kwargs.get("time"),
        category=kwargs.get("category"),
        labels=kwargs.get("labels"),
        node_attrs=kwargs.get("node_attrs"),
    )
    if focus is not None:
        if frames.animated:
            return "time_series", f"one focus node ({focus}) and several periods"
        return "ego", f"one focus node ({focus})"
    u = frames.union
    if has_coordinates(u, kwargs.get("coords")):
        what = "every node has coordinates"
        return "flow_map", what + (", animated over time" if frames.animated else "")
    if not frames.animated and _only_sends_or_receives(u):
        return "sankey", "every node only sends or only receives"
    return "network", "general graph" + (", animated over time" if frames.animated else "")


def auto(data: Any, kind: Kind | Literal["map"] = "auto", **kwargs: Any) -> go.Figure:
    """Draw a chart by name, or let the data choose one.

    ``auto(g, kind="sankey", top_n=8)`` is the same as
    ``plot.sankey(g, top_n=8)``. With ``kind="auto"`` the chart is chosen by
    the rules in :func:`choose_kind`, and the choice is recorded in the
    figure: ``fig.layout.meta["netviz"]`` holds ``{"kind": ..., "reason": ...}``
    and a line under the title names the chart and the reason.

    Every chart function uses the same argument names (``color_by``,
    ``size_by``, ``edge_width_by``, ``edge_color_by``, ``label_by``,
    ``show_labels``, ``top_n``, ``focus``, ``time``, ``category``, ``labels``,
    ``title``, ``height``), so the same keyword arguments work whichever
    chart is drawn, as long as that chart uses them.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, a mapping from period to graph, a
        period-indexed table, or a power-law fit.
    kind
        A chart name (see :data:`Kind`) or ``"auto"``.
    **kwargs
        Passed to the chart function.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Raises
    ------
    ValueError
        For an unknown ``kind``.
    TypeError
        If a keyword argument is not used by the chosen chart. The message
        names the chart, why it was chosen, and the arguments it accepts.

    Examples
    --------
    >>> import networkx as nx
    >>> import netviz_tools as nv
    >>> fig = nv.plot.auto(nx.karate_club_graph())
    >>> fig.layout.meta["netviz"]["kind"]
    'network'
    """
    reason = "requested"
    if kind == "auto":
        chosen, reason = choose_kind(data, **kwargs)
    elif kind in KINDS:
        chosen = "flow_map" if kind == "map" else kind
    else:
        raise ValueError(f"unknown kind {kind!r}; choose from {['auto', *KINDS]}")
    fn = KINDS[chosen]
    params = inspect.signature(fn).parameters
    accepted = [p for p, spec in params.items() if spec.kind is inspect.Parameter.KEYWORD_ONLY]
    positional = [
        p
        for p, spec in params.items()
        if spec.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD and p != "data" and p != "fit"
    ]
    bad = [k for k in kwargs if k not in accepted and k not in positional]
    if bad:
        why = f" (chosen because {reason})" if kind == "auto" else ""
        raise TypeError(
            f"plot.{chosen}{why} does not use {', '.join(f'{b}=' for b in bad)}; "
            f"it accepts: {', '.join(positional + accepted)}"
        )
    fig = fn(data, **kwargs)
    fig.update_layout(meta={"netviz": {"kind": chosen, "reason": reason}})
    if kind == "auto":
        head = fig.layout.title.text or ""
        note = f"<sup>Chart: {chosen.replace('_', ' ')} ({reason})</sup>"
        fig.update_layout(title={"text": f"{head}<br>{note}" if head else note})
        top = fig.layout.margin.t if fig.layout.margin and fig.layout.margin.t else 60
        fig.update_layout(margin={"t": max(int(top), 80)})
    return fig
