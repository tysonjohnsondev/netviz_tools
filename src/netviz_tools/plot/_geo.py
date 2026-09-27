"""Flow map: the largest flows drawn as great-circle arcs on a world map."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

from netviz_tools._nxutil import has_weights
from netviz_tools.plot._data import PlotData, Selector, to_frames
from netviz_tools.plot._render import ShowLabels, figure, prepare, resolve_focus
from netviz_tools.plot._resolve import EdgeSpec, NodeSpec, edge_values, node_values

__all__ = ["flow_map"]


def coordinates(g: nx.Graph[Any], coords: pd.DataFrame | None) -> pd.DataFrame:
    """Return a ``lon``/``lat`` table for the nodes of ``g``.

    Order of preference: ``coords``; the ``lon``/``lat`` node attributes when
    every node has them; the FAOSTAT country label points bundled with the
    package (only useful when nodes are FAOSTAT country names).
    """
    if coords is not None:
        missing = {"lon", "lat"} - set(coords.columns)
        if missing:
            raise ValueError(f"coords is missing columns {sorted(missing)}")
        return coords[["lon", "lat"]]
    attrs = {n: (d.get("lon"), d.get("lat")) for n, d in g.nodes(data=True)}
    if attrs and all(lon is not None and lat is not None for lon, lat in attrs.values()):
        return pd.DataFrame.from_dict(attrs, orient="index", columns=["lon", "lat"])
    from netviz_tools.datasets.faostat import countries

    return countries()[["lon", "lat"]]


def has_coordinates(g: nx.Graph[Any]) -> bool:
    """True when every node has coordinates (attributes or FAOSTAT country names)."""
    if g.number_of_nodes() == 0:
        return False
    table = coordinates(g, None)
    return all(n in table.index for n in g.nodes)


def geo_layout(projection: str) -> dict[str, Any]:
    """Return the Plotly ``geo`` layout used by map figures."""
    return {
        "projection": {"type": projection},
        "showland": True,
        "landcolor": "#efeeea",
        "showcountries": True,
        "countrycolor": "#d6d5cf",
        "showocean": False,
        "coastlinecolor": "#c3c2b7",
        "showframe": False,
    }


def flow_map(
    data: PlotData,
    *,
    color_by: NodeSpec = "auto",
    size_by: NodeSpec = "auto",
    edge_width_by: EdgeSpec = "auto",
    edge_color_by: EdgeSpec = None,
    label_by: str | None = None,
    show_labels: ShowLabels = False,
    top_n: int = 60,
    focus: Hashable | Iterable[Hashable] | None = None,
    time: Selector = None,
    category: Selector = None,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 550,
    coords: pd.DataFrame | None = None,
    projection: str = "natural earth",
    seed: int = 42,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Draw the largest flows on a world map, animated over time when there are several periods.

    Edges are drawn as great-circle lines between node coordinates. With a
    flow table spanning several periods (or a mapping of graphs), each period
    becomes an animation frame with a play button and a time slider, so you
    can watch flows between places change year by year: trade, migration,
    shipping, or anything else with an origin and a destination.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, or a mapping from period to graph.
    color_by, size_by, edge_width_by, edge_color_by, label_by
        As in :func:`~netviz_tools.plot.network`.
    show_labels
        Which nodes get a text label (none by default; hover shows names).
    top_n
        Number of largest edges (by ``edge_width_by``) to draw in each period.
    focus
        Node or nodes to follow: only flows touching them are drawn (the
        ``top_n`` largest), and they are outlined and labelled.
    time, category, labels, title, height, seed, node_attrs
        As in :func:`~netviz_tools.plot.network`.
    coords
        Table indexed by node with ``lon`` and ``lat`` columns. Defaults to
        node attributes ``lon``/``lat`` when every node has them, otherwise
        to the FAOSTAT country label points bundled with the package (derived
        from Natural Earth).
    projection
        Any Plotly geo projection type, for example ``"natural earth"`` or
        ``"equirectangular"``.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.

    Warns
    -----
    UserWarning
        If some endpoints of the selected edges have no coordinates. Those
        edges are skipped.
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    u = frames.union()
    focus_nodes = set(resolve_focus(u, focus))
    table = coordinates(u, coords)
    if edge_width_by == "auto":
        edge_width_by = "weight" if has_weights(u) else None
    directed = u.is_directed()

    missing: set[str] = set()

    def top_edges(g: nx.Graph[Any], _key: Hashable) -> list[tuple[Hashable, Hashable]]:
        cand = [
            (a, b, d)
            for a, b, d in g.edges(data=True)
            if not focus_nodes or a in focus_nodes or b in focus_nodes
        ]
        w = edge_values(cand, edge_width_by, arg="edge_width_by", directed=directed)
        order = (
            pd.to_numeric(w, errors="coerce")
            .fillna(0.0)
            .sort_values(ascending=False, kind="stable")
            if w is not None
            else pd.Series(range(len(cand)), dtype=float)
        )
        chosen = [cand[i] for i in order.index[:top_n]]
        for a, b, _ in chosen:
            missing.update(str(n) for n in (a, b) if n not in table.index)
        return [(a, b) for a, b, _ in chosen if a in table.index and b in table.index]

    selected: list[Hashable] = []
    for key, g in frames.graphs.items():
        for a, b in top_edges(g, key):
            for n in (a, b):
                if n not in selected:
                    selected.append(n)
    selected += [n for n in focus_nodes if n not in selected and n in table.index]
    if missing:
        warnings.warn(
            f"no coordinates for {len(missing)} node(s), their flows are skipped: "
            f"{sorted(missing)[:10]}",
            UserWarning,
            stacklevel=2,
        )
    labels_ = frames.labels
    size = node_values(u, size_by, arg="size_by", labels=labels_, seed=seed)
    color = node_values(u, color_by, arg="color_by", labels=labels_, seed=seed)
    if size is not None and pd.api.types.is_numeric_dtype(size.values):
        vals = size.values.reindex(selected).fillna(-1.0)
        selected = sorted(selected, key=lambda n: (-float(vals[n]), str(n)))
    pos = {n: (float(table.at[n, "lon"]), float(table.at[n, "lat"])) for n in selected}
    scene = prepare(
        frames,
        nodes=selected,
        pos=pos,
        geo=True,
        size_by=size_by,
        color_by=color_by,
        edge_width_by=edge_width_by,
        edge_color_by=edge_color_by,
        label_by=label_by,
        show_labels=show_labels,
        focus=sorted(focus_nodes, key=str),
        partition=None,
        seed=seed,
        size_range=(5.0, 26.0),
        edge_filter=top_edges,
        union=u,
        resolved=(size, color),
    )
    if focus_nodes:
        what = "flows of " + ", ".join(map(str, sorted(focus_nodes, key=str)))
    else:
        what = f"flows, {top_n} largest"
    fig = figure(scene, title=title, height=height, what=what)
    fig.update_layout(geo=geo_layout(projection), margin={"l": 0, "r": 0, "t": 60})
    return fig
