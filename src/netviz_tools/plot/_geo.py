"""Flow map: the largest flows drawn as great-circle arcs on a world map."""

from __future__ import annotations

import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools._nxutil import has_weights
from netviz_tools.plot._data import PlotData, Selector, to_frames
from netviz_tools.plot._render import ShowLabels, figure, prepare, resolve_focus
from netviz_tools.plot._resolve import (
    EdgeSpec,
    NodeSpec,
    edge_values,
    node_index,
    node_values,
    select_top,
)

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


def has_coordinates(g: nx.Graph[Any], coords: pd.DataFrame | None = None) -> bool:
    """Return True when every node of ``g`` has coordinates (see :func:`coordinates`)."""
    if g.number_of_nodes() == 0:
        return False
    index = coordinates(g, coords).index
    return all(n in index for n in g.nodes)


def positions(table: pd.DataFrame, nodes: list[Hashable]) -> dict[Hashable, tuple[float, float]]:
    """Return ``{node: (lon, lat)}`` for nodes listed in a coordinate table."""
    rows = table.reindex(node_index(nodes))
    lon = rows["lon"].to_numpy(dtype=float)
    lat = rows["lat"].to_numpy(dtype=float)
    return {n: (float(x), float(y)) for n, x, y in zip(nodes, lon, lat, strict=True)}


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
    u = frames.union
    focus_nodes = resolve_focus(u, focus)
    table = coordinates(u, coords)
    if edge_width_by == "auto":
        edge_width_by = "weight" if has_weights(u) else None
    directed = u.is_directed()

    every = [e for g in frames.graphs.values() for e in g.edges(data=True)]
    edge_values(every, edge_width_by, arg="edge_width_by", directed=directed)  # check the name

    # The top_n largest edges of each period (touching a focus node, if any).
    missing: set[str] = set()
    chosen: dict[Hashable, list[tuple[Hashable, Hashable]]] = {}
    for key, g in frames.graphs.items():
        cand = [
            (a, b, d)
            for a, b, d in g.edges(data=True)
            if a != b and (not focus_nodes or a in focus_nodes or b in focus_nodes)
        ]
        w = edge_values(cand, edge_width_by, arg="edge_width_by", directed=directed, strict=False)
        rank = (
            pd.to_numeric(w, errors="coerce").fillna(0.0).to_numpy()
            if w is not None
            else np.zeros(len(cand))
        )
        top = [cand[i] for i in np.argsort(-rank, kind="stable")[:top_n]]
        missing.update(str(n) for a, b, _ in top for n in (a, b) if n not in table.index)
        chosen[key] = [(a, b) for a, b, _ in top if a in table.index and b in table.index]
    if missing:
        warnings.warn(
            f"no coordinates for {len(missing)} node(s), their flows are skipped: "
            f"{sorted(missing)[:10]}",
            UserWarning,
            stacklevel=2,
        )
    ends = dict.fromkeys(n for pairs in chosen.values() for pair in pairs for n in pair)
    selected = [*ends, *(n for n in focus_nodes if n not in ends and n in table.index)]
    lab = frames.labels
    size = node_values(u, size_by, arg="size_by", labels=lab, seed=seed)
    color = node_values(u, color_by, arg="color_by", labels=lab, seed=seed)
    if size is not None and not size.categorical:
        selected = select_top(size.series.reindex(node_index(selected)), None)
    scene = prepare(
        frames,
        nodes=selected,
        pos=positions(table, selected),
        geo=True,
        size=size,
        color=color,
        edge_width_by=edge_width_by,
        edge_color_by=edge_color_by,
        label_by=label_by,
        show_labels=show_labels,
        focus=focus_nodes,
        size_range=(5.0, 26.0),
        edge_subset=chosen,
    )
    if focus_nodes:
        what = "flows of " + ", ".join(map(str, focus_nodes))
    else:
        what = f"flows, {top_n} largest"
    fig = figure(scene, title=title, height=height, what=what)
    fig.update_layout(geo=geo_layout(projection), margin={"l": 0, "r": 0, "t": 60})
    return fig
