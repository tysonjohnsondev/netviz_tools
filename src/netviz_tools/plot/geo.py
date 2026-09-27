"""Flow map: the largest flows drawn as great-circle arcs on a world map."""

from __future__ import annotations

import warnings
from collections.abc import Hashable
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.plot._style import SURFACE, categories, default_title, scale, unit_of

__all__ = ["flow_map"]


def _coordinates(g: nx.Graph[Any], coords: pd.DataFrame | None) -> pd.DataFrame:
    if coords is not None:
        missing = {"lon", "lat"} - set(coords.columns)
        if missing:
            raise ValueError(f"coords is missing columns {sorted(missing)}")
        return coords[["lon", "lat"]]
    attrs = {n: (d.get("lon"), d.get("lat")) for n, d in g.nodes(data=True)}
    if all(lon is not None and lat is not None for lon, lat in attrs.values()):
        return pd.DataFrame.from_dict(attrs, orient="index", columns=["lon", "lat"])
    from netviz_tools.datasets.faostat import countries

    return countries()[["lon", "lat"]]


def flow_map(
    g: nx.Graph[Any],
    *,
    top_n: int = 60,
    coords: pd.DataFrame | None = None,
    color_by: str | None = "continent",
    projection: str = "natural earth",
    title: str | None = None,
    height: int = 550,
) -> go.Figure:
    """Draw the largest flows of a graph on a world map.

    Parameters
    ----------
    g
        A graph from :func:`netviz_tools.build_graph`. Build it with
        ``node_attrs=faostat.countries()`` to colour nodes by continent.
    top_n
        Number of largest edges to draw.
    coords
        Table indexed by node with ``lon`` and ``lat`` columns. Defaults to
        node attributes ``lon``/``lat`` if every node has them, otherwise to
        the FAOSTAT country label points bundled with the package (derived
        from Natural Earth).
    color_by
        Node attribute used for marker colour and legend groups, or ``None``.
    projection
        Any Plotly geo projection type, for example ``"natural earth"`` or
        ``"equirectangular"``.
    title
        Figure title. Defaults to one built from the graph's item and year.
    height
        Figure height in pixels.

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
    table = _coordinates(g, coords)
    edges = sorted(g.edges(data="weight", default=1.0), key=lambda e: e[2], reverse=True)[:top_n]
    missing = sorted({str(n) for u, v, _ in edges for n in (u, v) if n not in table.index})
    if missing:
        warnings.warn(
            f"no coordinates for {len(missing)} node(s), their flows are skipped: {missing[:10]}",
            UserWarning,
            stacklevel=2,
        )
    edges = [(u, v, w) for u, v, w in edges if u in table.index and v in table.index]
    lonlat: dict[Hashable, tuple[float, float]] = {
        n: (float(lon), float(lat))
        for n, lon, lat in zip(table.index, table["lon"], table["lat"], strict=True)
    }
    unit = unit_of(g)
    fig = go.Figure()
    if edges:
        widths = scale([w for *_, w in edges], 0.6, 6.0)
        buckets = np.digitize(widths, np.linspace(0.6, 6.0, 5)[1:-1])
        for b in sorted(set(buckets.tolist())):
            lons: list[float | None] = []
            lats: list[float | None] = []
            for (u, v, _), bb in zip(edges, buckets, strict=True):
                if bb == b:
                    lons += [lonlat[u][0], lonlat[v][0], None]
                    lats += [lonlat[u][1], lonlat[v][1], None]
            fig.add_trace(
                go.Scattergeo(
                    lon=lons,
                    lat=lats,
                    mode="lines",
                    line={
                        "width": float(np.mean(widths[buckets == b])),
                        "color": "rgba(42, 120, 214, 0.45)",
                    },
                    hoverinfo="skip",
                    showlegend=False,
                    name=f"flows {b}",
                )
            )
    nodes: list[Hashable] = list(dict.fromkeys(n for u, v, _ in edges for n in (u, v)))
    strength = dict(g.degree(weight="weight"))
    sizes = dict(zip(nodes, scale([strength[n] for n in nodes], 5.0, 26.0), strict=True))
    labels, colors = categories(g, nodes, color_by, None)
    out_top: dict[Hashable, list[str]] = {}
    for u, v, w in edges:
        out_top.setdefault(u, []).append(f"{v}: {w:,.0f}{unit}")
    for cat, color in colors.items():
        members = [n for n in nodes if labels[n] == cat]
        if not members:
            continue
        fig.add_trace(
            go.Scattergeo(
                lon=[lonlat[n][0] for n in members],
                lat=[lonlat[n][1] for n in members],
                mode="markers",
                marker={
                    "size": [sizes[n] for n in members],
                    "color": color,
                    "line": {"width": 1, "color": SURFACE},
                },
                hovertext=[
                    f"{n}<br>strength {strength[n]:,.0f}{unit}"
                    + ("<br>" + "<br>".join(out_top[n][:5]) if n in out_top else "")
                    for n in members
                ],
                hoverinfo="text",
                name=cat or "nodes",
                showlegend=bool(cat),
            )
        )
    fig.update_layout(
        title={"text": title or default_title(g, f"trade, {len(edges)} largest flows")},
        template="plotly_white",
        height=height,
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        legend={"title": {"text": (color_by or "").replace("_", " ").capitalize()}},
        geo={
            "projection": {"type": projection},
            "showland": True,
            "landcolor": "#efeeea",
            "showcountries": True,
            "countrycolor": "#d6d5cf",
            "showocean": False,
            "coastlinecolor": "#c3c2b7",
            "showframe": False,
        },
    )
    return fig
