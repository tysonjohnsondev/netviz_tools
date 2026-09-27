"""Interactive node-link diagrams of any NetworkX graph."""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable, Iterable, Mapping
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools._nxutil import has_weights
from netviz_tools.plot._data import Frames, PlotData, Selector, to_frames
from netviz_tools.plot._geo import coordinates, geo_layout, positions
from netviz_tools.plot._render import ShowLabels, figure, prepare, resolve_focus
from netviz_tools.plot._resolve import (
    EdgeSpec,
    NodeSpec,
    compute_metric,
    node_values,
    select_top,
)

__all__ = ["Layout", "community_layout", "ego", "network"]

Layout: TypeAlias = Literal[
    "spring",
    "kamada_kawai",
    "circular",
    "shell",
    "spectral",
    "community",
    "bipartite",
    "multipartite",
    "geo",
]
"""Node placement for :func:`network`.

* ``"spring"``: Fruchterman-Reingold force layout, seeded (default).
* ``"kamada_kawai"``: path-length preserving layout; slow above about 500 nodes.
* ``"circular"``: nodes on a circle, largest first.
* ``"shell"``: concentric circles, one per colour category when ``color_by``
  is categorical.
* ``"spectral"``: eigenvectors of the graph Laplacian; fast on large graphs.
* ``"community"``: each community in its own cluster (see :func:`community_layout`).
* ``"bipartite"``: two columns, from the ``bipartite`` node attribute (0/1) or
  a two-colouring of the graph.
* ``"multipartite"``: one column per value of the ``subset`` node attribute
  (change it with ``layout_options={"subset_key": "layer"}``).
* ``"geo"``: nodes at their ``lon``/``lat`` attributes on a world map.
"""


def community_layout(
    g: nx.Graph[Any], partition: pd.Series, *, seed: int = 42, spread: float = 4.0
) -> dict[Hashable, tuple[float, float]]:
    """Place each community in its own cluster.

    Community centres are laid out with Kamada-Kawai on a ring of the
    communities, then each community is laid out around its centre, with a
    radius that grows with the square root of its size.

    Parameters
    ----------
    g
        The graph.
    partition
        Community id per node, as returned by
        :func:`netviz_tools.metrics.communities`.
    seed
        Seed for the within-community spring layouts.
    spread
        Distance scale between community centres.

    Returns
    -------
    dict
        Node to ``(x, y)``.
    """
    groups: dict[int, list[Hashable]] = {}
    for node, cid in partition.items():
        if node in g:
            groups.setdefault(int(cid), []).append(node)
    ring = nx.cycle_graph(len(groups)) if len(groups) > 2 else nx.path_graph(len(groups))
    centres = nx.kamada_kawai_layout(ring, scale=spread) if len(groups) > 1 else {0: np.zeros(2)}
    pos: dict[Hashable, tuple[float, float]] = {}
    for (_, members), centre in zip(sorted(groups.items()), centres.values(), strict=True):
        sub = g.subgraph(members)
        radius = 0.35 * math.sqrt(len(members))
        inner = nx.spring_layout(sub, seed=seed, center=centre, scale=radius, weight=None)
        pos.update({n: (float(x), float(y)) for n, (x, y) in inner.items()})
    return pos


def _bipartite_top(g: nx.Graph[Any]) -> list[Hashable]:
    marks = nx.get_node_attributes(g, "bipartite")
    if marks:
        return [n for n in g if marks.get(n) == 0]
    try:
        colouring = nx.bipartite.color(g)
    except nx.NetworkXError as exc:
        raise ValueError(
            "layout='bipartite' needs a bipartite graph or a 'bipartite' node attribute (0/1)"
        ) from exc
    return [n for n, c in colouring.items() if c == 0]


def _normalize(raw: Mapping[Hashable, Any]) -> dict[Hashable, tuple[float, float]]:
    arr = np.asarray([tuple(p) for p in raw.values()], dtype=float)
    arr = arr - arr.mean(axis=0)
    extent = float(np.abs(arr).max())
    if extent > 0:
        arr = arr / extent
    return {n: (float(x), float(y)) for n, (x, y) in zip(raw, arr, strict=True)}


def compute_layout(
    g: nx.Graph[Any],
    layout: Layout,
    *,
    order: list[Hashable],
    seed: int,
    partition: pd.Series | None,
    groups: pd.Series | None,
    options: Mapping[str, Any] | None,
) -> dict[Hashable, tuple[float, float]]:
    """Place the nodes of ``g`` and scale positions into ``[-1, 1]``."""
    opts = dict(options or {})
    n = g.number_of_nodes()
    if n == 0:
        return {}
    raw: Mapping[Hashable, Any]
    if layout == "spring":
        raw = nx.spring_layout(g, seed=seed, weight=None, **opts)
    elif layout == "kamada_kawai":
        if n > 500:
            warnings.warn(
                f"kamada_kawai on {n} nodes can take minutes; 'spring' or 'spectral' is faster",
                UserWarning,
                stacklevel=3,
            )
        raw = nx.kamada_kawai_layout(g, weight=None, **opts)
    elif layout == "circular":
        h: nx.Graph[Any] = nx.Graph()
        h.add_nodes_from(order)
        raw = nx.circular_layout(h, **opts)
    elif layout == "shell":
        nlist = None
        if groups is not None:
            nlist = [list(idx) for _, idx in groups.groupby(groups, sort=False).groups.items()]
        raw = nx.shell_layout(g, nlist=nlist, **opts)
    elif layout == "spectral":
        raw = nx.spectral_layout(g, weight=None, **opts) if n > 2 else nx.circular_layout(g)
    elif layout == "community":
        part = partition if partition is not None else compute_metric(g, "community", seed=0)
        raw = community_layout(g, part, seed=seed, **opts)
    elif layout == "bipartite":
        raw = nx.bipartite_layout(g, _bipartite_top(g), **opts)
    elif layout == "multipartite":
        key = opts.pop("subset_key", "subset")
        missing = [v for v in g if key not in g.nodes[v]]
        if missing:
            raise ValueError(
                f"layout='multipartite' needs a {key!r} attribute on every node "
                f"({len(missing)} missing); set layout_options={{'subset_key': ...}}"
            )
        raw = nx.multipartite_layout(g, subset_key=key, **opts)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    return _normalize(raw)


def _choose_nodes(
    u: nx.Graph[Any],
    size_values: pd.Series | None,
    top_n: int | None,
    max_nodes: int | None,
    focus: list[Hashable],
) -> list[Hashable]:
    if size_values is not None and pd.api.types.is_numeric_dtype(size_values):
        rank = size_values
    else:
        rank = pd.Series(dict(u.degree()), dtype=float).reindex(list(u.nodes))
    limit = top_n
    if limit is None and max_nodes is not None and u.number_of_nodes() > max_nodes:
        warnings.warn(
            f"the graph has {u.number_of_nodes()} nodes; drawing the {max_nodes} largest. "
            "Pass top_n= to choose, or max_nodes=None to draw all (slow in the browser).",
            UserWarning,
            stacklevel=3,
        )
        limit = max_nodes
    nodes = select_top(rank, limit)
    nodes += [f for f in focus if f not in set(nodes)]
    return nodes


def _draw_network(
    frames: Frames,
    *,
    layout: Layout,
    color_by: NodeSpec,
    size_by: NodeSpec,
    edge_width_by: EdgeSpec,
    edge_color_by: EdgeSpec,
    label_by: str | None,
    show_labels: ShowLabels,
    top_n: int | None,
    focus: Hashable | Iterable[Hashable] | None,
    title: str | None,
    height: int,
    pos: Mapping[Any, tuple[float, float]] | None,
    partition: pd.Series | None,
    seed: int,
    min_weight_quantile: float | None,
    max_nodes: int | None,
    layout_options: Mapping[str, Any] | None,
    what: str,
) -> go.Figure:
    u = frames.union
    focus_nodes = resolve_focus(u, focus)
    if edge_width_by == "auto":
        edge_width_by = "weight" if has_weights(u) else None
    labels = frames.labels
    size = node_values(u, size_by, arg="size_by", labels=labels, partition=partition, seed=seed)
    color = node_values(u, color_by, arg="color_by", labels=labels, partition=partition, seed=seed)
    nodes = _choose_nodes(u, size.series if size else None, top_n, max_nodes, focus_nodes)
    sub = u.subgraph(nodes)
    geo = layout == "geo"
    if pos is not None:
        missing = [n for n in nodes if n not in pos]
        if missing:
            raise ValueError(f"pos= has no position for {len(missing)} node(s): {missing[:5]}")
        placed = {n: (float(pos[n][0]), float(pos[n][1])) for n in nodes}
    elif geo:
        table = coordinates(sub, None)
        dropped = [str(n) for n in nodes if n not in table.index]
        if dropped:
            warnings.warn(
                f"no coordinates for {len(dropped)} node(s), they are left out: {dropped[:10]}",
                UserWarning,
                stacklevel=3,
            )
            nodes = [n for n in nodes if n in table.index]
        placed = positions(table, nodes)
    else:
        groups = None
        if color is not None and color.categorical:
            groups = color.series.reindex(nodes).astype(str)
        part = partition
        if part is None and color is not None and color.name == "community":
            part = color.series
        placed = compute_layout(
            sub,
            layout,
            order=nodes,
            seed=seed,
            partition=part,
            groups=groups,
            options=layout_options,
        )
    scene = prepare(
        frames,
        nodes=nodes,
        pos=placed,
        geo=geo,
        size=size,
        color=color,
        edge_width_by=edge_width_by,
        edge_color_by=edge_color_by,
        label_by=label_by,
        show_labels=show_labels,
        focus=focus_nodes,
        size_range=(6.0, 26.0) if geo else (8.0, 40.0),
        min_weight_quantile=min_weight_quantile,
    )
    fig = figure(scene, title=title, height=height, what=what)
    if geo:
        fig.update_layout(geo=geo_layout("natural earth"))
    return fig


def network(
    data: PlotData,
    *,
    layout: Layout = "spring",
    color_by: NodeSpec = "auto",
    size_by: NodeSpec = "auto",
    edge_width_by: EdgeSpec = "auto",
    edge_color_by: EdgeSpec = None,
    label_by: str | None = None,
    show_labels: ShowLabels = "auto",
    top_n: int | None = None,
    focus: Hashable | Iterable[Hashable] | None = None,
    time: Selector = None,
    category: Selector = None,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 650,
    pos: Mapping[Any, tuple[float, float]] | None = None,
    partition: pd.Series | None = None,
    seed: int = 42,
    min_weight_quantile: float | None = None,
    max_nodes: int | None = 2000,
    layout_options: Mapping[str, Any] | None = None,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Draw any NetworkX graph as an interactive node-link diagram.

    Works on :class:`networkx.Graph`, :class:`~networkx.DiGraph` and the
    multigraph classes (parallel edges are summed). Directed edges get an
    arrowhead pointing at the target. Hover over a node for its attributes and
    the values behind its colour and size; hover over an arrowhead or edge
    midpoint for the edge.

    With several periods (a flow table spanning several times, or a mapping
    of graphs) the figure is animated: one frame per period, with a play
    button and a time slider. Node positions stay fixed across frames.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, or a mapping from period to graph
        (for example ``nv.graphs_by(flows, by="time")``).
    layout
        Node placement, see :data:`Layout`. Ignored when ``pos`` is given.
    color_by
        Node colour: a node attribute name, a metric name
        (:data:`~netviz_tools.plot.NodeMetric`), a mapping or Series from node
        to value, ``None``, or ``"auto"`` (communities when the graph has
        edges). Text and other non-numeric values get a legend with up to
        eight colours; numbers get a colour bar.
    size_by
        Node size, same forms as ``color_by``. ``"auto"`` is ``"strength"``
        (weighted degree) when edges have a ``weight`` attribute, otherwise
        ``"degree"``. Sizes are log-scaled when values span more than two
        orders of magnitude.
    edge_width_by
        Edge attribute (or mapping from ``(u, v)``) for edge width.
        ``"auto"`` uses ``"weight"`` when present, otherwise equal widths.
    edge_color_by
        Edge attribute or mapping for edge colour, or ``"source"`` /
        ``"target"`` to colour each edge like that end node. ``None`` (default)
        draws edges grey.
    label_by
        Node attribute to use as label text. Defaults to the node itself.
    show_labels
        Which nodes get a text label, see :data:`~netviz_tools.plot.ShowLabels`.
    top_n
        Keep the ``top_n`` largest nodes by ``size_by`` (by degree when
        ``size_by`` is not numeric). ``None`` keeps all, up to ``max_nodes``.
    focus
        Node or nodes to emphasise: always drawn and labelled, outlined, with
        their edges darker while other nodes fade.
    time
        With a flow table or mapping of graphs: one period, or several to
        animate over. ``None`` uses every period.
    category
        With a flow table: the category to draw (required when the table
        holds more than one).
    labels
        Display labels for hover text, legends and captions, for example
        ``{"source": "Exporter", "weight": "Tonnes"}``. Merged over labels
        carried by the data.
    title
        Figure title. Defaults to one built from the graph's category, time
        or name.
    height
        Figure height in pixels.
    pos
        Your own node positions, ``{node: (x, y)}``; overrides ``layout``.
    partition
        Community id per node, used by ``color_by="community"`` and
        ``layout="community"``. Computed with Louvain (seed 0) when omitted.
    seed
        Seed for the layout, so the same call draws the same picture.
    min_weight_quantile
        Hide edges whose ``edge_width_by`` value is below this quantile
        (between 0 and 1) to reduce clutter.
    max_nodes
        Safety cap when ``top_n`` is ``None``: larger graphs are cut to the
        ``max_nodes`` largest nodes with a warning. ``None`` disables it.
    layout_options
        Extra keyword arguments for the NetworkX layout function.
    node_attrs
        With a flow table: a table indexed by node whose columns become node
        attributes (for example ``faostat.countries()``).

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown; call ``fig.show()`` yourself.

    Raises
    ------
    UnknownMetricError
        If ``color_by`` or ``size_by`` is neither a node attribute nor a metric.
    ValueError
        For an unknown layout or focus node, or ``pos`` missing nodes.

    Examples
    --------
    >>> import networkx as nx
    >>> import netviz_tools as nv
    >>> fig = nv.plot.network(nx.karate_club_graph(), color_by="club")
    >>> [t.name for t in fig.data if t.legendgroup and t.legendgroup.startswith("node-")]
    ['Mr. Hi', 'Officer']
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    return _draw_network(
        frames,
        layout=layout,
        color_by=color_by,
        size_by=size_by,
        edge_width_by=edge_width_by,
        edge_color_by=edge_color_by,
        label_by=label_by,
        show_labels=show_labels,
        top_n=top_n,
        focus=focus,
        title=title,
        height=height,
        pos=pos,
        partition=partition,
        seed=seed,
        min_weight_quantile=min_weight_quantile,
        max_nodes=max_nodes,
        layout_options=layout_options,
        what="network",
    )


def ego(
    data: PlotData,
    focus: Hashable | Iterable[Hashable],
    *,
    radius: int = 1,
    undirected: bool = True,
    layout: Layout = "spring",
    color_by: NodeSpec = "auto",
    size_by: NodeSpec = "auto",
    edge_width_by: EdgeSpec = "auto",
    edge_color_by: EdgeSpec = None,
    label_by: str | None = None,
    show_labels: ShowLabels = True,
    top_n: int | None = None,
    time: Selector = None,
    category: Selector = None,
    labels: Mapping[str, str] | None = None,
    title: str | None = None,
    height: int = 600,
    seed: int = 42,
    node_attrs: pd.DataFrame | None = None,
) -> go.Figure:
    """Draw the neighbourhood of one or more nodes.

    Keeps the focus node(s) and every node within ``radius`` steps, then
    draws them like :func:`network` with the focus emphasised.

    Parameters
    ----------
    data
        A NetworkX graph, a flow table, or a mapping from period to graph.
    focus
        The centre node, or several centres (their neighbourhoods are joined).
    radius
        Number of steps from the centre to include.
    undirected
        Follow edges in both directions on a directed graph (default). If
        ``False``, only successors are included.
    layout, color_by, size_by, edge_width_by, edge_color_by, label_by, show_labels
        As in :func:`network`. Every node is labelled by default.
    top_n
        Keep only the ``top_n`` largest neighbours (the centre always stays).
    time, category, labels, title, height, seed, node_attrs
        As in :func:`network`.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure. It is not shown.
    """
    frames = to_frames(data, time=time, category=category, labels=labels, node_attrs=node_attrs)
    u = frames.union
    centres = resolve_focus(u, focus)
    keep: set[Hashable] = set()
    for c in centres:
        keep |= set(nx.ego_graph(u, c, radius=radius, undirected=undirected))
    restricted = Frames(
        {k: g.subgraph([n for n in g if n in keep]).copy() for k, g in frames.graphs.items()},
        frames.labels,
        frames.unit,
    )
    if title is None:
        names = ", ".join(map(str, centres))
        steps = "1 step" if radius == 1 else f"{radius} steps"
        title = f"{names}: neighbours within {steps}"
        t = frames.first.graph.get("time")
        if not frames.animated and isinstance(t, int):
            title += f", {t}"
    return _draw_network(
        restricted,
        layout=layout,
        color_by=color_by,
        size_by=size_by,
        edge_width_by=edge_width_by,
        edge_color_by=edge_color_by,
        label_by=label_by,
        show_labels=show_labels,
        top_n=top_n,
        focus=centres,
        title=title,
        height=height,
        pos=None,
        partition=None,
        seed=seed,
        min_weight_quantile=None,
        max_nodes=None,
        layout_options=None,
        what="neighbourhood",
    )
