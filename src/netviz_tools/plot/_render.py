"""Shared node-link renderer for network diagrams and flow maps (private).

Both :func:`~netviz_tools.plot.network` and :func:`~netviz_tools.plot.flow_map`
reduce to: pick nodes and edges, place nodes, then draw edges and nodes with
the same colour, size, label and hover rules. With several periods every
period becomes one animation frame; every frame has the same traces in the
same order, which is what Plotly needs to animate smoothly.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.analysis import suggest
from netviz_tools.plot._data import Frames
from netviz_tools.plot._resolve import (
    Categories,
    EdgeSpec,
    NodeValues,
    categories,
    edge_values,
    is_numeric,
    metric_label,
    node_values,
)
from netviz_tools.plot._style import (
    ARROW,
    EDGE_COLOR,
    EDGE_FOCUS_COLOR,
    EN_DASH,
    MISSING_COLOR,
    MISSING_LABEL,
    OTHER_COLOR,
    OTHER_LABEL,
    SEQUENTIAL,
    SURFACE,
    TEXT_COLOR,
    Scaler,
    base_layout,
    default_title,
    fmt,
    lower_label,
    rgba,
    sample_sequential,
)

Position: TypeAlias = dict[Hashable, tuple[float, float]]
Edge: TypeAlias = tuple[Hashable, Hashable, Mapping[str, Any]]
EdgeSubset: TypeAlias = Mapping[Hashable, list[tuple[Hashable, Hashable]]]
ShowLabels: TypeAlias = "bool | int | Sequence[Hashable] | Literal['auto']"
"""``"auto"`` labels every node when there are at most 30, otherwise the 12
largest; an int labels that many of the largest nodes; ``True``/``False`` all
or none; a sequence labels exactly those nodes. Focus nodes are always
labelled."""

_AUTO_LABEL_ALL: Final = 30
_AUTO_LABEL_TOP: Final = 12
_N_WIDTH_BUCKETS: Final = 5
_N_COLOR_BINS: Final = 5
_EDGE_WIDTH_RANGE: Final = (0.5, 6.0)
_ARROW_AT: Final = 0.62
_MAX_HOVER_ATTRS: Final = 6
_FADED_NODE_OPACITY: Final = 0.45
_CAPTION_COLOR: Final = "#6b6a66"


@dataclass(frozen=True, eq=False)
class Scene:
    """Everything needed to draw one or more frames, resolved once for all of them."""

    frames: Frames
    nodes: list[Hashable]
    pos: Position
    geo: bool
    directed: bool
    focus: frozenset[Hashable]
    text: dict[Hashable, str]
    size: NodeValues | None
    size_by_frame: dict[Hashable, pd.Series]
    size_scaler: Scaler | None
    size_range: tuple[float, float]
    color: NodeValues | None
    color_by_frame: dict[Hashable, pd.Series]
    color_scaler: Scaler | None
    cats: Categories | None
    edge_subset: EdgeSubset | None
    width_spec: EdgeSpec
    width_scaler: Scaler | None
    min_weight: float | None
    edge_color_spec: EdgeSpec
    edge_color_scaler: Scaler | None
    edge_cats: Categories | None

    @property
    def labels(self) -> dict[str, str]:
        """Display labels."""
        return self.frames.labels

    @property
    def unit(self) -> str:
        """Unit of the edge weights."""
        return self.frames.unit


GroupKey: TypeAlias = tuple[str, int, bool]
"""Edge line group: (colour key, width bucket, touches a focus node)."""


@dataclass(frozen=True, eq=False)
class FrameEdges:
    """The edges drawn in one period, with their group, width value and colour value."""

    edges: list[Edge]
    groups: list[GroupKey]
    widths: list[float]
    colors: list[Any]


def resolve_focus(g: nx.Graph[Any], focus: Hashable | Iterable[Hashable] | None) -> list[Hashable]:
    """Normalize ``focus`` into a list of nodes that exist in ``g``.

    Strings, numbers and tuples are single nodes; other iterables hold several.
    """
    if focus is None:
        return []
    items = (
        [focus]
        if isinstance(focus, str | int | tuple) or not isinstance(focus, Iterable)
        else list(focus)
    )
    for n in items:
        if n not in g:
            hint = suggest(str(n), [str(x) for x in g.nodes])
            extra = f". Did you mean: {', '.join(map(repr, hint))}?" if hint else ""
            raise ValueError(f"focus node {n!r} is not in the graph{extra}")
    return items


def labelled_nodes(
    nodes: Sequence[Hashable], show: ShowLabels, focus: Iterable[Hashable]
) -> set[Hashable]:
    """Apply the :data:`ShowLabels` rule to nodes ordered largest first."""
    if show == "auto":
        chosen = set(nodes) if len(nodes) <= _AUTO_LABEL_ALL else set(nodes[:_AUTO_LABEL_TOP])
    elif show is True:
        chosen = set(nodes)
    elif show is False:
        chosen = set()
    elif isinstance(show, int):
        chosen = set(nodes[:show])
    else:
        chosen = set(show)
    return chosen | set(focus)


def _label_text(g: nx.Graph[Any], label_by: str | None) -> Callable[[Hashable], str]:
    if label_by is None:
        return str

    def text(n: Hashable) -> str:
        value = g.nodes[n].get(label_by) if n in g else None
        return str(n) if value is None else str(value)

    return text


def _frame_values(frames: Frames, g: nx.Graph[Any], resolved: NodeValues) -> pd.Series:
    """Values for one period: metrics are recomputed, attributes and mappings are not."""
    if not frames.animated or not resolved.metric or resolved.categorical:
        return resolved.series
    vals = node_values(g, resolved.name, arg="size_by", labels=frames.labels)
    assert vals is not None  # a metric name always resolves
    return vals.series


def _numeric_scaler(
    per_frame: Mapping[Hashable, pd.Series], nodes: list[Hashable], lo: float, hi: float
) -> Scaler:
    allv = np.concatenate([s.reindex(nodes).to_numpy(dtype=float) for s in per_frame.values()])
    return Scaler(allv, lo, hi)


def _all_edges(
    frames: Frames, nodes: list[Hashable], edge_subset: EdgeSubset | None
) -> dict[Hashable, list[Edge]]:
    node_set = set(nodes)
    out: dict[Hashable, list[Edge]] = {}
    for key, g in frames.graphs.items():
        if edge_subset is not None:
            out[key] = [(a, b, g[a][b]) for a, b in edge_subset[key]]
        else:
            out[key] = [
                (a, b, d) for a, b, d in g.edges(data=True) if a in node_set and b in node_set
            ]
    return out


def prepare(
    frames: Frames,
    *,
    nodes: list[Hashable],
    pos: Position,
    geo: bool,
    size: NodeValues | None,
    color: NodeValues | None,
    edge_width_by: EdgeSpec,
    edge_color_by: EdgeSpec,
    label_by: str | None,
    show_labels: ShowLabels,
    focus: Iterable[Hashable],
    size_range: tuple[float, float],
    min_weight_quantile: float | None = None,
    edge_subset: EdgeSubset | None = None,
) -> Scene:
    """Resolve the values and scales shared by all frames.

    ``size`` and ``color`` are node values already resolved on the union of
    all periods. Metrics are recomputed per period, and every scale spans all
    periods, so frames are comparable. ``edge_subset`` restricts the edges
    drawn in each period (by period key); by default every edge between two
    drawn nodes is drawn.
    """
    union = frames.union
    directed = union.is_directed()
    focus = list(focus)
    cats = (
        categories(color.series.reindex(nodes), name=color.name)
        if color is not None and color.categorical
        else None
    )
    size_by_frame: dict[Hashable, pd.Series] = {}
    color_by_frame: dict[Hashable, pd.Series] = {}
    for key, g in frames.graphs.items():
        if size is not None:
            size_by_frame[key] = _frame_values(frames, g, size)
        if color is not None and not color.categorical:
            color_by_frame[key] = _frame_values(frames, g, color)
    size_scaler = (
        _numeric_scaler(size_by_frame, nodes, *size_range)
        if size is not None and not size.categorical
        else None
    )
    color_scaler = _numeric_scaler(color_by_frame, nodes, 0.0, 1.0) if color_by_frame else None

    # Edge widths and colours, scaled across all frames.
    widths: list[float] = []
    ecolors: list[Any] = []
    for edges in _all_edges(frames, nodes, edge_subset).values():
        w = edge_values(edges, edge_width_by, arg="edge_width_by", directed=directed)
        if w is not None:
            widths.extend(pd.to_numeric(w, errors="coerce").tolist())
        if edge_color_by not in ("source", "target"):
            c = edge_values(edges, edge_color_by, arg="edge_color_by", directed=directed)
            if c is not None:
                ecolors.extend(c.tolist())
    width_scaler = Scaler(widths, *_EDGE_WIDTH_RANGE) if widths else None
    min_weight = None
    if widths and min_weight_quantile is not None and np.isfinite(widths).any():
        min_weight = float(np.nanquantile(np.asarray(widths, dtype=float), min_weight_quantile))
    edge_color_scaler = None
    edge_cats = None
    if ecolors:
        es = pd.Series(ecolors, dtype=object)
        if is_numeric(es):
            edge_color_scaler = Scaler(es.astype(float), 0.0, 1.0)
        else:
            edge_cats = categories(es, name=str(edge_color_by))

    text_of = _label_text(union, label_by)
    shown = labelled_nodes(nodes, show_labels, focus)
    return Scene(
        frames=frames,
        nodes=nodes,
        pos=pos,
        geo=geo,
        directed=directed,
        focus=frozenset(focus),
        text={n: text_of(n) for n in nodes if n in shown},
        size=size,
        size_by_frame=size_by_frame,
        size_scaler=size_scaler,
        size_range=size_range,
        color=color,
        color_by_frame=color_by_frame,
        color_scaler=color_scaler,
        cats=cats,
        edge_subset=edge_subset,
        width_spec=edge_width_by,
        width_scaler=width_scaler,
        min_weight=min_weight,
        edge_color_spec=edge_color_by,
        edge_color_scaler=edge_color_scaler,
        edge_cats=edge_cats,
    )


def _node_color(scene: Scene, n: Hashable, key: Hashable) -> str:
    """Colour of node ``n`` in period ``key`` (used by ``edge_color_by="source"``)."""
    if scene.cats is not None:
        return scene.cats.colors[str(scene.cats.labels.get(n))]
    if scene.color_scaler is not None:
        v = scene.color_by_frame[key].get(n, np.nan)
        if pd.isna(v):
            return MISSING_COLOR
        return sample_sequential(float(scene.color_scaler.unit([v])[0]))
    return SEQUENTIAL[3][1]


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _frame_edges(scene: Scene, key: Hashable, edges: list[Edge]) -> FrameEdges:
    """Drop edges below the weight cut, then group the rest by colour and width."""
    w = edge_values(edges, scene.width_spec, arg="edge_width_by", directed=scene.directed)
    if w is not None and scene.min_weight is not None:
        keep = pd.to_numeric(w, errors="coerce").fillna(-np.inf).to_numpy() >= scene.min_weight
        edges = [e for e, k in zip(edges, keep, strict=True) if k]
        w = w[keep].reset_index(drop=True)
    widths = (
        pd.to_numeric(w, errors="coerce").tolist() if w is not None else [math.nan] * len(edges)
    )
    if scene.width_scaler is not None:
        t = np.nan_to_num(scene.width_scaler.unit(widths), nan=0.0)
        buckets = np.minimum((t * _N_WIDTH_BUCKETS).astype(int), _N_WIDTH_BUCKETS - 1).tolist()
    else:
        buckets = [1] * len(edges)
    colors: list[Any] = [None] * len(edges)
    cspec = scene.edge_color_spec
    if cspec == "source":
        ckeys = ["node:" + _node_color(scene, u, key) for u, _, _ in edges]
    elif cspec == "target":
        ckeys = ["node:" + _node_color(scene, v, key) for _, v, _ in edges]
    elif cspec is not None:
        c = edge_values(edges, cspec, arg="edge_color_by", directed=scene.directed)
        colors = c.tolist() if c is not None else colors
        ckeys = [_edge_color_key(scene, v) for v in colors]
    else:
        ckeys = ["base"] * len(edges)
    focus = scene.focus
    groups = [
        (ck, b, bool(focus) and (u in focus or v in focus))
        for ck, b, (u, v, _) in zip(ckeys, buckets, edges, strict=True)
    ]
    return FrameEdges(edges, groups, widths, colors)


def _edge_color_key(scene: Scene, v: Any) -> str:
    if _is_missing(v):
        return "missing"
    if scene.edge_color_scaler is not None:
        t = float(scene.edge_color_scaler.unit([v])[0])
        return f"bin:{min(int(t * _N_COLOR_BINS), _N_COLOR_BINS - 1)}"
    assert scene.edge_cats is not None
    label = str(v)
    return "cat:" + (label if label in scene.edge_cats.colors else OTHER_LABEL)


def _group_style(scene: Scene, ck: str, focused: bool) -> tuple[str, str, bool]:
    """Return (colour, legend name, show in legend) for an edge colour key."""
    if ck == "base":
        if focused:
            return EDGE_FOCUS_COLOR, "", False
        return (rgba("#6e6e6e", 0.18) if scene.focus else EDGE_COLOR), "", False
    if ck == "missing":
        return rgba(MISSING_COLOR, 0.6), MISSING_LABEL, True
    alpha = 0.8 if focused or not scene.focus else 0.25
    if ck.startswith("node:"):
        return rgba(ck[5:], alpha * 0.7), "", False
    if ck.startswith("bin:"):
        b = int(ck[4:])
        color = sample_sequential((b + 0.5) / _N_COLOR_BINS)
        return rgba(color, alpha), _bin_label(scene, b), True
    assert scene.edge_cats is not None
    name = ck[4:]
    return rgba(scene.edge_cats.colors.get(name, OTHER_COLOR), alpha), name, True


def _bin_label(scene: Scene, b: int) -> str:
    s = scene.edge_color_scaler
    assert s is not None
    step = (s.vmax - s.vmin) / _N_COLOR_BINS
    lo, hi = s.inverse(s.vmin + step * b), s.inverse(s.vmin + step * (b + 1))
    return f"{fmt(lo)} to {fmt(hi)}"


def _trace_class(scene: Scene) -> tuple[str, str, type[go.Scatter] | type[go.Scattergeo]]:
    return ("lon", "lat", go.Scattergeo) if scene.geo else ("x", "y", go.Scatter)


def _line_width(bucket: int) -> float:
    lo, hi = _EDGE_WIDTH_RANGE
    return lo + (bucket + 0.5) / _N_WIDTH_BUCKETS * (hi - lo)


def frame_traces(
    scene: Scene, key: Hashable, fe: FrameEdges, group_keys: list[GroupKey]
) -> list[Any]:
    """Build the traces of one frame, in a fixed order.

    One line trace per edge group (empty when the group has no edge in this
    period), the edge hover/arrow markers, self-loops (not on maps), then
    the node traces.
    """
    g = scene.frames.graphs[key]
    xk, yk, cls = _trace_class(scene)
    pos = scene.pos
    traces: list[Any] = []
    in_legend: set[str] = set()
    for gk in group_keys:
        ck, bucket, focused = gk
        xs: list[float | None] = []
        ys: list[float | None] = []
        for (u, v, _), grp in zip(fe.edges, fe.groups, strict=True):
            if grp == gk and u != v:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        color, name, legend = _group_style(scene, ck, focused)
        traces.append(
            cls(
                **{xk: xs, yk: ys},
                mode="lines",
                line={"width": _line_width(bucket), "color": color},
                hoverinfo="skip",
                name=name or "edge lines",
                legendgroup=f"edge-{ck}",
                showlegend=legend and ck not in in_legend,
            )
        )
        in_legend.add(ck)
    traces.append(_edge_markers(scene, fe))
    if not scene.geo:
        traces.append(_self_loop_trace(scene, g))
    traces.extend(_node_traces(scene, g, key))
    return traces


def _edge_text(scene: Scene, u: Hashable, v: Hashable, w: float, c: Any) -> str:
    lines = [f"{u} {ARROW if scene.directed else EN_DASH} {v}"]
    spec = scene.width_spec
    if not math.isnan(w):
        if spec == "weight":
            lines.append(f"{scene.labels['weight']}: {fmt(w, scene.unit)}")
        elif isinstance(spec, str):
            lines.append(f"{spec.replace('_', ' ')}: {fmt(w)}")
        else:
            lines.append(f"value: {fmt(w)}")
    cspec = scene.edge_color_spec
    if isinstance(cspec, str) and cspec not in ("source", "target", spec) and not _is_missing(c):
        lines.append(f"{cspec.replace('_', ' ')}: {c if isinstance(c, str) else fmt(float(c))}")
    return "<br>".join(lines)


def _edge_markers(scene: Scene, fe: FrameEdges) -> Any:
    """Hover targets on the edges; on a directed node-link diagram, also arrowheads."""
    xk, yk, cls = _trace_class(scene)
    pos = scene.pos
    real = [
        (u, v, w, c)
        for (u, v, _), w, c in zip(fe.edges, fe.widths, fe.colors, strict=True)
        if u != v
    ]
    t = _ARROW_AT if scene.directed else 0.5
    xs = [pos[u][0] + t * (pos[v][0] - pos[u][0]) for u, v, _, _ in real]
    ys = [pos[u][1] + t * (pos[v][1] - pos[u][1]) for u, v, _, _ in real]
    text = [_edge_text(scene, u, v, w, c) for u, v, w, c in real]
    marker: dict[str, Any]
    if scene.directed and not scene.geo:
        # Plotly rotates markers clockwise from "up", so the angle of the
        # direction (dx, dy) is atan2(dx, dy). The y axis is scale-anchored
        # to x, so screen angles equal data angles.
        angles = [
            math.degrees(math.atan2(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]))
            for u, v, _, _ in real
        ]
        if scene.width_scaler is not None:
            unit = np.nan_to_num(scene.width_scaler.unit([w for _, _, w, _ in real]), nan=0.0)
            sizes = (7 + 5 * unit).tolist()
        else:
            sizes = [8.0] * len(real)
        colors = [
            "rgba(60, 60, 60, 0.85)"
            if not scene.focus or u in scene.focus or v in scene.focus
            else "rgba(110, 110, 110, 0.25)"
            for u, v, _, _ in real
        ]
        marker = {
            "symbol": "triangle-up",
            "angle": angles,
            "size": sizes,
            "color": colors,
            "line": {"width": 0},
        }
    else:
        marker = {"size": 8, "opacity": 0}
    return cls(
        **{xk: xs, yk: ys},
        mode="markers",
        marker=marker,
        hovertext=text,
        hoverinfo="text",
        showlegend=False,
        name="edges",
    )


def _self_loop_trace(scene: Scene, g: nx.Graph[Any]) -> go.Scatter:
    xs: list[float | None] = []
    ys: list[float | None] = []
    loops = [n for n in nx.nodes_with_selfloops(g) if n in scene.pos]
    if loops:
        arr = np.asarray(list(scene.pos.values()), dtype=float)
        r = 0.025 * float(max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1]), 1e-9))
        theta = np.linspace(0, 2 * math.pi, 17)
        for n in loops:
            x0, y0 = scene.pos[n]
            xs += [*(x0 + r * np.sin(theta)).tolist(), None]
            ys += [*(y0 + r + r * np.cos(theta)).tolist(), None]
    return go.Scatter(
        x=xs,
        y=ys,
        mode="lines",
        line={"width": 1.2, "color": EDGE_FOCUS_COLOR},
        hoverinfo="skip",
        showlegend=False,
        name="self-loops",
    )


def _hover(
    scene: Scene, g: nx.Graph[Any], n: Hashable, key: Hashable, size_v: Any, color_v: Any
) -> str:
    lab = scene.labels
    lines = [f"<b>{n}</b>"]
    if scene.frames.animated:
        lines.append(f"{lab['time']}: {key}")
    if n not in g:
        lines.append("not present in this period")
        return "<br>".join(lines)
    shown: set[str] = set()
    if scene.color is not None:
        if scene.cats is not None:
            text = str(scene.cats.labels.get(n))
        else:
            text = fmt(float(color_v), scene.unit if scene.color.in_weight_units else "")
        lines.append(f"{scene.color.label}: {text}")
        shown.add(scene.color.name)
    if scene.size is not None and scene.size.name not in shown:
        unit = scene.unit if scene.size.in_weight_units else ""
        text = str(size_v) if scene.size.categorical else fmt(float(size_v), unit)
        lines.append(f"{scene.size.label}: {text}")
        shown.add(scene.size.name)
    if "degree" not in shown:
        lines.append(f"{metric_label('degree', lab, directed=scene.directed)}: {g.degree(n)}")
    if g.has_edge(n, n):
        lines.append(f"self-loop: {fmt(float(g[n][n].get('weight', 1.0)), scene.unit)}")
    extra = [
        (k, v)
        for k, v in g.nodes[n].items()
        if k not in shown and k not in ("lon", "lat") and isinstance(v, str | int | float | bool)
    ]
    for k, v in extra[:_MAX_HOVER_ATTRS]:
        lines.append(f"{lab.get(k, k.replace('_', ' '))}: {fmt(v) if isinstance(v, float) else v}")
    return "<br>".join(lines)


def _node_traces(scene: Scene, g: nx.Graph[Any], key: Hashable) -> list[Any]:
    xk, yk, cls = _trace_class(scene)
    nodes = scene.nodes
    lo, hi = scene.size_range
    size_vals: dict[Hashable, Any]
    if scene.size is None:
        sizes = dict.fromkeys(nodes, (lo + hi) / 2)
        size_vals = {}
    elif scene.size_scaler is None:  # categorical size_by: equal sizes, value in the hover
        sizes = dict.fromkeys(nodes, hi * 0.6)
        size_vals = scene.size.series.reindex(nodes).to_dict()
    else:
        sv = scene.size_by_frame[key].reindex(nodes).to_numpy(dtype=float)
        sizes = dict(zip(nodes, scene.size_scaler(sv).tolist(), strict=True))
        size_vals = dict(zip(nodes, sv.tolist(), strict=True))
    for n in nodes:
        if n not in g:
            sizes[n] = 0.0
    color_vals: dict[Hashable, Any] = {}
    if scene.color is not None and scene.cats is None:
        color_vals = scene.color_by_frame[key].reindex(nodes).to_dict()
    focus = scene.focus

    def trace(members: list[Hashable], marker: dict[str, Any], name: str, legend: bool) -> Any:
        kwargs: dict[str, Any] = {
            xk: [scene.pos[m][0] for m in members],
            yk: [scene.pos[m][1] for m in members],
            "mode": "markers+text",
            "text": [scene.text.get(m, "") for m in members],
            "textposition": "top center",
            "marker": {
                "size": [sizes[m] for m in members],
                "opacity": [
                    1.0 if not focus or m in focus else _FADED_NODE_OPACITY for m in members
                ],
                "line": {
                    "width": [2.5 if m in focus else 1.0 for m in members],
                    "color": [TEXT_COLOR if m in focus else SURFACE for m in members],
                },
                **marker,
            },
            "hovertext": [
                _hover(scene, g, m, key, size_vals.get(m), color_vals.get(m)) for m in members
            ],
            "hoverinfo": "text",
            "name": name,
            "legendgroup": f"node-{name}" if legend else "nodes",
            "showlegend": legend,
        }
        if not scene.geo:
            kwargs["textfont"] = {"size": 11, "color": TEXT_COLOR}
        return cls(**kwargs)

    if scene.cats is not None:
        cats = scene.cats
        return [
            trace([n for n in nodes if cats.labels.get(n) == cat], {"color": color}, cat, True)
            for cat, color in cats.colors.items()
        ]
    if scene.color is not None and scene.color_scaler is not None:
        s = scene.color_scaler
        # Split on the values over all periods, so that every frame has the same
        # points in the same traces; nodes absent from a period get size 0.
        overall = scene.color.series
        miss = [n for n in nodes if pd.isna(overall.get(n))]
        ok = [n for n in nodes if pd.notna(overall.get(n))]
        tickvals, ticktext = s.ticks()
        marker = {
            "color": np.nan_to_num(s.transform([color_vals[n] for n in ok]), nan=s.vmin).tolist(),
            "cmin": s.vmin,
            "cmax": s.vmax if s.vmax > s.vmin else s.vmin + 1,
            "colorscale": [list(p) for p in SEQUENTIAL],
            "showscale": True,
            "colorbar": {
                "title": {"text": scene.color.label},
                "tickvals": tickvals,
                "ticktext": ticktext,
                "thickness": 14,
                "len": 0.6,
            },
        }
        missing = trace(miss, {"color": MISSING_COLOR}, MISSING_LABEL, bool(miss))
        return [trace(ok, marker, scene.color.label, False), missing]
    return [trace(nodes, {"color": SEQUENTIAL[3][1]}, "nodes", False)]


def _caption(scene: Scene) -> str:
    bits = []
    if scene.size is not None:
        bits.append(f"node size: {scene.size.label}")
    if isinstance(scene.width_spec, str) and scene.width_scaler is not None:
        name = (
            scene.labels["weight"]
            if scene.width_spec == "weight"
            else scene.width_spec.replace("_", " ")
        )
        bits.append(f"edge width: {name.lower()}")
    if scene.directed and not scene.geo:
        lab = scene.labels
        src, dst = lower_label(lab["source"]), lower_label(lab["target"])
        bits.append(f"arrows point from {src} to {dst}")
    text = "; ".join(bits)
    return text[:1].upper() + text[1:] + "." if text else ""


def figure(scene: Scene, *, title: str | None, height: int, what: str) -> go.Figure:
    """Assemble the figure, with animation frames when there are several periods."""
    keys = list(scene.frames.graphs)
    all_edges = _all_edges(scene.frames, scene.nodes, scene.edge_subset)
    per_key = {k: _frame_edges(scene, k, all_edges[k]) for k in keys}
    group_keys = sorted(
        {gk for fe in per_key.values() for gk in fe.groups}, key=lambda t: (t[2], t[0], t[1])
    )
    per_frame = [frame_traces(scene, k, per_key[k], group_keys) for k in keys]
    fig = go.Figure(data=per_frame[0])
    first = scene.frames.first
    if title is None:
        if scene.frames.animated:
            title = f"{default_title(_without_time(first), what)}, {keys[0]} to {keys[-1]}"
        else:
            title = default_title(first, what)
    layout = base_layout(title, height)
    layout.update(
        hovermode="closest",
        legend={
            "title": {"text": scene.color.label if scene.color and scene.cats else ""},
            "itemsizing": "constant",
        },
    )
    caption = _caption(scene)
    if caption:
        layout["annotations"] = [
            {
                "text": caption,
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.01,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "font": {"size": 11, "color": _CAPTION_COLOR},
            }
        ]
        layout["margin"] = {"l": 10, "r": 10, "t": 60, "b": 30}
    if not scene.geo:
        layout.update(xaxis={"visible": False}, yaxis={"visible": False, "scaleanchor": "x"})
    fig.update_layout(**layout)
    if scene.frames.animated:
        add_animation(fig, keys, per_frame, scene.labels["time"], redraw=scene.geo)
    return fig


def _without_time(g: nx.Graph[Any]) -> nx.Graph[Any]:
    h: nx.Graph[Any] = nx.Graph()
    h.graph.update({k: v for k, v in g.graph.items() if k != "time"})
    return h


def add_animation(
    fig: go.Figure,
    keys: Sequence[Hashable],
    per_frame: Sequence[Sequence[Any]],
    time_label: str,
    *,
    redraw: bool,
) -> None:
    """Attach frames, a play/pause button and a time slider to ``fig``.

    ``redraw`` must be True for geo traces, which Plotly cannot animate
    without a full redraw.
    """
    fig.frames = [
        go.Frame(data=list(tr), name=str(k)) for k, tr in zip(keys, per_frame, strict=True)
    ]
    play = {
        "frame": {"duration": 900, "redraw": redraw},
        "transition": {"duration": 300},
        "fromcurrent": True,
        "mode": "immediate",
    }
    pause = {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}
    step = {
        "frame": {"duration": 0, "redraw": redraw},
        "mode": "immediate",
        "transition": {"duration": 0},
    }
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.0,
                "y": 0.0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 40, "r": 10},
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None, play]},
                    {"label": "Pause", "method": "animate", "args": [[None], pause]},
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.12,
                "len": 0.88,
                "y": 0.0,
                "yanchor": "top",
                "pad": {"t": 30},
                "currentvalue": {"prefix": f"{time_label}: ", "font": {"size": 13}},
                "steps": [
                    {"label": str(k), "method": "animate", "args": [[str(k)], step]} for k in keys
                ],
            }
        ],
    )
    margin = fig.layout.margin.to_plotly_json() if fig.layout.margin else {}
    margin["b"] = max(int(margin.get("b") or 10), 90)
    fig.update_layout(margin=margin)
