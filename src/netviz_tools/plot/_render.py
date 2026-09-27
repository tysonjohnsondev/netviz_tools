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
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from netviz_tools.analysis import suggest
from netviz_tools.plot._data import Frames
from netviz_tools.plot._resolve import (
    Categories,
    EdgeSpec,
    NodeSpec,
    NodeValues,
    categories,
    edge_values,
    metric_label,
    node_values,
)
from netviz_tools.plot._style import (
    EDGE_COLOR,
    EDGE_FOCUS_COLOR,
    MISSING_COLOR,
    MISSING_LABEL,
    SEQUENTIAL,
    SURFACE,
    TEXT_COLOR,
    Scaler,
    base_layout,
    default_title,
    fmt,
    rgba,
    sample_sequential,
)

Position: TypeAlias = Mapping[Hashable, tuple[float, float]]
ShowLabels: TypeAlias = "bool | int | Sequence[Hashable] | Literal['auto']"
"""``"auto"`` labels every node when there are at most 30, otherwise the 12
largest; an int labels that many of the largest nodes; ``True``/``False`` all
or none; a sequence labels exactly those nodes. Focus nodes are always
labelled."""

_N_WIDTH_BUCKETS = 5
_ARROW_AT = 0.62
_MAX_HOVER_ATTRS = 6


@dataclass
class Scene:
    """Everything needed to draw one or more frames."""

    frames: Frames
    nodes: list[Hashable]
    pos: dict[Hashable, tuple[float, float]]
    geo: bool
    directed: bool
    labels: dict[str, str]
    unit: str
    size: NodeValues | None
    color: NodeValues | None
    cats: Categories | None
    size_scaler: Scaler | None
    color_scaler: Scaler | None
    width_spec: EdgeSpec
    width_scaler: Scaler | None
    edge_color_spec: EdgeSpec
    edge_cats: Categories | None
    edge_color_scaler: Scaler | None
    text: dict[Hashable, str]
    focus: set[Hashable]
    min_weight: float | None
    edge_filter: Callable[[nx.Graph[Any], Hashable], list[tuple[Hashable, Hashable]]] | None
    size_range: tuple[float, float]
    partition: pd.Series | None
    seed: int
    size_spec: NodeSpec
    color_spec: NodeSpec
    edge_width_range: tuple[float, float] = (0.5, 6.0)
    per_frame_size: dict[Hashable, pd.Series] = field(default_factory=dict)
    per_frame_color: dict[Hashable, pd.Series] = field(default_factory=dict)


def resolve_focus(g: nx.Graph[Any], focus: Hashable | Iterable[Hashable] | None) -> list[Hashable]:
    """Normalize ``focus`` into a list of nodes that exist in ``g``."""
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
    nodes: Sequence[Hashable], show: Any, focus: Iterable[Hashable]
) -> set[Hashable]:
    """Apply the :data:`ShowLabels` rule to nodes ordered largest first."""
    if show == "auto":
        chosen = set(nodes) if len(nodes) <= 30 else set(nodes[:12])
    elif show is True:
        chosen = set(nodes)
    elif show is False or show is None:
        chosen = set()
    elif isinstance(show, int):
        chosen = set(nodes[:show])
    else:
        chosen = set(show)
    return chosen | set(focus)


def label_text(g: nx.Graph[Any], label_by: str | None) -> Callable[[Hashable], str]:
    """Return a function giving the display text of a node."""
    if label_by is None:
        return str

    def text(n: Hashable) -> str:
        value = g.nodes[n].get(label_by) if n in g else None
        return str(n) if value is None else str(value)

    return text


def prepare(
    frames: Frames,
    *,
    nodes: list[Hashable],
    pos: dict[Hashable, tuple[float, float]],
    geo: bool,
    size_by: NodeSpec,
    color_by: NodeSpec,
    edge_width_by: EdgeSpec,
    edge_color_by: EdgeSpec,
    label_by: str | None,
    show_labels: Any,
    focus: list[Hashable],
    partition: pd.Series | None,
    seed: int,
    size_range: tuple[float, float],
    min_weight_quantile: float | None = None,
    edge_filter: Callable[[nx.Graph[Any], Hashable], list[tuple[Hashable, Hashable]]] | None = None,
    union: nx.Graph[Any] | None = None,
    resolved: tuple[NodeValues | None, NodeValues | None] | None = None,
) -> Scene:
    """Resolve values and scales shared by all frames.

    ``resolved`` passes already computed ``(size, color)`` values for the
    union graph, so expensive metrics are not computed twice.
    """
    u = union if union is not None else frames.union()
    labels = frames.labels
    if resolved is not None:
        size, color = resolved
    else:
        size = node_values(u, size_by, arg="size_by", labels=labels, partition=partition, seed=seed)
        color = node_values(
            u, color_by, arg="color_by", labels=labels, partition=partition, seed=seed
        )
    cats = (
        categories(color.values.reindex(nodes), name=color.name)
        if color and color.categorical
        else None
    )

    scene = Scene(
        frames=frames,
        nodes=nodes,
        pos=pos,
        geo=geo,
        directed=u.is_directed(),
        labels=labels,
        unit=frames.unit,
        size=size,
        color=color,
        cats=cats,
        size_scaler=None,
        color_scaler=None,
        width_spec=edge_width_by,
        width_scaler=None,
        edge_color_spec=edge_color_by,
        edge_cats=None,
        edge_color_scaler=None,
        text={},
        focus=set(focus),
        min_weight=None,
        edge_filter=edge_filter,
        size_range=size_range,
        partition=partition,
        seed=seed,
        size_spec=size_by,
        color_spec=color_by,
    )
    # Per-frame values for metrics that change between periods.
    for key, g in frames.graphs.items():
        scene.per_frame_size[key] = _frame_values(scene, g, size, size_by, "size_by")
        if color is not None and not color.categorical:
            scene.per_frame_color[key] = _frame_values(scene, g, color, color_by, "color_by")
    if size is not None and not size.categorical:
        allv = np.concatenate(
            [s.reindex(nodes).to_numpy(float) for s in scene.per_frame_size.values()]
        )
        scene.size_scaler = Scaler(allv, *size_range)
    if color is not None and not color.categorical:
        allc = np.concatenate(
            [s.reindex(nodes).to_numpy(float) for s in scene.per_frame_color.values()]
        )
        scene.color_scaler = Scaler(allc, 0.0, 1.0)

    # Edge widths and colours, scaled across all frames.
    widths: list[float] = []
    ecolors: list[Any] = []
    node_set = set(nodes)
    for key, g in frames.graphs.items():
        edges = _frame_edges(scene, g, key, node_set, apply_cut=False)
        w = edge_values(edges, edge_width_by, arg="edge_width_by", directed=scene.directed)
        if w is not None:
            widths.extend(pd.to_numeric(w, errors="coerce").tolist())
        if edge_color_by not in (None, "source", "target"):
            c = edge_values(edges, edge_color_by, arg="edge_color_by", directed=scene.directed)
            if c is not None:
                ecolors.extend(c.tolist())
    if widths:
        arr = np.asarray(widths, dtype=float)
        scene.width_scaler = Scaler(arr, *scene.edge_width_range)
        if min_weight_quantile is not None and np.isfinite(arr).any():
            scene.min_weight = float(np.nanquantile(arr, min_weight_quantile))
    if ecolors:
        es = pd.Series(ecolors, dtype=object)
        if all(isinstance(v, int | float) and not isinstance(v, bool) for v in es.dropna()):
            scene.edge_color_scaler = Scaler(es.astype(float), 0.0, 1.0)
        else:
            scene.edge_cats = categories(es, name=str(edge_color_by))

    text_of = label_text(u, label_by)
    shown = labelled_nodes(nodes, show_labels, focus)
    scene.text = {n: text_of(n) for n in nodes if n in shown}
    return scene


def _frame_values(
    scene: Scene, g: nx.Graph[Any], resolved: NodeValues | None, spec: NodeSpec, arg: str
) -> pd.Series:
    if resolved is None:
        return pd.Series(dtype=float)
    if not scene.frames.animated or resolved.categorical:
        return resolved.values
    is_attr = isinstance(spec, str) and any(resolved.name in d for _, d in g.nodes(data=True))
    if not isinstance(spec, str) or (is_attr and resolved.name not in ("community",)):
        return resolved.values
    vals = node_values(
        g, resolved.name, arg=arg, labels=scene.labels, partition=scene.partition, seed=scene.seed
    )
    assert vals is not None
    return vals.values


def _frame_edges(
    scene: Scene, g: nx.Graph[Any], key: Hashable, node_set: set[Hashable], *, apply_cut: bool
) -> list[tuple[Hashable, Hashable, dict[str, Any]]]:
    if scene.edge_filter is not None:
        pairs = scene.edge_filter(g, key)
        edges = [(a, b, g[a][b]) for a, b in pairs]
    else:
        edges = [(a, b, d) for a, b, d in g.edges(data=True) if a in node_set and b in node_set]
    if apply_cut and scene.min_weight is not None:
        w = edge_values(edges, scene.width_spec, arg="edge_width_by", directed=scene.directed)
        if w is not None:
            keep = pd.to_numeric(w, errors="coerce").fillna(-np.inf).to_numpy() >= scene.min_weight
            edges = [e for e, k in zip(edges, keep, strict=True) if k]
    return edges


def _node_color(scene: Scene, n: Hashable, key: Hashable) -> str:
    if scene.cats is not None:
        return scene.cats.colors[str(scene.cats.labels.get(n))]
    if scene.color is not None and scene.color_scaler is not None:
        vals = scene.per_frame_color.get(key, scene.color.values)
        v = vals.get(n, np.nan)
        return (
            MISSING_COLOR
            if pd.isna(v)
            else sample_sequential(float(scene.color_scaler.unit([v])[0]))
        )
    return SEQUENTIAL[3][1]


def _edge_groups(
    scene: Scene, g: nx.Graph[Any], key: Hashable
) -> tuple[
    list[tuple[Hashable, Hashable, dict[str, Any]]],
    list[tuple[str, int, bool]],
    list[float],
    list[Any],
]:
    node_set = set(scene.nodes)
    edges = _frame_edges(scene, g, key, node_set, apply_cut=True)
    w = edge_values(edges, scene.width_spec, arg="edge_width_by", directed=scene.directed)
    wvals = pd.to_numeric(w, errors="coerce").tolist() if w is not None else [math.nan] * len(edges)
    if w is not None and scene.width_scaler is not None:
        t = np.nan_to_num(scene.width_scaler.unit(wvals), nan=0.0)
        buckets = np.minimum((t * _N_WIDTH_BUCKETS).astype(int), _N_WIDTH_BUCKETS - 1).tolist()
    else:
        buckets = [1] * len(edges)
    spec = scene.edge_color_spec
    cvals: list[Any] = [None] * len(edges)
    ckeys: list[str] = []
    if spec in ("source", "target"):
        idx = 0 if spec == "source" else 1
        for e in edges:
            ckeys.append("node:" + _node_color(scene, e[idx], key))
    elif spec is not None:
        c = edge_values(edges, spec, arg="edge_color_by", directed=scene.directed)
        cvals = c.tolist() if c is not None else cvals
        for v in cvals:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                ckeys.append("missing")
            elif scene.edge_color_scaler is not None:
                b = int(min(float(scene.edge_color_scaler.unit([v])[0]) * 5, 4))
                ckeys.append(f"bin:{b}")
            else:
                ckeys.append("cat:" + _edge_cat_label(scene, v))
    else:
        ckeys = ["base"] * len(edges)
    groups = [
        (ck, b, bool(scene.focus) and (e[0] in scene.focus or e[1] in scene.focus))
        for ck, b, e in zip(ckeys, buckets, edges, strict=True)
    ]
    return edges, groups, wvals, cvals


def _edge_cat_label(scene: Scene, v: Any) -> str:
    assert scene.edge_cats is not None
    label = str(v)
    return label if label in scene.edge_cats.colors else "Other"


def _group_style(scene: Scene, ck: str, focused: bool) -> tuple[str, str, bool]:
    """Return (colour, legend name, show in legend) for an edge colour key."""
    if focused and ck == "base":
        return EDGE_FOCUS_COLOR, "", False
    if ck == "base":
        return EDGE_COLOR if not scene.focus else rgba("#6e6e6e", 0.18), "", False
    if ck == "missing":
        return rgba(MISSING_COLOR[:7], 0.6), MISSING_LABEL, True
    alpha = 0.8 if focused or not scene.focus else 0.25
    if ck.startswith("node:"):
        return rgba(ck[5:], alpha * 0.7), "", False
    if ck.startswith("bin:"):
        b = int(ck[4:])
        assert scene.edge_color_scaler is not None
        return rgba(sample_sequential((b + 0.5) / 5), alpha), f"{_bin_label(scene, b)}", True
    assert scene.edge_cats is not None
    name = ck[4:]
    return rgba(scene.edge_cats.colors.get(name, "#8a8986"), alpha), name, True


def _bin_label(scene: Scene, b: int) -> str:
    s = scene.edge_color_scaler
    assert s is not None
    lo_t, hi_t = s.vmin + (s.vmax - s.vmin) * b / 5, s.vmin + (s.vmax - s.vmin) * (b + 1) / 5
    inv = (lambda t: 10**t - 1) if s.log else (lambda t: t)
    return f"{fmt(inv(lo_t))} to {fmt(inv(hi_t))}"


def _xy(scene: Scene) -> tuple[str, str, type[go.Scatter] | type[go.Scattergeo]]:
    return ("lon", "lat", go.Scattergeo) if scene.geo else ("x", "y", go.Scatter)


def frame_traces(scene: Scene, key: Hashable, group_keys: list[tuple[str, int, bool]]) -> list[Any]:
    """Build the traces of one frame, in a fixed order."""
    g = scene.frames.graphs[key]
    xk, yk, cls = _xy(scene)
    traces: list[Any] = []
    edges, groups, wvals, cvals = _edge_groups(scene, g, key)
    pos = scene.pos
    seen_legend: set[str] = set()
    for gk in group_keys:
        ck, b, focused = gk
        xs: list[float | None] = []
        ys: list[float | None] = []
        for (u, v, _), grp in zip(edges, groups, strict=True):
            if grp == gk and u != v:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
        color, name, legend = _group_style(scene, ck, focused)
        width = scene.edge_width_range[0] + (b + 0.5) / _N_WIDTH_BUCKETS * (
            scene.edge_width_range[1] - scene.edge_width_range[0]
        )
        show = legend and ck not in seen_legend
        seen_legend.add(ck)
        traces.append(
            cls(
                **{xk: xs, yk: ys},
                mode="lines",
                line={"width": width, "color": color},
                hoverinfo="skip",
                name=name or f"edges {ck} {b}",
                legendgroup=f"edge-{ck}",
                showlegend=show,
            )
        )
    traces.append(_edge_hover_trace(scene, edges, wvals, cvals, key))
    if not scene.geo:
        traces.append(_self_loop_trace(scene, g))
    traces.extend(_node_traces(scene, g, key))
    return traces


def _edge_text(scene: Scene, u: Hashable, v: Hashable, w: float, c: Any) -> str:
    lab = scene.labels
    arrow = "→" if scene.directed else "–"
    lines = [f"{u} {arrow} {v}"]
    if isinstance(scene.width_spec, str) and not math.isnan(w):
        name = lab["weight"] if scene.width_spec == "weight" else scene.width_spec.replace("_", " ")
        unit = scene.unit if scene.width_spec == "weight" else ""
        lines.append(f"{name}: {fmt(w, unit)}")
    elif not math.isnan(w):
        lines.append(f"value: {fmt(w)}")
    if (
        isinstance(scene.edge_color_spec, str)
        and scene.edge_color_spec not in ("source", "target")
        and c is not None
    ):
        lines.append(
            f"{scene.edge_color_spec.replace('_', ' ')}: {c if isinstance(c, str) else fmt(float(c))}"
        )
    return "<br>".join(lines)


def _edge_hover_trace(
    scene: Scene,
    edges: list[tuple[Hashable, Hashable, dict[str, Any]]],
    wvals: list[float],
    cvals: list[Any],
    key: Hashable,
) -> Any:
    xk, yk, cls = _xy(scene)
    pos = scene.pos
    real = [(e, w, c) for e, w, c in zip(edges, wvals, cvals, strict=True) if e[0] != e[1]]
    t = _ARROW_AT if scene.directed else 0.5
    xs = [pos[u][0] + t * (pos[v][0] - pos[u][0]) for (u, v, _), _, _ in real]
    ys = [pos[u][1] + t * (pos[v][1] - pos[u][1]) for (u, v, _), _, _ in real]
    text = [_edge_text(scene, u, v, w, c) for (u, v, _), w, c in real]
    marker: dict[str, Any]
    if scene.directed and not scene.geo:
        angles = [
            math.degrees(math.atan2(pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]))
            for (u, v, _), _, _ in real
        ]
        if scene.width_scaler is not None:
            sizes = (
                7 + 5 * np.nan_to_num(scene.width_scaler.unit([w for _, w, _ in real]), nan=0.0)
            ).tolist()
        else:
            sizes = [8.0] * len(real)
        faded = bool(scene.focus)
        colors = [
            "rgba(60, 60, 60, 0.85)"
            if not faded or u in scene.focus or v in scene.focus
            else "rgba(110, 110, 110, 0.25)"
            for (u, v, _), _, _ in real
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
    if scene.pos:
        arr = np.asarray(list(scene.pos.values()), dtype=float)
        r = 0.025 * float(max(np.ptp(arr[:, 0]), np.ptp(arr[:, 1]), 1e-9))
    else:
        r = 0.0
    node_set = set(scene.nodes)
    theta = np.linspace(0, 2 * math.pi, 17)
    for n in nx.nodes_with_selfloops(g):
        if n in node_set:
            x0, y0 = scene.pos[n]
            xs += (x0 + r * np.sin(theta)).tolist() + [None]
            ys += (y0 + r + r * np.cos(theta)).tolist() + [None]
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
    scene: Scene, g: nx.Graph[Any], n: Hashable, key: Hashable, size_v: float, color_v: Any
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
        v = color_v
        if scene.cats is not None:
            text = str(scene.cats.labels.get(n))
        else:
            text = fmt(float(v), scene.unit if scene.color.in_weight_units else "")
        lines.append(f"{scene.color.label}: {text}")
        shown.add(scene.color.name)
    if scene.size is not None and scene.size.name not in shown:
        unit = scene.unit if scene.size.in_weight_units else ""
        lines.append(
            f"{scene.size.label}: {fmt(size_v, unit) if not isinstance(size_v, str) else size_v}"
        )
        shown.add(scene.size.name)
    if "degree" not in shown:
        lines.append(f"{metric_label('degree', lab, directed=scene.directed)}: {g.degree(n)}")
    if g.has_edge(n, n):
        lines.append(f"self-loop: {fmt(float(g[n][n].get('weight', 1.0)), scene.unit)}")
    extra = [
        (k, v)
        for k, v in g.nodes[n].items()
        if k not in shown and isinstance(v, str | int | float | bool) and k not in ("lon", "lat")
    ]
    for k, v in extra[:_MAX_HOVER_ATTRS]:
        lines.append(f"{lab.get(k, k.replace('_', ' '))}: {fmt(v) if isinstance(v, float) else v}")
    return "<br>".join(lines)


def _node_traces(scene: Scene, g: nx.Graph[Any], key: Hashable) -> list[Any]:
    xk, yk, cls = _xy(scene)
    nodes = scene.nodes
    size_s = scene.per_frame_size.get(key, pd.Series(dtype=float))
    lo, hi = scene.size_range
    if scene.size_scaler is not None:
        sv = size_s.reindex(nodes).to_numpy(dtype=float)
        sizes = dict(zip(nodes, scene.size_scaler(sv), strict=True))
        size_val = dict(zip(nodes, sv, strict=True))
    else:
        mid = (lo + hi) / 2 if scene.size is None else hi * 0.6
        sizes = dict.fromkeys(nodes, mid)
        size_val = {
            n: (str(scene.size.values.get(n)) if scene.size is not None else math.nan)
            for n in nodes
        }
    present = {n: n in g for n in nodes}
    for n in nodes:
        if not present[n]:
            sizes[n] = 0.0
    color_s = (
        scene.per_frame_color.get(key)
        if scene.color is not None and not scene.color.categorical
        else None
    )
    faded = bool(scene.focus)

    def marker_line(members: list[Hashable]) -> dict[str, Any]:
        return {
            "width": [2.5 if m in scene.focus else 1.0 for m in members],
            "color": [TEXT_COLOR if m in scene.focus else SURFACE for m in members],
        }

    def opacity(members: list[Hashable]) -> list[float]:
        return [1.0 if not faded or m in scene.focus else 0.45 for m in members]

    def trace(
        members: list[Hashable], marker: dict[str, Any], name: str, legend: bool, group: str
    ) -> Any:
        colors = scene.color.values if scene.color is not None else pd.Series(dtype=object)
        cv = color_s if color_s is not None else colors
        kwargs: dict[str, Any] = {
            xk: [scene.pos[m][0] for m in members],
            yk: [scene.pos[m][1] for m in members],
            "mode": "markers+text",
            "text": [scene.text.get(m, "") for m in members],
            "textposition": "top center",
            "textfont": {"size": 11, "color": TEXT_COLOR},
            "marker": {
                "size": [sizes[m] for m in members],
                "opacity": opacity(members),
                "line": marker_line(members),
                **marker,
            },
            "hovertext": [
                _hover(scene, g, m, key, size_val[m], cv.get(m, math.nan)) for m in members
            ],
            "hoverinfo": "text",
            "name": name,
            "legendgroup": group,
            "showlegend": legend,
        }
        if scene.geo:
            del kwargs["textfont"]
        return cls(**kwargs)

    out: list[Any] = []
    if scene.cats is not None:
        for cat, color in scene.cats.colors.items():
            members = [n for n in nodes if scene.cats.labels.get(n) == cat]
            out.append(trace(members, {"color": color}, cat, True, f"node-{cat}"))
        return out
    if scene.color is not None and scene.color_scaler is not None:
        assert color_s is not None
        vals = color_s.reindex(nodes)
        ok = [n for n in nodes if pd.notna(vals.get(n))]
        miss = [n for n in nodes if pd.isna(vals.get(n))]
        s = scene.color_scaler
        tick_t = np.linspace(s.vmin, s.vmax, 5)
        inv = (lambda t: 10**t - 1) if s.log else (lambda t: t)
        marker = {
            "color": s._t(vals.reindex(ok).to_numpy(float)).tolist(),
            "cmin": s.vmin,
            "cmax": s.vmax if s.vmax > s.vmin else s.vmin + 1,
            "colorscale": [list(p) for p in SEQUENTIAL],
            "showscale": True,
            "colorbar": {
                "title": {"text": scene.color.label},
                "tickvals": tick_t.tolist(),
                "ticktext": [fmt(inv(t)) for t in tick_t],
                "thickness": 14,
                "len": 0.6,
            },
        }
        out.append(trace(ok, marker, scene.color.label, False, "nodes"))
        out.append(
            trace(miss, {"color": MISSING_COLOR}, MISSING_LABEL, bool(miss), "nodes-missing")
        )
        return out
    out.append(trace(nodes, {"color": SEQUENTIAL[3][1]}, "nodes", False, "nodes"))
    return out


def figure(
    scene: Scene, *, title: str | None, height: int, what: str, caption: bool = True
) -> go.Figure:
    """Assemble the figure (with animation frames when there are several periods)."""
    keys = list(scene.frames.graphs)
    group_keys: list[tuple[str, int, bool]] = []
    for key in keys:
        _, groups, _, _ = _edge_groups(scene, scene.frames.graphs[key], key)
        for gk in groups:
            if gk not in group_keys:
                group_keys.append(gk)
    group_keys.sort(key=lambda t: (t[2], t[0], t[1]))
    per_frame = [frame_traces(scene, k, group_keys) for k in keys]
    fig = go.Figure(data=per_frame[0])
    first = scene.frames.first
    if title is None:
        if scene.frames.animated:
            head = default_title(_without_time(first), what)
            title = f"{head}, {keys[0]} to {keys[-1]}"
        else:
            title = default_title(first, what)
    layout = base_layout(title, height)
    layout.update(
        hovermode="closest",
        legend={
            "title": {"text": scene.color.label if scene.cats else ""},
            "itemsizing": "constant",
        },
    )
    if caption:
        bits = []
        if scene.size is not None:
            bits.append(f"node size: {scene.size.label}")
        if isinstance(scene.width_spec, str) and scene.width_scaler is not None:
            wl = (
                scene.labels["weight"]
                if scene.width_spec == "weight"
                else scene.width_spec.replace("_", " ")
            )
            bits.append(f"edge width: {wl.lower()}")
        if scene.directed and not scene.geo:
            bits.append("arrows point from source to target")
        if bits:
            text = "; ".join(bits)
            layout["annotations"] = [
                {
                    "text": text[:1].upper() + text[1:] + ".",
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0,
                    "y": -0.01,
                    "xanchor": "left",
                    "yanchor": "top",
                    "showarrow": False,
                    "font": {"size": 11, "color": "#6b6a66"},
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
    """Attach frames, a play/pause button and a time slider to ``fig``."""
    fig.frames = [
        go.Frame(data=list(tr), name=str(k)) for k, tr in zip(keys, per_frame, strict=True)
    ]
    anim = {
        "frame": {"duration": 900, "redraw": redraw},
        "transition": {"duration": 300},
        "fromcurrent": True,
        "mode": "immediate",
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
                    {"label": "Play", "method": "animate", "args": [None, anim]},
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"},
                        ],
                    },
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
                    {
                        "label": str(k),
                        "method": "animate",
                        "args": [
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": redraw},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    }
                    for k in keys
                ],
            }
        ],
    )
    margin = dict(fig.layout.margin.to_plotly_json()) if fig.layout.margin else {}
    margin["b"] = max(int(margin.get("b", 10) or 10), 90)
    fig.update_layout(margin=margin)
