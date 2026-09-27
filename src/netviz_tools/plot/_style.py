"""Shared colours, scaling, labels and number formatting for the plot modules."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Final

import networkx as nx
import numpy as np
import numpy.typing as npt

PALETTE: Final = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
"""Categorical palette in fixed order. It passes adjacent-pair colour-vision
deficiency checks; categories beyond eight are folded into ``"Other"``."""

SEQUENTIAL: Final = (
    (0.0, "#c6dbef"),
    (0.25, "#8fbbe0"),
    (0.5, "#4f93cc"),
    (0.75, "#2166ac"),
    (1.0, "#08306b"),
)
"""Single-hue sequential colour scale used for continuous values. It starts
at a light blue rather than white, so the smallest values stay visible on a
white background."""

DECREASE_COLOR: Final = "#c2410c"
INCREASE_COLOR: Final = "#2166ac"
OTHER_COLOR: Final = "#8a8986"
MISSING_COLOR: Final = "#cfcfcb"
OTHER_LABEL: Final = "Other"
MISSING_LABEL: Final = "No value"
EDGE_COLOR: Final = "rgba(110, 110, 110, 0.35)"
EDGE_FOCUS_COLOR: Final = "rgba(40, 40, 40, 0.7)"
TEXT_COLOR: Final = "#3d3d3a"
SURFACE: Final = "#ffffff"

CONTINENT_COLORS: Final[Mapping[str, str]] = {
    "Asia": PALETTE[0],
    "Europe": PALETTE[1],
    "Africa": PALETTE[2],
    "Northern America": PALETTE[3],
    "South America": PALETTE[4],
    "Oceania": PALETTE[5],
    "Central America": PALETTE[6],
    "Caribbean": PALETTE[7],
}
"""Colour per continent, as defined in :func:`netviz_tools.datasets.faostat.countries`.
Used whenever every category of ``color_by`` is one of these continents."""

DEFAULT_LABELS: Final[Mapping[str, str]] = {
    "source": "Source",
    "target": "Target",
    "time": "Time",
    "category": "Category",
    "weight": "Weight",
    "node": "Node",
    "out": "Outgoing",
    "in": "Incoming",
}
"""Display labels used when neither the data nor the caller gives one.
Override any of them with ``labels=``. FAOSTAT flows carry their own
(:data:`netviz_tools.datasets.faostat.LABELS`)."""


def scale(
    values: Iterable[float], lo: float, hi: float, *, log: bool = True
) -> npt.NDArray[np.float64]:
    """Map values linearly (optionally after ``log10(1 + x)``) onto ``[lo, hi]``.

    Parameters
    ----------
    values
        Non-negative numbers.
    lo, hi
        Output range.
    log
        Apply ``log10(1 + x)`` first, so that a few very large values do not
        flatten everything else.

    Returns
    -------
    numpy.ndarray
        Scaled values. If all inputs are equal, every output is ``hi``.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if log:
        arr = np.log10(1.0 + np.clip(arr, 0.0, None))
    span = float(np.ptp(arr))
    if span == 0.0:
        return np.full(arr.shape, float(hi))
    return np.asarray(lo + (arr - arr.min()) / span * (hi - lo), dtype=float)


def wants_log(values: Iterable[float]) -> bool:
    """Return True when positive values span more than two orders of magnitude."""
    arr = np.asarray([v for v in values if np.isfinite(v) and v > 0], dtype=float)
    return bool(arr.size > 1 and arr.max() / arr.min() > 100.0)


class Scaler:
    """Scale values onto a range using bounds fixed in advance.

    Values are log-scaled when they span more than two orders of magnitude.
    Fixing the bounds lets every frame of an animation share one scale.
    Missing values map to ``lo``.
    """

    def __init__(self, values: Iterable[float], lo: float, hi: float) -> None:
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
        self.log = wants_log(arr)
        t = self._t(arr)
        self.vmin = float(t.min()) if t.size else 0.0
        self.vmax = float(t.max()) if t.size else 0.0
        self.lo, self.hi = lo, hi

    def _t(self, arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        if self.log:
            return np.asarray(np.log10(1.0 + np.clip(arr, 0.0, None)), dtype=float)
        return arr

    def unit(self, values: Iterable[float]) -> npt.NDArray[np.float64]:
        """Return positions in ``[0, 1]`` (NaN stays NaN)."""
        arr = np.asarray(list(values), dtype=float)
        span = self.vmax - self.vmin
        if span == 0.0:
            return np.where(np.isfinite(arr), 1.0, np.nan)
        return np.asarray(np.clip((self._t(arr) - self.vmin) / span, 0.0, 1.0), dtype=float)

    def __call__(self, values: Iterable[float]) -> npt.NDArray[np.float64]:
        t = self.unit(values)
        return np.asarray(self.lo + np.nan_to_num(t, nan=0.0) * (self.hi - self.lo), dtype=float)


def rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a CSS ``rgba()`` string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def sample_sequential(t: float) -> str:
    """Return the :data:`SEQUENTIAL` colour at position ``t`` in ``[0, 1]``."""
    t = min(max(float(t), 0.0), 1.0)
    for (t0, c0), (t1, c1) in zip(SEQUENTIAL, SEQUENTIAL[1:], strict=False):
        if t <= t1:
            f = (t - t0) / (t1 - t0)
            a = [int(c0[i : i + 2], 16) for i in (1, 3, 5)]
            b = [int(c1[i : i + 2], 16) for i in (1, 3, 5)]
            mix = [round(x + (y - x) * f) for x, y in zip(a, b, strict=True)]
            return "#" + "".join(f"{v:02x}" for v in mix)
    return SEQUENTIAL[-1][1]  # pragma: no cover - t is clipped to [0, 1]


def merge_labels(*layers: Mapping[str, str] | None) -> dict[str, str]:
    """Merge display-label mappings over :data:`DEFAULT_LABELS`; later layers win."""
    out = dict(DEFAULT_LABELS)
    for layer in layers:
        if layer:
            out.update({str(k): str(v) for k, v in layer.items()})
    return out


def graph_labels(g: nx.Graph[Any]) -> Mapping[str, str] | None:
    """Return the display labels stored on a graph by ``build_graph``, if any."""
    labels = g.graph.get("labels")
    return labels if isinstance(labels, Mapping) else None


def unit_of(g: nx.Graph[Any]) -> str:
    """Return the graph's unit, for example ``"t"``, or ``""``."""
    unit = g.graph.get("unit")
    return unit if isinstance(unit, str) else ""


def fmt(value: float | None, unit: str = "") -> str:
    """Format a number for hover text and annotations."""
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    v = float(value)
    if v == int(v) and abs(v) < 1e15:
        text = f"{int(v):,}"
    elif abs(v) >= 100:
        text = f"{v:,.0f}"
    elif abs(v) >= 1:
        text = f"{v:,.2f}"
    else:
        text = f"{v:.3g}"
    return f"{text} {unit}" if unit else text


def fmt_pct(value: float) -> str:
    """Format a fraction as a signed percentage, for example ``"+12.5%"``."""
    if not math.isfinite(value):
        return "n/a"
    return f"{value * 100:+.1f}%"


def default_title(g: nx.Graph[Any], what: str) -> str:
    """Build a title such as ``"Wheat network, 2022"`` from graph metadata.

    Uses ``G.graph["category"]`` and ``G.graph["time"]`` when present, then
    ``G.graph["name"]``, then ``what`` alone.
    """
    category = g.graph.get("category")
    name = g.graph.get("name")
    if isinstance(category, str):
        head = f"{category} {what}"
    elif isinstance(name, str) and name:
        head = f"{name}: {what}"
    else:
        head = what[:1].upper() + what[1:]
    t = g.graph.get("time")
    if isinstance(t, int):
        return f"{head}, {t}"
    if isinstance(t, list) and t:
        agg = g.graph.get("aggregate") or "sum"
        return f"{head}, {min(t)} to {max(t)} ({agg})"
    return head


def base_layout(
    title: str | None, height: int, *, margin: Mapping[str, int] | None = None
) -> dict[str, Any]:
    """Return the layout settings shared by every figure."""
    return {
        "title": {"text": title or ""},
        "template": "plotly_white",
        "height": height,
        "font": {"color": TEXT_COLOR},
        "hoverlabel": {"align": "left"},
        "margin": dict(margin) if margin else {"l": 10, "r": 10, "t": 60, "b": 10},
    }


def lower_label(label: str) -> str:
    """Lower-case a display label for use mid-sentence, keeping acronyms such as ``PageRank``."""
    head, rest = label[:1], label[1:]
    return head.lower() + rest if rest == rest.lower() else label
