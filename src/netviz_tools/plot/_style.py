"""Shared colours, scaling and labelling helpers for the plot modules."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping, Sequence
from typing import Any, Final

import networkx as nx
import numpy as np
import numpy.typing as npt
import pandas as pd

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
deficiency checks; categories beyond eight are folded into ``OTHER``."""

OTHER_COLOR: Final = "#8a8986"
OTHER_LABEL: Final = "Other"
EDGE_COLOR: Final = "rgba(110, 110, 110, 0.35)"
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
"""Colour per continent, as defined in :func:`netviz_tools.datasets.faostat.countries`."""


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
        Apply ``log10(1 + x)`` first, so that a few very large flows do not
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


def rgba(hex_color: str, alpha: float) -> str:
    """Convert ``#rrggbb`` to a CSS ``rgba()`` string."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def categories(
    g: nx.Graph[Any],
    nodes: Sequence[Hashable],
    color_by: str | None,
    partition: pd.Series | None,
) -> tuple[dict[Hashable, str], dict[str, str]]:
    """Assign each node a category label and each label a colour.

    Returns
    -------
    tuple of dict
        ``(node -> label, label -> colour)``. Labels keep a stable order:
        continents follow :data:`CONTINENT_COLORS`; other attributes are
        ordered by the number of nodes in each category.
    """
    if color_by is None:
        return dict.fromkeys(nodes, ""), {"": PALETTE[0]}
    if color_by == "community":
        if partition is None:
            raise ValueError("color_by='community' needs partition= (see metrics.communities)")
        lookup = partition.to_dict()
        raw = {n: f"Community {int(lookup[n])}" if n in lookup else OTHER_LABEL for n in nodes}
    else:
        raw = {n: str(g.nodes[n].get(color_by, OTHER_LABEL) or OTHER_LABEL) for n in nodes}
    if color_by == "continent":
        colors = {c: CONTINENT_COLORS[c] for c in CONTINENT_COLORS if c in raw.values()}
    else:
        counts = pd.Series(list(raw.values())).value_counts()
        ordered = [c for c in counts.index if c != OTHER_LABEL]
        if color_by == "community":
            ordered.sort(key=lambda c: int(c.split()[-1]))
        colors = dict(zip(ordered[: len(PALETTE)], PALETTE, strict=False))
    labels = {n: (lab if lab in colors else OTHER_LABEL) for n, lab in raw.items()}
    if OTHER_LABEL in labels.values():
        colors[OTHER_LABEL] = OTHER_COLOR
    return labels, colors


def unit_of(g: nx.Graph[Any]) -> str:
    """Return the graph's unit suffix for hover text, for example ``" t"``."""
    unit = g.graph.get("unit")
    return f" {unit}" if isinstance(unit, str) and unit else ""


def default_title(g: nx.Graph[Any], what: str) -> str:
    """Build a title such as ``"Wheat trade network, 2022"`` from graph metadata."""
    category = g.graph.get("category")
    time = g.graph.get("time")
    head = f"{category} {what}" if isinstance(category, str) else what.capitalize()
    if isinstance(time, int):
        return f"{head}, {time}"
    if isinstance(time, list) and time:
        agg = g.graph.get("aggregate") or "sum"
        return f"{head}, {min(time)} to {max(time)} ({agg})"
    return head


def top_nodes(g: nx.Graph[Any], top_n: int | None) -> list[Hashable]:
    """Return nodes ordered by total strength, keeping at most ``top_n``."""
    strength = pd.Series(dict(g.degree(weight="weight")), dtype=float)
    order = strength.sort_values(ascending=False, kind="stable").index.tolist()
    return order if top_n is None else order[:top_n]
