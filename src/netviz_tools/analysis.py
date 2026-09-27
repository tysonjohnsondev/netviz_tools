"""Table-level questions about flows: who a node exchanges with, and how categories compare."""

from __future__ import annotations

import difflib
from collections.abc import Hashable, Iterable
from typing import Literal, TypeAlias, get_args

import pandas as pd

from netviz_tools._schema import FlowFrame
from netviz_tools.errors import MixedUnitError, SchemaError, UnknownMetricError, UnknownNodeError

__all__ = ["CategoryMetric", "Role", "compare_categories", "partners", "suggest"]

Role: TypeAlias = Literal["out", "in", "both"]
"""Which flows of the focal node to rank: outgoing (``"out"``, for trade its
export destinations), incoming (``"in"``, its import sources), or both side by
side."""

CategoryMetric: TypeAlias = Literal["total_weight", "n_flows", "n_sources", "n_targets", "n_nodes"]
"""Metrics for :func:`compare_categories`: total ``weight``, number of flow
rows, number of distinct sources, of distinct targets, and of distinct nodes
(sources and targets combined)."""


def suggest(name: str, choices: Iterable[str], n: int = 5) -> list[str]:
    """Return up to ``n`` choices that resemble ``name``.

    Substring matches (case-insensitive) come first, then fuzzy matches from
    :func:`difflib.get_close_matches`.

    Parameters
    ----------
    name
        The string that failed to match.
    choices
        Candidate strings.
    n
        Maximum number of suggestions.

    Returns
    -------
    list of str
        Suggestions, best first, without duplicates.
    """
    pool = sorted(set(choices))
    needle = name.lower().strip()
    lowered = {c.lower(): c for c in pool}
    hits = [c for c in pool if needle and needle in c.lower()]
    hits.sort(key=lambda c: (len(c), c))
    fuzzy = [lowered[m] for m in difflib.get_close_matches(needle, list(lowered), n=n, cutoff=0.6)]
    out: list[str] = []
    for c in (*hits, *fuzzy):
        if c not in out:
            out.append(c)
    return out[:n]


def _filter(
    flows: pd.DataFrame,
    time: int | Iterable[int] | None,
    category: Hashable | Iterable[Hashable] | None,
) -> pd.DataFrame:
    out = flows
    for col, wanted in (("time", time), ("category", category)):
        if wanted is None:
            continue
        if isinstance(wanted, str | bytes) or not isinstance(wanted, Iterable):
            out = out[out[col] == wanted]
        else:
            out = out[out[col].isin(list(wanted))]
    return out


def partners(
    flows: FlowFrame,
    node: str,
    role: Role = "out",
    *,
    top_n: int | None = 10,
    time: int | Iterable[int] | None = None,
    category: Hashable | Iterable[Hashable] | None = None,
) -> pd.DataFrame:
    """Rank the partners of one node.

    Parameters
    ----------
    flows
        A flow frame.
    node
        The focal node, matched exactly against ``source`` and ``target``.
    role
        ``"out"`` ranks where ``node`` sends flows (for trade, its export
        destinations); ``"in"`` ranks where its inflows come from (its import
        sources); ``"both"`` returns outflows and inflows per partner, ranked
        by their total.
    top_n
        Number of partners to return. ``None`` returns all.
    time, category
        Optional filters, each a value or an iterable of values, applied
        before ranking. Without them, flows over all times and categories in
        ``flows`` are summed.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``partner``. For ``"out"`` and ``"in"`` the columns are
        ``weight`` and ``share`` (of the node's total for that direction).
        For ``"both"`` the columns are ``out_weight``, ``in_weight`` and
        ``total_weight``.

    Raises
    ------
    UnknownNodeError
        If ``node`` does not appear in the (filtered) flows. The error lists
        close matches.
    MixedUnitError
        If the flows being summed have more than one unit.
    ValueError
        If ``role`` is not one of the supported values.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(items="Wheat", years=2021)
    >>> top = nv.partners(flows, "Ukraine", role="out", top_n=3)
    >>> list(top.columns)
    ['weight', 'share']
    """
    if role not in get_args(Role):
        raise ValueError(f"role must be one of {list(get_args(Role))}, not {role!r}")
    missing = [c for c in ("source", "target", "weight") if c not in flows.columns]
    if missing:
        raise SchemaError([f"missing columns {missing}"])
    sub = _filter(flows, time, category)
    names = set(sub["source"]).union(sub["target"])
    if node not in names:
        raise UnknownNodeError(node, suggest(node, map(str, names)))
    if "unit" in sub.columns:
        involved = sub[(sub["source"] == node) | (sub["target"] == node)]
        if involved["unit"].nunique() > 1:
            units = sorted(involved["unit"].unique())
            raise MixedUnitError(f"flows for {node!r} mix units {units}; filter by category first")

    def ranked(side: str, other: str) -> pd.Series:
        rows = sub[sub[side] == node]
        return rows.groupby(other)["weight"].sum().sort_values(ascending=False)

    if role == "both":
        out = pd.DataFrame(
            {"out_weight": ranked("source", "target"), "in_weight": ranked("target", "source")}
        ).fillna(0.0)
        out["total_weight"] = out["out_weight"] + out["in_weight"]
        out = out.sort_values(["total_weight", "out_weight"], ascending=False)
    else:
        side, other = ("source", "target") if role == "out" else ("target", "source")
        w = ranked(side, other)
        total = w.sum()
        out = pd.DataFrame({"weight": w, "share": w / total if total else w * 0.0})
    out.index.name = "partner"
    return out if top_n is None else out.head(top_n)


def compare_categories(
    flows: FlowFrame,
    metric: CategoryMetric = "total_weight",
    *,
    by_time: bool = False,
) -> pd.DataFrame | pd.Series:
    """Compare categories (for trade, commodities) on one metric.

    Parameters
    ----------
    flows
        A flow frame containing one or more categories.
    metric
        ``"total_weight"``, ``"n_flows"``, ``"n_sources"``, ``"n_targets"``
        or ``"n_nodes"`` (sources and targets combined).
    by_time
        Return one column per ``time`` value instead of a single total.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Indexed by category, sorted in descending order of the metric (of the
        latest time value when ``by_time`` is true). For ``total_weight`` the
        unit is part of the index so that different units are never compared
        directly.

    Raises
    ------
    UnknownMetricError
        If ``metric`` is not supported.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(years=2022)
    >>> nv.compare_categories(flows, "n_sources").index.tolist()
    ['Maize (corn)', 'Soya beans', 'Wheat']
    """
    if metric not in get_args(CategoryMetric):
        raise UnknownMetricError(
            f"unknown metric {metric!r}; choose from {list(get_args(CategoryMetric))}"
        )
    with_unit = metric == "total_weight" and "unit" in flows.columns
    keys = ["category", "unit"] if with_unit else ["category"]
    if by_time:
        keys = [*keys, "time"]
    grouped = flows.groupby(keys, observed=True)
    if metric == "total_weight":
        result = grouped["weight"].sum()
    elif metric == "n_flows":
        result = grouped.size()
    elif metric == "n_sources":
        result = grouped["source"].nunique()
    elif metric == "n_targets":
        result = grouped["target"].nunique()
    else:
        result = grouped[["source", "target"]].apply(
            lambda d: len(set(d["source"]).union(d["target"]))
        )
    result = result.rename(metric)
    if by_time:
        table = result.unstack("time")
        return table.sort_values(table.columns[-1], ascending=False)
    return result.sort_values(ascending=False)
