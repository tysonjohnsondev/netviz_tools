"""Table-level questions about flows: who trades with whom, and how items compare."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Literal, TypeAlias, get_args

import pandas as pd

from netviz_tools._schema import FlowFrame
from netviz_tools.errors import MixedUnitError, SchemaError, UnknownCountryError, UnknownMetricError

__all__ = ["ItemMetric", "Role", "compare_items", "partners", "suggest"]

Role: TypeAlias = Literal["exporter", "importer", "both"]
"""The role of the focal country: its export destinations, its import sources,
or both side by side."""

ItemMetric: TypeAlias = Literal[
    "total_quantity", "n_flows", "n_exporters", "n_importers", "n_countries"
]


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


def _filter(flows: pd.DataFrame, year: int | None, item: str | None) -> pd.DataFrame:
    out = flows
    if year is not None:
        out = out[out["year"] == year]
    if item is not None:
        out = out[out["item"] == item]
    return out


def partners(
    flows: FlowFrame,
    country: str,
    role: Role = "exporter",
    *,
    top_n: int | None = 10,
    year: int | None = None,
    item: str | None = None,
) -> pd.DataFrame:
    """Rank a country's trading partners.

    Parameters
    ----------
    flows
        A flow frame.
    country
        The focal node, matched exactly against ``exporter`` and ``importer``.
    role
        ``"exporter"`` ranks where ``country`` sends flows; ``"importer"``
        ranks where its inflows come from; ``"both"`` returns exports and
        imports per partner, ranked by their total.
    top_n
        Number of partners to return. ``None`` returns all.
    year, item
        Optional filters applied before ranking. Without them, flows over all
        years and items in ``flows`` are summed.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``partner``. For a single role the columns are ``quantity``
        and ``share`` (of the country's total for that role). For ``"both"``
        the columns are ``exports``, ``imports`` and ``total``.

    Raises
    ------
    UnknownCountryError
        If ``country`` does not appear in the (filtered) flows. The error
        lists close matches.
    MixedUnitError
        If the flows being summed have more than one unit.
    ValueError
        If ``role`` is not one of the supported values.

    Examples
    --------
    >>> import netviz_tools as nv
    >>> flows = nv.datasets.faostat.load_sample(items="Wheat", years=2021)
    >>> top = nv.partners(flows, "Ukraine", role="exporter", top_n=3)
    >>> list(top.columns)
    ['quantity', 'share']
    """
    if role not in get_args(Role):
        raise ValueError(f"role must be one of {list(get_args(Role))}, not {role!r}")
    missing = [c for c in ("exporter", "importer", "quantity") if c not in flows.columns]
    if missing:
        raise SchemaError([f"missing columns {missing}"])
    sub = _filter(flows, year, item)
    names = set(sub["exporter"]).union(sub["importer"])
    if country not in names:
        raise UnknownCountryError(country, suggest(country, map(str, names)))
    if "unit" in sub.columns:
        involved = sub[(sub["exporter"] == country) | (sub["importer"] == country)]
        if involved["unit"].nunique() > 1:
            units = sorted(involved["unit"].unique())
            raise MixedUnitError(f"flows for {country!r} mix units {units}; filter by item first")

    def ranked(side: str, other: str) -> pd.Series:
        rows = sub[sub[side] == country]
        return rows.groupby(other)["quantity"].sum().sort_values(ascending=False)

    if role == "both":
        out = pd.DataFrame(
            {"exports": ranked("exporter", "importer"), "imports": ranked("importer", "exporter")}
        ).fillna(0.0)
        out["total"] = out["exports"] + out["imports"]
        out = out.sort_values(["total", "exports"], ascending=False)
    else:
        side, other = ("exporter", "importer") if role == "exporter" else ("importer", "exporter")
        q = ranked(side, other)
        total = q.sum()
        out = pd.DataFrame({"quantity": q, "share": q / total if total else q * 0.0})
    out.index.name = "partner"
    return out if top_n is None else out.head(top_n)


def compare_items(
    flows: FlowFrame,
    metric: ItemMetric = "total_quantity",
    *,
    by_year: bool = False,
) -> pd.DataFrame | pd.Series:
    """Compare items (commodities) on one metric.

    Parameters
    ----------
    flows
        A flow frame containing one or more items.
    metric
        ``"total_quantity"``, ``"n_flows"``, ``"n_exporters"``,
        ``"n_importers"`` or ``"n_countries"`` (exporters and importers
        combined).
    by_year
        Return one column per year instead of a single total.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        Indexed by item, sorted in descending order of the metric (of the
        latest year when ``by_year`` is true). For ``total_quantity`` the unit
        is part of the index so that different units are never compared
        directly.

    Raises
    ------
    UnknownMetricError
        If ``metric`` is not supported.
    """
    if metric not in get_args(ItemMetric):
        raise UnknownMetricError(
            f"unknown metric {metric!r}; choose from {list(get_args(ItemMetric))}"
        )
    keys = ["item", "unit"] if metric == "total_quantity" and "unit" in flows.columns else ["item"]
    if by_year:
        keys = [*keys, "year"]
    grouped = flows.groupby(keys, observed=True)
    if metric == "total_quantity":
        result = grouped["quantity"].sum()
    elif metric == "n_flows":
        result = grouped.size()
    elif metric == "n_exporters":
        result = grouped["exporter"].nunique()
    elif metric == "n_importers":
        result = grouped["importer"].nunique()
    else:
        result = grouped[["exporter", "importer"]].apply(
            lambda d: len(set(d["exporter"]).union(d["importer"]))
        )
    result = result.rename(metric)
    if by_year:
        table = result.unstack("year")
        return table.sort_values(table.columns[-1], ascending=False)
    return result.sort_values(ascending=False)
