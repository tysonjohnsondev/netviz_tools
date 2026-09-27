"""The flow table schema shared by every function in the library.

A *flow frame* is a :class:`pandas.DataFrame` with one row per directed flow and
at least these columns:

========== ========================================================
column     meaning
========== ========================================================
exporter   origin node (for trade data, the exporting country)
importer   destination node (the importing country)
year       integer period label
item       what flows (a commodity, a product code, a migrant type)
quantity   non-negative amount
unit       unit of ``quantity``, for example ``"t"`` or ``"1000 USD"``
========== ========================================================

Extra columns are allowed and preserved.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeAlias

import numpy as np
import pandas as pd

from netviz_tools.errors import SchemaError

__all__ = ["FLOW_COLUMNS", "FlowFrame", "to_flowframe", "validate_flows"]

FLOW_COLUMNS: Final = ("exporter", "importer", "year", "item", "quantity", "unit")
"""Required columns of a flow frame, in canonical order."""

FlowFrame: TypeAlias = pd.DataFrame
"""A :class:`pandas.DataFrame` that satisfies :func:`validate_flows`."""

_KEY_COLUMNS: Final = ("exporter", "importer", "item", "unit")


def validate_flows(flows: pd.DataFrame) -> FlowFrame:
    """Check that a DataFrame follows the flow schema.

    The check is light: required columns exist, key columns have no missing
    values, ``year`` is an integer column, and ``quantity`` is numeric, finite
    and non-negative. All problems are collected and reported together.

    Parameters
    ----------
    flows
        The DataFrame to check.

    Returns
    -------
    FlowFrame
        The same object, unchanged, so the call can be chained.

    Raises
    ------
    SchemaError
        If any check fails.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"exporter": ["A"], "importer": ["B"], "year": [2020],
    ...                    "item": ["Wheat"], "quantity": [1.0], "unit": ["t"]})
    >>> validate_flows(df) is df
    True
    """
    problems: list[str] = []
    missing = [c for c in FLOW_COLUMNS if c not in flows.columns]
    if missing:
        problems.append(f"missing columns {missing}")
    for col in _KEY_COLUMNS:
        if col in flows.columns and flows[col].isna().any():
            problems.append(f"column {col!r} has {int(flows[col].isna().sum())} missing values")
    if "year" in flows.columns and not pd.api.types.is_integer_dtype(flows["year"]):
        problems.append(f"column 'year' must be integer, not {flows['year'].dtype}")
    if "quantity" in flows.columns:
        q = flows["quantity"]
        if not pd.api.types.is_numeric_dtype(q) or pd.api.types.is_bool_dtype(q):
            problems.append(f"column 'quantity' must be numeric, not {q.dtype}")
        else:
            values = q.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(values).all():
                problems.append("column 'quantity' has missing or infinite values")
            elif (values < 0).any():
                problems.append("column 'quantity' has negative values")
    if problems:
        raise SchemaError(problems)
    return flows


def to_flowframe(
    df: pd.DataFrame,
    columns: Mapping[str, str] | None = None,
    **constants: str | float,
) -> FlowFrame:
    """Convert any edge list into a flow frame.

    Use this to bring your own bilateral data (migration, shipping, payments)
    into the schema that the rest of the library expects.

    Parameters
    ----------
    df
        Source table. It is not modified.
    columns
        Mapping from column names in ``df`` to flow-frame column names, for
        example ``{"origin": "exporter", "dest": "importer", "n": "quantity"}``.
    **constants
        Values for required columns that ``df`` does not have, for example
        ``item="migrants", unit="people", year=2020``.

    Returns
    -------
    FlowFrame
        A new DataFrame with the six schema columns first, followed by any
        other columns of ``df``. ``year`` is cast to ``int64``.

    Raises
    ------
    SchemaError
        If the result does not satisfy :func:`validate_flows`, or if a
        constant is given for a column that already exists.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"], "n": [5, 7]})
    >>> ff = to_flowframe(raw, {"src": "exporter", "dst": "importer", "n": "quantity"},
    ...                   year=2020, item="people", unit="persons")
    >>> list(ff.columns)
    ['exporter', 'importer', 'year', 'item', 'quantity', 'unit']
    """
    out = df.rename(columns=dict(columns or {}))
    clashes = sorted(set(constants) & set(out.columns))
    if clashes:
        raise SchemaError([f"constant given for existing column(s) {clashes}"])
    unknown = sorted(set(constants) - set(FLOW_COLUMNS))
    if unknown:
        raise SchemaError([f"constants given for non-schema column(s) {unknown}"])
    out = out.copy()
    for col, value in constants.items():
        out[col] = value
    if "year" in out.columns and pd.api.types.is_numeric_dtype(out["year"]):
        years = out["year"].to_numpy(dtype=float, na_value=np.nan)
        if np.isfinite(years).all() and (years == np.round(years)).all():
            out["year"] = out["year"].astype("int64")
    ordered = [c for c in FLOW_COLUMNS if c in out.columns]
    ordered += [c for c in out.columns if c not in FLOW_COLUMNS]
    return validate_flows(out[ordered])
