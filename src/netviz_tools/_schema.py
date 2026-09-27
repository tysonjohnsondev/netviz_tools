"""The flow table schema shared by every function in the library.

A *flow frame* is a :class:`pandas.DataFrame` with one row per directed flow and
at least these columns:

========== ============================================================
column     meaning
========== ============================================================
source     origin node (for trade data, the exporting country)
target     destination node (for trade data, the importing country)
time       integer period label, for example a year
category   what flows (a commodity, a product code, a migrant type)
weight     non-negative amount
unit       unit of ``weight``, for example ``"t"``, ``"people"`` or ``"1000 USD"``
========== ============================================================

Extra columns are allowed and preserved. Domain-specific display names (such as
"Exporter" for ``source``) can be stored as a dict in ``flows.attrs["labels"]``;
:func:`netviz_tools.datasets.faostat.load` does this for trade data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final, TypeAlias

import numpy as np
import pandas as pd

from netviz_tools.errors import SchemaError

__all__ = ["FLOW_COLUMNS", "FlowFrame", "to_flowframe", "validate_flows"]

FLOW_COLUMNS: Final = ("source", "target", "time", "category", "weight", "unit")
"""Required columns of a flow frame, in canonical order."""

FlowFrame: TypeAlias = pd.DataFrame
"""A :class:`pandas.DataFrame` that satisfies :func:`validate_flows`."""

_KEY_COLUMNS: Final = ("source", "target", "category", "unit")


def validate_flows(flows: pd.DataFrame) -> FlowFrame:
    """Check that a DataFrame follows the flow schema.

    The check is light: required columns exist, key columns have no missing
    values, ``time`` is an integer column, and ``weight`` is numeric, finite
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
    >>> df = pd.DataFrame({"source": ["A"], "target": ["B"], "time": [2020],
    ...                    "category": ["Wheat"], "weight": [1.0], "unit": ["t"]})
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
    if "time" in flows.columns and not pd.api.types.is_integer_dtype(flows["time"]):
        problems.append(f"column 'time' must be integer, not {flows['time'].dtype}")
    if "weight" in flows.columns:
        w = flows["weight"]
        if not pd.api.types.is_numeric_dtype(w) or pd.api.types.is_bool_dtype(w):
            problems.append(f"column 'weight' must be numeric, not {w.dtype}")
        else:
            values = w.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(values).all():
                problems.append("column 'weight' has missing or infinite values")
            elif (values < 0).any():
                problems.append("column 'weight' has negative values")
    if problems:
        raise SchemaError(problems)
    return flows


def to_flowframe(
    df: pd.DataFrame,
    columns: Mapping[str, str] | None = None,
    **constants: str | float,
) -> FlowFrame:
    """Convert any edge list into a flow frame.

    Use this to bring your own flow data (trade, migration, shipping,
    payments) into the schema that the rest of the library expects.

    Parameters
    ----------
    df
        Source table. It is not modified.
    columns
        Mapping from column names in ``df`` to flow-frame column names, for
        example ``{"origin": "source", "dest": "target", "n": "weight"}``.
    **constants
        Values for required columns that ``df`` does not have, for example
        ``category="migrants", unit="people", time=2020``.

    Returns
    -------
    FlowFrame
        A new DataFrame with the six schema columns first, followed by any
        other columns of ``df``. A whole-number ``time`` column is cast to
        ``int64``. ``df.attrs`` (for example ``"labels"``) is carried over.

    Raises
    ------
    SchemaError
        If the result does not satisfy :func:`validate_flows`, or if a
        constant is given for a column that already exists.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "C"], "n": [5, 7]})
    >>> ff = to_flowframe(raw, {"src": "source", "dst": "target", "n": "weight"},
    ...                   time=2020, category="people", unit="persons")
    >>> list(ff.columns)
    ['source', 'target', 'time', 'category', 'weight', 'unit']
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
    if "time" in out.columns and pd.api.types.is_numeric_dtype(out["time"]):
        times = out["time"].to_numpy(dtype=float, na_value=np.nan)
        if np.isfinite(times).all() and (times == np.round(times)).all():
            out["time"] = out["time"].astype("int64")
    ordered = [c for c in FLOW_COLUMNS if c in out.columns]
    ordered += [c for c in out.columns if c not in FLOW_COLUMNS]
    return validate_flows(out[ordered])
