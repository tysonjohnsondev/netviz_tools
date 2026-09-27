from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import SchemaError


def test_validate_returns_same_object(flows: pd.DataFrame) -> None:
    assert nv.validate_flows(flows) is flows


def test_validate_collects_all_problems() -> None:
    bad = pd.DataFrame(
        {
            "exporter": ["A", None],
            "importer": ["B", "C"],
            "year": [2020.5, 2021.0],
            "item": ["x", "x"],
            "quantity": [1.0, -2.0],
        }
    )
    with pytest.raises(SchemaError) as exc:
        nv.validate_flows(bad)
    problems = exc.value.problems
    assert any("missing columns ['unit']" in p for p in problems)
    assert any("'exporter' has 1 missing" in p for p in problems)
    assert any("'year' must be integer" in p for p in problems)
    assert any("negative" in p for p in problems)
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize(
    ("quantity", "message"),
    [
        (["a", "b"], "must be numeric"),
        ([True, False], "must be numeric"),
        ([1.0, np.nan], "missing or infinite"),
        ([1.0, np.inf], "missing or infinite"),
    ],
)
def test_validate_quantity(quantity: list[object], message: str) -> None:
    df = pd.DataFrame(
        {
            "exporter": ["A", "B"],
            "importer": ["B", "C"],
            "year": [2020, 2020],
            "item": ["x", "x"],
            "quantity": quantity,
            "unit": ["t", "t"],
        }
    )
    with pytest.raises(SchemaError, match=message):
        nv.validate_flows(df)


def test_to_flowframe_renames_fills_and_orders() -> None:
    raw = pd.DataFrame({"n": [5, 7], "dst": ["B", "C"], "src": ["A", "B"], "note": ["p", "q"]})
    out = nv.to_flowframe(
        raw,
        {"src": "exporter", "dst": "importer", "n": "quantity"},
        year=2020,
        item="migrants",
        unit="people",
    )
    assert list(out.columns) == [*nv.FLOW_COLUMNS, "note"]
    assert out["year"].dtype == np.int64
    assert raw.columns.tolist() == ["n", "dst", "src", "note"]  # input untouched


def test_to_flowframe_casts_float_years() -> None:
    raw = pd.DataFrame(
        {"exporter": ["A"], "importer": ["B"], "year": [2020.0], "quantity": [1], "item": ["x"]}
    )
    out = nv.to_flowframe(raw, unit="t")
    assert out["year"].dtype == np.int64


def test_to_flowframe_rejects_bad_constants() -> None:
    raw = pd.DataFrame({"exporter": ["A"], "importer": ["B"], "quantity": [1.0]})
    with pytest.raises(SchemaError, match="existing column"):
        nv.to_flowframe(raw, exporter="Z", year=1, item="x", unit="t")
    with pytest.raises(SchemaError, match="non-schema"):
        nv.to_flowframe(raw, colour="red", year=1, item="x", unit="t")
