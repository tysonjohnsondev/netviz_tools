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
            "source": ["A", None],
            "target": ["B", "C"],
            "time": [2020.5, 2021.0],
            "category": ["x", "x"],
            "weight": [1.0, -2.0],
        }
    )
    with pytest.raises(SchemaError) as exc:
        nv.validate_flows(bad)
    problems = exc.value.problems
    assert any("missing columns ['unit']" in p for p in problems)
    assert any("'source' has 1 missing" in p for p in problems)
    assert any("'time' must be integer" in p for p in problems)
    assert any("negative" in p for p in problems)
    assert isinstance(exc.value, ValueError)


@pytest.mark.parametrize(
    ("weight", "message"),
    [
        (["a", "b"], "must be numeric"),
        ([True, False], "must be numeric"),
        ([1.0, np.nan], "missing or infinite"),
        ([1.0, np.inf], "missing or infinite"),
    ],
)
def test_validate_weight(weight: list[object], message: str) -> None:
    df = pd.DataFrame(
        {
            "source": ["A", "B"],
            "target": ["B", "C"],
            "time": [2020, 2020],
            "category": ["x", "x"],
            "weight": weight,
            "unit": ["t", "t"],
        }
    )
    with pytest.raises(SchemaError, match=message):
        nv.validate_flows(df)


def test_to_flowframe_renames_fills_and_orders() -> None:
    raw = pd.DataFrame({"n": [5, 7], "dst": ["B", "C"], "src": ["A", "B"], "note": ["p", "q"]})
    out = nv.to_flowframe(
        raw,
        {"src": "source", "dst": "target", "n": "weight"},
        time=2020,
        category="migrants",
        unit="people",
    )
    assert list(out.columns) == [*nv.FLOW_COLUMNS, "note"]
    assert out["time"].dtype == np.int64
    assert raw.columns.tolist() == ["n", "dst", "src", "note"]  # input untouched


def test_to_flowframe_casts_float_times() -> None:
    raw = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "time": [2020.0], "weight": [1], "category": ["x"]}
    )
    out = nv.to_flowframe(raw, unit="t")
    assert out["time"].dtype == np.int64


def test_to_flowframe_rejects_bad_constants() -> None:
    raw = pd.DataFrame({"source": ["A"], "target": ["B"], "weight": [1.0]})
    with pytest.raises(SchemaError, match="existing column"):
        nv.to_flowframe(raw, source="Z", time=1, category="x", unit="t")
    with pytest.raises(SchemaError, match="non-schema"):
        nv.to_flowframe(raw, colour="red", time=1, category="x", unit="t")


def test_to_flowframe_keeps_labels() -> None:
    raw = pd.DataFrame({"origin": ["A"], "dest": ["B"], "people": [3]})
    raw.attrs["labels"] = {"source": "Origin", "target": "Destination"}
    out = nv.to_flowframe(
        raw,
        {"origin": "source", "dest": "target", "people": "weight"},
        time=2020,
        category="migrants",
        unit="people",
    )
    assert out.attrs["labels"] == {"source": "Origin", "target": "Destination"}
