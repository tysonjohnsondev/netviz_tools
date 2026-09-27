from __future__ import annotations

import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import (
    MixedUnitError,
    SchemaError,
    UnknownMetricError,
    UnknownNameError,
    UnknownNodeError,
)
from netviz_tools.analysis import CategoryMetric, suggest


def test_partners_out(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "A", role="out")
    assert out.index.name == "partner"
    assert out.columns.tolist() == ["weight", "share"]
    assert out.index.tolist() == ["B", "C"]
    assert out.loc["B", "weight"] == 30.0
    assert out["share"].sum() == pytest.approx(1.0)
    pd.testing.assert_frame_equal(nv.partners(flows, "A"), out)  # "out" is the default


def test_partners_in_and_filters(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "A", role="in", time=2021)
    assert out.index.tolist() == ["B"]
    assert out.loc["B", "weight"] == 4.0
    assert nv.partners(flows, "A", category="x", top_n=1).shape == (1, 2)
    both_years = nv.partners(flows, "A", role="in", time=[2020, 2021])
    assert both_years.loc["C", "weight"] == 1.0
    assert both_years.loc["B", "weight"] == 4.0
    assert nv.partners(flows, "A", time=range(2020, 2021)).loc["B", "weight"] == 10.0


def test_partners_both(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "C", role="both", top_n=None)
    assert out.index.name == "partner"
    assert out.columns.tolist() == ["out_weight", "in_weight", "total_weight"]
    assert out.loc["A", "out_weight"] == 1.0
    assert out.loc["A", "in_weight"] == 5.0
    assert out.loc["A", "total_weight"] == 6.0
    assert out["total_weight"].is_monotonic_decreasing


def test_partners_zero_total() -> None:
    df = pd.DataFrame(
        {
            "source": ["A"],
            "target": ["B"],
            "time": [1],
            "category": ["x"],
            "weight": [0.0],
            "unit": ["t"],
        }
    )
    assert nv.partners(df, "A")["share"].tolist() == [0.0]


def test_partners_unknown_node_suggests(wheat_2022: pd.DataFrame) -> None:
    with pytest.raises(UnknownNodeError) as exc:
        nv.partners(wheat_2022, "Russia")
    assert exc.value.kind == "node"
    assert "Russian Federation" in exc.value.suggestions
    assert str(exc.value).startswith("unknown node 'Russia'. Did you mean")
    assert isinstance(exc.value, UnknownNameError)
    assert isinstance(exc.value, LookupError)


def test_partners_errors(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="role must be"):
        nv.partners(flows, "A", role="exporter")  # type: ignore[arg-type]
    with pytest.raises(SchemaError):
        nv.partners(flows.drop(columns="weight"), "A")
    mixed = flows.assign(unit=["t", "head", "t", "t", "t", "t", "t"])
    with pytest.raises(MixedUnitError, match="filter by category"):
        nv.partners(mixed, "A")
    with pytest.raises(UnknownNodeError):
        nv.partners(flows, "C", time=1999)


def test_partners_on_sample_matches_known_totals(wheat_2022: pd.DataFrame) -> None:
    ua = nv.partners(wheat_2022, "Ukraine", top_n=None)
    # Importer-reported wheat from Ukraine in 2022, summed over partners.
    assert ua["weight"].sum() == pytest.approx(
        wheat_2022.loc[wheat_2022.source == "Ukraine", "weight"].sum()
    )
    inflow = nv.partners(wheat_2022, "Egypt", role="in", top_n=None)
    assert inflow["weight"].sum() == pytest.approx(
        wheat_2022.loc[wheat_2022.target == "Egypt", "weight"].sum()
    )


def test_compare_categories() -> None:
    sample = nv.datasets.faostat.load_sample(years=[2021, 2022])
    total = nv.compare_categories(sample)
    assert isinstance(total, pd.Series)
    assert total.name == "total_weight"
    assert total.index.names == ["category", "unit"]
    assert total.is_monotonic_decreasing
    wide = nv.compare_categories(sample, "n_nodes", by_time=True)
    assert isinstance(wide, pd.DataFrame)
    assert [int(c) for c in wide.columns] == [2021, 2022]
    metrics: tuple[CategoryMetric, ...] = ("n_flows", "n_sources", "n_targets", "n_nodes")
    for metric in metrics:
        s = nv.compare_categories(sample, metric)
        assert isinstance(s, pd.Series)
        assert s.index.name == "category"
        assert len(s) == 3
    with pytest.raises(UnknownMetricError, match="total_weight"):
        nv.compare_categories(sample, "total_quantity")  # type: ignore[arg-type]


def test_compare_categories_values(flows: pd.DataFrame) -> None:
    df = pd.concat([flows, flows.assign(category="y", weight=1.0)], ignore_index=True)
    assert nv.compare_categories(df).loc[("x", "t")] == 45.0
    assert nv.compare_categories(df, "n_flows")["x"] == 7
    assert nv.compare_categories(df, "n_sources")["x"] == 3
    assert nv.compare_categories(df, "n_targets")["x"] == 3
    assert nv.compare_categories(df, "n_nodes")["y"] == 3
    by_time = nv.compare_categories(df, by_time=True)
    assert by_time.loc[("x", "t"), 2021] == 27.0
    no_unit = nv.compare_categories(df.drop(columns="unit"))
    assert no_unit.index.names == ["category"]


def test_suggest() -> None:
    names = ["Russian Federation", "Belarus", "Rwanda", "Ukraine"]
    assert suggest("russia", names)[0] == "Russian Federation"
    assert suggest("Ukrane", names) == ["Ukraine"]
    assert suggest("", names) == []
