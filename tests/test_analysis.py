from __future__ import annotations

import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import MixedUnitError, SchemaError, UnknownCountryError, UnknownMetricError
from netviz_tools.analysis import ItemMetric, suggest


def test_partners_exporter(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "A", role="exporter")
    assert out.index.name == "partner"
    assert out.index.tolist() == ["B", "C"]
    assert out.loc["B", "quantity"] == 30.0
    assert out["share"].sum() == pytest.approx(1.0)


def test_partners_importer_and_filters(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "A", role="importer", year=2021)
    assert out.index.tolist() == ["B"]
    assert out.loc["B", "quantity"] == 4.0
    assert nv.partners(flows, "A", item="x", top_n=1).shape == (1, 2)


def test_partners_both(flows: pd.DataFrame) -> None:
    out = nv.partners(flows, "C", role="both", top_n=None)
    assert out.columns.tolist() == ["exports", "imports", "total"]
    assert out.loc["A", "exports"] == 1.0
    assert out.loc["A", "imports"] == 5.0
    assert out["total"].is_monotonic_decreasing


def test_partners_zero_total() -> None:
    df = pd.DataFrame(
        {
            "exporter": ["A"],
            "importer": ["B"],
            "year": [1],
            "item": ["x"],
            "quantity": [0.0],
            "unit": ["t"],
        }
    )
    assert nv.partners(df, "A")["share"].tolist() == [0.0]


def test_partners_unknown_country_suggests(wheat_2022: pd.DataFrame) -> None:
    with pytest.raises(UnknownCountryError) as exc:
        nv.partners(wheat_2022, "Russia")
    assert "Russian Federation" in exc.value.suggestions
    assert "Did you mean" in str(exc.value)
    assert isinstance(exc.value, LookupError)


def test_partners_errors(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="role must be"):
        nv.partners(flows, "A", role="seller")  # type: ignore[arg-type]
    with pytest.raises(SchemaError):
        nv.partners(flows.drop(columns="quantity"), "A")
    mixed = flows.assign(unit=["t", "head", "t", "t", "t", "t", "t"])
    with pytest.raises(MixedUnitError):
        nv.partners(mixed, "A")


def test_partners_on_sample_matches_known_totals(wheat_2022: pd.DataFrame) -> None:
    ua = nv.partners(wheat_2022, "Ukraine", top_n=None)
    # Importer-reported wheat from Ukraine in 2022, summed over partners.
    assert ua["quantity"].sum() == pytest.approx(
        wheat_2022.loc[wheat_2022.exporter == "Ukraine", "quantity"].sum()
    )


def test_compare_items() -> None:
    sample = nv.datasets.faostat.load_sample(years=[2021, 2022])
    total = nv.compare_items(sample)
    assert isinstance(total, pd.Series)
    assert total.index.names == ["item", "unit"]
    assert total.is_monotonic_decreasing
    wide = nv.compare_items(sample, "n_countries", by_year=True)
    assert isinstance(wide, pd.DataFrame)
    assert [int(c) for c in wide.columns] == [2021, 2022]
    metrics: tuple[ItemMetric, ...] = ("n_flows", "n_exporters", "n_importers")
    for metric in metrics:
        s = nv.compare_items(sample, metric)
        assert len(s) == 3
    with pytest.raises(UnknownMetricError):
        nv.compare_items(sample, "volume")  # type: ignore[arg-type]


def test_suggest() -> None:
    names = ["Russian Federation", "Belarus", "Rwanda", "Ukraine"]
    assert suggest("russia", names)[0] == "Russian Federation"
    assert suggest("Ukrane", names) == ["Ukraine"]
    assert suggest("", names) == []
