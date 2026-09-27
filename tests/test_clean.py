from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import SchemaError
from netviz_tools._clean import (
    _COLUMN,
    BY_SOURCE,
    BY_TARGET,
    CleaningReport,
    CleaningStep,
    MirrorStats,
    clean,
)
from netviz_tools.datasets import faostat

FLOW = list(nv.FLOW_COLUMNS)
SRC, DST, TIME, CAT, W, UNIT = (
    _COLUMN[r] for r in ("source", "target", "time", "category", "weight", "unit")
)


def edges(**cols: list[Any]) -> pd.DataFrame:
    """A small raw edge table with FAOSTAT-style headers."""
    base: dict[str, list[Any]] = {
        "exporter": ["A", "B"],
        "importer": ["B", "C"],
        "year": [2020, 2020],
        "item": ["x", "x"],
        "quantity": [1.0, 2.0],
        "unit": ["t", "t"],
    }
    base.update(cols)
    return pd.DataFrame(base)


def n_edges(n: int, **cols: list[Any]) -> pd.DataFrame:
    """``n`` rows from distinct exporters E0, E1, ... to one importer."""
    base: dict[str, list[Any]] = {
        "exporter": [f"E{i}" for i in range(n)],
        "importer": ["I"] * n,
        "year": [2020] * n,
        "item": ["x"] * n,
        "quantity": [1.0] * n,
        "unit": ["t"] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


def reporter_table(rows: list[tuple[str, str, str, float]]) -> pd.DataFrame:
    """(reporter, partner, element, value) rows of one item, year and unit."""
    return pd.DataFrame(
        {
            "Reporter Countries": [r[0] for r in rows],
            "Partner Countries": [r[1] for r in rows],
            "Element": [r[2] for r in rows],
            "Value": [r[3] for r in rows],
            "Item": "Wheat",
            "Year": 2021,
            "Unit": "t",
        }
    )


def step(report: CleaningReport, name: str) -> CleaningStep:
    return next(s for s in report.steps if s.name == name)


def reasons(report: CleaningReport) -> list[str]:
    return report.dropped["reason"].tolist()


# ---------------------------------------------------------------------------
# Column mapping


def test_output_columns_follow_the_flow_schema() -> None:
    assert list(_COLUMN) == ["source", "target", "time", "category", "weight", "unit"]
    assert list(_COLUMN.values()) == FLOW


def test_guesses_edge_table_columns() -> None:
    raw = pd.DataFrame(
        {
            " Origin ": ["A"],
            "DESTINATION": ["B"],
            "period": [2020],
            "Commodity": ["x"],
            "Flow": [3],
            "Units": ["t"],
            "Origin Code": [1],
            "flag": ["A"],
            "notes": ["n"],
        }
    )
    flows, report = clean(raw)
    assert report.columns == {
        " Origin ": "source",
        "DESTINATION": "target",
        "period": "time",
        "Commodity": "category",
        "Flow": "weight",
        "Units": "unit",
    }
    assert report.guessed == ("source", "target", "time", "category", "weight", "unit")
    assert report.filled == {}
    assert flows.columns.tolist() == FLOW
    assert flows.iloc[0].tolist() == ["A", "B", 2020, "x", 3.0, "t"]
    assert "' Origin ' -> source (guessed)" in report.summary()


def test_guesses_underscored_headers() -> None:
    raw = pd.DataFrame(
        {"exporter_country": ["A"], "Importer  Country": ["B"], "Year": [2020], "value": [1]}
    )
    _, report = clean(raw, fill={"category": "x", "unit": "t"})
    assert report.columns["exporter_country"] == "source"
    assert report.columns["Importer  Country"] == "target"


def test_schema_names_are_guessed_too() -> None:
    raw = pd.DataFrame(
        {"source": ["A"], "target": ["B"], "time": [2020], "category": ["x"], "weight": [1]}
    )
    _, report = clean(raw, fill={"unit": "t"})
    assert set(report.columns.values()) == {"source", "target", "time", "category", "weight"}


def test_guessed_reporter_table_ignores_edge_only_roles() -> None:
    raw = reporter_table([("A", "B", "Export", 1.0)]).assign(**{"reported by": "x"})
    _, report = clean(raw)
    assert "reported by" not in report.columns


def test_given_columns_mix_with_guessed_ones() -> None:
    raw = pd.DataFrame(
        {"src": ["A"], "dst": ["B"], "n": [5], "year": [2020], "Unit": ["t"], "value": [9]}
    )
    flows, report = clean(raw, source="src", target="dst", weight="n", fill={"category": "x"})
    assert report.columns == {
        "src": "source",
        "dst": "target",
        "year": "time",
        "n": "weight",
        "Unit": "unit",
    }
    assert report.guessed == ("time", "unit")
    assert report.filled == {"category": "x"}
    assert flows[W].tolist() == [5.0]
    summary = report.summary()
    assert "'src' -> source (given)" in summary
    assert "'year' -> time (guessed)" in summary
    assert "Filled: category = 'x'" in summary


def test_fill_constants() -> None:
    raw = pd.DataFrame({"from": ["A", "B"], "to": ["B", "A"], "count": [1, 2]})
    flows, report = clean(
        raw, fill={"time": np.int64(2019), "category": " migrants ", "unit": "people"}
    )
    nv.validate_flows(flows)
    assert flows[TIME].tolist() == [2019, 2019]
    assert flows[TIME].dtype == "int64"
    assert set(flows[CAT]) == {"migrants"}
    assert set(flows[UNIT]) == {"people"}
    assert report.filled == {"time": 2019, "category": "migrants", "unit": "people"}


def test_fill_weight_for_unweighted_edges() -> None:
    raw = pd.DataFrame({"src": ["A", "A"], "dst": ["B", "C"]})
    flows, _ = clean(raw, fill={"weight": 1, "time": 2020, "category": "links", "unit": "edges"})
    assert flows[W].tolist() == [1.0, 1.0]


def test_fill_stops_guessing_that_role() -> None:
    _, report = clean(edges(), fill={"unit": "kg"})
    assert "unit" not in report.columns
    assert report.filled == {"unit": "kg"}


@pytest.mark.parametrize(
    ("raw", "kwargs", "message"),
    [
        (pd.DataFrame({"a": [1], "b": [2]}), {}, "could not tell which columns"),
        (edges(value=[1, 2]), {}, "more than one column could be weight"),
        (
            edges().assign(reporter="A", partner="B"),
            {},
            "both source/target and reporter/partner",
        ),
        (edges().drop(columns="quantity"), {}, "no column for 'weight'"),
        (edges().drop(columns="year"), {}, "fill={'time': ...}"),
        (edges(), {"unit": "unit", "fill": {"unit": "kg"}}, "unit is given both"),
        (reporter_table([("A", "B", "Export", 1)]).drop(columns="Element"), {}, "'direction'"),
        (edges(), {"source": "nope"}, "source='nope' is not a column"),
        (edges(), {"source": "item", "target": "item"}, "given for more than one role"),
        (edges(), {"source": "exporter", "reporter": "importer"}, "not both"),
        (edges(), {"reporter": "exporter", "partner": "importer"}, "no column for 'direction'"),
        (
            edges(),
            {"reporter": "exporter", "partner": "importer", "reported_by": "unit"},
            "reported_by only applies",
        ),
        (edges(), {"fill": {"source": "A"}}, "fill accepts the roles"),
    ],
)
def test_mapping_errors(raw: pd.DataFrame, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SchemaError, match=message) as exc:
        clean(raw, **kwargs)
    assert "Columns found" in str(exc.value) or "fill accepts" in str(exc.value)


def test_mapping_error_suggests_keyword_arguments() -> None:
    with pytest.raises(SchemaError, match="source='origin'"):
        clean(pd.DataFrame({"a": [1]}))


def test_duplicate_raw_column_names_are_rejected() -> None:
    raw = pd.DataFrame([["A", "B", 1]], columns=["exporter", "importer", "exporter"])
    with pytest.raises(SchemaError, match="duplicate column names"):
        clean(raw)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"prefer": "importer"}, "prefer must be one of"),
        ({"zeros": "nan"}, "zeros must be one of"),
        ({"duplicates": "mean"}, "duplicates must be one of"),
        ({"fill": {"time": 2020.5}}, "not a valid time"),
        ({"fill": {"time": True}}, "not a valid time"),
        ({"fill": {"time": "2020"}}, "not a valid time"),
        ({"fill": {"weight": -1}}, "not a valid weight"),
        ({"fill": {"weight": float("inf")}}, "not a valid weight"),
        ({"fill": {"category": " "}}, "not a valid category"),
        ({"fill": {"unit": 5}}, "not a valid unit"),
        ({"direction_values": {"X": "exports"}}, "must map to 'export' or 'import'"),
        ({"unit_conversions": {"t": ("kg", 0.0)}}, "must be positive"),
        ({"unit_conversions": {"t": ("kg", True)}}, "must be positive"),
        ({"unit_conversions": {"t": (" ", 2.0)}}, "empty unit"),
        ({"aliases": {"A": ""}}, "empty name"),
    ],
)
def test_bad_options(kwargs: dict[str, Any], message: str) -> None:
    fill = kwargs.get("fill", {})
    raw = edges().drop(
        columns=[
            c
            for c, r in {
                "year": "time",
                "quantity": "weight",
                "item": "category",
                "unit": "unit",
            }.items()
            if r in fill
        ]
    )
    with pytest.raises(ValueError, match=message):
        clean(raw, **kwargs)


# ---------------------------------------------------------------------------
# Orientation and mirror flows


MIRROR_ROWS = [
    ("A", "B", "Export quantity", 100.0),  # A -> B, both sides report
    ("B", "A", "Import quantity", 80.0),
    ("A", "C", "Export quantity", 10.0),  # A -> C, source only
    ("C", "B", "Import quantity", 5.0),  # B -> C, target only
]


def test_orientation_and_mirror_prefer_target() -> None:
    flows, report = clean(reporter_table(MIRROR_ROWS))
    assert flows.columns.tolist() == [*FLOW, "reported_by"]
    got = flows[[SRC, DST, W, "reported_by"]].values.tolist()
    assert got == [
        ["A", "B", 80.0, BY_TARGET],
        ["A", "C", 10.0, BY_SOURCE],
        ["B", "C", 5.0, BY_TARGET],
    ]
    assert report.mirror == MirrorStats(
        both=1, source_only=1, target_only=1, median_disagreement=0.2, prefer="target"
    )
    assert reasons(report) == ["mirror flow: the target's report was used"]
    assert report.dropped.index.tolist() == [0]
    assert "median disagreement 20.0%" in report.summary()
    assert "used the target's report" in report.summary()


@pytest.mark.parametrize(
    ("prefer", "weight", "side"),
    [("source", 100.0, BY_SOURCE), ("mean", 90.0, "both"), ("max", 100.0, "both")],
)
def test_mirror_prefer_modes(prefer: Any, weight: float, side: str) -> None:
    flows, report = clean(reporter_table(MIRROR_ROWS), prefer=prefer)
    ab = flows[(flows[SRC] == "A") & (flows[DST] == "B")]
    assert ab[[W, "reported_by"]].values.tolist() == [[weight, side]]
    assert len(flows) == 3
    assert report.mirror is not None
    assert report.mirror.prefer == prefer
    assert step(report, "mirror flows").rows_out == 3
    if prefer == "source":
        assert reasons(report) == ["mirror flow: the source's report was used"]
        assert report.dropped.index.tolist() == [1]
        assert "used the source's report" in report.summary()
    else:
        assert report.dropped.empty
        assert f"({prefer})" in step(report, "mirror flows").detail


def test_mirror_mean_sums_each_side_first_even_when_keeping_duplicates() -> None:
    rows = [*MIRROR_ROWS, ("A", "B", "Export quantity", 20.0), ("C", "B", "Import quantity", 6.0)]
    flows, report = clean(reporter_table(rows), prefer="mean", duplicates="keep")
    ab = flows[(flows[SRC] == "A") & (flows[DST] == "B")]
    assert ab[W].tolist() == [100.0]  # (120 + 80) / 2
    bc = flows[(flows[SRC] == "B") & (flows[DST] == "C")]
    assert bc[W].tolist() == [5.0, 6.0]
    assert "mean of the two reports" in report.summary()


def test_mirror_drop_lists_every_summed_raw_row() -> None:
    rows = [*MIRROR_ROWS, ("A", "B", "Export quantity", 20.0)]
    flows, report = clean(reporter_table(rows))
    assert report.dropped.index.tolist() == [0, 4]
    assert step(report, "duplicates").detail == "summed 2 rows that share a key into 1"
    assert flows[W].sum() == 95.0


def test_mirror_without_overlap_and_zero_reports() -> None:
    _, report = clean(reporter_table([("A", "B", "Export", 1.0)]))
    assert report.mirror is not None
    assert report.mirror.median_disagreement is None
    assert "median disagreement" not in report.summary()
    rows = [("A", "B", "Export", 0.0), ("B", "A", "Import", 0.0)]
    _, report = clean(reporter_table(rows), zeros="keep", prefer="max")
    assert report.mirror is not None
    assert report.mirror.median_disagreement == 0.0
    assert "larger of the two reports" in report.summary()


def test_unclassified_directions_are_dropped() -> None:
    rows = [
        ("A", "B", "Export", 1.0),
        ("A", "B", "Re-export/import", 1.0),
        ("A", "B", "Production", 1.0),
    ]
    raw = reporter_table(rows)
    raw.loc[3] = ["A", "B", None, 1.0, "Wheat", 2021, "t"]
    flows, report = clean(raw)
    assert len(flows) == 1
    assert reasons(report) == [
        "unrecognized direction 'Re-export/import'",
        "unrecognized direction 'Production'",
        "missing direction",
    ]
    detail = step(report, "direction").detail
    assert "'Export' -> export" in detail
    assert "'Production' -> dropped" in detail


def test_direction_values_override_detection() -> None:
    rows = [("A", "B", "X", 1.0), ("B", "C", " M ", 2.0), ("A", "B", "Export", 3.0)]
    flows, report = clean(reporter_table(rows), direction_values={"X": "export", "M": "import"})
    assert flows[[SRC, DST]].values.tolist() == [["A", "B"], ["C", "B"]]
    assert reasons(report) == ["unrecognized direction 'Export'"]


def test_direction_detail_is_capped() -> None:
    rows = [("A", "B", f"Export {i}", 1.0) for i in range(25)]
    _, report = clean(reporter_table(rows), duplicates="keep")
    assert step(report, "direction").detail.endswith("and 5 more values")


def test_given_reporter_columns() -> None:
    raw = reporter_table(MIRROR_ROWS).rename(columns={"Element": "kind", "Value": "tonnes"})
    raw["Value"] = 0.0
    flows, report = clean(
        raw,
        reporter="Reporter Countries",
        partner="Partner Countries",
        direction="kind",
        weight="tonnes",
    )
    assert report.guessed == ("time", "category", "unit")
    assert len(flows) == 3


def test_reported_by_column_in_edge_table() -> None:
    raw = n_edges(
        4,
        exporter=["A"] * 4,
        importer=["B", "B", "C", "C"],
        quantity=[10.0, 12.0, 3.0, 4.0],
    )
    raw["reported_by"] = pd.Series(["exporter", "Importer", "nobody", None], dtype=object)
    flows, report = clean(raw)
    assert report.columns["reported_by"] == "reported_by"
    assert flows[[DST, W, "reported_by"]].values.tolist() == [["B", 12.0, BY_TARGET]]
    assert reasons(report) == [
        "mirror flow: the target's report was used",
        "unrecognized reported_by 'nobody'",
        "missing reported_by",
    ]


def test_edge_table_without_reporter_has_no_mirror_stats() -> None:
    flows, report = clean(edges(), prefer="source")
    assert report.mirror is None
    assert "Mirror" not in report.summary()
    assert flows.columns.tolist() == FLOW


# ---------------------------------------------------------------------------
# Names


def test_whitespace_aliases_and_accent_merges() -> None:
    raw = n_edges(
        6,
        exporter=["Türkiye", "Türkiye", "Turkiye ", "turkey", "Cote d'Ivoire", "Egypt"],
        importer=["Egypt", "Iraq", "Iraq", "Iraq", "Egypt", "Côte d\u2019Ivoire"],
        item=["Wheat", "wheat", "Wheat", "Wheat", "Wheat", "Wheat"],
    )
    flows, report = clean(raw, aliases={" turkey": "Türkiye"})
    assert set(flows[SRC]) == {"Türkiye", "Cote d'Ivoire", "Egypt"}
    assert set(flows[DST]) == {"Egypt", "Iraq", "Cote d'Ivoire"}
    assert set(flows[CAT]) == {"Wheat"}
    assert report.renamed == {
        "Côte d\u2019Ivoire": "Cote d'Ivoire",
        "Turkiye ": "Türkiye",
        "turkey": "Türkiye",
        "wheat": "Wheat",
    }
    detail = step(report, "names").detail
    assert "4 spellings changed: 1 by aliases, 3 case, accent or punctuation variants" in detail
    # Türkiye -> Iraq was reported three times and is summed.
    assert flows.loc[flows[DST] == "Iraq", W].tolist() == [3.0]


def test_merge_ties_go_to_first_alphabetically() -> None:
    raw = edges(exporter=["Cote d'Ivoire", "Côte d'Ivoire"], importer=["X", "Y"])
    flows, report = clean(raw)
    assert set(flows[SRC]) == {"Cote d'Ivoire"}
    assert report.renamed == {"Côte d'Ivoire": "Cote d'Ivoire"}


def test_names_without_letters_or_digits_are_not_merged() -> None:
    flows, report = clean(edges(exporter=["-", "--"], importer=["X", "X"]))
    assert set(flows[SRC]) == {"-", "--"}
    assert report.renamed == {}


def test_possible_aliases_are_reported_not_merged() -> None:
    raw = n_edges(
        4,
        exporter=["Kazakhstan", "Kazakstan", "Niger", "Nigeria"],
        item=["Soya beans", "Soya bean", "Soya beans", "Soya beans"],
    )
    flows, report = clean(raw)
    assert {"Kazakhstan", "Kazakstan"} <= set(flows[SRC])
    pairs = {(a, b) for a, b, _ in report.possible_aliases}
    assert pairs == {("Kazakhstan", "Kazakstan"), ("Soya bean", "Soya beans")}
    assert all(0.9 <= r < 1 for _, _, r in report.possible_aliases)
    assert "'Kazakhstan' ~ 'Kazakstan' (0.95)" in report.summary()


def test_non_string_names_become_strings() -> None:
    flows, report = clean(edges(exporter=[1, 2], importer=[2, 3]))
    assert flows[SRC].tolist() == ["1", "2"]
    assert report.renamed == {}


# ---------------------------------------------------------------------------
# Values


def test_weight_coercion_and_drops() -> None:
    values: list[Any] = [
        "1,234.5",
        " 12 ",
        7,
        "",
        None,
        "abc",
        "inf",
        np.nan,
        -1,
        True,
        "1.234,5",
        "-2,000",
        "NaN",
    ]
    flows, report = clean(n_edges(len(values), quantity=values))
    assert flows[W].tolist() == [1234.5, 12.0, 7.0]
    assert flows[W].dtype == "float64"
    assert reasons(report) == [
        "missing weight",
        "missing weight",
        "unparseable weight",
        "infinite weight",
        "missing weight",
        "negative weight",
        "unparseable weight",
        "unparseable weight",
        "negative weight",
        "missing weight",
    ]
    assert step(report, "weight").detail == (
        "1 infinite weight, 4 missing weight, 3 unparseable weight"
    )


def test_numeric_weight_with_missing_and_infinite() -> None:
    flows, report = clean(n_edges(3, quantity=[1.0, np.nan, np.inf]))
    assert len(flows) == 1
    assert reasons(report) == ["missing weight", "infinite weight"]


def test_time_coercion_and_drops() -> None:
    years: list[Any] = [2021, 2021.0, "2021", " 2022 ", "20x", 2021.5, None, "inf"]
    flows, report = clean(n_edges(len(years), year=years))
    assert flows[TIME].dtype == "int64"
    assert sorted(flows[TIME].tolist()) == [2021, 2021, 2021, 2022]
    assert reasons(report) == ["unparseable time", "invalid time", "missing time", "invalid time"]


def test_time_from_datetimes() -> None:
    raw = edges().assign(year=pd.to_datetime(pd.Series(["2020-03-01", None])))
    flows, report = clean(raw)
    assert flows[TIME].tolist() == [2020]
    assert reasons(report) == ["missing time"]


def test_missing_names_categories_and_units_are_dropped() -> None:
    raw = n_edges(
        5,
        exporter=["A", None, "  ", "B", "B"],
        importer=["B", "C", "C", None, "C"],
        item=["x", "x", "x", "x", np.nan],
    )
    flows, report = clean(raw)
    assert len(flows) == 1
    assert reasons(report) == [
        "missing source",
        "missing source",
        "missing target",
        "missing category",
    ]
    assert step(report, "missing values").detail == (
        "1 missing category, 2 missing source, 1 missing target"
    )


def test_unit_conversions() -> None:
    raw = n_edges(
        3,
        exporter=["A", "A", "B"],
        importer=["B", "C", "C"],
        item=["Horses"] * 3,
        quantity=[0.5, 12.0, 3.0],
        unit=[" 1000 An", "An", "t"],
    )
    conv = {"1000 An": ("head", 1000.0), "An": ("head", 1.0)}
    flows, report = clean(raw, unit_conversions=conv)
    assert flows[[W, UNIT]].values.tolist() == [[500.0, "head"], [12.0, "head"], [3.0, "t"]]
    assert step(report, "unit conversions").detail == (
        "1 rows '1000 An' -> 'head' (x1000); 1 rows 'An' -> 'head' (x1)"
    )
    assert report.units == {"Horses": {"head": 2, "t": 1}}
    summary = report.summary()
    assert "Categories with more than one unit" in summary
    assert "'Horses': head (2 rows), t (1 rows)" in summary
    _, report = clean(raw, unit_conversions={"kg": ("t", 0.001)})
    assert step(report, "unit conversions").detail == "no matching units"


# ---------------------------------------------------------------------------
# Filters and duplicates


def test_drop_nodes() -> None:
    raw = n_edges(3, exporter=["A", "World ", "B"], importer=["World", "B", "C"])
    flows, report = clean(raw, drop_nodes="World")
    assert flows[[SRC, DST]].values.tolist() == [["B", "C"]]
    assert reasons(report) == ["node listed in drop_nodes"] * 2
    assert step(report, "drop nodes").detail == "dropped rows touching 'World'"
    # Names in drop_nodes follow aliases and merges.
    flows, _ = clean(raw, drop_nodes=["Wörld"])
    assert len(flows) == 3
    flows, _ = clean(raw, drop_nodes=["Globe"], aliases={"Globe": "World"})
    assert len(flows) == 1
    raw = n_edges(3, exporter=["A", "WORLD", "WORLD"], importer=["World", "B", "C"])
    flows, _ = clean(raw, drop_nodes=["World"])
    assert len(flows) == 0


def test_zeros_drop_and_keep() -> None:
    raw = edges(quantity=[0.0, 2.0])
    flows, report = clean(raw)
    assert flows[W].tolist() == [2.0]
    assert reasons(report) == ["zero weight"]
    flows, report = clean(raw, zeros="keep")
    assert flows[W].tolist() == [0.0, 2.0]
    assert step(report, "zeros").detail == "kept 1 zero rows"


def test_self_loops_drop_and_keep() -> None:
    raw = edges(importer=["A", "C"])
    flows, report = clean(raw)
    assert flows[[SRC, DST]].values.tolist() == [["B", "C"]]
    assert reasons(report) == ["self-loop"]
    flows, report = clean(raw, self_loops=True)
    assert len(flows) == 2
    assert step(report, "self-loops").detail == "kept 1 self-loops"


def test_exact_duplicates_and_key_duplicates() -> None:
    raw = n_edges(4, exporter=["A"] * 4, importer=["B"] * 4, quantity=[1.0, 1.0, 2.0, 1.0]).assign(
        month=[1, 1, 2, 3]
    )
    flows, report = clean(raw)
    assert flows[W].tolist() == [4.0]
    assert reasons(report) == ["exact duplicate of an earlier row"]
    assert report.dropped.index.tolist() == [1]
    assert report.dropped["month"].tolist() == [1]
    assert step(report, "duplicates").detail == "summed 3 rows that share a key into 1"
    flows, report = clean(raw, duplicates="keep")
    assert flows[W].tolist() == [1.0, 2.0, 1.0]
    assert step(report, "duplicates").detail == "kept 3 rows that share a key"


# ---------------------------------------------------------------------------
# Output and report


def test_output_is_sorted_valid_and_raw_is_untouched() -> None:
    raw = n_edges(
        3,
        exporter=["C", "A", "B"],
        importer=["A", "C", "A"],
        year=[2021, 2021, 2020],
        item=["y", "x", "x"],
        quantity=["3", "1", "2"],
    )
    raw.index = pd.Index([10, 20, 30])
    before = raw.copy()
    flows, report = clean(raw)
    pd.testing.assert_frame_equal(raw, before)
    nv.validate_flows(flows)
    assert flows.index.equals(pd.RangeIndex(3))
    assert flows[[CAT, TIME, SRC]].values.tolist() == [
        ["x", 2020, "B"],
        ["x", 2021, "A"],
        ["y", 2021, "C"],
    ]
    assert report.rows_in == 3
    assert report.rows_out == 3
    assert report.dropped.columns.tolist() == [*raw.columns, "reason"]


def test_empty_result_is_a_valid_flow_frame() -> None:
    flows, report = clean(edges(quantity=[0, 0]))
    nv.validate_flows(flows)
    assert flows.empty
    assert flows.columns.tolist() == FLOW
    assert report.units == {}


def test_summary_and_to_frame() -> None:
    names = [f"Place {i:02d} " for i in range(25)]
    _, report = clean(n_edges(25, exporter=names, quantity=[1.0] * 24 + [0.0]))
    frame = report.to_frame()
    assert frame.columns.tolist() == ["step", "rows_in", "rows_out", "removed", "detail"]
    assert frame["step"].tolist() == [s.name for s in report.steps]
    assert frame.set_index("step").loc["zeros", "removed"] == 1
    summary = report.summary()
    lines = summary.splitlines()
    assert lines[0] == "Cleaned 25 raw rows into 24 flows."
    assert lines[1].startswith("Columns: 'exporter' -> source (guessed)")
    assert "- zeros: 25 -> 24 rows" in lines
    assert "Renamed 25 spellings:" in lines
    assert "  ... and 5 more" in lines
    assert "\u2014" not in summary


def test_possible_alias_list_is_capped_in_summary() -> None:
    names = [f"Region number {chr(65 + i)}" for i in range(8)]
    _, report = clean(n_edges(8, exporter=names))
    assert len(report.possible_aliases) == 28
    assert "  ... and 8 more" in report.summary()


def test_nothing_is_printed(capsys: pytest.CaptureFixture[str]) -> None:
    clean(reporter_table(MIRROR_ROWS))[1].summary()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_report_types_are_frozen() -> None:
    _, report = clean(edges())
    with pytest.raises(AttributeError):
        report.rows_in = 0  # type: ignore[misc]
    with pytest.raises(AttributeError):
        report.steps[0].name = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# FAOSTAT sample in the bulk-file layout


def test_clean_raw_sample_matches_combined_load_sample() -> None:
    flows, report = clean(faostat.raw_sample(items="Wheat", years=2022))
    expected = faostat.load_sample(items="Wheat", years=2022, reporter="combined")
    assert report.columns["Element"] == "direction"
    assert set(report.guessed) == set(report.columns.values())
    key = [CAT, TIME, SRC, DST]
    got = flows[FLOW].sort_values(key).reset_index(drop=True)
    want = expected[FLOW].sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want, check_exact=False, rtol=1e-12)
    m = report.mirror
    assert m is not None
    assert min(m.both, m.source_only, m.target_only) > 0
    assert m.median_disagreement is not None
    assert 0 < m.median_disagreement < 1
    assert m.both + m.source_only + m.target_only == len(flows)


def test_clean_raw_sample_drops_self_loops_like_load_sample() -> None:
    raw = faostat.raw_sample(items="Soya beans")
    assert (raw["Reporter Countries"] == raw["Partner Countries"]).any()
    flows, report = clean(raw)
    expected = faostat.load_sample(items="Soya beans", reporter="combined")
    assert len(flows) == len(expected)
    assert step(report, "self-loops").rows_out < step(report, "self-loops").rows_in
