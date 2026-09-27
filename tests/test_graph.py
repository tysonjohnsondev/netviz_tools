from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import netviz_tools as nv
from netviz_tools import MixedSliceError, MixedUnitError, SchemaError


def test_single_slice(flows: pd.DataFrame) -> None:
    g = nv.build_graph(flows[flows.time == 2020])
    assert isinstance(g, nx.DiGraph)
    assert g["A"]["B"]["weight"] == 10.0
    assert g.graph == {"aggregate": None, "unit": "t", "time": 2020, "category": "x"}


def test_mixed_times_raise_with_guidance(flows: pd.DataFrame) -> None:
    with pytest.raises(
        MixedSliceError, match=r"2 values of 'time' \(2020, 2021\).*time= or category=.*graphs_by"
    ):
        nv.build_graph(flows)


def test_mixed_categories_raise() -> None:
    df = pd.DataFrame(
        {
            "source": list("ABCDEF"),
            "target": list("BCDEFA"),
            "time": [1] * 6,
            "category": list("pqrstu"),
            "weight": [1.0] * 6,
            "unit": ["t"] * 6,
        }
    )
    with pytest.raises(MixedSliceError, match=r"6 values of 'category' \(p, q, r, s, \.\.\.\)"):
        nv.build_graph(df)


def test_time_filter_scalar_and_iterable(flows: pd.DataFrame) -> None:
    g = nv.build_graph(flows, time=2020)
    assert g.graph["time"] == 2020
    assert g["A"]["B"]["weight"] == 10.0
    both = nv.build_graph(flows, time=[2020, 2021], aggregate="sum")
    assert both["A"]["B"]["weight"] == 30.0
    assert both.graph["time"] == [2020, 2021]
    assert nv.build_graph(flows, time=range(2021, 2022)).graph["time"] == 2021
    with pytest.raises(MixedSliceError):
        nv.build_graph(flows, time=(2020, 2021))


def test_category_filter_selects_one_slice() -> None:
    sample = nv.datasets.faostat.load_sample(years=2021)
    with pytest.raises(MixedSliceError, match="category"):
        nv.build_graph(sample)
    g = nv.build_graph(sample, category="Soya beans")
    assert g.graph["category"] == "Soya beans"
    two = nv.build_graph(sample, category=["Wheat", "Maize (corn)"], aggregate="sum")
    assert two.graph["category"] == ["Maize (corn)", "Wheat"]
    assert two.graph["n_slices"] == 2


def test_unknown_filter_values_raise(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match=r"time=1999\. Available values \(2\): 2020, 2021$"):
        nv.build_graph(flows, time=1999)
    with pytest.raises(
        ValueError,
        match=r"category='X' among the selected time values\. Did you mean: 'x'\? Available",
    ):
        nv.build_graph(flows, time=2020, category="X")
    with pytest.raises(ValueError, match="among the selected time values"):
        nv.build_graph(flows, time=2020, category=["x", "y"])
    with pytest.raises(ValueError, match="at least one value"):
        nv.build_graph(flows, time=[])
    many = flows.assign(category=[f"c{i}" for i in range(7)])
    with pytest.raises(ValueError, match=r"Available values \(7\): 'c0', .*'c6'$"):
        nv.build_graph(many, category="zzz")
    wide = pd.concat([flows.assign(time=t) for t in range(1990, 2002)], ignore_index=True)
    with pytest.raises(ValueError, match=r"\(12\): 1990, .*1999, \.\.\.$"):
        nv.build_graph(wide, time=2030)
    with pytest.raises(SchemaError, match="missing columns"):
        nv.build_graph(flows.drop(columns="category"), category="x")
    mixed_types = flows.assign(category=[1, "a", 1, "a", 1, "a", 1])
    with pytest.raises(ValueError, match=r"Available values \(2\): 1, 'a'$"):
        nv.build_graph(mixed_types, category="b")


def test_labels_propagate(flows: pd.DataFrame) -> None:
    df = flows.copy()
    df.attrs["labels"] = {"source": "Origin", "target": "Destination"}
    g = nv.build_graph(df, time=2020)
    assert g.graph["labels"] == {"source": "Origin", "target": "Destination"}
    g.graph["labels"]["source"] = "changed"
    assert df.attrs["labels"]["source"] == "Origin"  # a copy, not the same dict
    back = nv.graph_to_flows(nv.build_graph(df, time=2021))
    assert back.attrs["labels"] == {"source": "Origin", "target": "Destination"}
    assert "labels" not in nv.build_graph(flows, time=2020).graph
    assert "labels" not in nv.graph_to_flows(nv.build_graph(flows, time=2020)).attrs


def test_labels_reach_every_group(flows: pd.DataFrame) -> None:
    df = flows.copy()
    df.attrs["labels"] = {"node": "Region"}
    graphs = nv.graphs_by(df)
    assert len(graphs) == 2
    assert all(g.graph["labels"] == {"node": "Region"} for g in graphs.values())
    sample = nv.datasets.faostat.load_sample(years=[2021, 2022])
    for g in nv.graphs_by(sample, by="category", time=2022).values():
        assert g.graph["labels"]["source"] == "Exporter"
        assert g.graph["time"] == 2022


def test_aggregate_sum_and_mean(flows: pd.DataFrame) -> None:
    total = nv.build_graph(flows, aggregate="sum")
    mean = nv.build_graph(flows, aggregate="mean")
    assert total["A"]["B"]["weight"] == 30.0
    assert mean["A"]["B"]["weight"] == 15.0
    # A->C appears only in 2020, so its mean counts 2021 as zero.
    assert mean["A"]["C"]["weight"] == 2.5
    assert total.graph["time"] == [2020, 2021]
    assert total.graph["n_slices"] == 2


def test_bad_aggregate(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="aggregate must be"):
        nv.build_graph(flows, aggregate="max")  # type: ignore[arg-type]


def test_mixed_units_always_raise(flows: pd.DataFrame) -> None:
    df = flows.assign(unit=["t"] * 6 + ["head"])
    with pytest.raises(MixedUnitError):
        nv.build_graph(df, aggregate="sum")


def test_self_loops_and_zero_weights(flows: pd.DataFrame) -> None:
    df = flows[flows.time == 2021]
    g = nv.build_graph(df)
    assert not g.has_edge("C", "C")
    g2 = nv.build_graph(df, self_loops=True)
    assert g2["C"]["C"]["weight"] == 3.0
    zero = df.assign(weight=[0.0, 4.0, 3.0])
    assert not nv.build_graph(zero).has_edge("A", "B")


def test_duplicate_rows_are_summed(flows: pd.DataFrame) -> None:
    df = pd.concat([flows[flows.time == 2020]] * 2)
    assert nv.build_graph(df)["A"]["B"]["weight"] == 20.0


def test_undirected_sums_both_directions(flows: pd.DataFrame) -> None:
    g = nv.build_graph(flows[flows.time == 2020], directed=False)
    assert not g.is_directed()
    assert g["A"]["C"]["weight"] == 6.0  # A->C 5 plus C->A 1


def test_schema_errors_in_build(flows: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        nv.build_graph(flows.drop(columns="weight"))
    with pytest.raises(SchemaError, match="numeric"):
        nv.build_graph(flows[flows.time == 2020].assign(weight="x"))
    with pytest.raises(SchemaError, match="negative"):
        nv.build_graph(flows[flows.time == 2020].assign(weight=-1.0))


def test_minimal_columns_without_slice_metadata() -> None:
    df = pd.DataFrame({"source": ["a", "b"], "target": ["b", "c"], "weight": [1, 2]})
    g = nv.build_graph(df)
    assert g.graph == {"aggregate": None}
    assert nv.build_graph(df, aggregate="mean")["b"]["c"]["weight"] == 2.0


def test_node_attrs(wheat_graph: nx.DiGraph) -> None:  # type: ignore[type-arg]
    assert wheat_graph.nodes["Ukraine"]["continent"] == "Europe"
    assert wheat_graph.nodes["Egypt"]["iso3"] == "EGY"


def test_graphs_by_keys(flows: pd.DataFrame) -> None:
    by_time = nv.graphs_by(flows, by="time")
    assert list(by_time) == [2020, 2021]
    assert all(isinstance(k, int) for k in by_time)
    both = nv.graphs_by(flows)
    assert list(both) == [(2020, "x"), (2021, "x")]
    with pytest.raises(SchemaError):
        nv.graphs_by(flows, by="colour")


def test_graphs_by_passes_kwargs() -> None:
    df = pd.DataFrame(
        {
            "source": ["A", "A"],
            "target": ["B", "B"],
            "time": [1, 1],
            "category": ["p", "q"],
            "weight": [1.0, 2.0],
            "unit": ["t", "t"],
        }
    )
    with pytest.raises(MixedSliceError):
        nv.graphs_by(df, by="time")
    assert nv.graphs_by(df, by="time", aggregate="sum")[1]["A"]["B"]["weight"] == 3.0
    assert nv.graphs_by(df, by="time", category="q")[1]["A"]["B"]["weight"] == 2.0


def test_graph_to_flows_round_trip(flows: pd.DataFrame) -> None:
    sub = flows[(flows.time == 2020)]
    back = nv.graph_to_flows(nv.build_graph(sub))
    nv.validate_flows(back)
    assert back.columns.tolist() == list(nv.FLOW_COLUMNS)
    expected = sub.sort_values(["source", "target"], ignore_index=True)
    pd.testing.assert_frame_equal(back, expected[back.columns.tolist()])


def test_graph_to_flows_aggregated_has_no_time(flows: pd.DataFrame) -> None:
    back = nv.graph_to_flows(nv.build_graph(flows, aggregate="sum"))
    assert back.columns.tolist() == ["source", "target", "category", "weight", "unit"]
    assert back["category"].unique().tolist() == ["x"]


# ---------------------------------------------------------------------------
# Property-based invariants

NAMES = st.sampled_from(list("ABCDEFG"))
ROWS = st.lists(
    st.tuples(
        NAMES,
        NAMES,
        st.sampled_from([2019, 2020, 2021]),
        st.sampled_from(["wheat", "maize"]),
        st.floats(min_value=0, max_value=1e9, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=60,
)


def _frame(rows: list[tuple[str, str, int, str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["source", "target", "time", "category", "weight"])
    return df.assign(unit="t", time=df["time"].astype("int64"))


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.source != df.target) & (df.weight > 0)]


@settings(max_examples=150, deadline=None)
@given(ROWS)
def test_total_out_strength_equals_total_weight(
    rows: list[tuple[str, str, int, str, float]],
) -> None:
    df = _frame(rows)
    g = nv.build_graph(df, aggregate="sum")
    assert isinstance(g, nx.DiGraph)
    out_strength = sum(s for _, s in g.out_degree(weight="weight"))
    in_strength = sum(s for _, s in g.in_degree(weight="weight"))
    expected = _real(df)["weight"].sum()
    assert np.isclose(out_strength, expected, rtol=1e-9)
    assert np.isclose(in_strength, expected, rtol=1e-9)
    undirected = nv.build_graph(df, aggregate="sum", directed=False)
    assert np.isclose(undirected.size(weight="weight"), expected, rtol=1e-9)


@settings(max_examples=150, deadline=None)
@given(ROWS)
def test_node_set_is_endpoints_of_real_flows(rows: list[tuple[str, str, int, str, float]]) -> None:
    df = _frame(rows)
    real = _real(df)
    g = nv.build_graph(df, aggregate="sum")
    assert set(g.nodes) == set(real.source) | set(real.target)
    assert g.number_of_edges() == len(real[["source", "target"]].drop_duplicates())


@settings(max_examples=150, deadline=None)
@given(ROWS)
def test_graphs_by_round_trip(rows: list[tuple[str, str, int, str, float]]) -> None:
    df = _frame(rows)
    graphs = nv.graphs_by(df, by=["time", "category"])
    parts = [nv.graph_to_flows(g) for g in graphs.values()]
    back: pd.DataFrame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    keys = ["source", "target", "time", "category"]
    expected = _real(df).groupby(keys, as_index=False).agg(weight=("weight", "sum"))
    if expected.empty:
        assert back.empty or back["weight"].sum() == 0
        return
    back = back.sort_values(keys, ignore_index=True)
    expected = expected.sort_values(keys, ignore_index=True)
    pd.testing.assert_frame_equal(back[keys], expected[keys], check_dtype=False)
    assert np.allclose(back["weight"], expected["weight"], rtol=1e-12)
