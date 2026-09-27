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
    g = nv.build_graph(flows[flows.year == 2020])
    assert isinstance(g, nx.DiGraph)
    assert g["A"]["B"]["weight"] == 10.0
    assert g.graph == {
        "weight": "quantity",
        "aggregate": None,
        "unit": "t",
        "year": 2020,
        "item": "x",
    }


def test_mixed_years_raise_with_guidance(flows: pd.DataFrame) -> None:
    with pytest.raises(MixedSliceError, match=r"2 values of 'year' \(2020, 2021\).*graphs_by"):
        nv.build_graph(flows)


def test_mixed_items_raise() -> None:
    df = pd.DataFrame(
        {
            "exporter": list("ABCDEF"),
            "importer": list("BCDEFA"),
            "year": [1] * 6,
            "item": list("pqrstu"),
            "quantity": [1.0] * 6,
            "unit": ["t"] * 6,
        }
    )
    with pytest.raises(MixedSliceError, match=r"6 values of 'item' \(p, q, r, s, \.\.\.\)"):
        nv.build_graph(df)


def test_aggregate_sum_and_mean(flows: pd.DataFrame) -> None:
    total = nv.build_graph(flows, aggregate="sum")
    mean = nv.build_graph(flows, aggregate="mean")
    assert total["A"]["B"]["weight"] == 30.0
    assert mean["A"]["B"]["weight"] == 15.0
    # A->C appears only in 2020, so its mean counts 2021 as zero.
    assert mean["A"]["C"]["weight"] == 2.5
    assert total.graph["year"] == [2020, 2021]
    assert total.graph["n_slices"] == 2


def test_bad_aggregate(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="aggregate must be"):
        nv.build_graph(flows, aggregate="max")  # type: ignore[arg-type]


def test_mixed_units_always_raise(flows: pd.DataFrame) -> None:
    df = flows.assign(unit=["t"] * 6 + ["head"])
    with pytest.raises(MixedUnitError):
        nv.build_graph(df, aggregate="sum")


def test_self_loops_and_zero_weights(flows: pd.DataFrame) -> None:
    df = flows[flows.year == 2021]
    g = nv.build_graph(df)
    assert not g.has_edge("C", "C")
    g2 = nv.build_graph(df, self_loops=True)
    assert g2["C"]["C"]["weight"] == 3.0
    zero = df.assign(quantity=[0.0, 4.0, 3.0])
    assert not nv.build_graph(zero).has_edge("A", "B")


def test_duplicate_rows_are_summed(flows: pd.DataFrame) -> None:
    df = pd.concat([flows[flows.year == 2020]] * 2)
    assert nv.build_graph(df)["A"]["B"]["weight"] == 20.0


def test_undirected_sums_both_directions(flows: pd.DataFrame) -> None:
    g = nv.build_graph(flows[flows.year == 2020], directed=False)
    assert not g.is_directed()
    assert g["A"]["C"]["weight"] == 6.0  # A->C 5 plus C->A 1


def test_schema_errors_in_build(flows: pd.DataFrame) -> None:
    with pytest.raises(SchemaError, match="missing columns"):
        nv.build_graph(flows, weight="value")
    with pytest.raises(SchemaError, match="numeric"):
        nv.build_graph(flows[flows.year == 2020].assign(quantity="x"))
    with pytest.raises(SchemaError, match="negative"):
        nv.build_graph(flows[flows.year == 2020].assign(quantity=-1.0))


def test_generic_columns_without_slice_metadata() -> None:
    df = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"], "w": [1, 2]})
    g = nv.build_graph(df, source="src", target="dst", weight="w")
    assert g.graph == {"weight": "w", "aggregate": None}
    assert nv.build_graph(df, "src", "dst", "w", aggregate="mean")["b"]["c"]["weight"] == 2.0


def test_node_attrs(wheat_graph: nx.DiGraph) -> None:  # type: ignore[type-arg]
    assert wheat_graph.nodes["Ukraine"]["continent"] == "Europe"
    assert wheat_graph.nodes["Egypt"]["iso3"] == "EGY"


def test_graphs_by_keys(flows: pd.DataFrame) -> None:
    by_year = nv.graphs_by(flows, by="year")
    assert list(by_year) == [2020, 2021]
    assert all(isinstance(k, int) for k in by_year)
    both = nv.graphs_by(flows)
    assert list(both) == [(2020, "x"), (2021, "x")]
    with pytest.raises(SchemaError):
        nv.graphs_by(flows, by="colour")


def test_graphs_by_passes_kwargs() -> None:
    df = pd.DataFrame(
        {
            "exporter": ["A", "A"],
            "importer": ["B", "B"],
            "year": [1, 1],
            "item": ["p", "q"],
            "quantity": [1.0, 2.0],
            "unit": ["t", "t"],
        }
    )
    with pytest.raises(MixedSliceError):
        nv.graphs_by(df, by="year")
    assert nv.graphs_by(df, by="year", aggregate="sum")[1]["A"]["B"]["weight"] == 3.0


def test_graph_to_flows_round_trip(flows: pd.DataFrame) -> None:
    sub = flows[(flows.year == 2020)]
    back = nv.graph_to_flows(nv.build_graph(sub))
    nv.validate_flows(back)
    expected = sub.sort_values(["exporter", "importer"], ignore_index=True)
    pd.testing.assert_frame_equal(back, expected[back.columns.tolist()])


def test_graph_to_flows_aggregated_has_no_year(flows: pd.DataFrame) -> None:
    back = nv.graph_to_flows(nv.build_graph(flows, aggregate="sum"))
    assert "year" not in back.columns
    assert back["item"].unique().tolist() == ["x"]


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
    df = pd.DataFrame(rows, columns=["exporter", "importer", "year", "item", "quantity"])
    return df.assign(unit="t", year=df["year"].astype("int64"))


def _real(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.exporter != df.importer) & (df.quantity > 0)]


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
    expected = _real(df)["quantity"].sum()
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
    assert set(g.nodes) == set(real.exporter) | set(real.importer)
    assert g.number_of_edges() == len(real[["exporter", "importer"]].drop_duplicates())


@settings(max_examples=150, deadline=None)
@given(ROWS)
def test_graphs_by_round_trip(rows: list[tuple[str, str, int, str, float]]) -> None:
    df = _frame(rows)
    graphs = nv.graphs_by(df, by=["year", "item"])
    parts = [nv.graph_to_flows(g) for g in graphs.values()]
    back: pd.DataFrame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    keys = ["exporter", "importer", "year", "item"]
    expected = _real(df).groupby(keys, as_index=False).agg(quantity=("quantity", "sum"))
    if expected.empty:
        assert back.empty or back["quantity"].sum() == 0
        return
    back = back.sort_values(keys, ignore_index=True)
    expected = expected.sort_values(keys, ignore_index=True)
    pd.testing.assert_frame_equal(back[keys], expected[keys], check_dtype=False)
    assert np.allclose(back["quantity"], expected["quantity"], rtol=1e-12)
