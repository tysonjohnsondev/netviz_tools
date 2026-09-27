from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import UnknownMetricError


def _max_weight(g: nx.Graph[Any]) -> float:
    return max(float(d["weight"]) for _, _, d in g.edges(data=True))


def test_metric_series_values(flows: pd.DataFrame) -> None:
    graphs = nv.graphs_by(flows, by="year")
    ts = nv.temporal.metric_series(
        graphs,
        [
            "n_nodes",
            "n_edges",
            "density",
            "total_weight",
            "mean_strength",
            "reciprocity",
            "out_strength_hhi",
            "in_strength_hhi",
        ],
        custom={"max_weight": _max_weight},
        index_name="year",
    )
    assert ts.index.tolist() == [2020, 2021]
    assert ts.index.name == "year"
    r2020 = ts.loc[2020]
    assert r2020["n_nodes"] == 3
    assert r2020["n_edges"] == 4
    assert r2020["total_weight"] == 18.0
    assert r2020["density"] == pytest.approx(4 / 6)
    assert r2020["reciprocity"] == pytest.approx(0.5)  # A<->C is the only mutual pair
    # out-strength shares: A 15, B 2, C 1 of 18.
    assert r2020["out_strength_hhi"] == pytest.approx((15**2 + 2**2 + 1**2) / 18**2)
    assert r2020["max_weight"] == 10.0
    assert ts.columns[-1] == "max_weight"


def test_metrics_edge_cases() -> None:
    single: nx.DiGraph[str] = nx.DiGraph()
    single.add_node("a")
    summary = nv.temporal.graph_summary(single, ["density", "reciprocity", "out_strength_hhi"])
    assert summary["density"] == 0.0
    assert np.isnan(summary["reciprocity"])
    assert np.isnan(summary["out_strength_hhi"])
    und: nx.Graph[str] = nx.Graph()
    und.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 3.0)])
    s = nv.temporal.graph_summary(und, ["reciprocity", "in_strength_hhi", "mean_strength"])
    assert np.isnan(s["reciprocity"])
    assert s["in_strength_hhi"] == pytest.approx((1 + 16 + 9) / 64)
    assert s["mean_strength"] == pytest.approx(8 / 3)
    empty: nx.DiGraph[str] = nx.DiGraph()
    assert np.isnan(nv.temporal.graph_summary(empty, ["mean_strength"])["mean_strength"])


def test_unknown_metric(flows: pd.DataFrame) -> None:
    with pytest.raises(UnknownMetricError):
        nv.temporal.metric_series(nv.graphs_by(flows, by="year"), ["diameter"])  # type: ignore[list-item]


def test_tuple_keys_make_multiindex(flows: pd.DataFrame) -> None:
    ts = nv.temporal.metric_series(nv.graphs_by(flows), ["n_edges"], index_name=["year", "item"])
    assert isinstance(ts.index, pd.MultiIndex)
    assert ts.index.names == ["year", "item"]


def test_centrality_series(flows: pd.DataFrame) -> None:
    graphs = nv.graphs_by(flows, by="year")
    cs = nv.temporal.centrality_series(graphs, "out_strength")
    assert cs.loc[2021, "A"] == 20.0
    assert cs.columns.name == "node"
    picked = nv.temporal.centrality_series(graphs, "in_strength", nodes=["C", "Z"])
    assert picked.columns.tolist() == ["C", "Z"]
    assert picked["Z"].tolist() == [0.0, 0.0]
