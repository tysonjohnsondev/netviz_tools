from __future__ import annotations

import json
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import pytest

import netviz_tools as nv
from netviz_tools.plot import CONTINENT_COLORS, PALETTE, scale


def as_json(fig: go.Figure) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(fig.to_json())
    return data


def test_scale() -> None:
    assert scale([], 1, 2).size == 0
    assert scale([5, 5], 1, 2).tolist() == [2.0, 2.0]
    out = scale([0, 9, 99], 0, 1)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert out[1] == pytest.approx(0.5)
    assert scale([0, 5, 10], 0, 10, log=False).tolist() == [0.0, 5.0, 10.0]


def test_network_structure(wheat_graph: nx.DiGraph[Any]) -> None:
    fig = nv.plot.network(wheat_graph, top_n=20)
    spec = as_json(fig)
    names = [t.get("name") for t in spec["data"]]
    node_traces = [t for t in spec["data"] if t.get("mode") == "markers+text"]
    assert sum(len(t["x"]) for t in node_traces) == 20
    assert {t["name"] for t in node_traces} <= set(CONTINENT_COLORS) | {"Other"}
    for t in node_traces:
        if t["name"] in CONTINENT_COLORS:
            assert t["marker"]["color"] == CONTINENT_COLORS[t["name"]]
    assert "flows" in names
    assert spec["layout"]["title"]["text"] == "Wheat trade network, 2022"
    assert spec["layout"]["xaxis"]["visible"] is False
    flows_trace = next(t for t in spec["data"] if t.get("name") == "flows")
    assert all("→" in h for h in flows_trace["hovertext"])


def test_network_options(wheat_graph: nx.DiGraph[Any]) -> None:
    part = nv.metrics.communities(wheat_graph, seed=0)
    fig = nv.plot.network(
        wheat_graph,
        top_n=None,
        layout="community",
        color_by="community",
        partition=part,
        min_weight_quantile=0.5,
        title="custom",
    )
    spec = as_json(fig)
    node_traces = [t for t in spec["data"] if t.get("mode") == "markers+text"]
    assert node_traces[0]["name"] == "Community 0"
    assert node_traces[0]["marker"]["color"] == PALETTE[0]
    assert len(node_traces) <= len(PALETTE) + 1
    assert spec["layout"]["title"]["text"] == "custom"
    layouts: tuple[nv.plot.Layout, ...] = ("kamada_kawai", "circular")
    for layout in layouts:
        nv.plot.network(wheat_graph, top_n=10, layout=layout, color_by=None)


def test_network_generic_attribute_and_errors() -> None:
    g: nx.Graph[Any] = nx.Graph()
    g.add_weighted_edges_from([("a", "b", 3.0), ("b", "c", 1.0), ("c", "d", 2.0)])
    nx.set_node_attributes(g, {"a": "x", "b": "x", "c": "y"}, "kind")
    fig = nv.plot.network(g, color_by="kind")
    spec = as_json(fig)
    node_traces = {t["name"]: t for t in spec["data"] if t.get("mode") == "markers+text"}
    assert set(node_traces) == {"x", "y", "Other"}
    flows_trace = next(t for t in spec["data"] if t.get("name") == "flows")
    assert "↔" in flows_trace["hovertext"][0]
    assert spec["layout"]["title"]["text"] == "Trade network"
    with pytest.raises(ValueError, match="partition"):
        nv.plot.network(g, layout="community")
    with pytest.raises(ValueError, match="partition"):
        nv.plot.network(g, color_by="community")
    with pytest.raises(ValueError, match="unknown layout"):
        nv.plot.network(g, layout="random")  # type: ignore[arg-type]


def test_network_many_categories_fold_into_other() -> None:
    g: nx.Graph[Any] = nx.path_graph(12)
    nx.set_edge_attributes(g, dict.fromkeys(g.edges, 1.0), "weight")
    nx.set_node_attributes(g, {n: f"k{n}" for n in g}, "kind")
    spec = as_json(nv.plot.network(g, color_by="kind", top_n=None))
    names = [t["name"] for t in spec["data"] if t.get("mode") == "markers+text"]
    assert len(names) == len(PALETTE) + 1
    assert names[-1] == "Other"


def test_community_layout_single_group() -> None:
    g = nx.complete_graph(4)
    pos = nv.plot.community_layout(g, pd.Series(dict.fromkeys(g, 0)))
    assert set(pos) == set(g)


def test_sankey_structure(wheat_graph: nx.DiGraph[Any]) -> None:
    fig = nv.plot.sankey(wheat_graph, top_n=5)
    trace = as_json(fig)["data"][0]
    assert trace["type"] == "sankey"
    labels = trace["node"]["label"]
    assert len(labels) == 12  # 5 + other on each side
    assert labels[5] == "Other exporters"
    assert labels[-1] == "Other importers"
    total = sum(float(d["weight"]) for _, _, d in wheat_graph.edges(data=True))
    assert sum(trace["link"]["value"]) == pytest.approx(total)
    assert max(trace["link"]["source"]) < 6 <= min(trace["link"]["target"])
    assert trace["valuesuffix"] == " t"
    narrow = as_json(nv.plot.sankey(wheat_graph, top_n=3, other=False, color_by=None))["data"][0]
    assert len(narrow["node"]["label"]) == 6
    assert sum(narrow["link"]["value"]) < total


def test_sankey_requires_directed() -> None:
    with pytest.raises(ValueError, match="directed"):
        nv.plot.sankey(nx.Graph())


def test_flow_map_structure(wheat_graph: nx.DiGraph[Any]) -> None:
    fig = nv.plot.flow_map(wheat_graph, top_n=25)
    spec = as_json(fig)
    lines = [t for t in spec["data"] if t.get("mode") == "lines"]
    markers = [t for t in spec["data"] if t.get("mode") == "markers"]
    assert all(t["type"] == "scattergeo" for t in spec["data"])
    assert sum(t["lon"].count(None) for t in lines) == 25
    assert markers
    assert spec["layout"]["geo"]["projection"]["type"] == "natural earth"


def test_flow_map_coordinate_sources() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_weighted_edges_from([("p", "q", 5.0), ("q", "r", 1.0)])
    coords = pd.DataFrame({"lon": [0.0, 10.0], "lat": [0.0, 5.0]}, index=["p", "q"])
    with pytest.warns(UserWarning, match="no coordinates for 1 node"):
        fig = nv.plot.flow_map(g, coords=coords, color_by=None)
    lines = [t for t in as_json(fig)["data"] if t.get("mode") == "lines"]
    assert sum(t["lon"].count(None) for t in lines) == 1
    with pytest.raises(ValueError, match="missing columns"):
        nv.plot.flow_map(g, coords=coords[["lon"]])
    nx.set_node_attributes(g, {"p": 1.0, "q": 2.0, "r": 3.0}, "lon")
    nx.set_node_attributes(g, {"p": 1.0, "q": 2.0, "r": 3.0}, "lat")
    spec = as_json(nv.plot.flow_map(g, color_by=None))
    assert spec["data"][-1]["lon"] == [1.0, 2.0, 3.0]


def test_time_series(flows: pd.DataFrame) -> None:
    ts = nv.temporal.metric_series(nv.graphs_by(flows, by="time"), ["total_weight", "density"])
    fig = nv.plot.time_series(ts, ["total_weight"], title="t", y_title="tonnes")
    spec = as_json(fig)
    assert spec["data"][0]["x"] == [2020, 2021]
    assert spec["layout"]["yaxis"]["title"]["text"] == "tonnes"
    faceted = as_json(nv.plot.time_series(ts, facet=True))
    assert len(faceted["data"]) == 2
    assert "yaxis2" in faceted["layout"]
    wide = pd.DataFrame({f"c{i}": [1, 2] for i in range(9)})
    with pytest.raises(ValueError, match="too many"):
        nv.plot.time_series(wide)
    with pytest.raises(ValueError, match="not in df"):
        nv.plot.time_series(ts, ["nope"])
    multi = nv.temporal.metric_series(nv.graphs_by(flows), ["n_edges"])
    assert as_json(nv.plot.time_series(multi))["data"][0]["x"][0] == "(2020, 'x')"


def test_degree_distribution(wheat_graph: nx.DiGraph[Any]) -> None:
    fit = nv.stats.degree_distribution_fit(wheat_graph, "degree")
    spec = as_json(nv.plot.degree_distribution(fit))
    assert spec["data"][0]["name"] == "observed"
    assert len(spec["data"]) == 4
    assert spec["layout"]["xaxis"]["type"] == "log"
    assert spec["data"][1]["y"][0] == pytest.approx(fit.n_tail / fit.n)
    strength = nv.stats.degree_distribution_fit(wheat_graph, "out_strength")
    assert (
        "out strength" in as_json(nv.plot.degree_distribution(strength))["layout"]["title"]["text"]
    )


def test_plots_never_show(monkeypatch: pytest.MonkeyPatch, wheat_graph: nx.DiGraph[Any]) -> None:
    def boom(*_: object, **__: object) -> None:
        raise AssertionError("show() must not be called")

    monkeypatch.setattr(go.Figure, "show", boom)
    nv.plot.network(wheat_graph, top_n=5)
    nv.plot.sankey(wheat_graph, top_n=3)
    nv.plot.flow_map(wheat_graph, top_n=3)


def test_aggregated_title(flows: pd.DataFrame) -> None:
    g = nv.build_graph(flows, aggregate="mean")
    spec = as_json(nv.plot.sankey(g))
    assert spec["layout"]["title"]["text"].endswith("2020 to 2021 (mean)")
