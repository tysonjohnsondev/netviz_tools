"""Plots on flow tables and FAOSTAT data: time, category, focus, labels, auto."""

from __future__ import annotations

import json
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

import netviz_tools as nv
from netviz_tools.errors import MixedSliceError
from netviz_tools.plot import CONTINENT_COLORS, scale
from netviz_tools.plot._style import DECREASE_COLOR, INCREASE_COLOR, nice_ticks

faostat = nv.datasets.faostat


def spec(fig: go.Figure) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(fig.to_json())
    return data


def node_traces(fig: go.Figure) -> list[dict[str, Any]]:
    """Traces of node markers (not the legend-only entries drawn when there is a focus)."""
    return [
        t
        for t in spec(fig)["data"]
        if str(t.get("legendgroup", "")).startswith("node")
        and (t.get("x") or t.get("lon") or [0]) != [None]
    ]


def labels_shown(fig: go.Figure) -> list[str]:
    """Non-empty node labels (they live in their own trace, drawn above the markers)."""
    trace = next(t for t in spec(fig)["data"] if t.get("name") == "labels")
    return [x for x in trace.get("text") or [] if x]


def edge_hover(fig: go.Figure) -> list[str]:
    trace = next(t for t in spec(fig)["data"] if t.get("name") == "edges")
    return list(trace.get("hovertext") or [])


def caption(fig: go.Figure) -> str:
    text: str = spec(fig)["layout"]["annotations"][0]["text"]
    return text


def total_weight(g: nx.Graph[Any]) -> float:
    return float(sum(w for _, _, w in g.edges(data="weight", default=0.0)))


@pytest.fixture(scope="module")
def sample() -> pd.DataFrame:
    """Every item and year of the bundled FAOSTAT sample."""
    return faostat.load_sample()


@pytest.fixture(scope="module")
def wheat(sample: pd.DataFrame) -> pd.DataFrame:
    """Wheat flows, 2021 to 2023, keeping the FAOSTAT display labels."""
    out = sample[(sample["category"] == "Wheat") & sample["time"].between(2021, 2023)]
    assert out.attrs["labels"]["source"] == "Exporter"
    return out


@pytest.fixture(scope="module")
def soya(sample: pd.DataFrame) -> pd.DataFrame:
    return sample[(sample["category"] == "Soya beans") & sample["time"].between(2020, 2022)]


# --- helpers -------------------------------------------------------------------------------


def test_scale_and_ticks() -> None:
    assert scale([], 1, 2).size == 0
    assert scale([5, 5], 1, 2).tolist() == [2.0, 2.0]
    out = scale([0, 9, 99], 0, 1)
    assert out[0] == 0.0
    assert out[-1] == 1.0
    assert out[1] == pytest.approx(0.5)
    assert scale([0, 5, 10], 0, 10, log=False).tolist() == [0.0, 5.0, 10.0]
    assert nice_ticks(3, 3e6, log=True) == [10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
    assert nice_ticks(0, 34, log=True) == [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]
    assert nice_ticks(12, 15, log=True) == [12.0, 13.0, 14.0, 15.0]  # too narrow for log ticks
    assert nice_ticks(-5e6, 3e6) == [-4e6, -2e6, 0.0, 2e6]
    assert nice_ticks(2, 2) == [2.0]


# --- network on flow tables ----------------------------------------------------------------


def test_network_on_flows_with_time_and_category(sample: pd.DataFrame) -> None:
    fig = nv.plot.network(sample, category="Wheat", time=2022, top_n=15)
    assert not fig.frames
    assert spec(fig)["layout"]["title"]["text"] == "Wheat network, 2022"
    assert sum(len(t["x"]) for t in node_traces(fig)) == 15
    assert caption(fig) == (
        "Node size: Exports + imports; edge width: quantity; "
        "arrows point from exporter to importer."
    )
    assert all(h.endswith(" t") and "→" in h and "Quantity: " in h for h in edge_hover(fig))
    hover = node_traces(fig)[0]["hovertext"][0]
    assert "Exports + imports: " in hover
    assert " t<br>" in hover
    with pytest.raises(MixedSliceError, match="3 categories"):
        nv.plot.network(sample, time=2022)
    with pytest.raises(ValueError, match=r"category=\['Rice'\] not found"):
        nv.plot.network(sample, category="Rice")
    with pytest.raises(ValueError, match="time"):
        nv.plot.network(sample, category="Wheat", time=1990)
    no_time = sample[sample["time"] == 2022].drop(columns=["time"])
    with pytest.raises(ValueError, match="no 'time' column"):
        nv.plot.network(no_time, category="Wheat", time=2022)
    no_cat = sample[sample["category"] == "Wheat"].drop(columns=["category"])
    with pytest.raises(ValueError, match="no 'category' column"):
        nv.plot.network(no_cat, category="Wheat")


def test_node_attrs_colour_by_continent(wheat_2022: pd.DataFrame) -> None:
    fig = nv.plot.network(
        wheat_2022, node_attrs=faostat.countries(), color_by="continent", top_n=25
    )
    traces = node_traces(fig)
    assert {t["name"] for t in traces} <= set(CONTINENT_COLORS) | {"Other", "No value"}
    for t in traces:
        if t["name"] in CONTINENT_COLORS:
            assert t["marker"]["color"] == CONTINENT_COLORS[t["name"]]
    assert spec(fig)["layout"]["legend"]["title"]["text"] == "Continent"


def test_network_animation(flows: pd.DataFrame) -> None:
    fig = nv.plot.network(flows, color_by="pagerank")
    assert [f.name for f in fig.frames] == ["2020", "2021"]
    assert {len(f.data) for f in fig.frames} == {len(fig.data)}
    s = spec(fig)
    slider = s["layout"]["sliders"][0]
    assert [step["label"] for step in slider["steps"]] == ["2020", "2021"]
    assert slider["currentvalue"]["prefix"] == "Time: "
    assert slider["steps"][0]["args"][1]["frame"]["redraw"] is False
    assert s["layout"]["title"]["text"] == "x network, 2020 to 2021"
    # sizes are recomputed per period (strength changes) on one shared scale
    sizes = [
        list(next(t for t in f.data if t.legendgroup == "nodes").marker.size) for f in fig.frames
    ]
    assert sizes[0] != sizes[1]
    # build_graph drops the self-loop C -> C, so C has no flows in 2021
    last = next(t for t in fig.frames[1].data if t.legendgroup == "nodes")
    i = next(i for i, h in enumerate(last.hovertext) if h.startswith("<b>C</b>"))
    assert last.marker.size[i] == 0
    assert "not present in this period" in last.hovertext[i]
    one = nv.plot.network(flows, time=[2021])
    assert not one.frames
    assert spec(one)["layout"]["title"]["text"] == "x network, 2021"


def test_ego_on_flows_animates(flows: pd.DataFrame) -> None:
    fig = nv.plot.ego(flows, "A")
    assert len(fig.frames) == 2
    assert spec(fig)["layout"]["title"]["text"] == "A: neighbours within 1 step"
    single = nv.plot.ego(flows, "A", time=2020)
    assert spec(single)["layout"]["title"]["text"] == "A: neighbours within 1 step, 2020"


# --- flow_map -------------------------------------------------------------------------------


def test_flow_map_animation(soya: pd.DataFrame) -> None:
    fig = nv.plot.flow_map(soya, top_n=10)
    assert [f.name for f in fig.frames] == ["2020", "2021", "2022"]
    assert {len(f.data) for f in fig.frames} == {len(fig.data)}
    s = spec(fig)
    assert all(t["type"] == "scattergeo" for t in s["data"])
    slider = s["layout"]["sliders"][0]
    assert [step["label"] for step in slider["steps"]] == ["2020", "2021", "2022"]
    assert slider["currentvalue"]["prefix"] == "Year: "
    assert all(step["args"][1]["frame"]["redraw"] is True for step in slider["steps"])
    assert s["layout"]["title"]["text"] == "Soya beans flows, 10 largest, 2020 to 2022"
    for frame in fig.frames:
        hover = next(t for t in frame.data if t.name == "edges").hovertext
        assert len(hover) == 10
    assert "arrows" not in caption(fig)


def test_flow_map_focus_and_labels(wheat_2022: pd.DataFrame) -> None:
    fig = nv.plot.flow_map(wheat_2022, focus="Egypt", top_n=8, show_labels=True)
    hover = edge_hover(fig)
    assert len(hover) == 8
    assert all("Egypt" in h.split("<br>")[0] for h in hover)
    assert spec(fig)["layout"]["title"]["text"] == "Wheat flows of Egypt, 2022"
    texts = labels_shown(fig)
    assert "Egypt" in texts
    assert len(texts) == 9
    egypt = [
        w
        for t in node_traces(fig)
        for w, h in zip(t["marker"]["line"]["width"], t["hovertext"], strict=True)
        if h.startswith("<b>Egypt</b>")
    ]
    assert egypt == [2.5]


def test_flow_map_coords_and_missing_coordinates(flows: pd.DataFrame) -> None:
    coords = pd.DataFrame(
        {"lon": [0.0, 10.0, 20.0], "lat": [0.0, 5.0, 10.0]}, index=["A", "B", "C"]
    )
    fig = nv.plot.flow_map(flows, coords=coords, time=2020, color_by=None)
    lons = sorted(x for t in node_traces(fig) for x in t["lon"])
    assert lons == [0.0, 10.0, 20.0]
    with pytest.warns(UserWarning, match=r"no coordinates for 1 node\(s\).*\['C'\]"):
        partial = nv.plot.flow_map(flows, coords=coords.loc[["A", "B"]], time=2020)
    assert all("C" not in h for h in edge_hover(partial))
    with pytest.raises(ValueError, match="coords is missing columns"):
        nv.plot.flow_map(flows, coords=coords[["lon"]])


def test_flow_map_from_node_attributes() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_edge("p", "q", weight=3.0)
    g.add_edge("q", "p", weight=1.0)
    nx.set_node_attributes(g, {"p": 1.0, "q": 2.0}, "lon")
    nx.set_node_attributes(g, {"p": 3.0, "q": 4.0}, "lat")
    fig = nv.plot.flow_map(g, size_by="degree", edge_width_by=None, projection="equirectangular")
    assert len(edge_hover(fig)) == 2
    assert spec(fig)["layout"]["geo"]["projection"]["type"] == "equirectangular"


# --- labels ---------------------------------------------------------------------------------


def test_faostat_labels_propagate(wheat_graph: nx.DiGraph[Any]) -> None:
    assert wheat_graph.graph["labels"]["source"] == "Exporter"
    heat = spec(nv.plot.adjacency(wheat_graph, top_n=10))
    assert heat["layout"]["xaxis"]["title"]["text"] == "Importer"
    assert heat["layout"]["yaxis"]["title"]["text"] == "Exporter"
    assert heat["data"][0]["colorbar"]["title"]["text"] == "Quantity (t)"
    cells = [h for row in heat["data"][0]["text"] for h in row if h]
    assert cells
    assert all(h.startswith("Exporter: ") and "<br>Importer: " in h for h in cells)
    bars = spec(nv.plot.ranking(wheat_graph, size_by="out_strength", top_n=5))
    assert bars["layout"]["xaxis"]["title"]["text"] == "Exports (t)"
    assert bars["layout"]["title"]["text"] == "Wheat: top 5 by exports, 2022"
    fig = nv.plot.network(wheat_graph, top_n=10, color_by=None)
    assert caption(fig).endswith("arrows point from exporter to importer.")


def test_user_labels_override(wheat_graph: nx.DiGraph[Any]) -> None:
    labels = {"source": "Origin", "target": "Destination", "out_strength": "Shipped"}
    trace = spec(nv.plot.sankey(wheat_graph, top_n=3, labels=labels))["data"][0]
    assert trace["node"]["label"][3] == "Other origins"
    assert trace["node"]["label"][-1] == "Other destinations"
    bars = spec(nv.plot.ranking(wheat_graph, size_by="out_strength", top_n=3, labels=labels))
    assert bars["layout"]["xaxis"]["title"]["text"] == "Shipped (t)"
    fig = nv.plot.network(wheat_graph, top_n=5, labels={"weight": "Tonnes"}, color_by=None)
    assert all("Tonnes: " in h for h in edge_hover(fig))


# --- sankey ---------------------------------------------------------------------------------


def test_sankey_on_faostat(wheat_graph: nx.DiGraph[Any], sample: pd.DataFrame) -> None:
    trace = spec(nv.plot.sankey(wheat_graph, top_n=5))["data"][0]
    assert trace["type"] == "sankey"
    assert trace["valuesuffix"] == " t"
    labels = trace["node"]["label"]
    assert len(labels) == 12  # 5 + "Other" on each side
    assert labels[5] == "Other exporters"
    assert labels[-1] == "Other importers"
    assert max(trace["link"]["source"]) < 6 <= min(trace["link"]["target"])
    assert "Exporter: " in trace["link"]["customdata"][0]
    total = total_weight(wheat_graph)
    assert sum(trace["link"]["value"]) == pytest.approx(total)
    everything = spec(nv.plot.sankey(wheat_graph, top_n=1000, color_by=None))["data"][0]
    assert "Other exporters" not in everything["node"]["label"]
    assert sum(everything["link"]["value"]) == pytest.approx(total)
    layout = spec(nv.plot.sankey(wheat_graph, top_n=5))["layout"]
    assert [a["text"] for a in layout["annotations"]] == ["<b>Exporter</b>", "<b>Importer</b>"]
    assert layout["title"]["text"] == "Wheat flows, top exporters to top importers, 2022"
    focused = spec(nv.plot.sankey(sample, category="Wheat", time=2022, focus="Egypt"))
    assert "Egypt" in focused["data"][0]["node"]["label"]
    with pytest.raises(ValueError, match="one period"):
        nv.plot.sankey(sample, category="Wheat", time=[2021, 2022])


def test_sankey_continent_legend(wheat_graph: nx.DiGraph[Any]) -> None:
    fig = nv.plot.sankey(wheat_graph, top_n=4, color_by="continent")
    legend = [t for t in spec(fig)["data"] if t["type"] == "scatter"]
    assert legend
    assert all(t["name"] in CONTINENT_COLORS or t["name"] == "Other" for t in legend)
    assert spec(fig)["layout"]["legend"]["title"]["text"] == "Continent"


# --- ranking --------------------------------------------------------------------------------


def test_ranking_previous_period(wheat: pd.DataFrame) -> None:
    fig = nv.plot.ranking(wheat, time=[2022, 2023], size_by="out_strength", top_n=6)
    bar, tick = spec(fig)["data"]
    assert bar["name"] == "2023"
    assert tick["name"] == "2022"
    assert tick["marker"]["symbol"] == "line-ns"
    assert bar["x"] == sorted(bar["x"])  # largest at the top
    assert "2022: " in bar["hovertext"][0]
    assert "change: " in bar["hovertext"][0]
    assert spec(fig)["layout"]["title"]["text"] == "Wheat: top 6 by exports, 2023"


def test_ranking_change(wheat: pd.DataFrame) -> None:
    fig = nv.plot.ranking(wheat, time=[2022, 2023], change=True, top_n=8, focus="Egypt")
    bar = spec(fig)["data"][0]
    x, names = bar["x"], bar["y"]
    assert names[0] == "Egypt"  # a focus node outside the top 8 is added (at the bottom)
    ranked = [abs(v) for v in x[1:]]
    assert ranked == sorted(ranked)  # sorted by absolute change, largest at the top
    colors = bar["marker"]["color"]
    expected = [INCREASE_COLOR if v >= 0 else DECREASE_COLOR for v in x]
    assert colors == expected
    drops = {n: v for n, v in zip(names, x, strict=True) if v < 0}
    assert min(drops, key=lambda n: drops[n]) == "Argentina"
    assert drops["Argentina"] == pytest.approx(-9.84e6, rel=0.01)
    assert all(t.endswith("%") or t == "new" for t in bar["text"])
    layout = spec(fig)["layout"]
    assert layout["title"]["text"] == "Wheat: largest changes in exports + imports, 2022 to 2023"
    assert layout["xaxis"]["title"]["text"] == "Change in exports + imports (t)"
    lo, hi = layout["xaxis"]["range"]  # room for the labels outside the longest bars
    assert lo < min(x) * 1.1
    assert hi > max(x) * 1.1


# --- compare --------------------------------------------------------------------------------


def test_compare_brazil(sample: pd.DataFrame) -> None:
    fig = nv.plot.compare(sample, "Brazil", time=[2023, 2024])
    data = spec(fig)["data"]
    bars = {t["name"]: dict(zip(t["y"], t["x"], strict=True)) for t in data if t["type"] == "bar"}
    assert set(bars) == {"Exports", "Imports"}
    assert set(bars["Exports"]) == {"Wheat", "Maize (corn)", "Soya beans"}
    assert bars["Exports"]["Soya beans"] > bars["Imports"]["Soya beans"]
    assert bars["Exports"]["Maize (corn)"] > bars["Imports"]["Maize (corn)"]
    assert bars["Imports"]["Wheat"] > bars["Exports"]["Wheat"]
    ticks = [t for t in data if t["type"] == "scatter"]
    assert sorted(t["showlegend"] for t in ticks) == [False, True]  # one legend entry
    assert ticks[0]["name"] == "2023"
    layout = spec(fig)["layout"]
    assert layout["title"]["text"] == "Brazil: exports and imports by item, 2024"
    assert layout["xaxis"]["title"]["text"] == "Quantity (t)"
    assert layout["scattermode"] == "group"
    hover = next(t for t in data if t.get("name") == "Exports")["hovertext"][0]
    assert "Exports, 2024: " in hover
    assert "Exports, 2023: " in hover
    assert "change: " in hover


def test_compare_options_and_errors(sample: pd.DataFrame) -> None:
    two = spec(nv.plot.compare(sample, "Brazil", category=["Wheat", "Soya beans"], role="in"))
    assert [t["name"] for t in two["data"] if t["type"] == "bar"] == ["Imports"]
    assert set(two["data"][0]["y"]) == {"Wheat", "Soya beans"}
    top = spec(nv.plot.compare(sample, "Brazil", time=2024, top_n=1))
    assert top["data"][0]["y"] == ["Soya beans"]
    assert not [t for t in top["data"] if t["type"] == "scatter"]  # one period, no ticks
    with pytest.raises(ValueError, match="Did you mean: 'Brazil'"):
        nv.plot.compare(sample, "Brasil")
    mixed = pd.concat([sample.head(50), sample.head(50).assign(category="Value", unit="USD")])
    with pytest.raises(ValueError, match="2 units"):
        nv.plot.compare(mixed, mixed["source"].iloc[0])
    with pytest.raises(ValueError, match="not found"):
        nv.plot.compare(sample, "Brazil", category="Rice")
    with pytest.raises(ValueError, match="role"):
        nv.plot.compare(sample, "Brazil", role="sideways")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="flow table"):
        nv.plot.compare(nx.DiGraph(), "Brazil")  # type: ignore[arg-type]


# --- time_series ----------------------------------------------------------------------------


def test_time_series_on_flows(sample: pd.DataFrame) -> None:
    recent = sample[sample["time"] >= 2020]
    fig = nv.plot.time_series(recent)
    data = spec(fig)["data"]
    assert [t["name"] for t in data] == ["Maize (corn)", "Soya beans", "Wheat"]
    assert data[0]["x"] == [2020, 2021, 2022, 2023, 2024]
    layout = spec(fig)["layout"]
    assert layout["yaxis"]["title"]["text"] == "Quantity (t)"
    assert layout["xaxis"]["title"]["text"] == "Year"
    assert layout["title"]["text"] == "Total quantity"
    assert "change: " in data[0]["hovertext"][1]
    assert "change" not in data[0]["hovertext"][0]
    wheat = spec(nv.plot.time_series(recent, category="Wheat"))
    assert [t["name"] for t in wheat["data"]] == ["Wheat"]
    assert wheat["layout"]["title"]["text"] == "Wheat, total quantity"


def test_time_series_focus(sample: pd.DataFrame) -> None:
    fig = nv.plot.time_series(sample, focus="Ukraine", category="Wheat", time=range(2018, 2025))
    data = spec(fig)["data"]
    assert [t["name"] for t in data] == ["Exports", "Imports"]
    assert data[0]["y"][0] > data[1]["y"][0]
    assert spec(fig)["layout"]["title"]["text"] == "Ukraine: exports and imports (Wheat)"
    out = spec(nv.plot.time_series(sample, focus="Ukraine", category="Wheat", role="out"))
    assert [t["name"] for t in out["data"]] == ["Exports"]
    items = spec(nv.plot.time_series(sample, focus="Brazil", role="in"))
    assert [t["name"] for t in items["data"]] == ["Maize (corn)", "Soya beans", "Wheat"]
    pair = spec(nv.plot.time_series(sample, focus=["Brazil", "India"], category="Wheat"))
    assert [t["name"] for t in pair["data"]][:2] == ["Brazil: Exports", "Brazil: Imports"]
    change = spec(nv.plot.time_series(sample, focus="Ukraine", category="Wheat", change=True))
    assert change["layout"]["yaxis"]["title"]["text"] == "Change from previous period (%)"
    assert change["layout"]["title"]["text"].endswith(", change from previous period")
    assert change["data"][0]["y"][0] is None  # no previous period
    assert change["layout"]["shapes"]  # zero line


def test_time_series_errors(sample: pd.DataFrame) -> None:
    table = pd.DataFrame({"a": [1.0, 2.0]}, index=[2020, 2021])
    with pytest.raises(ValueError, match="columns= applies"):
        nv.plot.time_series(sample, ["weight"])
    with pytest.raises(ValueError, match="apply to flows"):
        nv.plot.time_series(table, focus="a")
    with pytest.raises(ValueError, match="Did you mean: 'Ukraine'"):
        nv.plot.time_series(sample, focus="Ukrane", category="Wheat")
    mixed = pd.concat([sample.head(20), sample.head(20).assign(category="V", unit="USD")])
    with pytest.raises(ValueError, match="2 units"):
        nv.plot.time_series(mixed)
    with pytest.raises(ValueError, match="'time' column"):
        nv.plot.time_series(sample.drop(columns=["time"]))
    with pytest.raises(ValueError, match="no 'category' column"):
        nv.plot.time_series(sample.drop(columns=["category"]), category="Wheat")


def test_time_series_on_mapping_of_graphs(flows: pd.DataFrame) -> None:
    graphs = nv.graphs_by(flows, by="time")
    fig = nv.plot.time_series(graphs, focus="A")
    data = spec(fig)["data"]
    assert [t["name"] for t in data] == ["Outgoing", "Incoming"]
    assert data[0]["y"] == [15.0, 20.0]


def test_time_series_on_metric_table() -> None:
    table = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [3.0, 3.0, 0.0]}, index=[2020, 2021, 2022])
    fig = nv.plot.time_series(table, ["b", "a"], y_title="Count", title="t")
    s = spec(fig)
    assert [t["name"] for t in s["data"]] == ["b", "a"]
    assert s["layout"]["yaxis"]["title"]["text"] == "Count"
    assert "change: +100.0%" in s["data"][1]["hovertext"][1]
    assert "change: -100.0%" in s["data"][0]["hovertext"][2]
    faceted = spec(nv.plot.time_series(table, facet=True))
    assert len([k for k in faceted["layout"] if k.startswith("yaxis")]) == 2
    assert faceted["layout"]["height"] == 440
    wide = pd.DataFrame(np.ones((2, 9)), columns=[f"c{i}" for i in range(9)], index=[1, 2])
    with pytest.raises(ValueError, match="too many"):
        nv.plot.time_series(wide)
    assert len(spec(nv.plot.time_series(wide, facet=True))["data"]) == 9
    with pytest.raises(ValueError, match="columns not in the table"):
        nv.plot.time_series(table, ["z"])


# --- degree_distribution --------------------------------------------------------------------


def test_degree_distribution(wheat_graph: nx.DiGraph[Any]) -> None:
    fit = nv.stats.degree_distribution_fit(wheat_graph, "out_strength")
    fig = nv.plot.degree_distribution(fit, labels={"out_strength": "Exports (t)"})
    s = spec(fig)
    assert s["data"][1]["name"].startswith("power law (alpha ")
    assert s["layout"]["xaxis"]["title"]["text"] == "Exports (t)"
    assert s["layout"]["xaxis"]["type"] == "log"
    assert s["layout"]["title"]["text"].startswith("Exports (t) distribution<br><sup>")
    degree = nv.stats.degree_distribution_fit(wheat_graph, "degree")
    assert spec(nv.plot.degree_distribution(degree))["layout"]["xaxis"]["title"]["text"] == (
        "degree"
    )


# --- auto -----------------------------------------------------------------------------------


def test_auto_rules_on_flow_tables(sample: pd.DataFrame, flows: pd.DataFrame) -> None:
    kind, reason = nv.plot.choose_kind(sample, focus="Brazil")
    assert (kind, reason) == ("compare", "one focus node (Brazil) and several categories")
    assert nv.plot.choose_kind(sample, focus="Brazil", category="Wheat")[0] == "time_series"
    assert nv.plot.choose_kind(sample, focus="Brazil", category="Wheat", time=2022)[0] == "ego"
    assert nv.plot.choose_kind(sample, category="Wheat", time=2022) == (
        "flow_map",
        "every node has coordinates",
    )
    assert nv.plot.choose_kind(sample, category="Wheat") == (
        "flow_map",
        "every node has coordinates, animated over time",
    )
    assert nv.plot.choose_kind(flows) == ("network", "general graph, animated over time")
    coords = pd.DataFrame({"lon": [0.0, 1.0, 2.0], "lat": [0.0, 1.0, 2.0]}, index=["A", "B", "C"])
    assert nv.plot.choose_kind(flows, coords=coords)[0] == "flow_map"
    one_way = pd.DataFrame(
        {"source": ["f1", "f2", "f1"], "target": ["m1", "m1", "m2"], "time": 2020, "weight": 1.0}
    )
    assert nv.plot.choose_kind(one_way)[0] == "sankey"
    fig = nv.plot.auto(sample, focus="Brazil", time=2024)
    assert fig.layout.meta["netviz"]["kind"] == "compare"
    note = "Chart: compare (one focus node (Brazil) and several categories)"
    assert note in fig.layout.title.text
    ts = nv.plot.auto(sample, focus="Ukraine", category="Wheat", role="out")
    assert ts.layout.meta["netviz"]["kind"] == "time_series"
    with pytest.raises(TypeError, match=r"plot.flow_map \(chosen because every node"):
        nv.plot.auto(sample, category="Wheat", time=2022, layout="circular")


def test_plots_never_show(
    monkeypatch: pytest.MonkeyPatch, sample: pd.DataFrame, wheat_2022: pd.DataFrame
) -> None:
    def boom(*_: object, **__: object) -> None:
        raise AssertionError("show() must not be called")

    monkeypatch.setattr(go.Figure, "show", boom)
    for kind in ("network", "flow_map", "sankey", "adjacency", "ranking"):
        nv.plot.auto(wheat_2022, kind=kind, top_n=5)
    nv.plot.auto(wheat_2022, kind="ego", focus="Egypt", top_n=5)
    nv.plot.compare(sample, "Brazil")
    nv.plot.time_series(sample, focus="Brazil", category="Wheat")
    nv.plot.auto(wheat_2022)


# --- edge cases -----------------------------------------------------------------------------


def test_edge_cases(flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="no graphs to plot"):
        nv.plot.network({})
    with pytest.raises(ValueError, match="no flows left"):
        nv.plot.network(flows.iloc[:0])
    summed = nv.build_graph(flows, aggregate="sum")
    assert spec(nv.plot.network(summed))["layout"]["title"]["text"] == (
        "x network, 2020 to 2021 (sum)"
    )
    part = pd.Series({"A": 0, "B": 1, "C": 1})
    fig = nv.plot.network(summed, color_by="community", partition=part, layout="community")
    assert [t["name"] for t in node_traces(fig)] == ["Community 0", "Community 1"]
    edgeless: nx.Graph[Any] = nx.empty_graph(3)
    assert [t["name"] for t in node_traces(nv.plot.network(edgeless, color_by="community"))] == [
        "Community 0"
    ]
    # edge colour follows a continuous node colour; categorical size_by gives equal sizes
    g: nx.DiGraph[Any] = nx.DiGraph(summed)
    nx.set_node_attributes(g, {"A": "big", "B": "small", "C": "small"}, "kind")
    nx.set_edge_attributes(g, dict.fromkeys(g.edges, 1.0), "trips")
    fig = nv.plot.network(
        g, color_by="pagerank", edge_color_by="target", size_by="kind", edge_width_by="trips"
    )
    assert "Kind: big" in node_traces(fig)[0]["hovertext"][0]
    assert len(set(node_traces(fig)[0]["marker"]["size"])) == 1
    assert any("trips: 1" in h for h in edge_hover(fig))
    # size_by=None ranks by degree for top_n; a bipartite graph without the attribute
    path: nx.Graph[Any] = nx.path_graph(5)
    top3 = nv.plot.network(path, size_by=None, top_n=3, color_by=None)
    assert len(node_traces(top3)[0]["x"]) == 3
    xs = node_traces(nv.plot.network(path, layout="bipartite", color_by=None))[0]["x"]
    assert len({round(x, 6) for x in xs}) == 2
    # auto on a flow table without a category column, and on a power-law fit
    no_cat = flows.drop(columns=["category"])
    assert nv.plot.choose_kind(no_cat, focus="A") == (
        "time_series",
        "one focus node (A) and several periods",
    )
    fit = nv.stats.degree_distribution_fit(nx.barabasi_albert_graph(200, 2, seed=1), "degree")
    assert nv.plot.auto(fit).layout.meta["netviz"]["kind"] == "degree_distribution"
