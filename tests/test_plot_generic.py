"""Plots on plain NetworkX graphs (no flow table involved)."""

from __future__ import annotations

import json
import warnings
from typing import Any

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import pytest

import netviz_tools as nv
from netviz_tools.errors import UnknownMetricError
from netviz_tools.plot import PALETTE, SEQUENTIAL


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


def edge_hover(fig: go.Figure) -> dict[str, Any]:
    return next(t for t in spec(fig)["data"] if t.get("name") == "edges")


def n_nodes(fig: go.Figure) -> int:
    return sum(len(t.get("x") or t.get("lon") or []) for t in node_traces(fig))


@pytest.fixture
def karate() -> nx.Graph[Any]:
    g: nx.Graph[Any] = nx.karate_club_graph()
    return g


def node_xs(s: dict[str, Any]) -> list[float]:
    return [
        x
        for t in s["data"]
        if str(t.get("legendgroup", "")).startswith("node")
        for x in t["x"]
        if x is not None  # legend-only entries have no points
    ]


def test_karate_categorical_attribute(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.network(karate, color_by="club")
    traces = node_traces(fig)
    assert [t["name"] for t in traces] == ["Mr. Hi", "Officer"]
    assert traces[0]["marker"]["color"] == PALETTE[0]
    assert n_nodes(fig) == 34
    assert spec(fig)["layout"]["legend"]["title"]["text"] == "Club"
    assert spec(fig)["layout"]["title"]["text"] == "Zachary's Karate Club: network"
    # 34 nodes is above the "label everything" threshold of 30: the 12 largest are labelled
    assert len(labels_shown(fig)) == 12
    hover = traces[0]["hovertext"][0]
    assert "Club: Mr. Hi" in hover
    assert "Total weight" in hover


def test_auto_defaults_are_generic(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.network(karate)
    names = [t["name"] for t in node_traces(fig)]
    assert names[0] == "Community 0"
    caption = spec(fig)["layout"]["annotations"][0]["text"]
    assert caption.startswith("Node size: Total weight; edge width: weight")
    unweighted = nx.florentine_families_graph()
    fig2 = nv.plot.network(unweighted, color_by=None)
    caption2 = spec(fig2)["layout"]["annotations"][0]["text"]
    assert caption2 == "Node size: Degree."
    assert len(node_traces(fig2)) == 1


def test_continuous_colour_has_colorbar(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.network(karate, color_by="pagerank", size_by="betweenness")
    main = node_traces(fig)[0]
    assert main["marker"]["showscale"] is True
    assert main["marker"]["colorbar"]["title"]["text"] == "PageRank"
    assert main["marker"]["colorscale"][0][1] == SEQUENTIAL[0][1]
    ticks = main["marker"]["colorbar"]["ticktext"]
    assert 3 <= len(ticks) <= 7
    assert all(len(t) <= 5 for t in ticks)  # round values such as "0.04"


def test_numeric_attribute_mapping_and_missing() -> None:
    g: nx.Graph[Any] = nx.path_graph(5)
    nx.set_node_attributes(g, {0: 1.0, 1: 5.0, 2: 2.0}, "score")
    fig = nv.plot.network(g, color_by="score", size_by={0: 1, 1: 2, 2: 3, 3: 4, 4: 5})
    names = [t["name"] for t in node_traces(fig)]
    assert names[-1] == "No value"
    series = pd.Series({n: "odd" if n % 2 else "even" for n in g}, name="parity")
    fig2 = nv.plot.network(g, color_by=series)
    assert [t["name"] for t in node_traces(fig2)] == ["even", "odd"]
    assert spec(fig2)["layout"]["legend"]["title"]["text"] == "parity"


def test_many_categories_fold_into_other() -> None:
    g: nx.Graph[Any] = nx.path_graph(12)
    nx.set_node_attributes(g, {n: f"k{n:02d}" for n in g}, "kind")
    names = [t["name"] for t in node_traces(nv.plot.network(g, color_by="kind"))]
    assert len(names) == len(PALETTE) + 1
    assert names[-1] == "Other"


def test_booleans_and_continents() -> None:
    g: nx.Graph[Any] = nx.path_graph(4)
    nx.set_node_attributes(g, {0: True, 1: False, 2: True, 3: False}, "flag")
    assert [t["name"] for t in node_traces(nv.plot.network(g, color_by="flag"))] == [
        "False",
        "True",
    ]
    nx.set_node_attributes(g, {0: "Europe", 1: "Asia", 2: "Asia", 3: "Europe"}, "continent")
    names = [t["name"] for t in node_traces(nv.plot.network(g, color_by="continent"))]
    assert names == ["Asia", "Europe"]


def test_unknown_names_raise(karate: nx.Graph[Any]) -> None:
    with pytest.raises(UnknownMetricError, match="neither a node attribute nor a metric"):
        nv.plot.network(karate, color_by="nope")
    with pytest.raises(UnknownMetricError, match="not an edge attribute"):
        nv.plot.network(karate, edge_width_by="nope")
    with pytest.raises(ValueError, match="focus node 99"):
        nv.plot.network(karate, focus=99)
    with pytest.raises(ValueError, match="unknown layout"):
        nv.plot.network(karate, layout="random")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="pos= has no position"):
        nv.plot.network(karate, pos={0: (0.0, 0.0)})
    with pytest.raises(TypeError, match="expected a networkx graph"):
        nv.plot.network([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="single graph has one slice"):
        nv.plot.network(karate, time=2020)
    with pytest.raises(ValueError, match="node_attrs"):
        nv.plot.network(karate, node_attrs=pd.DataFrame())


@pytest.mark.parametrize(
    "layout", ["spring", "kamada_kawai", "circular", "shell", "spectral", "community"]
)
def test_layouts_are_seeded_and_complete(layout: nv.plot.Layout) -> None:
    g = nx.les_miserables_graph()
    a = spec(nv.plot.network(g, layout=layout, color_by="community"))
    b = spec(nv.plot.network(g, layout=layout, color_by="community"))
    assert node_xs(a) == node_xs(b)
    assert len(node_xs(a)) == g.number_of_nodes()
    assert all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in node_xs(a))


def test_bipartite_and_multipartite_layouts() -> None:
    b = nx.complete_bipartite_graph(3, 4)
    s = spec(nv.plot.network(b, layout="bipartite", color_by=None))
    xs = sorted({round(x, 6) for t in s["data"] if t.get("legendgroup") == "nodes" for x in t["x"]})
    assert len(xs) == 2
    with pytest.raises(ValueError, match="bipartite"):
        nv.plot.network(nx.cycle_graph(3), layout="bipartite")
    g: nx.Graph[Any] = nx.path_graph(6)
    nx.set_node_attributes(g, {n: n // 2 for n in g}, "layer")
    nv.plot.network(g, layout="multipartite", layout_options={"subset_key": "layer"})
    with pytest.raises(ValueError, match="subset"):
        nv.plot.network(g, layout="multipartite")


def test_kamada_kawai_warns_on_large_graphs(monkeypatch: pytest.MonkeyPatch) -> None:
    # The real layout takes most of a minute on 501 nodes; only the warning matters here.
    monkeypatch.setattr(nx, "kamada_kawai_layout", lambda g, **_: nx.circular_layout(g))
    with pytest.warns(UserWarning, match="kamada_kawai"):
        nv.plot.network(nx.path_graph(501), layout="kamada_kawai")


def test_directed_edges_get_arrows_and_self_loops() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_weighted_edges_from([("a", "b", 5.0), ("b", "a", 1.0), ("b", "c", 2.0), ("c", "c", 1.0)])
    fig = nv.plot.network(g, color_by=None)
    arrows = edge_hover(fig)
    assert arrows["marker"]["symbol"] == "triangle-up"
    assert len(arrows["marker"]["angle"]) == 3  # self-loop has no arrow
    assert any("a → b" in h for h in arrows["hovertext"])
    loops = next(t for t in spec(fig)["data"] if t.get("name") == "self-loops")
    assert loops["x"]
    caption = spec(fig)["layout"]["annotations"][0]["text"]
    assert "arrows point from source to target" in caption
    hover = " ".join(t["hovertext"][0] for t in node_traces(fig))
    assert "Degree (in + out)" in hover or "Outgoing + incoming" in hover


def test_undirected_edges_have_invisible_hover_markers(karate: nx.Graph[Any]) -> None:
    e = edge_hover(nv.plot.network(karate))
    assert e["marker"]["opacity"] == 0
    assert "\u2013" in e["hovertext"][0]  # en dash between the ends


def test_multigraph_is_collapsed() -> None:
    m: nx.MultiDiGraph[Any] = nx.MultiDiGraph([(1, 2), (1, 2), (2, 3)])
    fig = nv.plot.network(m, color_by=None)
    assert any("Weight: 2" in h for h in edge_hover(fig)["hovertext"])


def test_edge_colour_options() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_edge("a", "b", weight=1.0, mode="air", co2=1.0)
    g.add_edge("b", "c", weight=3.0, mode="sea", co2=50.0)
    g.add_edge("c", "a", weight=2.0, mode="sea")
    s = spec(nv.plot.network(g, edge_color_by="mode", color_by=None))
    legend = {t["name"] for t in s["data"] if t.get("showlegend") and t.get("mode") == "lines"}
    assert legend == {"air", "sea"}
    s2 = spec(nv.plot.network(g, edge_color_by="co2", color_by=None))
    names = {t["name"] for t in s2["data"] if t.get("showlegend") and t.get("mode") == "lines"}
    assert "No value" in names
    assert any(" to " in n for n in names)
    s3 = spec(nv.plot.network(g, edge_color_by="source", color_by="community"))
    lines = [t["line"]["color"] for t in s3["data"] if t.get("mode") == "lines"]
    assert any(c.startswith("rgba(42, 120, 214") for c in lines)
    s4 = spec(nv.plot.network(g, edge_color_by={("a", "b"): "x"}, edge_width_by={("a", "b"): 3.0}))
    assert s4["data"]


def test_top_n_focus_labels_and_quantile() -> None:
    g = nx.les_miserables_graph()
    fig = nv.plot.network(g, top_n=10, focus="Gervais", show_labels=3, min_weight_quantile=0.5)
    assert n_nodes(fig) == 11
    labelled = labels_shown(fig)
    assert "Gervais" in labelled
    assert len(labelled) == 4
    widths = [
        w
        for t in node_traces(fig)
        for w, name in zip(t["marker"]["line"]["width"], t["hovertext"], strict=True)
        if "Gervais" in name
    ]
    assert widths == [2.5]
    fig2 = nv.plot.network(g, show_labels=["Valjean"], label_by=None)
    assert labels_shown(fig2) == ["Valjean"]
    fig3 = nv.plot.network(g, show_labels=False)
    assert not labels_shown(fig3)
    nx.set_node_attributes(g, {n: n.upper() for n in g}, "caps")
    fig4 = nv.plot.network(g, show_labels=True, label_by="caps", top_n=5)
    assert "VALJEAN" in labels_shown(fig4)


def test_pos_passthrough() -> None:
    g: nx.Graph[Any] = nx.path_graph(3)
    pos = {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 5.0)}
    fig = nv.plot.network(g, pos=pos, color_by=None, size_by="degree")
    t = node_traces(fig)[0]
    assert sorted(zip(t["x"], t["y"], strict=True)) == [(0.0, 0.0), (1.0, 0.0), (2.0, 5.0)]


def test_geo_layout_uses_coordinates() -> None:
    g: nx.Graph[Any] = nx.Graph()
    g.add_edge("p", "q", weight=1.0)
    g.add_edge("q", "r", weight=2.0)
    nx.set_node_attributes(g, {"p": 1.0, "q": 2.0, "r": 3.0}, "lon")
    nx.set_node_attributes(g, {"p": 4.0, "q": 5.0, "r": 6.0}, "lat")
    s = spec(nv.plot.network(g, layout="geo", color_by=None))
    assert all(t["type"] == "scattergeo" for t in s["data"])
    assert s["layout"]["geo"]["projection"]["type"] == "natural earth"


def test_geo_layout_warns_for_missing_coordinates() -> None:
    g: nx.Graph[Any] = nx.Graph([("Ukraine", "Egypt"), ("Egypt", "Atlantis")])
    with pytest.warns(UserWarning, match="no coordinates for 1 node"):
        s = spec(nv.plot.network(g, layout="geo", color_by=None))
    assert sum(len(t["lon"]) for t in s["data"] if t.get("legendgroup") == "nodes") == 2


def test_large_graph_is_capped_with_warning() -> None:
    g = nx.barabasi_albert_graph(300, 1, seed=1)
    with pytest.warns(UserWarning, match="drawing the 100 largest"):
        fig = nv.plot.network(g, max_nodes=100, layout="spectral")
    assert n_nodes(fig) == 100
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert n_nodes(nv.plot.network(g, top_n=50, layout="spectral")) == 50


def test_edgeless_and_disconnected_graphs() -> None:
    assert n_nodes(nv.plot.network(nx.empty_graph(4))) == 4
    g = nx.disjoint_union(nx.cycle_graph(4), nx.path_graph(3))
    assert n_nodes(nv.plot.network(g)) == 7
    assert n_nodes(nv.plot.network(nx.Graph())) == 0


def test_community_layout_single_group() -> None:
    g = nx.complete_graph(4)
    pos = nv.plot.community_layout(g, pd.Series(dict.fromkeys(g, 0)))
    assert set(pos) == set(g)


def test_ego(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.ego(karate, 0)
    assert n_nodes(fig) == karate.degree(0) + 1
    assert spec(fig)["layout"]["title"]["text"] == "0: neighbours within 1 step"
    # both centres are already among the 5 largest nodes, so nothing is added
    fig2 = nv.plot.ego(karate, [0, 33], radius=2, top_n=5)
    assert n_nodes(fig2) == 5
    d: nx.DiGraph[Any] = nx.DiGraph([(1, 2), (3, 1)])
    assert n_nodes(nv.plot.ego(d, 1, undirected=False)) == 2
    assert "2 steps" in spec(nv.plot.ego(karate, 0, radius=2))["layout"]["title"]["text"]


def test_adjacency(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.adjacency(karate, top_n=None)
    s = spec(fig)
    heat = s["data"][0]
    assert heat["type"] == "heatmap"
    assert len(heat["z"]) == 34
    assert s["layout"]["shapes"]  # community boundaries
    z = heat["z"]
    assert z[0][0] is None  # no self-loop
    assert all(z[i][j] == z[j][i] for i in range(34) for j in range(34))
    by_degree = spec(nv.plot.adjacency(karate, sort_by="degree", top_n=10, focus=16))
    assert len(by_degree["data"][0]["z"]) == 11
    assert "<b>16</b>" in by_degree["layout"]["xaxis"]["ticktext"]
    assert not by_degree["layout"].get("shapes")
    plain = spec(nv.plot.adjacency(karate, sort_by=None, top_n=5))
    assert len(plain["data"][0]["x"]) == 5
    grouped = spec(nv.plot.adjacency(karate, sort_by="club"))
    assert len(grouped["layout"]["shapes"]) == 2


def test_adjacency_directed_log_scale() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 10_000.0), ("c", "a", 50.0)])
    s = spec(nv.plot.adjacency(g, sort_by=None))
    heat = s["data"][0]
    assert heat["colorbar"]["ticktext"]
    assert s["layout"]["xaxis"]["title"]["text"] == "Target"


def test_ranking_on_a_graph(karate: nx.Graph[Any]) -> None:
    fig = nv.plot.ranking(karate, size_by="degree", top_n=5, focus=16)
    bar = spec(fig)["data"][0]
    assert bar["type"] == "bar"
    assert bar["y"][-1] == "33"  # largest at the top
    assert len(bar["y"]) == 6
    assert bar["marker"]["color"][0] == PALETTE[0]  # the focus node
    with pytest.raises(ValueError, match="numeric size_by"):
        nv.plot.ranking(karate, size_by="club")
    with pytest.raises(ValueError, match="two periods"):
        nv.plot.ranking(karate, change=True)


def test_sankey_on_a_graph() -> None:
    g: nx.DiGraph[Any] = nx.DiGraph()
    g.add_weighted_edges_from(
        [(f"s{i}", f"t{j}", float(i + j + 1)) for i in range(4) for j in range(4)]
    )
    fig = nv.plot.sankey(g, top_n=2, color_by=None)
    trace = spec(fig)["data"][0]
    assert trace["node"]["label"][2] == "Other sources"
    assert trace["node"]["label"][-1] == "Other targets"
    total = sum(float(w) for _, _, w in g.edges(data="weight", default=0.0))
    assert sum(trace["link"]["value"]) == pytest.approx(total)
    focused = spec(nv.plot.sankey(g, focus="s0", color_by="community"))
    assert set(focused["data"][0]["node"]["label"]) >= {"s0"}
    with pytest.raises(ValueError, match="directed"):
        nv.plot.sankey(nx.Graph([(1, 2)]))
    nx.set_node_attributes(g, {n: float(i) for i, n in enumerate(g)}, "x")
    with pytest.raises(ValueError, match="categorical"):
        nv.plot.sankey(g, color_by="x")


def test_mapping_of_graphs_animates() -> None:
    g1: nx.DiGraph[Any] = nx.DiGraph()
    g1.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 2.0)])
    g2: nx.DiGraph[Any] = nx.DiGraph()
    g2.add_weighted_edges_from([("a", "b", 5.0), ("c", "d", 1.0)])
    fig = nv.plot.network({2020: g1, 2021: g2}, color_by=None)
    assert [f.name for f in fig.frames] == ["2020", "2021"]
    assert len({len(f.data) for f in fig.frames}) == 1  # fixed trace structure
    assert len(fig.frames[0].data) == len(fig.data)
    s = spec(fig)
    assert s["layout"]["sliders"][0]["currentvalue"]["prefix"] == "Time: "
    assert s["layout"]["updatemenus"][0]["buttons"][0]["label"] == "Play"
    assert s["layout"]["title"]["text"] == "Network, 2020 to 2021"
    # node d is absent in 2020: size 0 and a note in the hover
    first_nodes = next(t for t in fig.frames[0].data if t.legendgroup == "nodes")
    i = list(first_nodes.hovertext).index(next(h for h in first_nodes.hovertext if "<b>d</b>" in h))
    assert first_nodes.marker.size[i] == 0
    assert "not present" in first_nodes.hovertext[i]
    only = nv.plot.network({2020: g1, 2021: g2}, time=2021)
    assert not only.frames
    with pytest.raises(ValueError, match="not found"):
        nv.plot.network({2020: g1}, time=1999)
    with pytest.raises(ValueError, match="category= needs a flow table"):
        nv.plot.network({2020: g1}, category="x")
    with pytest.raises(ValueError, match="one period"):
        nv.plot.sankey({2020: g1, 2021: g2})
    ranked = spec(nv.plot.ranking({2020: g1, 2021: g2}, size_by="out_strength"))
    assert ranked["data"][1]["name"] == "2020"
    change = spec(nv.plot.ranking({2020: g1, 2021: g2}, size_by="out_strength", change=True))
    assert "new" in change["data"][0]["text"] or "+400.0%" in change["data"][0]["text"]


def test_auto_rules(karate: nx.Graph[Any]) -> None:
    assert nv.plot.choose_kind(karate)[0] == "network"
    assert nv.plot.choose_kind(karate, focus=0)[0] == "ego"
    d: nx.DiGraph[Any] = nx.DiGraph([("f1", "m1"), ("f2", "m1")])
    assert nv.plot.choose_kind(d)[0] == "sankey"
    geo: nx.Graph[Any] = nx.Graph([("Ukraine", "Egypt")])
    assert nv.plot.choose_kind(geo)[0] == "flow_map"
    assert nv.plot.choose_kind({1: karate, 2: karate})[0] == "network"
    assert nv.plot.choose_kind({1: karate, 2: karate}, focus=0)[0] == "time_series"
    table = pd.DataFrame({"a": [1, 2]}, index=[2020, 2021])
    assert nv.plot.choose_kind(table)[0] == "time_series"
    fig = nv.plot.auto(karate)
    assert fig.layout.meta["netviz"] == {"kind": "network", "reason": "general graph"}
    assert "Chart: network (general graph)" in fig.layout.title.text
    named = nv.plot.auto(karate, kind="adjacency", top_n=5)
    assert named.layout.meta["netviz"]["reason"] == "requested"
    assert "Chart:" not in named.layout.title.text
    with pytest.warns(UserWarning, match="no coordinates for 3 node"):
        assert nv.plot.auto(d, kind="map").layout.meta["netviz"]["kind"] == "flow_map"
    with pytest.raises(TypeError, match=r"plot.sankey \(chosen because"):
        nv.plot.auto(d, size_by="degree")
    with pytest.raises(ValueError, match="unknown kind"):
        nv.plot.auto(karate, kind="pie")  # type: ignore[arg-type]
    untitled = nv.plot.auto(table)
    assert untitled.layout.title.text.startswith("<sup>Chart: time series")


def test_plots_never_show(monkeypatch: pytest.MonkeyPatch, karate: nx.Graph[Any]) -> None:
    def boom(*_: object, **__: object) -> None:
        raise AssertionError("show() must not be called")

    monkeypatch.setattr(go.Figure, "show", boom)
    for kind in ("network", "adjacency", "ranking"):
        nv.plot.auto(karate, kind=kind)
    nv.plot.ego(karate, 0)


def test_mapping_of_unweighted_graphs() -> None:
    g1: nx.DiGraph[Any] = nx.DiGraph([(1, 2), (2, 3)])
    g2: nx.DiGraph[Any] = nx.DiGraph([(1, 2), (3, 4)])
    fig = nv.plot.network({2020: g1, 2021: g2})
    assert spec(fig)["layout"]["annotations"][0]["text"].startswith("Node size: Degree (in + out)")
    weighted: nx.DiGraph[Any] = nx.DiGraph()
    weighted.add_edge(1, 2, weight=3.0)
    mixed = nv.plot.network({2020: g1, 2021: weighted})
    assert "edge width: weight" in spec(mixed)["layout"]["annotations"][0]["text"]
    with pytest.raises(UnknownMetricError, match="not an edge attribute"):
        nv.plot.network({2020: g1, 2021: g2}, edge_width_by="weight")


def test_tuple_nodes() -> None:
    grid = nx.grid_2d_graph(4, 4)
    fig = nv.plot.network(grid, color_by="pagerank")
    assert n_nodes(fig) == 16
    assert "(0, 0)" in " ".join(labels_shown(fig))
    assert n_nodes(nv.plot.network(grid, color_by={(0, 0): "corner"}, layout="community")) == 16
    assert len(spec(nv.plot.adjacency(grid, top_n=None))["data"][0]["z"]) == 16
    assert spec(nv.plot.ranking(grid, top_n=3))["data"][0]["y"][-1].startswith("(")
    assert n_nodes(nv.plot.ego(grid, (0, 0))) == 3


@pytest.mark.parametrize("layout", ["spectral", "kamada_kawai"])
def test_disconnected_layouts_do_not_collapse(layout: nv.plot.Layout) -> None:
    parts = [nx.cycle_graph(8), nx.path_graph(5), nx.star_graph(4), nx.empty_graph(3)]
    g = nx.disjoint_union_all(parts)
    xs = node_xs(spec(nv.plot.network(g, layout=layout, color_by=None, size_by=None)))
    ys = [
        y
        for t in spec(nv.plot.network(g, layout=layout, color_by=None, size_by=None))["data"]
        if t.get("legendgroup") == "nodes"
        for y in t["y"]
    ]
    points = {(round(x, 3), round(y, 3)) for x, y in zip(xs, ys, strict=True)}
    assert len(points) == g.number_of_nodes()


def test_layout_options_override_defaults() -> None:
    g = nx.les_miserables_graph()
    a = node_xs(spec(nv.plot.network(g, layout_options={"weight": "weight"})))
    b = node_xs(spec(nv.plot.network(g)))
    assert a != b
    c = node_xs(spec(nv.plot.network(g, layout_options={"seed": 7})))
    assert c != b


def test_layout_uses_drawn_edges() -> None:
    from netviz_tools.plot._network import _layout_graph

    g: nx.Graph[Any] = nx.Graph()
    g.add_weighted_edges_from([("a", "b", 1.0), ("b", "c", 10.0), ("c", "d", 20.0), ("d", "a", 2)])
    kept = _layout_graph(g, "weight", 0.5)
    assert set(kept) == set(g)
    assert sorted(kept.edges) == [("b", "c"), ("c", "d")]
    assert _layout_graph(g, "weight", None) is g  # sparse: every edge
    assert _layout_graph(g, None, 0.5) is g
    dense = nx.complete_graph(9)
    nx.set_edge_attributes(dense, {(u, v): float(u * v) for u, v in dense.edges}, "weight")
    backbone = _layout_graph(dense, "weight", None)
    assert backbone.number_of_nodes() == 9
    assert backbone.number_of_edges() < dense.number_of_edges()
    assert all(backbone.degree(n) >= 3 for n in backbone)


def test_animation_hides_absent_labels_and_keeps_legend() -> None:
    g1: nx.DiGraph[Any] = nx.DiGraph([("a", "b"), ("b", "c")])
    g2: nx.DiGraph[Any] = nx.DiGraph([("a", "b"), ("x", "y")])
    for g in (g1, g2):
        nx.set_node_attributes(g, {n: "late" if n in "xy" else "early" for n in g}, "kind")
    fig = nv.plot.network({2020: g1, 2021: g2}, color_by="kind", show_labels=True)
    first = {t.name: t for t in fig.frames[0].data}
    labels = dict(zip(first["labels"].x, first["labels"].text, strict=True))
    assert sorted(t for t in labels.values() if t) == ["a", "b", "c"]  # x and y are absent
    legend = [t for t in fig.frames[0].data if t.showlegend is not False and t.mode == "markers"]
    assert sorted(t.name for t in legend if t.x == (None,)) == ["early", "late"]
    s = spec(fig)
    button, slider = s["layout"]["updatemenus"][0], s["layout"]["sliders"][0]
    assert button["xanchor"] == "right"
    assert button["x"] <= slider["x"]  # buttons end where the slider starts
