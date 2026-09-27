# Getting started

This page walks through the pipeline: a raw table, cleaned into a flow table, built into NetworkX graphs, and drawn as Plotly figures. FAOSTAT trade data is the worked example, and every example runs offline on the bundled sample; the same steps apply to migration or any other flow data (see [Bringing your own data](#bringing-your-own-data)). If you already have a NetworkX graph, skip to [Plots](#plots).

```python
import netviz_tools as nv
from netviz_tools.datasets import faostat
```

## The flow frame

Every function in the library works on a *flow frame*: a pandas DataFrame with one row per directed flow and these six columns (`nv.FLOW_COLUMNS`):

| Column | Meaning |
| --- | --- |
| `source` | origin node (for trade data, the exporting country) |
| `target` | destination node (for trade data, the importing country) |
| `time` | integer period label, for example a year |
| `category` | what flows: a commodity, a product code, a type of migrant |
| `weight` | non-negative amount |
| `unit` | unit of `weight`, for example `"t"`, `"persons"` or `"1000 USD"` |

Extra columns are allowed and kept. `nv.validate_flows(df)` checks the schema and reports every problem at once:

```python
>>> nv.validate_flows(bad)
SchemaError: invalid flow data: column 'target' has 1 missing values;
column 'time' must be integer, not float64; column 'weight' has negative values
```

The sample holds quantity flows in tonnes for Wheat, Maize (corn) and Soya beans, 2010 to 2024. The FAOSTAT loaders return the generic columns (`source` is the exporter, `target` the importer, `time` the year, `category` the FAOSTAT item and `weight` the quantity):

```python
>>> flows = faostat.load_sample(items="Wheat", years=2022)
>>> flows.head(3)
    source                            target  time category  weight unit
0  Algeria                            France  2022    Wheat   28.36    t
1  Algeria      Netherlands (Kingdom of the)  2022    Wheat  280.00    t
2   Angola  Democratic Republic of the Congo  2022    Wheat  565.76    t
>>> len(flows)
1646
```

Display names travel with the data. Every FAOSTAT frame carries `faostat.LABELS` in `flows.attrs["labels"]`, graphs copy it into `G.graph["labels"]`, and the plots use it, so FAOSTAT charts say "Exporter" and "Importer" without you typing them:

```python
>>> flows.attrs["labels"]
{'source': 'Exporter', 'target': 'Importer', 'time': 'Year', 'category': 'Item',
 'weight': 'Quantity', 'node': 'Country', 'out': 'Exports', 'in': 'Imports'}
```

For your own data, set `df.attrs["labels"]` to a dict with any of these keys (for example `{"source": "Origin", "target": "Destination"}`).

Items can be given by FAOSTAT name (case-insensitive), by item code (`15` is wheat), or by the snake_case slug used in netviz_tools 0.x. A misspelt name raises `UnknownItemError` with suggestions: `unknown item 'wheet'. Did you mean: 'Wheat', 'Sheep'?`. `faostat.items("soya")` searches the catalogue of 558 traded items.

## Cleaning a raw table

`nv.clean` turns a messy table into a flow frame and returns a `CleaningReport`. The FAOSTAT bulk file is a good example of messy input: each row is one country's report, so every flow can appear twice (once as the exporter's "Export quantity", once as the importer's "Import quantity"), usually with different numbers. `faostat.raw_sample()` gives the bundled sample in that layout:

```python
>>> raw = faostat.raw_sample(items="Wheat")
>>> raw[["Reporter Countries", "Partner Countries", "Element", "Year", "Value", "Unit"]].head(2)
>>> flows, report = nv.clean(raw)
>>> print(report.summary())
Cleaned 44,738 raw rows into 29,627 flows.
Columns: 'Reporter Countries' -> reporter (guessed), 'Partner Countries' -> partner (guessed),
'Element' -> direction (guessed), 'Year' -> time (guessed), 'Item' -> category (guessed),
'Value' -> weight (guessed), 'Unit' -> unit (guessed)
...
- self-loops: 44,738 -> 44,722 rows
- mirror flows: 44,722 -> 29,627 rows (dropped 15,095 exporter reports of flows both sides reported)
Mirror flows: 15,095 reported by both sides, 6,994 by the source only, 7,538 by the target only;
median disagreement 23.9%; used the target's report where both exist.
```

The two reports of the same flow differ by 23.9% in the median case, which is why the choice matters. `prefer="target"` (the default) keeps the importer's report where there is one, the same rule as `faostat.load(..., reporter="combined")`; `"source"`, `"mean"` and `"max"` are the alternatives.

What `clean` does, in order, and what it records:

| Step | What happens |
| --- | --- |
| columns | each role (`source`, `target`, `time`, `category`, `weight`, `unit`, or `reporter`, `partner`, `direction`) is taken from the column you name, or guessed from the headers; ambiguous or missing columns raise `SchemaError` with the columns found |
| duplicates | rows identical in every raw column are dropped |
| missing values | rows without a source, target or unit are dropped |
| direction | reporter tables become source-to-target flows (`"Export ..."` rows keep the reporter as source, `"Import ..."` rows reverse it) |
| weight, time | text such as `"1,234.5"` becomes a number; unparseable, missing, infinite and negative weights are dropped; `"2021"` or `2021.0` becomes `2021` |
| units | `unit_conversions={"1000 An": ("head", 1000.0)}` rescales and renames |
| names | whitespace is normalized, `aliases={"Turkey": "Türkiye"}` applied, and spellings that differ only in case, accents or punctuation are merged; near matches (typos) are listed in `report.possible_aliases` but never merged |
| drop nodes | `drop_nodes=["World"]` removes aggregates |
| zeros, self-loops | dropped unless `zeros="keep"` or `self_loops=True` |
| repeated flows | rows with the same source, target, time, category and unit are summed (`duplicates="keep"` keeps them) |
| mirror flows | one report per flow, chosen by `prefer` |

Every removed row is in `report.dropped`, with the original columns and a `reason`. `report.to_frame()` gives the steps as a table, and `report.renamed` every spelling change. The output also carries display labels in `flows.attrs["labels"]`, taken from the raw column names (or Exporter and Importer for reporter tables), so the charts use your words.

For your own data, name the columns with the vocabulary words and fill in what the table lacks:

```python
flows, report = nv.clean(
    raw,
    source="origin",
    target="destination",
    weight="persons",
    fill={"time": 2020, "category": "migrants", "unit": "persons"},
    aliases={"Czech Rep.": "Czechia"},
    drop_nodes=["Total"],
)
```

`faostat.load()` and `faostat.load_sample()` return data that are already clean (the same steps run in DuckDB), so FAOSTAT users can skip this step.

## Building graphs

`nv.build_graph` turns one slice of a flow frame (one `time` value, one `category`) into a weighted `networkx.DiGraph`. Edge amounts are stored under `"weight"`, the default weight name of NetworkX algorithms, and the slice is recorded in `G.graph`:

```python
>>> g = nv.build_graph(flows, node_attrs=faostat.countries())
>>> g.number_of_nodes(), g.number_of_edges()
(166, 1646)
>>> {k: v for k, v in g.graph.items() if k != "labels"}
{'aggregate': None, 'unit': 't', 'time': 2022, 'category': 'Wheat'}
```

`time=` and `category=` select a slice from a larger frame before building. Each takes one value or an iterable of values:

```python
>>> sample = faostat.load_sample()  # all three items, 2010 to 2024
>>> g = nv.build_graph(sample, time=2022, category="Wheat", node_attrs=faostat.countries())
>>> g.number_of_edges()
1646
```

A value that matches no flows raises `ValueError` and lists what is there: `no flows with time=2030. Available values (15): 2010, 2011, ...`, or for a misspelt category `no flows with category='wheat'. Did you mean: 'Wheat'? ...`.

`node_attrs` attaches columns of a table indexed by node name as node attributes. `faostat.countries()` gives each FAOSTAT area its UN M49 region, a `continent` column (M49 regions with the Americas split into Northern America, Central America, the Caribbean and South America), and a label point (`lon`, `lat`). The plots use `continent` for colour.

Other options: `directed=False` sums both directions of each pair into an undirected graph, and `self_loops=True` keeps flows from a country to itself (dropped by default).

### Why mixed slices raise

Adding up flows from different periods or different categories is almost never what you want by accident: a graph of "wheat plus maize, 2020 to 2022" has edge weights that mean nothing in particular. So `build_graph` refuses to do it silently:

```python
>>> nv.build_graph(sample, category="Wheat", time=[2021, 2022])
MixedSliceError: flows contain 2 values of 'time' (2021, 2022). Select one slice with
time= or category=, build one graph per slice with graphs_by(), or pass aggregate='sum'
or aggregate='mean'.
```

If you do want a combined graph, say how to combine:

- `aggregate="sum"` totals the flows.
- `aggregate="mean"` divides the total by the number of slices present, so a pair missing from one period counts as zero for that period.

```python
>>> g2 = nv.build_graph(sample, category="Wheat", time=[2021, 2022], aggregate="mean")
>>> {k: v for k, v in g2.graph.items() if k != "labels"}
{'aggregate': 'mean', 'unit': 't', 'time': [2021, 2022], 'category': 'Wheat', 'n_slices': 2}
```

Units are never mixed: if the flows contain more than one `unit`, `build_graph` raises `MixedUnitError` whatever `aggregate` says.

### One graph per slice

`nv.graphs_by` groups a flow frame and builds one graph per group. The default groups by `("time", "category")`:

```python
>>> gs = nv.graphs_by(faostat.load_sample(years=range(2020, 2023)))
>>> list(gs)[:3]
[(2020, 'Maize (corn)'), (2020, 'Soya beans'), (2020, 'Wheat')]
>>> wheat_by_year = nv.graphs_by(sample, by="time", category="Wheat")
>>> sorted(wheat_by_year)[:3]
[2010, 2011, 2012]
```

`time=` and `category=` filter the whole frame before grouping; other keyword arguments go to `build_graph`. Grouping by `"time"` over several categories therefore needs `category=` or `aggregate=`. Every graph carries the frame's `labels`. `nv.graph_to_flows(g)` converts a graph back into a flow table (and its labels back into `df.attrs["labels"]`).

## Partners and category comparisons

Two table-level helpers do not need a graph:

```python
>>> nv.partners(flows, "Ukraine", role="out", top_n=3)
             weight  share
partner
Türkiye  2059037.61  0.246
Spain    1017543.91  0.122
Poland    522878.19  0.063
```

`role="out"` ranks where a node's flows go (for trade, its export destinations), `role="in"` ranks where its inflows come from (its import sources), and `role="both"` puts `out_weight` and `in_weight` per partner side by side, ranked by `total_weight`. `time=` and `category=` filter first. Node names are matched exactly; `nv.partners(flows, "Russia")` raises `UnknownNodeError: unknown node 'Russia'. Did you mean: 'Russian Federation', 'Tunisia', 'Austria'?`.

`nv.compare_categories(flows, metric)` ranks categories by `"total_weight"` (with the unit in the index, so units are never compared directly), `"n_flows"`, or the number of distinct sources, targets or nodes (`"n_sources"`, `"n_targets"`, `"n_nodes"`), optionally per period with `by_time=True`:

```python
>>> nv.compare_categories(sample, "n_sources", by_time=True).iloc[:, -3:]
time          2022  2023  2024
category
Maize (corn)   148   147   145
Wheat          119   124   122
Soya beans     128   128   120
```

## Metrics

`nv.metrics.centrality` computes several node measures at once and returns a DataFrame sorted by the first one:

```python
>>> nv.metrics.centrality(g, ["out_strength", "reverse_pagerank", "pagerank"]).head(3)
                    out_strength  reverse_pagerank  pagerank
node
Australia            24963439.63          0.061037  0.003561
Russian Federation   22444993.82          0.023815  0.001523
France               17663116.08          0.026335  0.028522
```

The measures are `in_strength`, `out_strength`, `strength` (weighted degree, in the graph's unit), `in_degree`, `out_degree`, `degree` (number of partners), `pagerank` (high when large flows arrive from high-ranking nodes: an important destination), `reverse_pagerank` (PageRank on the reversed graph: an important source) and `betweenness` (weighted, with `1 / weight` as the edge length, so large flows count as short routes).

Graph-level metrics live in `nv.temporal`: `graph_summary(g)` for one graph, and `metric_series(graphs)` for a mapping of graphs such as the output of `graphs_by(..., by="time")`:

```python
>>> ts = nv.temporal.metric_series(wheat_by_year, ["total_weight", "out_strength_hhi"])
>>> ts.loc[2021:2023]
      total_weight  out_strength_hhi
2021  1.707297e+08          0.085522
2022  1.665691e+08          0.084138
2023  1.753090e+08          0.099115
```

`out_strength_hhi` is the Herfindahl-Hirschman index of sources' shares of total volume (for trade, exporters' shares): 1 means one supplier, `1/n` means `n` equal suppliers. `centrality_series(graphs, kind, nodes=[...])` tracks one centrality measure per node across the same graphs; a node absent in a period gets `fill_value` (0 by default).

## Communities

```python
>>> part = nv.metrics.communities(g, method="louvain", seed=0)
>>> part.nunique(), round(nv.metrics.modularity(g, part), 3)
(7, 0.504)
```

`communities` returns a Series of community ids indexed by node. Ids are numbered by decreasing community size with a deterministic tie-break, so the same seed gives the same labels. `method="greedy_modularity"` uses the Clauset-Newman-Moore algorithm instead. Directed graphs use the directed form of modularity. `nv.metrics.community_graph(g, part)` collapses each community into one node, with flows between communities as edges and flows inside a community as self-loops. The [community notebook](gallery/community-structure.ipynb) checks how stable the partition is across seeds.

## Power-law fits

A straight line on a log-log plot is weak evidence of a power law. `nv.stats.degree_distribution_fit` follows Clauset, Shalizi and Newman (2009): it chooses the lower bound `xmin` of the tail by Kolmogorov-Smirnov distance, fits the exponent by maximum likelihood, and compares the power law with lognormal and exponential tails using Vuong's likelihood-ratio test. Degrees use discrete likelihoods; strengths use continuous ones.

```python
>>> fit = nv.stats.degree_distribution_fit(g, "out_strength")
>>> print(fit.summary())
out_strength: alpha=1.632, xmin=326460, tail 39 of 119, KS=0.117. vs lognormal:
R=-0.48, p=0.537 (inconclusive). vs exponential: R=17.32, p=0.013 (power_law).
```

Positive `R` favours the power law; the p-value says whether the sign of `R` can be trusted. Here the data reject an exponential tail but cannot tell a power law from a lognormal. `nv.stats.fit_power_law(values, discrete=...)` runs the same fit on any sample. The [power-law notebook](gallery/power-law-vs-lognormal.ipynb) covers interpretation in more detail.

## Plots

Every plotting function takes the data first (a NetworkX graph, a flow table, or a mapping from period to graph) and returns a `plotly.graph_objects.Figure` without showing or saving it. They share one set of argument names; [Concepts](concepts.md) has the full table.

```python
g = nv.build_graph(flows, time=2022, node_attrs=faostat.countries())
fig = nv.plot.network(g, top_n=40, color_by="continent", size_by="out_strength")
fig.update_layout(height=500)
fig.write_html("wheat-2022.html")
```

| Function | What it draws |
| --- | --- |
| `nv.plot.network(data, ...)` | node-link diagram of any graph; layouts `"spring"`, `"kamada_kawai"`, `"circular"`, `"shell"`, `"spectral"`, `"community"`, `"bipartite"`, `"multipartite"`, `"geo"`; arrows on directed edges |
| `nv.plot.ego(data, node, radius=1)` | a node and its neighbours, drawn like `network` with the centre emphasised |
| `nv.plot.flow_map(data, top_n=60)` | the largest flows as lines on a world map; `focus=` follows one country |
| `nv.plot.sankey(data, top_n=10)` | flows from the largest sources (left) to the largest targets (right), the rest grouped as "Other" |
| `nv.plot.adjacency(data, sort_by="community")` | the weighted adjacency matrix, with communities as blocks on the diagonal |
| `nv.plot.ranking(data, size_by=..., change=False)` | top nodes by a value, with the previous period marked; `change=True` ranks by the change |
| `nv.plot.compare(flows, node, role="both")` | one node's outgoing and incoming totals per category |
| `nv.plot.time_series(data, focus=..., category=...)` | totals per period, or one node's flows per period, with the change from the previous period in the hover text |
| `nv.plot.degree_distribution(fit)` | empirical CCDF with the fitted power-law, lognormal and exponential tails |
| `nv.plot.auto(data, kind="auto")` | any of the above by name, or a chart chosen from the data |

### Colour, size and labels

`color_by` and `size_by` take a node attribute (`"continent"`), a metric (`"degree"`, `"strength"`, `"out_strength"`, `"pagerank"`, `"betweenness"`, `"community"`, ...), a mapping or Series of your own values, `"auto"`, or `None`. Text values get a legend with up to eight colours (the rest are "Other"); numbers get a colour bar. Sizes and colours are log-scaled when values span more than two orders of magnitude. `edge_width_by` and `edge_color_by` do the same for edges, and `edge_color_by="source"` colours each flow like its exporter. `show_labels` decides which nodes get a text label and `label_by` which attribute supplies the text. `focus="Ukraine"` outlines a node, labels it, darkens its edges and fades the rest.

### Time

A flow table spanning several years draws every year. Network diagrams and flow maps become animations with a play button and a year slider, with node positions and scales fixed so that movement means change:

```python
all_years = nv.clean(faostat.raw_sample(items="Soya beans"))[0]
nv.plot.flow_map(all_years, top_n=40)  # 2010 to 2024, one frame per year
nv.plot.flow_map(all_years, time=range(2019, 2025), focus="Brazil")  # Brazil's flows only
```

Bar charts compare the last period with the one before:

```python
wheat = nv.clean(faostat.raw_sample(items="Wheat"))[0]
nv.plot.ranking(wheat, time=[2022, 2023], size_by="out_strength", change=True)
nv.plot.time_series(wheat, focus="Ukraine")  # Ukraine's wheat exports and imports per year
```

### Choosing a chart automatically

`nv.plot.auto(g)` picks a chart from the data: a flow map when every node has coordinates, a Sankey when every node only sends or only receives, an ego view or a comparison when you pass one `focus` node, a network otherwise. The rules are listed on the [Concepts](concepts.md#plotauto-name-the-chart-or-let-it-choose) page. The figure records what was chosen and why, in `fig.layout.meta["netviz"]` and under the title.

## Bringing your own data

For tables that need cleaning, use `nv.clean` as shown [above](#cleaning-a-raw-table). For a table that is already tidy, `nv.to_flowframe` is the lighter option: map your column names to schema names, and give constants for the columns you do not have. Migration between regions, for example:

```python
import pandas as pd

raw = pd.DataFrame(
    {
        "origin": ["A", "A", "B"],
        "dest": ["B", "C", "C"],
        "people": [120, 40, 75],  # toy numbers
    }
)
ff = nv.to_flowframe(
    raw,
    {"origin": "source", "dest": "target", "people": "weight"},
    time=2020,
    category="migrants",
    unit="persons",
)
ff.attrs["labels"] = {"source": "Origin", "target": "Destination", "node": "Region"}
g = nv.build_graph(ff)
```

The result has the six schema columns first, then any other columns of `raw`, and a whole-number `time` is cast to `int64`. `attrs` set on `raw` are kept. From here every metric and plot works the same way. Continent colouring needs a `continent` node attribute (pass `node_attrs=`), or use `color_by=None`; flow maps need `coords=` unless the node names are FAOSTAT country names.

## The full FAOSTAT store

The sample covers three items. For the other items, and for trade values, build the local store once:

```python
from netviz_tools.datasets import faostat

faostat.build_store()  # one-time: downloads about 420 MB
flows = faostat.load("Coffee, green", years=range(2015, 2025))
```

`build_store()` downloads the FAOSTAT bulk zip (about 420 MB), checks its SHA-256 against the value pinned in this release, extracts the CSV (about 9 GB of temporary disk space), and converts it with DuckDB into a Parquet store partitioned by item. It takes about a minute on a laptop, and the finished store is about 400 MB. Later calls return immediately unless you pass `force=True`. If you already have the zip, pass `zip_path=`; it is checked against the same hash.

**Cache directory.** The zip and the store live in `faostat.default_cache_dir()`: the operating system's user cache directory (for example `~/.cache/netviz_tools` on Linux), unless the environment variable `NETVIZ_TOOLS_CACHE` is set. Every function that touches the store also takes `cache_dir=`. `faostat.store_path()` shows where the store is, and `faostat.store_info()` returns its `manifest.json`. Calling `load()` before the store exists raises `StoreNotFoundError` with instructions.

**Reporter perspective.** Each trade flow can be reported twice, by the exporting country and by the importing country, and the two reports often differ. `load()` and `load_sample()` take `reporter=` (these FAOSTAT-specific options keep their trade names):

- `"importer"` (default): flows as reported by the importing country. More countries report imports than exports: for wheat in 2021 the sample has 155 reporting importers against 99 reporting exporters.
- `"exporter"`: flows as reported by the exporting country.
- `"combined"`: the importer's report where one exists, otherwise the exporter's, decided per exporter, importer, item and year.

The choice matters. The Russian Federation reports no wheat exports after 2021, so in exporter-reported data it has no wheat exports at all in 2022. Importer-reported data still show about 22.4 million tonnes that year. See the [wheat notebook](gallery/wheat-2022-shock.ipynb).

**Trade value.** `load(..., measure="value")` returns trade value in `1000 USD` in the `weight` column instead of physical quantity, and sets `labels["weight"]` to `"Value"`. Quantities are in `t`, `head` or `number`. Values and quantities are never mixed in one frame, but the live-animal items (cattle, sheep, chickens and so on) have quantity rows in both `head` and `t`; filter on `unit` before building a graph, or `build_graph` raises `MixedUnitError`. The bundled sample contains quantities only.

**Details.** `load(..., details=True)` adds `item_code`, `source_code` and `target_code` (FAOSTAT area codes), `reported_by` (`"exporter"` or `"importer"`) and the FAOSTAT `flag`. `self_loops=True` keeps flows from a country to itself (re-imports), which the store keeps but `load` drops by default.

See [Data and licences](data.md) for the transformation rules, the pinned release, and how to cite FAOSTAT.
