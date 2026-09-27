# Getting started

This page walks through the library from a flow table to a figure. Every example runs offline on the bundled FAOSTAT sample.

```python
import netviz_tools as nv
from netviz_tools.datasets import faostat
```

## The flow frame

Every function in the library works on a *flow frame*: a pandas DataFrame with one row per directed flow and these six columns (`nv.FLOW_COLUMNS`):

| Column | Meaning |
| --- | --- |
| `exporter` | origin node (for trade data, the exporting country) |
| `importer` | destination node (the importing country) |
| `year` | integer period label |
| `item` | what flows: a commodity, a product code, a type of migrant |
| `quantity` | non-negative amount |
| `unit` | unit of `quantity`, for example `"t"` or `"1000 USD"` |

Extra columns are allowed and kept. `nv.validate_flows(df)` checks the schema and reports every problem at once:

```python
>>> nv.validate_flows(bad)
SchemaError: invalid flow data: column 'importer' has 1 missing values;
column 'year' must be integer, not float64; column 'quantity' has negative values
```

The sample holds quantity flows in tonnes for Wheat, Maize (corn) and Soya beans, 2010 to 2024:

```python
>>> flows = faostat.load_sample(items="Wheat", years=2022)
>>> flows.head(3)
  exporter                          importer  year   item  quantity unit
0  Algeria                            France  2022  Wheat     28.36    t
1  Algeria      Netherlands (Kingdom of the)  2022  Wheat    280.00    t
2   Angola  Democratic Republic of the Congo  2022  Wheat    565.76    t
>>> len(flows)
1646
```

Items can be given by FAOSTAT name (case-insensitive), by item code (`15` is wheat), or by the snake_case slug used in netviz_tools 0.x. A misspelt name raises `UnknownItemError` with suggestions: `unknown item 'wheet'. Did you mean: 'Wheat', 'Sheep'?`. `faostat.items("soya")` searches the catalogue of 558 traded items.

## Building graphs

`nv.build_graph` turns one slice of a flow frame (one year, one item) into a weighted `networkx.DiGraph`. Edge amounts are stored under `"weight"`, the default weight name of NetworkX algorithms, and the slice is recorded in `G.graph`:

```python
>>> g = nv.build_graph(flows, node_attrs=faostat.countries())
>>> g.number_of_nodes(), g.number_of_edges()
(166, 1646)
>>> g.graph
{'weight': 'quantity', 'aggregate': None, 'unit': 't', 'year': 2022, 'item': 'Wheat'}
```

`node_attrs` attaches columns of a table indexed by node name as node attributes. `faostat.countries()` gives each FAOSTAT area its UN M49 region, a `continent` column (M49 regions with the Americas split into Northern America, Central America, the Caribbean and South America), and a label point (`lon`, `lat`). The plots use `continent` for colour.

Other options: `directed=False` sums both directions of each pair into an undirected graph, and `self_loops=True` keeps flows from a country to itself (dropped by default).

### Why mixed years and items raise

Adding up flows from different years or different commodities is almost never what you want by accident: a graph of "wheat plus maize, 2020 to 2022" has edge weights that mean nothing in particular. So `build_graph` refuses to do it silently:

```python
>>> nv.build_graph(faostat.load_sample(items="Wheat", years=[2021, 2022]))
MixedSliceError: flows contain 2 values of 'year' (2021, 2022). Build one graph per
slice with graphs_by(), filter the flows first, or pass aggregate='sum' or aggregate='mean'.
```

If you do want a combined graph, say how to combine:

- `aggregate="sum"` totals the flows.
- `aggregate="mean"` divides the total by the number of slices present, so a pair missing from one year counts as zero for that year.

```python
>>> g2 = nv.build_graph(faostat.load_sample(items="Wheat", years=[2021, 2022]), aggregate="mean")
>>> g2.graph
{'weight': 'quantity', 'aggregate': 'mean', 'unit': 't', 'year': [2021, 2022],
 'item': 'Wheat', 'n_slices': 2}
```

Units are never mixed: if the flows contain more than one `unit`, `build_graph` raises `MixedUnitError` whatever `aggregate` says.

### One graph per slice

`nv.graphs_by` groups a flow frame and builds one graph per group. The default groups by `("year", "item")`:

```python
>>> gs = nv.graphs_by(faostat.load_sample(years=range(2020, 2023)))
>>> list(gs)[:3]
[(2020, 'Maize (corn)'), (2020, 'Soya beans'), (2020, 'Wheat')]
>>> wheat_by_year = nv.graphs_by(faostat.load_sample(items="Wheat"), by="year")
>>> sorted(wheat_by_year)[:3]
[2010, 2011, 2012]
```

Extra keyword arguments go to `build_graph`. Grouping by `"year"` over several items therefore needs `aggregate=` as well. `nv.graph_to_flows(g)` converts a graph back into a flow table.

## Partners and item comparisons

Two table-level helpers do not need a graph:

```python
>>> nv.partners(flows, "Ukraine", role="exporter", top_n=3)
           quantity  share
partner
Türkiye  2059037.61  0.246
Spain    1017543.91  0.122
Poland    522878.19  0.063
```

`role="importer"` ranks where a country's imports come from, and `role="both"` puts exports and imports per partner side by side. Country names are matched exactly; `nv.partners(flows, "Russia")` raises `UnknownCountryError: unknown country 'Russia'. Did you mean: 'Russian Federation', 'Tunisia', 'Austria'?`.

`nv.compare_items(flows, metric)` ranks items by total quantity, number of flows, or number of exporting, importing or participating countries, optionally per year.

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

Graph-level metrics live in `nv.temporal`: `graph_summary(g)` for one graph, and `metric_series(graphs)` for a mapping of graphs such as the output of `graphs_by(..., by="year")`:

```python
>>> ts = nv.temporal.metric_series(wheat_by_year, ["total_weight", "out_strength_hhi"])
>>> ts.loc[2021:2023]
      total_weight  out_strength_hhi
2021  1.707297e+08          0.085522
2022  1.665691e+08          0.084138
2023  1.753090e+08          0.099115
```

`out_strength_hhi` is the Herfindahl-Hirschman index of exporters' shares of total volume: 1 means one supplier, `1/n` means `n` equal suppliers. `centrality_series(graphs, kind, nodes=[...])` tracks one centrality measure per country across the same graphs; a country absent in a year gets `fill_value` (0 by default).

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

Every plotting function returns a `plotly.graph_objects.Figure` and never calls `show()`. You decide whether to show it, change it, or save it:

```python
fig = nv.plot.network(g, top_n=40, color_by="continent")
fig.update_layout(height=500)
fig.write_html("wheat-2022.html")
```

| Function | What it draws |
| --- | --- |
| `nv.plot.network(g, ...)` | node-link diagram; `layout="spring"`, `"kamada_kawai"`, `"circular"` or `"community"`; colour by a node attribute or by community |
| `nv.plot.sankey(g, top_n=10)` | flows from the largest exporters (left) to the largest importers (right), with the rest grouped as "Other" |
| `nv.plot.flow_map(g, top_n=60)` | the largest flows as arcs on a world map, placed at the bundled country label points |
| `nv.plot.time_series(df, columns, facet=...)` | one line per column of a period-indexed table, such as `metric_series` output |
| `nv.plot.degree_distribution(fit)` | empirical CCDF with the fitted power-law, lognormal and exponential tails on log-log axes |
| `nv.plot.community_layout(g, partition)` | node positions with each community in its own cluster (used by `layout="community"`) |

Node sizes and edge widths are log-scaled so that a few very large flows do not hide the rest. Colours come from a fixed eight-colour palette; further categories are grouped as "Other", and legend entries carry the category name so that identity does not depend on colour alone. `flow_map` needs no extra dependencies: it uses Plotly's built-in geography and the coordinates in `faostat.countries()`. Pass `coords=` (a table with `lon` and `lat`) for nodes that are not FAOSTAT countries.

## Bringing your own data

`nv.to_flowframe` converts any edge list into a flow frame. Map your column names to schema names, and give constants for the columns you do not have:

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
    {"origin": "exporter", "dest": "importer", "people": "quantity"},
    year=2020,
    item="migrants",
    unit="persons",
)
g = nv.build_graph(ff)
```

The result has the six schema columns first, then any other columns of `raw`, and `year` is cast to `int64`. From here every metric and plot works the same way. Continent colouring needs a `continent` node attribute (pass `node_attrs=`), or use `color_by=None`; flow maps need `coords=` unless the node names are FAOSTAT country names.

## The full FAOSTAT store

The sample covers three items. For the other items, and for trade values, build the local store once:

```python
from netviz_tools.datasets import faostat

faostat.build_store()  # one-time: downloads about 420 MB
flows = faostat.load("Coffee, green", years=range(2015, 2025))
```

`build_store()` downloads the FAOSTAT bulk zip (about 420 MB), checks its SHA-256 against the value pinned in this release, extracts the CSV (about 9 GB of temporary disk space), and converts it with DuckDB into a Parquet store partitioned by item. It takes about a minute on a laptop, and the finished store is about 400 MB. Later calls return immediately unless you pass `force=True`. If you already have the zip, pass `zip_path=`; it is checked against the same hash.

**Cache directory.** The zip and the store live in `faostat.default_cache_dir()`: the operating system's user cache directory (for example `~/.cache/netviz_tools` on Linux), unless the environment variable `NETVIZ_TOOLS_CACHE` is set. Every function that touches the store also takes `cache_dir=`. `faostat.store_path()` shows where the store is, and `faostat.store_info()` returns its `manifest.json`. Calling `load()` before the store exists raises `StoreNotFoundError` with instructions.

**Reporter perspective.** Each trade flow can be reported twice, by the exporting country and by the importing country, and the two reports often differ. `load()` and `load_sample()` take `reporter=`:

- `"importer"` (default): flows as reported by the importing country. More countries report imports than exports: for wheat in 2021 the sample has 155 reporting importers against 99 reporting exporters.
- `"exporter"`: flows as reported by the exporting country.
- `"combined"`: the importer's report where one exists, otherwise the exporter's, decided per exporter, importer, item and year.

The choice matters. The Russian Federation reports no wheat exports after 2021, so in exporter-reported data it has no wheat exports at all in 2022. Importer-reported data still show about 22.4 million tonnes that year. See the [wheat notebook](gallery/wheat-2022-shock.ipynb).

**Trade value.** `load(..., measure="value")` returns trade value in `1000 USD` instead of physical quantity. Quantities are in `t`, `head` or `number`. Values and quantities are never mixed in one frame, but a few items (mostly live animals) have quantity rows in both `head` and `t`; filter on `unit` before building a graph, or `build_graph` raises `MixedUnitError`. The bundled sample contains quantities only.

**Details.** `load(..., details=True)` adds `item_code`, `exporter_code`, `importer_code`, `reported_by` and the FAOSTAT `flag`. `self_loops=True` keeps flows from a country to itself (re-imports), which the store keeps but `load` drops by default.

See [Data and licences](data.md) for the transformation rules, the pinned release, and how to cite FAOSTAT.
