# netviz-tools

netviz-tools is a small, typed Python library for bilateral flow networks: who sends how much of what to whom, and in which year. It turns an edge table into NetworkX graphs, measures them, tests their degree distributions, and draws them with Plotly.

The flagship dataset is the FAOSTAT Detailed Trade Matrix: bilateral trade in about 560 agricultural products between about 220 countries and territories since 1986. A sample with wheat, maize and soya beans for 2010 to 2024 ships inside the package, so every example on this site runs offline.

## Features

- **One schema.** A flow frame is a pandas DataFrame with `exporter`, `importer`, `year`, `item`, `quantity` and `unit`. `nv.to_flowframe` converts any edge list (migration, shipping, payments) into it, and `nv.validate_flows` checks it.
- **Graphs that know their slice.** `nv.build_graph` builds one graph per year and item, and raises `MixedSliceError` instead of silently adding up different years or commodities. Aggregation is explicit: `aggregate="sum"` or `aggregate="mean"`. `nv.graphs_by` builds one graph per group.
- **Metrics.** Strength, degree, PageRank, reverse PageRank and weighted betweenness in one call (`nv.metrics.centrality`); Louvain and greedy-modularity communities with fixed seeds; modularity; community graphs.
- **Time series.** `nv.temporal.metric_series` tracks graph-level metrics such as total volume and supplier concentration (Herfindahl-Hirschman index) across years; `centrality_series` tracks one measure per country.
- **Heavy-tail tests.** `nv.stats.degree_distribution_fit` fits a power-law tail by maximum likelihood following Clauset, Shalizi and Newman (2009), and compares it with lognormal and exponential tails using Vuong's likelihood-ratio test.
- **Plots that return figures.** Network diagrams, Sankey diagrams, flow maps on a world map, time series and degree-distribution plots. Every function returns a `plotly.graph_objects.Figure` and never calls `show()`.
- **Reproducible data.** `nv.datasets.faostat.build_store()` downloads the FAOSTAT bulk file once, checks it against a pinned SHA-256, converts it into a Parquet store partitioned by item, and writes a manifest with the source URL, hash, retrieval time and row counts. `load()` then reads only the partitions for the items you ask for.
- **Typed errors.** Every exception derives from `nv.NetvizError` and from the matching built-in type. Unknown item and country names come with suggestions (`unknown country 'Russia'. Did you mean: 'Russian Federation', ...`).

## Install

```bash
pip install netviz-tools
```

Python 3.11 or newer. The dependencies are pandas, NumPy, SciPy, NetworkX, Plotly, DuckDB, PyArrow and pooch.

## Quickstart

```python
import netviz_tools as nv
from netviz_tools.datasets import faostat

flows = faostat.load_sample(items="Wheat", years=2022)  # 1,646 flows, tonnes
g = nv.build_graph(flows, node_attrs=faostat.countries())

nv.metrics.centrality(g, ["out_strength", "pagerank"]).head()
nv.partners(flows, "Ukraine", role="exporter", top_n=5)

fig = nv.plot.network(g, top_n=40)  # coloured by continent
fig.show()
```

The 2022 wheat graph has 166 countries and 1,646 flows. Australia is the largest exporter that year in the importer-reported data (25.0 million tonnes), followed by the Russian Federation (22.4 million tonnes).

## Where to go next

- [Getting started](getting-started.md): the schema, graphs, metrics, plots, your own data, and the full FAOSTAT store.
- [Gallery](gallery/index.md): three executed notebooks on the 2022 wheat shock, community structure, and power-law tests.
- [Data and licences](data.md): where the data come from, how they are transformed, and how to cite them.
- [API reference](api/index.md): every public function, with signatures and docstrings.
- [Changelog](changelog.md).

The code is MIT-licensed. FAOSTAT data are published by FAO under CC BY 4.0; cite them as described on the [data page](data.md#citing-faostat).
