# netviz-tools

netviz-tools is a small, typed Python library for flow networks: who sends how much of what to whom, and when. It is a pipeline from a messy flow table to a clean flow frame, then to NetworkX graphs, metrics and Plotly charts. Trade, migration or any other flow data fits the same schema.

The worked example is the FAOSTAT Detailed Trade Matrix: bilateral trade in about 560 agricultural products between about 220 countries and territories since 1986. A sample with wheat, maize and soya beans for 2010 to 2024 ships inside the package, so every example on this site runs offline.

## Features

- **One schema.** A flow frame is a pandas DataFrame with `source`, `target`, `time`, `category`, `weight` and `unit`. `nv.to_flowframe` converts any edge list (trade, migration, shipping, payments) into it, and `nv.validate_flows` checks it. Domain display names ride along in `flows.attrs["labels"]`; the FAOSTAT loaders set them to Exporter, Importer, Year, Item and so on.
- **Graphs that know their slice.** `nv.build_graph` builds one graph per `time` and `category` (select one with `time=` and `category=`), and raises `MixedSliceError` instead of silently adding up different periods or commodities. Aggregation is explicit: `aggregate="sum"` or `aggregate="mean"`. `nv.graphs_by` builds one graph per group.
- **Metrics.** Strength, degree, PageRank, reverse PageRank and weighted betweenness in one call (`nv.metrics.centrality`); Louvain and greedy-modularity communities with fixed seeds; modularity; community graphs.
- **Time series.** `nv.temporal.metric_series` tracks graph-level metrics such as total volume and supplier concentration (Herfindahl-Hirschman index) across periods; `centrality_series` tracks one measure per node.
- **Heavy-tail tests.** `nv.stats.degree_distribution_fit` fits a power-law tail by maximum likelihood following Clauset, Shalizi and Newman (2009), and compares it with lognormal and exponential tails using Vuong's likelihood-ratio test.
- **Plots that return figures.** Network diagrams, Sankey diagrams, flow maps on a world map, time series and degree-distribution plots. Every function returns a `plotly.graph_objects.Figure` and never calls `show()`.
- **Reproducible data.** `nv.datasets.faostat.build_store()` downloads the FAOSTAT bulk file once, checks it against a pinned SHA-256, converts it into a Parquet store partitioned by item, and writes a manifest with the source URL, hash, retrieval time and row counts. `load()` then reads only the partitions for the items you ask for.
- **Typed errors.** Every exception derives from `nv.NetvizError` and from the matching built-in type. Unknown items and node names come with suggestions (`unknown node 'Russia'. Did you mean: 'Russian Federation', ...`).

## Install

```bash
pip install netviz-tools
```

Python 3.11 or newer. The dependencies are pandas, NumPy, SciPy, NetworkX, Plotly, DuckDB, PyArrow and pooch.

## Quickstart

```python
import netviz_tools as nv
from netviz_tools.datasets import faostat

flows = faostat.load_sample()  # wheat, maize and soya beans, 2010 to 2024, tonnes
g = nv.build_graph(flows, time=2022, category="Wheat", node_attrs=faostat.countries())

nv.metrics.centrality(g, ["out_strength", "pagerank"]).head()
nv.partners(flows, "Ukraine", role="out", time=2022, category="Wheat", top_n=5)

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
