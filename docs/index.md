# netviz-tools

Network data rarely arrives in a shape NetworkX can use. Columns are named differently in every source, the same place is spelled three ways, trade statistics list each flow twice with different numbers, and units get mixed. Once the data is clean, there are still several plotting packages to choose from, each with its own input format and quirks.

netviz-tools standardizes the path from a raw edge table to an interactive chart:

1. **Clean.** `nv.clean(raw)` works out which column holds the source, target, weight, time and category, fixes spellings, resolves flows reported by both sides, drops what cannot be used, and returns a report of every change.
2. **Build.** `nv.build_graph(flows, time=2022)` turns one slice into a NetworkX graph, and refuses to mix periods or categories by accident.
3. **Plot.** `nv.plot.network`, `flow_map`, `sankey`, `adjacency`, `ranking`, `compare`, `time_series` and others take the same arguments and return Plotly figures. `nv.plot.auto` draws a chart by name or picks one from the data.

Any NetworkX graph can go straight to step 3. FAOSTAT agricultural trade data ships with the package as the worked example of messy input: a sample of wheat, maize and soya bean trade from 2010 to 2024, so every example on this site runs offline.

## Quickstart

```python
import netviz_tools as nv

raw = nv.datasets.faostat.raw_sample(items="Wheat")  # FAOSTAT rows as published
flows, report = nv.clean(raw)  # 44,738 raw rows -> 29,627 flows
g = nv.build_graph(flows, time=2022)  # 186 countries, 2,143 flows
fig = nv.plot.auto(g)  # a flow map, because every node has coordinates
fig.show()
```

With a graph you already have:

```python
import networkx as nx

nv.plot.network(nx.karate_club_graph(), color_by="club", size_by="betweenness").show()
```

## Install

```bash
pip install netviz-tools
```

Python 3.11 or newer. The dependencies are pandas, NumPy, SciPy, NetworkX, Plotly, DuckDB, PyArrow and pooch.

## What is in the box

- **One vocabulary.** A clean flow table has the columns `source`, `target`, `time`, `category`, `weight` and `unit`; for trade data, source = exporter. The same words are the arguments of `clean`, `build_graph` and every plot. See [Concepts](concepts.md).
- **Cleaning with a paper trail.** `nv.clean` guesses columns from headers (and says which it guessed), handles reporter/partner tables such as FAOSTAT and UN Comtrade, merges spelling variants, lists look-alike names without merging them, converts units, and keeps every dropped row with its reason.
- **Charts for any graph.** Colour and size nodes by an attribute, a metric (`degree`, `strength`, `pagerank`, `betweenness`, `community`) or your own values; arrows on directed edges; hover text with every attribute; nine layouts including community clusters, bipartite columns and world maps.
- **Time as a dimension.** Pass several periods and network diagrams and flow maps animate with a play button and a slider. Rankings and comparisons mark the previous period, and time series show period-on-period change.
- **Analysis that feeds the charts.** Centrality, seeded communities, modularity, concentration over time, and power-law tail tests following Clauset, Shalizi and Newman (2009).
- **Reproducible FAOSTAT data.** `nv.datasets.faostat.build_store()` downloads the full bulk file once, checks it against a pinned SHA-256, and converts it with DuckDB into a Parquet store with a manifest. `faostat.load()` then reads only the items you ask for.
- **One backend, no side effects.** Every chart is a Plotly figure with a shared style, returned and never shown or saved for you. Nothing is downloaded, written or logged at import.

## Where to go next

- [Getting started](getting-started.md): the pipeline step by step, on FAOSTAT data and on your own tables.
- [Concepts](concepts.md): the shared argument names, the value forms for `color_by` and `size_by`, time animation, display labels, and the rules of `plot.auto`.
- [Gallery](gallery/index.md): executed notebooks, starting with plain NetworkX graphs and the FAOSTAT pipeline from raw rows to charts.
- [Data and licences](data.md): where the data come from, how they are transformed, and how to cite them.
- [API reference](api/index.md): every public function, with signatures and docstrings.
- [Changelog](changelog.md).

The code is MIT-licensed. FAOSTAT data are published by FAO under CC BY 4.0; cite them as described on the [data page](data.md#citing-faostat).
