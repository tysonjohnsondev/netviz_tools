# netviz-tools

[![CI](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/ci.yml/badge.svg)](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/netviz-tools)](https://pypi.org/project/netviz-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/netviz-tools)](https://pypi.org/project/netviz-tools/)
[![Docs](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/docs.yml/badge.svg)](https://tysonjohnsondev.github.io/netviz_tools/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tysonjohnsondev/netviz_tools/blob/main/LICENSE)

Network data rarely arrives in a shape NetworkX can use. Columns are named
differently in every source, the same place is spelled three ways, trade
statistics list each flow twice (once by each country) with different numbers,
and units get mixed. Once the data is clean, you still have to choose among
several plotting packages, each with its own input format and quirks.

netviz-tools standardizes the path from a raw edge table to an interactive
chart: three steps, the same argument names in every function, and one
rendering backend (Plotly). FAOSTAT agricultural trade data ships as the worked
example of messy input, but the steps work for migration, shipping, payments,
citations, or any NetworkX graph you already have.

![Two views of the bundled FAOSTAT sample after cleaning: the 2022 wheat trade network coloured by continent, and the 60 largest soya bean flows of 2024 on a world map](https://raw.githubusercontent.com/tysonjohnsondev/netviz_tools/main/docs/assets/hero.png)

## Install

```bash
pip install netviz-tools
```

Python 3.11 or newer. The wheel ships a 0.6 MB FAOSTAT sample, so every example
below runs offline.

## Quickstart: raw table to chart in three steps

```python
import netviz_tools as nv

raw = nv.datasets.faostat.raw_sample(items="Wheat")  # FAOSTAT rows as published
flows, report = nv.clean(raw)  # 1. clean, and record every change
g = nv.build_graph(flows, time=2022)  # 2. one NetworkX graph for one slice
fig = nv.plot.auto(g)  # 3. a Plotly figure; the chart type is chosen from the data
fig.write_html("wheat_2022.html")  # nothing is shown or saved unless you ask
```

`raw` has one row per reporting country, as in the FAOSTAT bulk file:
`Reporter Countries`, `Partner Countries`, `Element` (`Export quantity` or
`Import quantity`), `Year`, `Item`, `Value`, `Unit`. `clean` works out which
column is which, turns each report into a source-to-target flow, keeps one
report per flow, and says what it did. `print(report.summary())` begins:

```text
Cleaned 44,738 raw rows into 29,627 flows.
Columns: 'Reporter Countries' -> reporter (guessed), 'Partner Countries' -> partner (guessed), ...
...
Mirror flows: 15,095 reported by both sides, 6,994 by the source only, 7,538 by the target only;
median disagreement 23.9%; used the target's report where both exist.
```

Every dropped row is in `report.dropped` with its reason. `nv.plot.auto` chose a
flow map here, because every node is a country with known coordinates, and it
says so under the title and in `fig.layout.meta`. Name the chart instead with
`nv.plot.auto(g, kind="network")`, or call `nv.plot.network(g)` directly.

## One vocabulary, one backend

A clean flow table has the columns `source`, `target`, `time`, `category`,
`weight` and `unit`. For trade data, source = exporter and target = importer;
for migration, origin and destination. The same words are the arguments of
`clean` (which raw column holds each role), `build_graph` and the plots
(which slice to use):

| Argument | Used by | Meaning |
| --- | --- | --- |
| `source=`, `target=`, `weight=`, `time=`, `category=` | `clean` | raw column holding each role (guessed when omitted) |
| `time=`, `category=` | `build_graph`, every plot | which period(s) and category to use |
| `color_by=`, `size_by=` | `network`, `ego`, `flow_map`, `ranking` | a node attribute, a metric such as `"pagerank"`, a mapping, or `"auto"` |
| `edge_width_by=`, `edge_color_by=` | `network`, `ego`, `flow_map` | an edge attribute such as `"weight"`, a mapping, or `"source"`/`"target"` |
| `label_by=`, `show_labels=`, `top_n=`, `focus=` | most plots | label text, which nodes to label, how many to keep, what to emphasise |
| `labels=`, `title=`, `height=` | every plot | display labels such as `{"source": "Exporter"}`, title, size |

A chart that cannot use an argument raises `TypeError` instead of ignoring it.
Every chart is a Plotly figure with the same colours and hover style, returned
to you and never shown or saved for you. The
[Concepts page](https://tysonjohnsondev.github.io/netviz_tools/concepts/) has
the full table and the rules `plot.auto` follows.

| Chart | What it shows |
| --- | --- |
| `plot.network` | node-link diagram of any graph; nine layouts; arrows on directed edges; animated over time |
| `plot.flow_map` | flows as lines on a world map; animated over time with a play button and slider |
| `plot.ego` | one node and its neighbours |
| `plot.sankey` | largest sources on the left, largest targets on the right |
| `plot.adjacency` | adjacency matrix ordered by community, readable for dense graphs |
| `plot.ranking` | top nodes by any value, with the previous period marked, or ranked by change |
| `plot.compare` | one node's outgoing and incoming totals across categories |
| `plot.time_series` | totals per period, or one node's flows per period, with period-on-period change |
| `plot.degree_distribution` | a fitted power-law tail against lognormal and exponential alternatives |
| `plot.auto` | any of the above by name, or chosen from the data |

## Finding the story in FAOSTAT data

All three examples use the bundled sample (wheat, maize and soya beans,
2010 to 2024), cleaned once:

```python
flows, report = nv.clean(nv.datasets.faostat.raw_sample())
```

**Which exporters fell the most?** Rank exporters by the change in their wheat
exports from one year to the next:

```python
nv.plot.ranking(flows, category="Wheat", time=[2022, 2023], size_by="out_strength", change=True)
```

Argentina has the largest drop: its wheat exports went from 14.5 million
tonnes in 2022 to 3.1 million tonnes in 2023, 11.4 million tonnes less, ahead
of India (7.1 million less) and France (6.4 million less). The chart shows the
numbers; why they changed is for you to find out.

**What does one country send and receive?** Compare one country across items:

```python
nv.plot.compare(flows, "Brazil", time=2024)
```

In 2024 Brazil exported 99.8 million tonnes of soya beans and 43.1 million
tonnes of maize, and imported 6.6 million tonnes of wheat while exporting
2.9 million.

**How did flows change over time?** Pass several years and the map becomes an
animation with a play button and a year slider:

```python
nv.plot.flow_map(flows, category="Soya beans", top_n=40)
```

Brazil's soya bean exports to mainland China grow from 18.6 million tonnes in
2010 to 74.6 million in 2024. The same call works for migration or any other
table with origin, destination and year columns, once it has been through
`clean` (or `nv.to_flowframe`). `focus="Brazil"` keeps only Brazil's flows.

## Already have a NetworkX graph?

Skip the first two steps. Every plot takes any `Graph`, `DiGraph`,
`MultiGraph` or `MultiDiGraph`:

```python
import networkx as nx
import netviz_tools as nv

g = nx.karate_club_graph()
nv.plot.network(g, color_by="club", size_by="betweenness").show()
nv.plot.network(g, layout="community")  # colour and layout by Louvain community
nv.plot.adjacency(nx.les_miserables_graph())  # matrix ordered by community
nv.plot.ego(g, 0, radius=2)  # one node's neighbourhood
```

`color_by` and `size_by` take a node attribute name, a metric (`degree`,
`strength`, `pagerank`, `betweenness`, `community`, and in/out variants), or a
mapping of your own values. By default nodes are coloured by community and
sized by strength (weighted degree, or degree when edges have no weight), and
edges are as wide as their `weight`. Hover over any node for its attributes and
values. Graphs over 2,000 nodes are cut to the 2,000 largest with a warning;
pass `top_n=` or `max_nodes=None` to choose.

## Analysis that feeds the charts

- `nv.metrics.centrality`: in and out strength, degree, PageRank and reverse
  PageRank, weighted betweenness. `nv.metrics.communities`: seeded Louvain or
  greedy modularity, `modularity` and `community_graph`.
- `nv.temporal.metric_series`: density, total weight, reciprocity, and
  supplier or buyer concentration (Herfindahl-Hirschman index) per period.
- `nv.stats.degree_distribution_fit`: power-law tail fits following Clauset,
  Shalizi and Newman (2009), with likelihood-ratio tests against lognormal and
  exponential tails.
- `nv.partners(flows, "Ukraine", role="out")`: a node's partners and shares.
- Typed errors: all derive from `nv.NetvizError`; unknown names suggest close
  matches (`unknown node 'Russia'. Did you mean: 'Russian Federation', ...`).
- No side effects at import: nothing is downloaded, written or logged until you
  ask.

## The full FAOSTAT dataset

The bundled sample covers wheat, maize and soya beans from 2010 to 2024. For
all 558 items since 1986, build the local store once:

```python
from netviz_tools.datasets import faostat

faostat.build_store()  # downloads about 420 MB, needs about 9 GB of temporary disk
flows = faostat.load(["Wheat", "potatoes_frozen", 56], years=range(2015, 2025))
```

`faostat.load` returns a clean flow table directly (the conversion from the bulk
layout, including the choice between reports, happens in DuckDB), with display
labels attached so charts say Exporter and Importer. Use `nv.clean` for your own
tables. `build_store` downloads the FAOSTAT bulk file with pooch, checks its SHA-256
against the release pinned in this version, and converts it with DuckDB into a
Parquet dataset partitioned by item (about 400 MB, about a minute on a laptop).
A `manifest.json` next to the data records the source URL, the SHA-256 actually
used, the retrieval time, row counts and the library version. `load` then
queries only the items you ask for.

- The cache lives in your user cache directory; set `NETVIZ_TOOLS_CACHE` or
  pass `cache_dir=` to move it.
- FAO replaces the bulk file in place when it publishes an update. The build
  then raises `SourceHashMismatchError`. To build from the newer file, pass
  `known_hash=None` explicitly.
- Each flow can be reported by the exporter, the importer, or both.
  `load(..., reporter=...)` accepts `"importer"` (default, more countries
  report imports), `"exporter"`, or `"combined"` (the importer's report where
  one exists, otherwise the exporter's). This matters: the Russian Federation
  reports no wheat exports after 2021, so exporter-reported data drop it from
  the 2022 network.
- `measure="value"` returns trade value in thousands of US dollars instead of
  quantities.

## Data and attribution

FAOSTAT data are licensed under
[CC BY 4.0](https://www.fao.org/contact-us/terms/db-terms-of-use/en/). If you
publish results, cite them, for example:

> FAO. 2025. FAOSTAT: Detailed trade matrix. Accessed on 27 September 2026.
> https://www.fao.org/faostat/en/#data/TM. Licence: CC-BY-4.0.

FAO does not endorse this library or any analysis made with it. Country regions
come from the UN Statistics Division's
[M49 standard](https://unstats.un.org/unsd/methodology/m49/overview/); country
label points for flow maps come from
[Natural Earth](https://www.naturalearthdata.com/) (public domain). The small
metadata tables in the package are generated by `scripts/build_metadata.py`,
and every manual override is listed on the
[data page](https://tysonjohnsondev.github.io/netviz_tools/data/) of the docs.

## What changed from 0.x

Version 1.0 is a rewrite. The 0.x releases bundled about 736 MB of FAOSTAT CSV
files in the wheel, required Python 3.13, and were built around stateful
manager classes that printed, logged and plotted as side effects. They are
superseded. The package name and import name are unchanged, but the API is new:
functions instead of classes, a clean, build, plot pipeline that works on any
edge table or NetworkX graph, data downloaded on request instead of bundled,
and figures returned instead of shown. Item slugs such as `"potatoes_frozen"`
from 0.x still work in `faostat.load`.

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest          # tests, doctests and the 90% coverage gate
uv run ruff check . && uv run ruff format --check .
uv run mypy            # strict
uv run --extra docs mkdocs serve
```

See [CONTRIBUTING.md](https://github.com/tysonjohnsondev/netviz_tools/blob/main/CONTRIBUTING.md). Documentation:
<https://tysonjohnsondev.github.io/netviz_tools/>.

## License

MIT. See [LICENSE](https://github.com/tysonjohnsondev/netviz_tools/blob/main/LICENSE).
