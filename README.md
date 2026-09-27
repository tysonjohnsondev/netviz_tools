# netviz-tools

[![CI](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/ci.yml/badge.svg)](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/netviz-tools)](https://pypi.org/project/netviz-tools/)
[![Python versions](https://img.shields.io/pypi/pyversions/netviz-tools)](https://pypi.org/project/netviz-tools/)
[![Docs](https://github.com/tysonjohnsondev/netviz_tools/actions/workflows/docs.yml/badge.svg)](https://tysonjohnsondev.github.io/netviz_tools/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/tysonjohnsondev/netviz_tools/blob/main/LICENSE)

A small, typed library for building, measuring and plotting bilateral flow
networks: trade, migration, payments, anything with an origin, a destination
and an amount. FAOSTAT agricultural trade is the built-in dataset, served from
a local DuckDB + Parquet store.

![Wheat trade in 2022, the 80 largest flows](https://raw.githubusercontent.com/tysonjohnsondev/netviz_tools/main/docs/assets/hero.png)

## Install

```bash
pip install netviz-tools
```

Python 3.11 or newer. The wheel is small: it ships a 0.6 MB data sample, not
the dataset.

## Quickstart

```python
import netviz_tools as nv
from netviz_tools.datasets import faostat

flows = faostat.load_sample(items="Wheat", years=2022)  # bundled sample, works offline
g = nv.build_graph(flows, node_attrs=faostat.countries())  # one year, one item
print(nv.metrics.centrality(g).head())  # strengths, PageRank, betweenness
print(nv.partners(flows, "Ukraine", top_n=5))  # where Ukraine's wheat went
fit = nv.stats.degree_distribution_fit(g, "out_strength")
print(fit.summary())  # power law against lognormal and exponential tails
fig = nv.plot.network(g, top_n=30)  # a Plotly figure; nothing is shown or saved for you
fig.write_html("wheat_2022.html")
```

The last `print` gives:

```text
out_strength: alpha=1.632, xmin=326460, tail 39 of 119, KS=0.117. vs lognormal: R=-0.48, p=0.537 (inconclusive). vs exponential: R=17.32, p=0.013 (power_law).
```

The tail of wheat export volumes is heavier than exponential, but the data
cannot tell a power law from a lognormal. That is the usual answer, and the
reason the fit reports both tests.

## Features

- **One schema.** Every function takes a flow frame with columns `exporter`,
  `importer`, `year`, `item`, `quantity`, `unit`. `to_flowframe` converts your
  own edge lists; `validate_flows` checks them.
- **Graphs that mean one thing.** `build_graph` builds one year and one item.
  Mixing years or items raises `MixedSliceError` unless you choose
  `aggregate="sum"` or `"mean"`; mixing units always raises. Construction is
  vectorized. `graphs_by` builds one graph per slice.
- **Metrics.** In and out strength, degree, PageRank and reverse PageRank,
  weighted betweenness (`metrics.centrality`); seeded Louvain or greedy
  modularity communities, modularity and the community graph
  (`metrics.communities`).
- **Change over time.** `temporal.metric_series` (density, total weight,
  reciprocity, supplier and buyer concentration, or your own functions) and
  `temporal.centrality_series`.
- **Heavy-tail tests.** `stats.degree_distribution_fit` follows Clauset,
  Shalizi and Newman (2009): maximum-likelihood exponent, `xmin` chosen by KS
  distance, and likelihood-ratio tests against lognormal and exponential tails.
  It returns a frozen dataclass.
- **Plots.** `plot.network`, `plot.sankey` (exporters left, importers right),
  `plot.flow_map`, `plot.time_series` and `plot.degree_distribution` return
  Plotly figures and never call `show()`.
- **Typed errors.** `UnknownItemError` and `UnknownCountryError` suggest close
  matches; all errors derive from `NetvizError` and the matching built-in.
- **No side effects at import.** Nothing is downloaded, written or logged until
  you ask for it.

## The full FAOSTAT dataset

The bundled sample covers wheat, maize and soya beans from 2010 to 2024. For
all 558 items since 1986, build the local store once:

```python
from netviz_tools.datasets import faostat

faostat.build_store()  # downloads about 420 MB, needs about 9 GB of temporary disk
flows = faostat.load(["Wheat", "potatoes_frozen", 56], years=range(2015, 2025))
```

`build_store` downloads the FAOSTAT bulk file with pooch, checks its SHA-256
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
functions instead of classes, data downloaded on request instead of bundled,
and results returned instead of shown. Item slugs such as `"potatoes_frozen"`
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
