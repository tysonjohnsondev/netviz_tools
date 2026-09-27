# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-28

A ground-up rewrite around one pipeline: clean a raw edge table, build
NetworkX graphs, and plot them with Plotly, using the same argument names at
every step. The 0.x API is not compatible.

### Added

- `clean`: turns a messy edge or trade table into a flow table. It guesses
  which column holds each role (and says which it guessed), handles
  reporter/partner tables with export and import rows, resolves flows reported
  by both sides (`prefer=`), parses numbers and years written as text, converts
  units, merges spelling variants, lists look-alike names without merging
  them, drops aggregates, zeros and self-loops, and returns a `CleaningReport`
  with per-step counts, renames, mirror-report statistics and every dropped
  row with its reason.
- Flow-table schema with generic columns `source`, `target`, `time`,
  `category`, `weight`, `unit`, plus `validate_flows` and `to_flowframe`.
  Display labels travel with the data in `flows.attrs["labels"]` and
  `G.graph["labels"]`.
- `build_graph` (with `time=` and `category=` filters), `graphs_by` and
  `graph_to_flows`. Graphs hold one slice; mixing slices raises
  `MixedSliceError` unless `aggregate="sum"` or `"mean"` is given, and mixing
  units raises `MixedUnitError`.
- Plots for any NetworkX graph (including multigraphs), flow table, or mapping
  from period to graph, all sharing one set of arguments (`color_by`,
  `size_by`, `edge_width_by`, `edge_color_by`, `label_by`, `show_labels`,
  `top_n`, `focus`, `time`, `category`, `labels`, `title`, `height`):
  - `plot.network`: nine layouts (spring, Kamada-Kawai, circular, shell,
    spectral, community, bipartite, multipartite, geo), arrows on directed
    edges, self-loops, colour and size by attribute, metric or mapping, legend
    or colour bar, hover text with every attribute, `pos=` for your own
    layout, and a node cap with a warning for large graphs.
  - `plot.ego`, `plot.flow_map`, `plot.sankey`, `plot.adjacency`,
    `plot.ranking`, `plot.compare`, `plot.time_series` and
    `plot.degree_distribution`.
  - Time animation: with several periods, `network`, `ego` and `flow_map` get
    one frame per period with a play button and a slider; `ranking` and
    `compare` mark the previous period; `time_series` shows period-on-period
    change.
  - `plot.auto` draws a chart by name or chooses one from the data by
    documented rules, recording the choice in `fig.layout.meta`.
    `plot.choose_kind` returns the choice without drawing.
  - No function calls `show()` or writes a file.
- `metrics.centrality` (in/out strength, degree, PageRank, reverse PageRank,
  weighted betweenness), `metrics.communities` (seeded Louvain, greedy
  modularity), `metrics.modularity` and `metrics.community_graph`.
- `temporal.metric_series`, `temporal.centrality_series` and
  `temporal.graph_summary`.
- `stats.degree_distribution_fit` and `stats.fit_power_law`: discrete and
  continuous power-law fits with KS-selected `xmin` and likelihood-ratio tests
  against lognormal and exponential tails, returned as frozen dataclasses.
- `partners` (`role="out"`, `"in"` or `"both"`) and `compare_categories`.
- `datasets.faostat`: pooch download of the FAOSTAT Detailed Trade Matrix with
  a pinned SHA-256, a DuckDB-built Parquet store partitioned by item, a build
  manifest, `load` with importer, exporter or combined reporting and quantity
  or value measures, a bundled 0.6 MB sample, `raw_sample` (the sample in the
  bulk-file layout, for trying `clean`), `LABELS`, the item catalogue, and
  country metadata from UNSD M49 and Natural Earth.
- Typed errors with close-match suggestions (`UnknownItemError`,
  `UnknownNodeError`) under a common `NetvizError` base.
- Documentation site with a Concepts page, an API reference and executed
  example notebooks.
- CI on Linux (Python 3.11 to 3.13) and Windows, a lowest-dependency job,
  PyPI Trusted Publishing, and Dependabot.

### Changed

- Minimum Python is now 3.11 (was 3.13).
- Data are downloaded on request instead of shipped in the wheel.

### Removed

- The `DataManager`, `FAOSTATManager`, `TradeNetwork` and `TradeSeries`
  classes, the bundled CSV files, the unsourced `nodes.csv` country table, and
  logging and directory creation at import time.

## [0.2.0] - 2025

Last release of the 0.x series. Superseded by 1.0.0.

[Unreleased]: https://github.com/tysonjohnsondev/netviz_tools/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tysonjohnsondev/netviz_tools/releases/tag/v1.0.0
[0.2.0]: https://pypi.org/project/netviz-tools/0.2.0/
