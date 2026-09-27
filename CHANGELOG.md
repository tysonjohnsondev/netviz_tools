# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-09-27

A ground-up rewrite. The 0.x API is not compatible.

### Added

- Flow-frame schema (`exporter`, `importer`, `year`, `item`, `quantity`,
  `unit`) with `validate_flows` and `to_flowframe`.
- `build_graph`, `graphs_by` and `graph_to_flows`. Graphs hold one year and
  one item; mixing slices raises `MixedSliceError` unless `aggregate="sum"` or
  `"mean"` is given, and mixing units raises `MixedUnitError`.
- `metrics.centrality` (in/out strength, degree, PageRank, reverse PageRank,
  weighted betweenness), `metrics.communities` (seeded Louvain, greedy
  modularity), `metrics.modularity` and `metrics.community_graph`.
- `temporal.metric_series`, `temporal.centrality_series` and
  `temporal.graph_summary`.
- `stats.degree_distribution_fit` and `stats.fit_power_law`: discrete and
  continuous power-law fits with KS-selected `xmin` and likelihood-ratio tests
  against lognormal and exponential tails, returned as frozen dataclasses.
- `partners` and `compare_items`.
- Plotly figures: `plot.network` (including a community layout),
  `plot.sankey`, `plot.flow_map`, `plot.time_series` and
  `plot.degree_distribution`. No function calls `show()`.
- `datasets.faostat`: pooch download of the FAOSTAT Detailed Trade Matrix with
  a pinned SHA-256, a DuckDB-built Parquet store partitioned by item, a build
  manifest, `load` with importer, exporter or combined reporting and quantity
  or value measures, a bundled 0.6 MB sample, the item catalogue, and country
  metadata from UNSD M49 and Natural Earth.
- Typed errors with close-match suggestions (`UnknownItemError`,
  `UnknownCountryError`) under a common `NetvizError` base.
- Documentation site with an API reference and executed example notebooks.
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
