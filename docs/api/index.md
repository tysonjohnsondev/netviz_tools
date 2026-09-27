# API reference

One page per module. Signatures are generated from the source with their type annotations; descriptions come from the numpy-style docstrings.

- [netviz_tools](package.md): what the top-level package re-exports.
- [clean](clean.md): `clean`, `CleaningReport`, `CleaningStep`, `MirrorStats`.
- [graph](graph.md): `build_graph`, `graphs_by`, `graph_to_flows`.
- [analysis](analysis.md): `partners`, `compare_categories`.
- [metrics](metrics.md): `centrality`, `communities`, `modularity`, `community_graph`.
- [temporal](temporal.md): `metric_series`, `centrality_series`, `graph_summary`.
- [stats](stats.md): `degree_distribution_fit`, `fit_power_law`, `PowerLawFit`, `DistributionComparison`.
- [plot](plot.md): `auto`, `network`, `ego`, `flow_map`, `sankey`, `adjacency`, `ranking`, `compare`, `time_series`, `degree_distribution`, `community_layout`, `choose_kind`.
- [datasets.faostat](faostat.md): `load`, `load_sample`, `build_store`, `download`, `store_info`, `store_path`, `default_cache_dir`, `items`, `countries`, `CITATION`, `LABELS`.
- [errors](errors.md): the exception hierarchy.
- [schema](schema.md): the flow-frame schema, `to_flowframe`, `validate_flows`.
