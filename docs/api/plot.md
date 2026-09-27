# plot

Plotly figures for any NetworkX graph, flow table, or mapping from period to graph. Every function returns a `plotly.graph_objects.Figure` and never calls `show()`. Call `fig.show()`, `fig.write_html(...)` or `fig.write_image(...)` yourself (static image export needs Kaleido, which is not a dependency).

Every function uses the same argument names for the same things; the [Concepts](../concepts.md) page lists them and says which chart uses which. All functions below are available as `nv.plot.<name>`.

## Choosing a chart

::: netviz_tools.plot.auto
    options:
      show_root_full_path: false

::: netviz_tools.plot.choose_kind
    options:
      show_root_full_path: false

::: netviz_tools.plot.Kind
    options:
      show_root_full_path: false
      show_if_no_docstring: true

::: netviz_tools.plot.KINDS
    options:
      show_root_full_path: false
      show_if_no_docstring: true

## Node-link diagrams

::: netviz_tools.plot.network
    options:
      show_root_full_path: false

::: netviz_tools.plot.ego
    options:
      show_root_full_path: false

::: netviz_tools.plot.flow_map
    options:
      show_root_full_path: false

::: netviz_tools.plot.Layout
    options:
      show_root_full_path: false
      show_if_no_docstring: true

::: netviz_tools.plot.community_layout
    options:
      show_root_full_path: false

## Other charts

::: netviz_tools.plot.sankey
    options:
      show_root_full_path: false

::: netviz_tools.plot.adjacency
    options:
      show_root_full_path: false

::: netviz_tools.plot.ranking
    options:
      show_root_full_path: false

::: netviz_tools.plot.compare
    options:
      show_root_full_path: false

::: netviz_tools.plot.time_series
    options:
      show_root_full_path: false

::: netviz_tools.plot.degree_distribution
    options:
      show_root_full_path: false

## Argument types

::: netviz_tools.plot.PlotData
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.Selector
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.NodeSpec
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.NodeMetric
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.EdgeSpec
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.ShowLabels
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.Role
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

## Styling

::: netviz_tools.plot.PALETTE
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.SEQUENTIAL
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.CONTINENT_COLORS
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.DEFAULT_LABELS
    options:
      show_root_full_path: false
      show_if_no_docstring: true
      heading_level: 3

::: netviz_tools.plot.scale
    options:
      show_root_full_path: false
      heading_level: 3

