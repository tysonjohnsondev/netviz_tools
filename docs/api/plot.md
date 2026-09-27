# plot

Plotly figures for flow graphs. Every function returns a `plotly.graph_objects.Figure` and never calls `show()`. Call `fig.show()`, `fig.write_html(...)` or `fig.write_image(...)` yourself (static image export needs Kaleido, which is not a dependency).

All functions below are available as `nv.plot.<name>`.

::: netviz_tools.plot.network.network
    options:
      show_root_full_path: false

::: netviz_tools.plot.network.community_layout
    options:
      show_root_full_path: false

::: netviz_tools.plot.network.Layout
    options:
      show_root_full_path: false
      show_if_no_docstring: true

::: netviz_tools.plot.sankey.sankey
    options:
      show_root_full_path: false

::: netviz_tools.plot.geo.flow_map
    options:
      show_root_full_path: false

::: netviz_tools.plot.charts.time_series
    options:
      show_root_full_path: false

::: netviz_tools.plot.charts.degree_distribution
    options:
      show_root_full_path: false

## Styling helpers

::: netviz_tools.plot._style.PALETTE
    options:
      show_root_full_path: false
      heading_level: 3

::: netviz_tools.plot._style.CONTINENT_COLORS
    options:
      show_root_full_path: false
      heading_level: 3

::: netviz_tools.plot._style.scale
    options:
      show_root_full_path: false
      heading_level: 3
