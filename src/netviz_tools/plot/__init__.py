"""Plotly figures for flow graphs.

Every function returns a :class:`plotly.graph_objects.Figure` and never calls
``show()``, so figures can be customised, saved, or embedded.
"""

from netviz_tools.plot._style import CONTINENT_COLORS, PALETTE, scale
from netviz_tools.plot.charts import degree_distribution, time_series
from netviz_tools.plot.geo import flow_map
from netviz_tools.plot.network import Layout, community_layout, network
from netviz_tools.plot.sankey import sankey

__all__ = [
    "CONTINENT_COLORS",
    "PALETTE",
    "Layout",
    "community_layout",
    "degree_distribution",
    "flow_map",
    "network",
    "sankey",
    "scale",
    "time_series",
]
