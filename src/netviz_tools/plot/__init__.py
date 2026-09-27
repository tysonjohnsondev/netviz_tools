"""Plotly figures for any NetworkX graph or flow table.

Every function returns a :class:`plotly.graph_objects.Figure` and never calls
``show()``, so figures can be customised, saved, or embedded. Every function
uses the same argument names for the same things; see the Concepts page of
the documentation.
"""

from netviz_tools.plot._auto import KINDS, Kind, auto, choose_kind
from netviz_tools.plot._bars import Role, compare, ranking
from netviz_tools.plot._charts import degree_distribution, time_series
from netviz_tools.plot._data import PlotData, Selector
from netviz_tools.plot._geo import flow_map
from netviz_tools.plot._matrix import adjacency
from netviz_tools.plot._network import Layout, community_layout, ego, network
from netviz_tools.plot._render import ShowLabels
from netviz_tools.plot._resolve import EdgeSpec, NodeMetric, NodeSpec
from netviz_tools.plot._sankey import sankey
from netviz_tools.plot._style import CONTINENT_COLORS, DEFAULT_LABELS, PALETTE, SEQUENTIAL, scale

__all__ = [
    "CONTINENT_COLORS",
    "DEFAULT_LABELS",
    "KINDS",
    "PALETTE",
    "SEQUENTIAL",
    "EdgeSpec",
    "Kind",
    "Layout",
    "NodeMetric",
    "NodeSpec",
    "PlotData",
    "Role",
    "Selector",
    "ShowLabels",
    "adjacency",
    "auto",
    "choose_kind",
    "community_layout",
    "compare",
    "degree_distribution",
    "ego",
    "flow_map",
    "network",
    "ranking",
    "sankey",
    "scale",
    "time_series",
]
