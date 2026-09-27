"""netviz_tools: turn a flow table into NetworkX graphs, metrics and Plotly charts.

Every function works on a *flow frame* with the columns ``source``, ``target``,
``time``, ``category``, ``weight`` and ``unit`` (see :data:`FLOW_COLUMNS`).
FAOSTAT trade data is the worked example, but migration or any other flow data
fits the same schema via :func:`to_flowframe`. The main entry points are
re-exported here::

    import netviz_tools as nv

    flows = nv.datasets.faostat.load_sample()
    g = nv.build_graph(
        flows, time=2022, category="Wheat", node_attrs=nv.datasets.faostat.countries()
    )
    nv.metrics.centrality(g)
    nv.partners(flows, "Ukraine", role="out", time=2022, category="Wheat")
    nv.plot.network(g)

Submodules: :mod:`~netviz_tools.datasets`, :mod:`~netviz_tools.metrics`,
:mod:`~netviz_tools.temporal`, :mod:`~netviz_tools.stats` and
:mod:`~netviz_tools.plot`.
"""

from importlib.metadata import PackageNotFoundError, version

from netviz_tools import datasets, metrics, plot, stats, temporal
from netviz_tools._schema import FLOW_COLUMNS, FlowFrame, to_flowframe, validate_flows
from netviz_tools.analysis import compare_categories, partners
from netviz_tools.errors import (
    InsufficientDataError,
    MixedSliceError,
    MixedUnitError,
    NetvizError,
    SchemaError,
    SourceHashMismatchError,
    StoreNotFoundError,
    UnknownItemError,
    UnknownMetricError,
    UnknownNameError,
    UnknownNodeError,
)
from netviz_tools.graph import build_graph, graph_to_flows, graphs_by

try:
    __version__ = version("netviz-tools")
except PackageNotFoundError:  # pragma: no cover - only when running from a bare checkout
    __version__ = "0.0.0"

__all__ = [
    "FLOW_COLUMNS",
    "FlowFrame",
    "InsufficientDataError",
    "MixedSliceError",
    "MixedUnitError",
    "NetvizError",
    "SchemaError",
    "SourceHashMismatchError",
    "StoreNotFoundError",
    "UnknownItemError",
    "UnknownMetricError",
    "UnknownNameError",
    "UnknownNodeError",
    "__version__",
    "build_graph",
    "compare_categories",
    "datasets",
    "graph_to_flows",
    "graphs_by",
    "metrics",
    "partners",
    "plot",
    "stats",
    "temporal",
    "to_flowframe",
    "validate_flows",
]
