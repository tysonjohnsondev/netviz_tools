"""netviz_tools: from a messy edge table to interactive network charts.

Three steps, one vocabulary (``source``, ``target``, ``time``, ``category``,
``weight``, ``unit``)::

    import netviz_tools as nv

    raw = nv.datasets.faostat.raw_sample(items="Wheat")
    flows, report = nv.clean(raw)                   # 1. clean, with a report
    g = nv.build_graph(flows, time=2022)            # 2. one NetworkX graph
    fig = nv.plot.auto(g)                           # 3. a Plotly figure

Any NetworkX graph can go straight to step 3. FAOSTAT trade data is the worked
example; migration or any other origin-destination data fits the same schema.

Submodules: :mod:`~netviz_tools.datasets`, :mod:`~netviz_tools.metrics`,
:mod:`~netviz_tools.temporal`, :mod:`~netviz_tools.stats` and
:mod:`~netviz_tools.plot`.
"""

from importlib.metadata import PackageNotFoundError, version

from netviz_tools import datasets, metrics, plot, stats, temporal
from netviz_tools._clean import CleaningReport, CleaningStep, MirrorStats, clean
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
    "CleaningReport",
    "CleaningStep",
    "FlowFrame",
    "InsufficientDataError",
    "MirrorStats",
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
    "clean",
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
