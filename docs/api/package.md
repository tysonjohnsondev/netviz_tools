# netviz_tools

```python
import netviz_tools as nv
```

The top-level package re-exports the functions you use most. Each one is documented on its module page.

| Name | Module page |
| --- | --- |
| `nv.build_graph`, `nv.graphs_by`, `nv.graph_to_flows` | [graph](graph.md) |
| `nv.partners`, `nv.compare_categories` | [analysis](analysis.md) |
| `nv.to_flowframe`, `nv.validate_flows`, `nv.FLOW_COLUMNS`, `nv.FlowFrame` | [schema](schema.md) |
| `nv.metrics` | [metrics](metrics.md) |
| `nv.temporal` | [temporal](temporal.md) |
| `nv.stats` | [stats](stats.md) |
| `nv.plot` | [plot](plot.md) |
| `nv.datasets.faostat` | [datasets.faostat](faostat.md) |
| `nv.errors` and every exception class (`nv.MixedSliceError`, `nv.UnknownNodeError`, `nv.UnknownNameError`, ...) | [errors](errors.md) |

`nv.__version__` holds the installed version string.
