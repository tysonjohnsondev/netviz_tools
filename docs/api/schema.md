# schema

The flow-frame schema. Everything on this page is re-exported from the top-level package (`nv.FLOW_COLUMNS`, `nv.FlowFrame`, `nv.to_flowframe`, `nv.validate_flows`); import it from there rather than from the private `netviz_tools._schema` module.

A *flow frame* is a `pandas.DataFrame` with one row per directed flow and at least these columns:

| Column | Meaning |
| --- | --- |
| `source` | origin node (for trade data, the exporting country) |
| `target` | destination node (for trade data, the importing country) |
| `time` | integer period label, for example a year |
| `category` | what flows (a commodity, a product code, a migrant type) |
| `weight` | non-negative amount |
| `unit` | unit of `weight`, for example `"t"`, `"persons"` or `"1000 USD"` |

Extra columns are allowed and preserved. Display names for these terms (for example "Exporter" for `source`) can be stored as a dict in `flows.attrs["labels"]`; the FAOSTAT loaders set it from `faostat.LABELS`, `build_graph` copies it into `G.graph["labels"]`, and the plots use it.

::: netviz_tools._schema.FLOW_COLUMNS

::: netviz_tools._schema.FlowFrame

::: netviz_tools._schema.validate_flows

::: netviz_tools._schema.to_flowframe
