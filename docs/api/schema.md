# schema

The flow-frame schema. Everything on this page is re-exported from the top-level package (`nv.FLOW_COLUMNS`, `nv.FlowFrame`, `nv.to_flowframe`, `nv.validate_flows`); import it from there rather than from the private `netviz_tools._schema` module.

A *flow frame* is a `pandas.DataFrame` with one row per directed flow and at least these columns:

| Column | Meaning |
| --- | --- |
| `exporter` | origin node (for trade data, the exporting country) |
| `importer` | destination node (the importing country) |
| `year` | integer period label |
| `item` | what flows (a commodity, a product code, a migrant type) |
| `quantity` | non-negative amount |
| `unit` | unit of `quantity`, for example `"t"` or `"1000 USD"` |

Extra columns are allowed and preserved.

::: netviz_tools._schema.FLOW_COLUMNS

::: netviz_tools._schema.FlowFrame

::: netviz_tools._schema.validate_flows

::: netviz_tools._schema.to_flowframe
