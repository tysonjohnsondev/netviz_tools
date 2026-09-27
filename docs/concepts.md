# Concepts

netviz-tools standardizes the path from a messy edge table to a chart. It has three steps, and every step uses the same words for the same things.

```text
raw table ──clean()──▶ flow table ──build_graph()──▶ NetworkX graph ──plot.<kind>()──▶ Plotly figure
                        (FlowFrame)                                   plot.auto()
```

You can enter at any step. If you already have a NetworkX graph, skip straight to the plots. The plots also accept a flow table directly and build the graphs for you.

## The column vocabulary

A clean flow table (a *flow frame*) has these six columns, in this order:

| Column | Meaning | For trade data |
| --- | --- | --- |
| `source` | where the flow starts | exporter |
| `target` | where the flow ends | importer |
| `time` | integer period, such as a year | year |
| `category` | what flows | item (wheat, maize) |
| `weight` | non-negative amount | quantity or value |
| `unit` | unit of `weight` | `"t"`, `"1000 USD"` |

For trade data, source = exporter and target = importer. For migration, source is the origin country and target the destination; `category` might be a visa type and `unit` `"persons"`. Extra columns are allowed and kept.

In a graph built from a flow table, nodes are the sources and targets, each edge carries its amount under the `"weight"` attribute (the name NetworkX algorithms use by default), and `G.graph` records the slice: `time`, `category`, `unit`, and the display `labels`.

## Step 1: `clean`

`nv.clean(raw, ...)` turns a messy table into a flow frame and returns a `CleaningReport` that lists every change and every dropped row with its reason. Its arguments name columns of the raw table using the same vocabulary:

```python
flows, report = nv.clean(raw, source="Origin", target="Destination", weight="Persons", time="Year")
```

Any role you leave out is guessed from the column headers, and the report says which ones were guessed. Tables that list each flow once per reporting country (as FAOSTAT and UN Comtrade do) use `reporter=`, `partner=` and `direction=` instead of `source`/`target`; see [Getting started](getting-started.md#cleaning-a-raw-table).

## Step 2: `build_graph`

After cleaning, `time=` and `category=` select values, not columns: `nv.build_graph(flows, time=2022, category="Wheat")`. A graph always holds one slice. Mixing periods or categories raises `MixedSliceError` unless you choose `aggregate="sum"` or `"mean"`. `nv.graphs_by(flows, by="time")` builds one graph per period.

## Step 3: plots

Every plot function takes the data first and everything else by keyword. The same keyword means the same thing in every function:

| Argument | Meaning |
| --- | --- |
| `color_by` | node colour: an attribute, a metric, a mapping, `"auto"` or `None` |
| `size_by` | node size (bar length in `ranking`), same forms as `color_by` |
| `edge_width_by` | edge width: an edge attribute or mapping; `"auto"` uses `weight` |
| `edge_color_by` | edge colour: an edge attribute, a mapping, or `"source"`/`"target"` to match an end node |
| `label_by` | node attribute used as label text |
| `show_labels` | which nodes get a label: `"auto"`, a number, `True`, `False`, or a list |
| `top_n` | how many nodes (or flows, on a map) to keep, largest first |
| `focus` | node or nodes to emphasise (or follow, in `flow_map`, `sankey`, `compare`, `time_series`) |
| `time` | period or periods to use; several periods animate `network` and `flow_map` |
| `category` | which category (item) to draw |
| `labels` | display labels, for example `{"source": "Exporter"}` |
| `title`, `height` | figure title and height in pixels |

A function that cannot use an argument does not accept it, so passing one raises `TypeError` instead of being silently ignored. `nv.plot.auto` adds the chart name and the reason for the choice to that message.

Which chart uses which argument:

| | network | ego | flow_map | sankey | adjacency | ranking | compare | time_series |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| `color_by` | ✓ | ✓ | ✓ | ✓ | | | | |
| `size_by` | ✓ | ✓ | ✓ | | | ✓ | | |
| `edge_width_by`, `edge_color_by` | ✓ | ✓ | ✓ | | | | | |
| `label_by`, `show_labels` | ✓ | ✓ | ✓ | | | | | |
| `top_n` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
| `focus` | ✓ | required | ✓ | ✓ | ✓ | ✓ | required | ✓ |
| `time` | ✓ animates | ✓ animates | ✓ animates | one period | one period | last vs previous | last vs previous | ✓ |
| `category` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ several | ✓ several |
| `labels`, `title`, `height` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Values for `color_by` and `size_by`

- **A node attribute name**, such as `"club"` on the karate club graph or `"continent"` on a graph built with `node_attrs=faostat.countries()`. Text values get a legend (eight colours, the rest shown as "Other"); numbers get a colour bar.
- **A metric name**: `degree`, `in_degree`, `out_degree`, `strength`, `in_strength`, `out_strength`, `pagerank`, `reverse_pagerank`, `betweenness`, or `community`. On a directed graph, `degree` and `strength` count incoming plus outgoing edges. A node attribute with the same name wins over the metric.
- **A mapping or Series** from node to value, for anything you computed yourself.
- **`"auto"`**: `color_by="auto"` colours by community when the graph has edges; `size_by="auto"` uses strength when edges have a `weight`, otherwise degree.
- **`None`**: no colour or equal sizes.

Sizes and colours are log-scaled when values span more than two orders of magnitude, so a few very large nodes do not flatten the rest. Hover over any node to see the exact values.

### Time

With a flow table spanning several periods, or a mapping `{period: graph}`:

- `network` and `flow_map` become animations with a play button and a time slider, one frame per period. Node positions and scales stay fixed across frames, so movement means change.
- `ranking` and `compare` show the last period and mark the one before, and hover gives the change. `ranking(..., change=True)` ranks by the change itself.
- `time_series` draws one line per category, or a focus node's outgoing and incoming totals per period. `change=True` plots the change from the previous period.

### Display labels

Charts use generic words (Source, Target, Weight) unless told otherwise. Pass `labels=` to rename anything, including metric names:

```python
nv.plot.ranking(g, size_by="out_strength", labels={"out_strength": "Departures"})
```

Flow tables from `faostat.load` and `faostat.load_sample` carry `faostat.LABELS` in `flows.attrs["labels"]` (Exporter, Importer, Item, Exports, Imports), and `build_graph` copies them into `G.graph["labels"]`, so FAOSTAT charts say Exporter and Importer without extra arguments.

## `plot.auto`: name the chart or let it choose

`nv.plot.auto(data, kind="network", ...)` is the same as `nv.plot.network(data, ...)`. With `kind="auto"` (the default) it picks a chart from the data by these rules, in order:

| Condition | Chart |
| --- | --- |
| the data is a power-law fit | `degree_distribution` |
| the data is a table indexed by period (not a flow table) | `time_series` |
| one `focus` node and a flow table with several categories | `compare` |
| one `focus` node and several periods | `time_series` |
| one `focus` node | `ego` |
| every node has coordinates (`lon`/`lat` attributes, `coords=`, or FAOSTAT country names) | `flow_map`, animated when there are several periods |
| directed, and no node both sends and receives | `sankey` |
| anything else | `network`, animated when there are several periods |

The figure records the choice in `fig.layout.meta["netviz"]` and in a line under the title. `nv.plot.choose_kind(data, **kwargs)` returns the choice without drawing.

## One rendering backend

Every chart is a Plotly figure with the same colours, fonts and hover style. No function calls `show()` or writes a file: you decide whether to call `fig.show()`, `fig.write_html(...)`, or change the figure first with the usual Plotly methods.
