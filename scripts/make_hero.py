"""Render the README hero image from the bundled sample.

Static export needs kaleido and a Chrome build, which are not dependencies of
the library::

    uv run --with kaleido python scripts/make_hero.py

(Run ``plotly_get_chrome`` once if kaleido cannot find Chrome.)
"""

from __future__ import annotations

from pathlib import Path

import netviz_tools as nv

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "hero.png"


def main() -> None:
    faostat = nv.datasets.faostat
    flows = faostat.load_sample(items="Wheat", years=2022)
    g = nv.build_graph(flows, node_attrs=faostat.countries())
    fig = nv.plot.flow_map(
        g,
        top_n=80,
        title="Wheat trade in 2022: the 80 largest flows (FAOSTAT, importer-reported)",
        height=560,
    )
    fig.update_layout(
        width=1100,
        margin={"l": 0, "r": 0, "t": 50, "b": 0},
        legend={
            "orientation": "h",
            "y": -0.02,
            "x": 0.5,
            "xanchor": "center",
            "title": {"text": ""},
        },
        title={"x": 0.5, "xanchor": "center", "font": {"size": 18}},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(OUT, scale=1.5)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
