"""Render the README hero image from the bundled sample.

The image puts two views of the same cleaned data side by side: the 2022 wheat
trade network (coloured by continent, arrows from exporter to importer) and
the 2024 soya bean flow map. Static export needs kaleido, a Chrome build and
Pillow, which are not dependencies of the library::

    uv run --with kaleido --with pillow python scripts/make_hero.py

(Run ``plotly_get_chrome`` once if kaleido cannot find Chrome.)
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

import netviz_tools as nv

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "hero.png"
WIDTH, HEIGHT, SCALE = 760, 560, 1.5


def main() -> None:
    faostat = nv.datasets.faostat
    flows, _ = nv.clean(faostat.raw_sample())
    countries = faostat.countries()
    net = nv.plot.network(
        flows,
        category="Wheat",
        time=2022,
        node_attrs=countries,
        color_by="continent",
        size_by="out_strength",
        top_n=30,
        show_labels=8,
        min_weight_quantile=0.8,
        title="Wheat trade network, 2022: the 30 largest exporters",
    )
    geo = nv.plot.flow_map(
        flows,
        category="Soya beans",
        time=2024,
        node_attrs=countries,
        color_by="continent",
        top_n=60,
        title="Soya bean trade, 2024: the 60 largest flows",
    )
    panels = []
    for fig in (net, geo):
        fig.update_layout(
            width=WIDTH,
            height=HEIGHT,
            showlegend=fig is geo,
            margin={"l": 10, "r": 10, "t": 60, "b": 70},
            legend={
                "orientation": "h",
                "y": -0.07,
                "x": 0.5,
                "xanchor": "center",
                "title": {"text": ""},
            },
            title={"x": 0.5, "xanchor": "center", "font": {"size": 16}},
        )
        panels.append(Image.open(io.BytesIO(fig.to_image(format="png", scale=SCALE))))
    canvas = Image.new(
        "RGB", (sum(p.width for p in panels), max(p.height for p in panels)), "white"
    )
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    OUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
