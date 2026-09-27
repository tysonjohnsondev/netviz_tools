"""Cut the bundled sample Parquet file from a local FAOSTAT store.

The sample keeps quantity flows in tonnes, reported by both exporters and
importers, for Wheat (15), Maize (corn) (56) and Soya beans (236), 2010 onward.
It has the same columns as the store, so ``load_sample`` and ``load`` share one
query. Build the store first with ``faostat.build_store()``.

Usage::

    uv run python scripts/make_sample.py [--cache DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

from netviz_tools.datasets import faostat

ITEM_CODES = (15, 56, 236)
FIRST_YEAR = 2010
OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "netviz_tools"
    / "datasets"
    / "sample"
    / faostat.SAMPLE_FILE
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cache", type=Path, default=None)
    args = ap.parse_args()
    store = faostat.store_path(args.cache)
    info = faostat.store_info(args.cache)
    files = [(store / f"item_code={c}" / "*.parquet").as_posix() for c in ITEM_CODES]
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT exporter, importer, year, item, exporter_code, importer_code,
                   reported_by, measure, unit, value, flag, item_code
            FROM read_parquet({files!r}, hive_partitioning = true)
            WHERE measure = 'quantity' AND unit = 't' AND year >= {FIRST_YEAR}
            ORDER BY item_code, year, reported_by, exporter, importer
        ) TO '{OUT.as_posix()}'
        (FORMAT parquet, COMPRESSION zstd, COMPRESSION_LEVEL 19, ROW_GROUP_SIZE 1000000)
        """
    )
    rows = con.execute(f"SELECT count(*) FROM '{OUT.as_posix()}'").fetchone()
    print(
        f"Wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB, {rows[0] if rows else 0} rows) "
        f"from source SHA-256 {info['source_sha256']}"
    )


if __name__ == "__main__":
    main()
