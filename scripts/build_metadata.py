"""Regenerate the metadata tables bundled in ``netviz_tools.datasets._meta``.

The script is re-runnable. It fetches its three sources, joins them, applies the
manual overrides listed below, and writes:

* ``faostat_countries.csv``: one row per FAOSTAT area code that appears in the
  Detailed Trade Matrix, with M49 code, ISO3, region, continent and a label point.
* ``faostat_items.csv``: one row per FAOSTAT item code that has trade rows.
* ``sources.json``: source URLs, retrieval date and hashes.

Sources
-------
1. FAOSTAT Detailed Trade Matrix bulk zip (CC BY 4.0). Area codes, M49 codes and
   item names come from this file.
2. UN Statistics Division, "Standard country or area codes for statistical use
   (M49)", overview table. Region, sub-region and intermediate region.
3. Natural Earth 1:10m Admin 0 Countries, v5.1.2 (public domain). Label points
   (``LABEL_X``, ``LABEL_Y``) used as node positions in flow maps.

Usage::

    uv run python scripts/build_metadata.py --zip /path/to/Trade_...zip --cache /tmp/nv

Without ``--zip`` the FAOSTAT file is downloaded (about 420 MB) and verified
against the pinned SHA-256.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import tempfile
import unicodedata
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pooch

from netviz_tools.datasets import _registry

M49_URL = "https://unstats.un.org/unsd/methodology/m49/overview/"
NATURAL_EARTH_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/v5.1.2/"
    "geojson/ne_10m_admin_0_countries.geojson"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "src" / "netviz_tools" / "datasets" / "_meta"

# FAOSTAT areas that are missing from the current UNSD M49 table. Each entry
# gives the M49 code whose region to borrow, and the reason.
REGION_OVERRIDES: dict[str, tuple[str, str]] = {
    "058": ("056", "Belgium-Luxembourg (to 1999) is not in current M49; region of Belgium."),
    "128": (
        "296",
        "Canton and Enderbury Islands (historical) is not in current M49; now Kiribati.",
    ),
    "158": ("156", "M49 158 is not listed by UNSD; FAOSTAT places it in Eastern Asia with China."),
    "200": ("203", "Czechoslovakia (to 1992) is not in current M49; region of Czechia."),
    "230": ("231", "Ethiopia PDR (to 1992) is not in current M49; region of Ethiopia."),
    "396": ("581", "Johnston Island (historical code) is part of US Minor Outlying Islands."),
    "488": ("581", "Midway Island (historical code) is part of US Minor Outlying Islands."),
    "736": ("729", "Sudan (former, to 2011) is not in current M49; region of Sudan."),
    "810": ("643", "USSR (to 1991) is not in current M49; region of the Russian Federation."),
    "872": ("581", "Wake Island (historical code) is part of US Minor Outlying Islands."),
    "890": ("688", "Yugoslav SFR (to 1991) is not in current M49; region of Serbia."),
    "891": ("688", "Serbia and Montenegro (1992-2005) is not in current M49; region of Serbia."),
}

# Areas without their own Natural Earth admin-0 feature. Either borrow another
# code's label point, or give an explicit (lon, lat) with a reason.
COORD_OVERRIDES: dict[str, tuple[str | tuple[float, float], str]] = {
    "058": ("056", "Historical area; label point of Belgium."),
    "128": ("296", "Historical area; label point of Kiribati."),
    "200": ("203", "Historical area; label point of Czechia."),
    "230": ("231", "Historical area; label point of Ethiopia."),
    "396": ((-169.53, 16.73), "Not a separate Natural Earth feature; approximate atoll location."),
    "488": ((-177.37, 28.21), "Not a separate Natural Earth feature; approximate atoll location."),
    "872": ((166.63, 19.29), "Not a separate Natural Earth feature; approximate atoll location."),
    "736": ("729", "Historical area; label point of Sudan."),
    "810": ("643", "Historical area; label point of the Russian Federation."),
    "890": ("688", "Historical area; label point of Serbia."),
    "891": ("688", "Historical area; label point of Serbia."),
    "254": ((-53.2, 3.9), "Part of France in Natural Earth admin-0; approximate centroid."),
    "312": ((-61.55, 16.2), "Part of France in Natural Earth admin-0; approximate centroid."),
    "474": ((-61.0, 14.65), "Part of France in Natural Earth admin-0; approximate centroid."),
    "638": ((55.53, -21.13), "Part of France in Natural Earth admin-0; approximate centroid."),
    "744": ((16.0, 78.5), "Part of Norway in Natural Earth admin-0; approximate centroid."),
    "074": ((3.4, -54.42), "Not a separate Natural Earth admin-0 feature; approximate location."),
    "772": (
        (-171.85, -9.17),
        "Not a separate Natural Earth admin-0 feature; approximate location.",
    ),
}

# ISO3 codes for areas the UNSD table does not list.
ISO3_OVERRIDES: dict[str, tuple[str, str]] = {
    "158": ("TWN", "ISO 3166-1 alpha-3 code; M49 158 is not listed in the UNSD table."),
}

# Continent used for colouring: the M49 region, except that the Americas are
# split into Northern America and the three Latin American intermediate regions.
AMERICAS_SPLIT = {"Northern America", "South America", "Central America", "Caribbean"}


class _M49TableParser(HTMLParser):
    """Collect the rows of the English M49 overview table."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.rows: list[list[str]] = []
        self._row: list[str] = []
        self._cell: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table" and dict(attrs).get("id") == "downloadTableEN":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self._row = []
        elif self.in_table and tag == "td":
            self._cell = ""

    def handle_endtag(self, tag: str) -> None:
        if not self.in_table:
            return
        if tag == "td" and self._cell is not None:
            self._row.append(self._cell.strip())
            self._cell = None
        elif tag == "tr":
            self.rows.append(self._row)
        elif tag == "table":
            self.in_table = False

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell += data


def slugify(name: str) -> str:
    """Return the snake_case slug used by netviz_tools 0.x for item names."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower().replace(" ", "_")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_m49(cache: Path) -> tuple[pd.DataFrame, str]:
    path = Path(pooch.retrieve(M49_URL, known_hash=None, fname="m49_overview.html", path=cache))
    parser = _M49TableParser()
    parser.feed(path.read_text(encoding="utf-8"))
    header, *rows = parser.rows
    df = pd.DataFrame(rows, columns=header)
    df = df.rename(
        columns={
            "M49 Code": "m49",
            "ISO-alpha3 Code": "iso3",
            "Region Name": "region",
            "Sub-region Name": "subregion",
            "Intermediate Region Name": "intermediate_region",
        }
    )
    return df[["m49", "iso3", "region", "subregion", "intermediate_region"]], sha256(path)


def load_natural_earth(cache: Path) -> tuple[pd.DataFrame, str]:
    path = Path(
        pooch.retrieve(
            NATURAL_EARTH_URL, known_hash=None, fname="ne_10m_admin_0_countries.geojson", path=cache
        )
    )
    features = json.loads(path.read_text(encoding="utf-8"))["features"]
    rows = []
    for feat in features:
        p = feat["properties"]
        code = str(p["ISO_N3_EH"])
        if code in {"-99", ""}:
            continue
        rows.append({"m49": code.zfill(3), "lon": p["LABEL_X"], "lat": p["LABEL_Y"]})
    return pd.DataFrame(rows).drop_duplicates("m49"), sha256(path)


def load_faostat(zip_path: Path, workdir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(_registry.FAOSTAT_TRADE_CSV, workdir)
    csv = (workdir / _registry.FAOSTAT_TRADE_CSV).as_posix()
    con = duckdb.connect()
    con.execute(f"CREATE VIEW raw AS SELECT * FROM read_csv('{csv}', header = true)")
    areas = con.execute(
        """
        SELECT code AS fao_code, any_value(m49) AS m49, any_value(name) AS name
        FROM (
            SELECT "Reporter Country Code" AS code, "Reporter Country Code (M49)" AS m49,
                   "Reporter Countries" AS name FROM raw
            UNION ALL
            SELECT "Partner Country Code", "Partner Country Code (M49)", "Partner Countries"
            FROM raw
        )
        GROUP BY code ORDER BY code
        """
    ).df()
    items = con.execute(
        """
        SELECT "Item Code" AS item_code,
               arg_max(Item, Year) AS item,
               replace(any_value("Item Code (CPC)"), '''', '') AS cpc_code,
               min(Year) AS first_year,
               max(Year) AS last_year,
               string_agg(DISTINCT Unit, '|' ORDER BY Unit)
                   FILTER (WHERE Element LIKE '%quantity') AS quantity_units
        FROM raw
        GROUP BY "Item Code" ORDER BY "Item Code"
        """
    ).df()
    areas["m49"] = areas["m49"].str.replace("'", "", regex=False)
    return areas, items


def continent_of(row: pd.Series) -> str:
    if row["subregion"] == "Northern America":
        return "Northern America"
    if row["intermediate_region"] in AMERICAS_SPLIT:
        return str(row["intermediate_region"])
    return str(row["region"])


def build(zip_path: Path, cache: Path) -> dict[str, Any]:
    cache.mkdir(parents=True, exist_ok=True)
    m49, m49_sha = load_m49(cache)
    ne, ne_sha = load_natural_earth(cache)
    with tempfile.TemporaryDirectory(dir=cache) as tmp:
        areas, items = load_faostat(zip_path, Path(tmp))

    m49_idx = m49.set_index("m49")
    ne_idx = ne.set_index("m49")
    records = []
    notes: list[dict[str, str]] = []
    for area in areas.itertuples(index=False):
        code = str(area.m49)
        region_src = code
        if code in REGION_OVERRIDES:
            region_src, reason = REGION_OVERRIDES[code]
            notes.append({"m49": code, "name": str(area.name), "field": "region", "note": reason})
        if region_src not in m49_idx.index:
            raise SystemExit(
                f"No M49 region for {area.name} ({code}); add a REGION_OVERRIDES entry"
            )
        reg = m49_idx.loc[region_src]
        iso3 = reg["iso3"] if region_src == code else ""
        if code in ISO3_OVERRIDES:
            iso3, reason = ISO3_OVERRIDES[code]
            notes.append({"m49": code, "name": str(area.name), "field": "iso3", "note": reason})
        if code in COORD_OVERRIDES:
            target, reason = COORD_OVERRIDES[code]
            notes.append({"m49": code, "name": str(area.name), "field": "lon/lat", "note": reason})
            if isinstance(target, tuple):
                lon, lat = target
            else:
                lon, lat = ne_idx.loc[target, ["lon", "lat"]]
        elif code in ne_idx.index:
            lon, lat = ne_idx.loc[code, ["lon", "lat"]]
        else:
            raise SystemExit(
                f"No label point for {area.name} ({code}); add a COORD_OVERRIDES entry"
            )
        records.append(
            {
                "fao_code": int(area.fao_code),
                "m49": code,
                "iso3": iso3,
                "name": area.name,
                "region": reg["region"],
                "subregion": reg["subregion"],
                "continent": continent_of(reg),
                "lon": round(float(lon), 4),
                "lat": round(float(lat), 4),
            }
        )
    countries = pd.DataFrame(records)
    items["slug"] = items["item"].map(slugify)
    if items["slug"].duplicated().any():
        raise SystemExit("Item slugs are not unique")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    countries.to_csv(OUT_DIR / "faostat_countries.csv", index=False, lineterminator="\n")
    items[
        ["item_code", "item", "slug", "cpc_code", "quantity_units", "first_year", "last_year"]
    ].to_csv(OUT_DIR / "faostat_items.csv", index=False, lineterminator="\n")
    sources = {
        "generated_by": "scripts/build_metadata.py",
        "retrieved": dt.date.today().isoformat(),
        "sources": [
            {
                "name": "FAOSTAT Detailed Trade Matrix (TM), bulk normalized file",
                "url": _registry.FAOSTAT_TRADE_URL,
                "sha256": sha256(zip_path),
                "licence": "CC BY 4.0",
            },
            {
                "name": "UNSD Standard country or area codes for statistical use (M49)",
                "url": M49_URL,
                "sha256": m49_sha,
            },
            {
                "name": "Natural Earth 1:10m Admin 0 Countries v5.1.2",
                "url": NATURAL_EARTH_URL,
                "sha256": ne_sha,
                "licence": "Public domain",
            },
        ],
        "overrides": notes,
    }
    (OUT_DIR / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    return sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", type=Path, help="local copy of the FAOSTAT bulk zip")
    ap.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / "netviz_meta")
    args = ap.parse_args()
    zip_path = args.zip
    if zip_path is None:
        zip_path = Path(
            pooch.retrieve(
                _registry.FAOSTAT_TRADE_URL,
                known_hash=f"sha256:{_registry.FAOSTAT_TRADE_SHA256}",
                fname=_registry.FAOSTAT_TRADE_FILENAME,
                path=args.cache,
                progressbar=True,
            )
        )
    sources = build(zip_path, args.cache)
    print(f"Wrote metadata to {OUT_DIR} ({len(sources['overrides'])} overrides)")


if __name__ == "__main__":
    main()
