from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools.datasets import _registry

RAW_HEADER = [
    "Reporter Country Code",
    "Reporter Country Code (M49)",
    "Reporter Countries",
    "Partner Country Code",
    "Partner Country Code (M49)",
    "Partner Countries",
    "Item Code",
    "Item Code (CPC)",
    "Item",
    "Element Code",
    "Element",
    "Year Code",
    "Year",
    "Unit",
    "Value",
    "Flag",
]

COUNTRIES = {
    185: ("'643", "Russian Federation"),
    230: ("'804", "Ukraine"),
    59: ("'818", "Egypt"),
    223: ("'792", "Türkiye"),
}

ELEMENTS = {
    ("Export quantity", "t"): 5910,
    ("Import quantity", "t"): 5610,
    ("Export value", "1000 USD"): 5922,
    ("Import value", "1000 USD"): 5622,
    ("Export quantity", "An"): 5908,
    ("Import quantity", "1000 An"): 5609,
    ("Export quantity", "No"): 5907,
}

ITEMS = {15: ("'0111", "Wheat"), 1096: ("'02131", "Horses"), 1181: ("'02196", "Bees")}


def raw_row(
    reporter: int, partner: int, item: int, element: str, unit: str, year: int, value: float
) -> list[str]:
    rm49, rname = COUNTRIES[reporter]
    pm49, pname = COUNTRIES[partner]
    cpc, iname = ITEMS[item]
    return [
        str(reporter),
        rm49,
        rname,
        str(partner),
        pm49,
        pname,
        str(item),
        cpc,
        iname,
        str(ELEMENTS[(element, unit)]),
        element,
        str(year),
        str(year),
        unit,
        f"{value:.6f}",
        "A",
    ]


# (reporter, partner, item, element, unit, year, value)
FAKE_ROWS: list[tuple[int, int, int, str, str, int, float]] = [
    # Wheat 2021, reported by both sides (they disagree).
    (185, 59, 15, "Export quantity", "t", 2021, 1000.0),
    (59, 185, 15, "Import quantity", "t", 2021, 900.0),
    (230, 59, 15, "Export quantity", "t", 2021, 500.0),
    (59, 230, 15, "Import quantity", "t", 2021, 450.0),
    (185, 223, 15, "Export quantity", "t", 2021, 800.0),
    # Wheat 2022: Russia stops reporting exports; Türkiye reports only what it exported.
    (59, 185, 15, "Import quantity", "t", 2022, 950.0),
    (223, 185, 15, "Import quantity", "t", 2022, 700.0),
    (230, 59, 15, "Export quantity", "t", 2022, 200.0),
    (223, 59, 15, "Export quantity", "t", 2022, 50.0),
    # A zero value (dropped) and a self-loop (kept in store, dropped by load).
    (230, 223, 15, "Export quantity", "t", 2022, 0.0),
    (223, 223, 15, "Export quantity", "t", 2022, 5.0),
    # Values.
    (185, 59, 15, "Export value", "1000 USD", 2021, 250.0),
    (59, 185, 15, "Import value", "1000 USD", 2021, 260.0),
    # Live animals: "An" becomes "head"; "1000 An" is multiplied by 1000.
    (230, 223, 1096, "Export quantity", "An", 2021, 12.0),
    (223, 230, 1096, "Import quantity", "1000 An", 2021, 0.5),
    (230, 59, 1181, "Export quantity", "No", 2021, 40.0),
]


def make_fake_zip(path: Path, rows: list[tuple[Any, ...]] | None = None) -> Path:
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL, lineterminator="\n")
    writer.writerow(RAW_HEADER)
    for r in rows if rows is not None else FAKE_ROWS:
        writer.writerow(raw_row(*r))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_registry.FAOSTAT_TRADE_CSV, buf.getvalue())
        zf.writestr(
            "Trade_DetailedTradeMatrix_E_Flags.csv", "Flag, Description\nA,Official figure\n"
        )
    return path


@pytest.fixture
def fake_zip(tmp_path: Path) -> Path:
    return make_fake_zip(tmp_path / "trade.zip")


@pytest.fixture
def fake_store(tmp_path: Path, fake_zip: Path) -> Path:
    cache = tmp_path / "cache"
    nv.datasets.faostat.build_store(cache, zip_path=fake_zip, known_hash=None)
    return cache


@pytest.fixture
def flows() -> pd.DataFrame:
    """A small hand-made flow frame with two years and one item."""
    return pd.DataFrame(
        {
            "exporter": ["A", "A", "B", "C", "A", "B", "C"],
            "importer": ["B", "C", "C", "A", "B", "A", "C"],
            "year": [2020, 2020, 2020, 2020, 2021, 2021, 2021],
            "item": ["x"] * 7,
            "quantity": [10.0, 5.0, 2.0, 1.0, 20.0, 4.0, 3.0],
            "unit": ["t"] * 7,
        }
    )


@pytest.fixture(scope="session")
def wheat_2022() -> pd.DataFrame:
    return nv.datasets.faostat.load_sample(items="Wheat", years=2022)


@pytest.fixture(scope="session")
def wheat_graph(wheat_2022: pd.DataFrame) -> nx.DiGraph[Any]:
    g = nv.build_graph(wheat_2022, node_attrs=nv.datasets.faostat.countries())
    assert isinstance(g, nx.DiGraph)
    return g
