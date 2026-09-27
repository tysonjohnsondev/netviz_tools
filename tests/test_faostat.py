from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import netviz_tools as nv
from netviz_tools import SchemaError, SourceHashMismatchError, StoreNotFoundError, UnknownItemError
from netviz_tools.datasets import _registry, faostat
from tests.conftest import make_fake_zip

# ---------------------------------------------------------------------------
# Bundled metadata


def test_items_catalogue() -> None:
    cat = faostat.items()
    assert len(cat) == 558
    assert cat["item_code"].is_unique
    assert cat["slug"].is_unique
    wheat = cat.set_index("item_code").loc[15]
    assert wheat["item"] == "Wheat"
    assert wheat["slug"] == "wheat"
    assert faostat.items("potatoes, frozen")["slug"].tolist() == ["potatoes_frozen"]
    assert faostat.items("potatoes_frozen")["item"].tolist() == ["Potatoes, frozen"]


def test_countries_metadata() -> None:
    c = faostat.countries()
    assert c.index.name == "name"
    assert c.index.is_unique
    assert set(c["continent"]) == set(nv.plot.CONTINENT_COLORS)
    assert c["lon"].between(-180, 180).all()
    assert c["lat"].between(-90, 90).all()
    assert c.loc["Republic of Korea", "iso3"] == "KOR"
    assert c.loc["Mexico", "continent"] == "Central America"
    assert c.loc["USSR", "region"] == "Europe"  # manual override
    assert c.loc["China, Taiwan Province of", "iso3"] == "TWN"


def test_metadata_sources_record_overrides() -> None:
    from importlib import resources

    text = resources.files("netviz_tools.datasets").joinpath("_meta", "sources.json").read_text()
    sources = json.loads(text)
    assert {s["name"].split()[0] for s in sources["sources"]} == {"FAOSTAT", "UNSD", "Natural"}
    assert all(o["note"] for o in sources["overrides"])


# ---------------------------------------------------------------------------
# Sample


def test_load_sample_schema_and_filters() -> None:
    df = faostat.load_sample(items="wheat", years=range(2021, 2023))
    nv.validate_flows(df)
    assert df.columns.tolist() == list(nv.FLOW_COLUMNS)
    assert sorted(df["time"].unique()) == [2021, 2022]
    assert set(df["category"]) == {"Wheat"}
    assert (df["source"] != df["target"]).all()
    assert set(df["unit"]) == {"t"}


def test_labels_attached_to_every_frame() -> None:
    assert dict(faostat.LABELS) == {
        "source": "Exporter",
        "target": "Importer",
        "time": "Year",
        "category": "Item",
        "weight": "Quantity",
        "node": "Country",
        "out": "Exports",
        "in": "Imports",
    }
    assert "LABELS" in faostat.__all__
    df = faostat.load_sample(items="Wheat", years=2022)
    assert df.attrs["labels"] == dict(faostat.LABELS)
    df.attrs["labels"]["source"] = "changed"
    assert faostat.LABELS["source"] == "Exporter"  # each frame gets its own copy
    assert faostat.load_sample(items="Wheat", years=2022).attrs["labels"]["source"] == "Exporter"
    g = nv.build_graph(faostat.load_sample(items="Wheat", years=2022))
    assert g.graph["labels"]["target"] == "Importer"


def test_load_sample_defaults_and_details() -> None:
    df = faostat.load_sample(details=True)
    assert set(df["category"]) == {"Wheat", "Maize (corn)", "Soya beans"}
    assert df["time"].min() == 2010
    assert set(df["reported_by"]) == {"importer"}
    assert df.columns.tolist() == [
        *nv.FLOW_COLUMNS,
        "item_code",
        "source_code",
        "target_code",
        "reported_by",
        "flag",
    ]


def test_raw_sample_layout() -> None:
    raw = faostat.raw_sample(items=["Wheat", "Maize (corn)"], years=[2021, 2022])
    assert raw.columns.tolist() == [
        "Reporter Country Code",
        "Reporter Countries",
        "Partner Country Code",
        "Partner Countries",
        "Item Code",
        "Item",
        "Element Code",
        "Element",
        "Year Code",
        "Year",
        "Unit",
        "Value",
        "Flag",
    ]
    assert set(raw["Year"]) == {2021, 2022}
    assert set(raw["Item"]) == {"Wheat", "Maize (corn)"}
    assert set(zip(raw["Element Code"], raw["Element"], strict=True)) == {
        (5910, "Export quantity"),
        (5610, "Import quantity"),
    }
    assert set(raw["Unit"]) == {"t"}
    # The Russian Federation reports wheat exports in 2021 and none in 2022.
    ru = raw[(raw["Reporter Countries"] == "Russian Federation") & (raw["Item"] == "Wheat")]
    assert set(ru.loc[ru["Element"] == "Export quantity", "Year"]) == {2021}
    assert len(faostat.raw_sample()) == len(pd.read_parquet(faostat._sample_path()))
    with pytest.raises(nv.UnknownItemError):
        faostat.raw_sample(items="Rice")


def test_reporter_perspectives_on_sample() -> None:
    def russia(reporter: faostat.Reporter) -> float:
        df = faostat.load_sample(items=15, years=2022, reporter=reporter)
        return float(df.loc[df.source == "Russian Federation", "weight"].sum())

    # The Russian Federation reports no wheat exports after 2021.
    assert russia("exporter") == 0.0
    assert russia("importer") > 1e7
    assert russia("combined") == pytest.approx(russia("importer"))
    combined = faostat.load_sample(items=15, years=2022, reporter="combined", details=True)
    assert set(combined["reported_by"]) == {"importer", "exporter"}
    keys = ["source", "target", "category", "time"]
    assert not combined.duplicated(keys).any()


def test_load_sample_self_loops() -> None:
    with_loops = faostat.load_sample(items="Wheat", self_loops=True)
    assert (with_loops["source"] == with_loops["target"]).any()


def test_load_sample_errors() -> None:
    with pytest.raises(UnknownItemError) as exc:
        faostat.load_sample(items="Rice")
    assert "Wheat" in exc.value.suggestions
    with pytest.raises(ValueError, match="reporter"):
        faostat.load_sample(reporter="both")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="years"):
        faostat.load_sample(years=[])
    with pytest.raises(ValueError, match="years"):
        faostat.load_sample(years=["2020"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="years"):
        faostat.load_sample(years=True)


def test_item_resolution() -> None:
    assert faostat._resolve_items(["Wheat", 15, "wheat", " WHEAT "]) == [(15, "Wheat")]
    assert faostat._resolve_items("potatoes_frozen")[0][1] == "Potatoes, frozen"
    with pytest.raises(UnknownItemError, match="Did you mean: 'Wheat'"):
        faostat._resolve_items("wheet")
    with pytest.raises(UnknownItemError):
        faostat._resolve_items(99999)
    with pytest.raises(UnknownItemError):
        faostat._resolve_items([True])
    with pytest.raises(ValueError, match="at least one"):
        faostat._resolve_items([])
    assert faostat._resolve_items(99999, extra={99999: "New thing"}) == [(99999, "New thing")]


# ---------------------------------------------------------------------------
# Store: build, manifest, load


def test_default_cache_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(faostat.ENV_CACHE_DIR, str(tmp_path / "x"))
    assert faostat.default_cache_dir() == tmp_path / "x"
    assert faostat.store_path() == tmp_path / "x" / "faostat_trade"
    monkeypatch.delenv(faostat.ENV_CACHE_DIR)
    # platformdirs appends "Cache" on Windows (...\netviz_tools\Cache)
    assert "netviz_tools" in faostat.default_cache_dir().parts
    assert not (tmp_path / "x").exists()


def test_store_missing(tmp_path: Path) -> None:
    with pytest.raises(StoreNotFoundError, match="build_store"):
        faostat.load("Wheat", cache_dir=tmp_path)
    with pytest.raises(StoreNotFoundError):
        faostat.store_info(tmp_path)
    assert isinstance(StoreNotFoundError("x"), FileNotFoundError)


def test_build_store_and_manifest(fake_store: Path, fake_zip: Path) -> None:
    info = faostat.store_info(fake_store)
    assert info["source_url"] == _registry.FAOSTAT_TRADE_URL
    assert info["source_sha256"] == faostat._sha256(fake_zip)
    assert info["matches_pinned"] is False
    assert info["source"] == "local file"
    assert info["netviz_tools_version"] == nv.__version__
    assert info["rows"] == 15  # 16 rows minus one zero value
    assert info["items"] == {"15": "Wheat", "1096": "Horses", "1181": "Bees"}
    assert {"retrieved_at", "built_at", "rows_by_measure", "licence"} <= set(info)
    parts = sorted(p.name for p in faostat.store_path(fake_store).iterdir())
    assert parts == ["item_code=1096", "item_code=1181", "item_code=15", "manifest.json"]
    assert not list(fake_store.glob(".build-*"))


def test_load_from_store(fake_store: Path) -> None:
    imp = faostat.load("Wheat", cache_dir=fake_store)
    nv.validate_flows(imp)
    assert imp.attrs["labels"]["weight"] == "Quantity"
    assert set(zip(imp.source, imp.target, imp.time, strict=True)) == {
        ("Russian Federation", "Egypt", 2021),
        ("Ukraine", "Egypt", 2021),
        ("Russian Federation", "Egypt", 2022),
        ("Russian Federation", "Türkiye", 2022),
    }
    exp = faostat.load("Wheat", 2022, reporter="exporter", cache_dir=fake_store)
    assert set(exp.source) == {"Ukraine", "Türkiye"}  # self-loop dropped, zero dropped
    comb = faostat.load(
        "Wheat", [2021, 2022], reporter="combined", cache_dir=fake_store, details=True
    )
    ru_eg_2021 = comb[
        (comb.source == "Russian Federation") & (comb.time == 2021) & (comb.target == "Egypt")
    ]
    assert ru_eg_2021["weight"].tolist() == [900.0]  # importer report wins
    assert ru_eg_2021["reported_by"].tolist() == ["importer"]
    ru_tr_2021 = comb[
        (comb.source == "Russian Federation") & (comb.target == "Türkiye") & (comb.time == 2021)
    ]
    assert ru_tr_2021["reported_by"].tolist() == ["exporter"]  # only the exporter reported
    loops = faostat.load("Wheat", 2022, reporter="exporter", self_loops=True, cache_dir=fake_store)
    assert ("Türkiye", "Türkiye") in set(zip(loops.source, loops.target, strict=True))
    ru_eg = comb[(comb.source == "Russian Federation") & (comb.target == "Egypt")]
    assert set(ru_eg["source_code"]) == {185}
    assert set(ru_eg["target_code"]) == {59}


def test_units_and_values_normalized(fake_store: Path) -> None:
    horses = faostat.load("Horses", reporter="combined", cache_dir=fake_store, details=True)
    by_reporter = horses.set_index("reported_by")
    assert set(horses["unit"]) == {"head"}
    assert by_reporter.loc["importer", "weight"] == 500.0  # 0.5 thousand head
    assert faostat.load("Bees", reporter="exporter", cache_dir=fake_store)["unit"].tolist() == [
        "number"
    ]
    value = faostat.load(15, 2021, measure="value", cache_dir=fake_store)
    assert value["unit"].tolist() == ["1000 USD"]
    assert value["weight"].tolist() == [260.0]
    assert value.attrs["labels"]["weight"] == "Value"
    assert faostat.LABELS["weight"] == "Quantity"
    with pytest.raises(ValueError, match="measure"):
        faostat.load(15, measure="price", cache_dir=fake_store)  # type: ignore[arg-type]


def test_known_item_without_rows_returns_empty(fake_store: Path) -> None:
    empty = faostat.load("Maize (corn)", cache_dir=fake_store)
    assert empty.empty
    assert empty.columns.tolist() == list(nv.FLOW_COLUMNS)
    assert empty.attrs["labels"] == dict(faostat.LABELS)


def test_build_store_reuses_existing(fake_store: Path, tmp_path: Path) -> None:
    before = faostat.store_info(fake_store)["built_at"]
    other = make_fake_zip(tmp_path / "other.zip", rows=[])
    faostat.build_store(fake_store, zip_path=other, known_hash=None)
    assert faostat.store_info(fake_store)["built_at"] == before
    faostat.build_store(fake_store, zip_path=other, known_hash=None, force=True, threads=1)
    assert faostat.store_info(fake_store)["rows"] == 0


def test_build_store_rejects_wrong_hash(tmp_path: Path, fake_zip: Path) -> None:
    with pytest.raises(SourceHashMismatchError) as exc:
        faostat.build_store(tmp_path / "c", zip_path=fake_zip)
    assert exc.value.expected == _registry.FAOSTAT_TRADE_SHA256
    assert exc.value.actual == faostat._sha256(fake_zip)
    assert "known_hash=None" in str(exc.value)
    assert not faostat.store_path(tmp_path / "c").exists()


def test_build_store_rejects_bad_csv(tmp_path: Path) -> None:
    import zipfile

    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr(_registry.FAOSTAT_TRADE_CSV, "a,b,Value,Flag\n1,2,3,A\n")
    with pytest.raises(SchemaError, match="missing columns"):
        faostat.build_store(tmp_path / "c", zip_path=bad, known_hash=None)
    assert not faostat.store_path(tmp_path / "c").exists()


def test_build_store_downloads_via_pooch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_zip: Path
) -> None:
    import pooch

    calls: list[dict[str, object]] = []

    def fake_retrieve(
        url: str, known_hash: str | None, fname: str, path: Path, progressbar: bool
    ) -> str:
        calls.append({"url": url, "known_hash": known_hash, "fname": fname, "path": path})
        return str(fake_zip)

    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    store = faostat.build_store(tmp_path / "cache", known_hash=None)
    assert calls[0]["url"] == _registry.FAOSTAT_TRADE_URL
    assert calls[0]["known_hash"] is None
    assert calls[0]["path"] == tmp_path / "cache"
    assert faostat.store_info(tmp_path / "cache")["source"] == "downloaded"
    assert store == faostat.store_path(tmp_path / "cache")
    faostat.download(tmp_path / "cache")
    assert calls[-1]["known_hash"] == f"sha256:{_registry.FAOSTAT_TRADE_SHA256}"


def test_download_translates_hash_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import pooch

    def mismatch(*_: object, **__: object) -> str:
        raise ValueError("SHA256 hash of downloaded file (x) does not match the known hash")

    monkeypatch.setattr(pooch, "retrieve", mismatch)
    with pytest.raises(SourceHashMismatchError, match="known_hash=None"):
        faostat.download(tmp_path)

    def other(*_: object, **__: object) -> str:
        raise ValueError("something else")

    monkeypatch.setattr(pooch, "retrieve", other)
    with pytest.raises(ValueError, match="something else"):
        faostat.download(tmp_path)


def test_citation_template() -> None:
    text = faostat.CITATION.format(year=2025, accessed="27 September 2026")
    assert text.endswith("Licence: CC-BY-4.0.")
    assert "https://www.fao.org/faostat/en/#data/TM" in text


def test_loaded_frames_are_plain_objects() -> None:
    df = faostat.load_sample(items="Wheat", years=2022)
    assert df["source"].dtype == object
    assert df["category"].dtype == object
    assert pd.api.types.is_integer_dtype(df["time"])


def test_connect_survives_progress_bar_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """DuckDB rejects the progress-bar setting in Jupyter without ipywidgets."""
    import duckdb

    real_connect = duckdb.connect

    class _Refusing:
        def __init__(self) -> None:
            self._con = real_connect(":memory:")

        def execute(self, query: str) -> object:
            if "enable_progress_bar" in query:
                raise duckdb.InvalidInputException("required package 'ipywidgets' is missing")
            return self._con.execute(query)

    monkeypatch.setattr(duckdb, "connect", lambda *a, **k: _Refusing())
    con = faostat._connect()
    assert con.execute("SELECT 1").fetchone() == (1,)
