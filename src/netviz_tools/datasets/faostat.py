"""FAOSTAT Detailed Trade Matrix: download, local Parquet store, and queries.

The Detailed Trade Matrix (FAOSTAT domain TM) records bilateral trade in about
560 agricultural products for roughly 220 countries and territories from 1986.
FAO publishes it as a single zip of about 420 MB (8.5 GB of CSV).

This module works in two steps:

1. :func:`build_store` downloads the zip once with :mod:`pooch`, checks its
   SHA-256 against the value pinned in this release, and converts it with
   DuckDB into a Parquet dataset partitioned by item. A ``manifest.json``
   written next to the data records the source URL, the SHA-256 actually used,
   the retrieval time, row counts and the library version.
2. :func:`load` runs a DuckDB query over only the partitions you ask for and
   returns a flow frame. Queries take well under a second per item.

:func:`load_sample` runs the same query over a small Parquet file that ships
with the package (wheat, maize and soya beans, 2010 to 2024), so the
examples and tests work offline.

Flow frames and labels
----------------------
The store keeps FAOSTAT's own vocabulary, but :func:`load` and
:func:`load_sample` return the library's generic flow frame
(:data:`netviz_tools.FLOW_COLUMNS`):

* ``source``: the exporting country;
* ``target``: the importing country;
* ``time``: the year;
* ``category``: the FAOSTAT item (commodity);
* ``weight``: the quantity, or the trade value with ``measure="value"``;
* ``unit``: ``t``, ``head``, ``number`` or ``1000 USD``.

Every returned frame also carries a copy of :data:`LABELS` in
``flows.attrs["labels"]`` (with ``"weight": "Value"`` for trade values).
:func:`netviz_tools.build_graph` copies it into ``G.graph["labels"]``, and the
plots use these display labels, so FAOSTAT charts say "Exporter" and
"Importer" without you having to type them.

Reporting perspective
---------------------
Every trade flow can be reported twice: by the exporting country and by the
importing country. The two reports often differ, and some countries stop
reporting (the Russian Federation reports no exports after 2021). ``load``
lets you choose:

* ``"importer"`` (default): flows as reported by the importing country. More
  countries report imports than exports (for wheat in 2021, 155 against 99).
* ``"exporter"``: flows as reported by the exporting country.
* ``"combined"``: the importer's report where one exists, otherwise the
  exporter's, decided per (exporter, importer, item, year).

The ``reporter`` parameter and the ``reported_by`` column of ``details=True``
keep the FAOSTAT words ``"importer"`` and ``"exporter"``.

Data licence
------------
FAOSTAT data are published under CC BY 4.0. Cite them as, for example:
"FAO. 2025. FAOSTAT: Detailed trade matrix. Accessed on 27 September 2026.
https://www.fao.org/faostat/en/#data/TM. Licence: CC-BY-4.0." Use
:data:`CITATION` for the template.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import functools
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, TypeAlias, get_args

import pandas as pd

from netviz_tools._schema import FlowFrame
from netviz_tools.analysis import suggest
from netviz_tools.datasets import _registry
from netviz_tools.errors import (
    SchemaError,
    SourceHashMismatchError,
    StoreNotFoundError,
    UnknownItemError,
)

if TYPE_CHECKING:
    import duckdb

__all__ = [
    "CITATION",
    "ENV_CACHE_DIR",
    "LABELS",
    "Measure",
    "Reporter",
    "build_store",
    "countries",
    "default_cache_dir",
    "download",
    "items",
    "load",
    "load_sample",
    "store_info",
    "store_path",
]

Reporter: TypeAlias = Literal["importer", "exporter", "combined"]
"""Whose report of each flow to use. See the module documentation."""

Measure: TypeAlias = Literal["quantity", "value"]
"""``"quantity"`` returns physical amounts (``t``, ``head`` or ``number``);
``"value"`` returns trade value in ``1000 USD``."""

ENV_CACHE_DIR: Final = "NETVIZ_TOOLS_CACHE"
"""Environment variable that overrides the default cache directory."""

CITATION: Final = (
    "FAO. {year}. FAOSTAT: Detailed trade matrix. Accessed on {accessed}. "
    "https://www.fao.org/faostat/en/#data/TM. Licence: CC-BY-4.0."
)
"""Citation template required by the FAO database terms of use."""

LABELS: Final[Mapping[str, str]] = {
    "source": "Exporter",
    "target": "Importer",
    "time": "Year",
    "category": "Item",
    "weight": "Quantity",
    "node": "Country",
    "out": "Exports",
    "in": "Imports",
}
"""Display labels for FAOSTAT trade flows, keyed by generic flow-frame term.

:func:`load` and :func:`load_sample` attach a copy to every frame as
``flows.attrs["labels"]`` (with ``"weight": "Value"`` when
``measure="value"``); graphs and plots pick them up from there."""

STORE_DIRNAME: Final = "faostat_trade"
SAMPLE_FILE: Final = "faostat_trade_sample.parquet"

_RAW_COLUMNS: Final = (
    "Reporter Country Code",
    "Reporter Countries",
    "Partner Country Code",
    "Partner Countries",
    "Item Code",
    "Item",
    "Element",
    "Year",
    "Unit",
    "Value",
    "Flag",
)

# Raw CSV -> store. Each row is re-oriented so that `exporter` is always the
# sending country, whoever reported it. Units are normalized: FAOSTAT's "An"
# (animals) becomes "head", "1000 An" is multiplied by 1000 and becomes "head",
# and "No" becomes "number". Zero and missing values are dropped.
_STORE_SELECT: Final = """
SELECT
    CASE WHEN is_export THEN reporter ELSE partner END AS exporter,
    CASE WHEN is_export THEN partner ELSE reporter END AS importer,
    CAST(year AS SMALLINT) AS year,
    item,
    CASE WHEN is_export THEN reporter_code ELSE partner_code END AS exporter_code,
    CASE WHEN is_export THEN partner_code ELSE reporter_code END AS importer_code,
    CASE WHEN is_export THEN 'exporter' ELSE 'importer' END AS reported_by,
    CASE WHEN is_value THEN 'value' ELSE 'quantity' END AS measure,
    CASE unit WHEN 'An' THEN 'head' WHEN '1000 An' THEN 'head' WHEN 'No' THEN 'number'
              ELSE unit END AS unit,
    CASE unit WHEN '1000 An' THEN value * 1000 ELSE value END AS value,
    flag,
    item_code
FROM (
    SELECT
        "Reporter Country Code"::SMALLINT AS reporter_code,
        "Reporter Countries" AS reporter,
        "Partner Country Code"::SMALLINT AS partner_code,
        "Partner Countries" AS partner,
        "Item Code"::INTEGER AS item_code,
        "Item" AS item,
        starts_with("Element", 'Export') AS is_export,
        ends_with("Element", 'value') AS is_value,
        "Year" AS year,
        "Unit" AS unit,
        "Value"::DOUBLE AS value,
        "Flag" AS flag
    FROM {source}
    WHERE "Value" > 0
)
"""


# ---------------------------------------------------------------------------
# Paths and bundled metadata


def default_cache_dir() -> Path:
    """Return the cache directory used when none is given.

    The ``NETVIZ_TOOLS_CACHE`` environment variable takes precedence; otherwise
    the operating system's user cache directory is used (via
    :func:`pooch.os_cache`). Nothing is created until data is downloaded.

    Returns
    -------
    pathlib.Path
        The cache directory.
    """
    env = os.environ.get(ENV_CACHE_DIR)
    if env:
        return Path(env).expanduser()
    import pooch

    return Path(pooch.os_cache("netviz_tools"))


def store_path(cache_dir: str | os.PathLike[str] | None = None) -> Path:
    """Return the directory that holds (or will hold) the Parquet store.

    Parameters
    ----------
    cache_dir
        Cache directory. Defaults to :func:`default_cache_dir`.

    Returns
    -------
    pathlib.Path
        ``<cache_dir>/faostat_trade``.
    """
    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    return base / STORE_DIRNAME


def _meta_file(name: str) -> Any:
    return resources.files("netviz_tools.datasets").joinpath("_meta", name)


@functools.cache
def _items_table() -> pd.DataFrame:
    with _meta_file("faostat_items.csv").open("rb") as f:
        return pd.read_csv(f, dtype={"cpc_code": "string"})


@functools.cache
def _countries_table() -> pd.DataFrame:
    with _meta_file("faostat_countries.csv").open("rb") as f:
        return pd.read_csv(f, dtype={"m49": "string", "iso3": "string"}, keep_default_na=False)


def items(search: str | None = None) -> pd.DataFrame:
    """Return the catalogue of FAOSTAT trade items.

    The catalogue ships with the package, so it works before any download.

    Parameters
    ----------
    search
        Optional case-insensitive substring to filter item names and slugs.

    Returns
    -------
    pandas.DataFrame
        Columns ``item_code``, ``item`` (the FAOSTAT name), ``slug`` (the
        snake_case name used by netviz_tools 0.x), ``cpc_code``,
        ``quantity_units``, ``first_year`` and ``last_year``.

    Examples
    --------
    >>> from netviz_tools.datasets import faostat
    >>> faostat.items("wheat")["item"].head(2).tolist()
    ['Wheat', 'Wheat and meslin flour']
    """
    table = _items_table().copy()
    if search:
        needle = search.lower()
        mask = table["item"].str.lower().str.contains(needle, regex=False) | table[
            "slug"
        ].str.contains(needle.replace(" ", "_"), regex=False)
        table = table[mask].reset_index(drop=True)
    return table


def countries() -> pd.DataFrame:
    """Return metadata for every FAOSTAT area in the trade matrix.

    Regions come from the UN M49 standard; label points (``lon``, ``lat``)
    come from Natural Earth. Pass the result to
    :func:`netviz_tools.build_graph` as ``node_attrs`` to colour nodes by
    continent.

    Returns
    -------
    pandas.DataFrame
        Indexed by FAOSTAT area ``name``, with columns ``fao_code``, ``m49``,
        ``iso3``, ``region``, ``subregion``, ``continent``, ``lon`` and
        ``lat``. ``continent`` is the M49 region with the Americas split into
        Northern America, Central America, the Caribbean and South America.
    """
    return _countries_table().set_index("name")


def _resolve_items(
    requested: str | int | Iterable[str | int], extra: dict[int, str] | None = None
) -> list[tuple[int, str]]:
    table = _items_table()
    by_code = dict(zip(table["item_code"], table["item"], strict=True))
    by_code.update(extra or {})
    by_name = {n.lower(): c for c, n in by_code.items()}
    by_slug = dict(zip(table["slug"], table["item_code"], strict=True))
    wanted = [requested] if isinstance(requested, str | int) else list(requested)
    if not wanted:
        raise ValueError("at least one item is required")
    out: list[tuple[int, str]] = []
    for w in wanted:
        if isinstance(w, bool):
            raise UnknownItemError(w)
        if isinstance(w, int):
            code = w if w in by_code else None
        else:
            key = w.strip()
            code = by_name.get(key.lower(), by_slug.get(key.lower()))
        if code is None:
            raise UnknownItemError(w, suggest(str(w), by_code.values()))
        if (code, by_code[code]) not in out:
            out.append((code, by_code[code]))
    return out


def _resolve_years(years: int | Iterable[int] | None) -> list[int] | None:
    if years is None:
        return None
    values: list[object] = [years] if isinstance(years, int) else list(years)
    if not values or not all(isinstance(y, int) and not isinstance(y, bool) for y in values):
        raise ValueError(f"years must be an int or a non-empty iterable of ints, got {years!r}")
    return sorted({int(y) for y in values})  # type: ignore[call-overload]


# ---------------------------------------------------------------------------
# Download and build


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    known_hash: str | None = _registry.FAOSTAT_TRADE_SHA256,
    progressbar: bool = False,
) -> Path:
    """Download the FAOSTAT bulk zip into the cache, or reuse a cached copy.

    Parameters
    ----------
    cache_dir
        Cache directory. Defaults to :func:`default_cache_dir`.
    known_hash
        Expected SHA-256. Defaults to the value pinned in this release. Pass
        ``None`` explicitly to accept whatever file FAO currently serves.
    progressbar
        Show a download progress bar (requires ``tqdm``).

    Returns
    -------
    pathlib.Path
        Path to the zip.

    Raises
    ------
    SourceHashMismatchError
        If the file does not match ``known_hash``.
    """
    import pooch

    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    try:
        path = pooch.retrieve(
            _registry.FAOSTAT_TRADE_URL,
            known_hash=f"sha256:{known_hash}" if known_hash else None,
            fname=_registry.FAOSTAT_TRADE_FILENAME,
            path=base,
            progressbar=progressbar,
        )
    except ValueError as exc:
        if "hash" not in str(exc).lower():
            raise
        raise SourceHashMismatchError(_registry.FAOSTAT_TRADE_URL, str(known_hash)) from exc
    return Path(path)


def _sql_str(value: str) -> str:
    """Quote a string as a SQL literal (for statements that cannot take parameters)."""
    return "'" + value.replace("'", "''") + "'"


def _connect(threads: int | None = None) -> duckdb.DuckDBPyConnection:
    import duckdb

    con = duckdb.connect(":memory:")
    # Inside Jupyter, DuckDB refuses this setting when ipywidgets is not
    # installed. The default connection still runs queries, so carry on.
    with contextlib.suppress(duckdb.InvalidInputException):
        con.execute("SET enable_progress_bar = false")
    if threads is not None:
        con.execute(f"SET threads = {int(threads)}")
    return con


def build_store(
    cache_dir: str | os.PathLike[str] | None = None,
    *,
    zip_path: str | os.PathLike[str] | None = None,
    known_hash: str | None = _registry.FAOSTAT_TRADE_SHA256,
    force: bool = False,
    progressbar: bool = False,
    threads: int | None = None,
) -> Path:
    """Build the local Parquet store from the FAOSTAT bulk zip.

    This is a one-time step. It needs network access (unless ``zip_path`` is
    given), about 9 GB of temporary disk space for the extracted CSV, and a
    minute or two on a laptop. The finished store is about 400 MB.

    Parameters
    ----------
    cache_dir
        Cache directory for the zip and the store. Defaults to
        :func:`default_cache_dir`.
    zip_path
        Use a zip you already have instead of downloading it. It is still
        checked against ``known_hash``.
    known_hash
        Expected SHA-256 of the zip. Defaults to the value pinned in this
        release. FAO replaces the file in place when it publishes an update;
        to build from a newer file, pass ``known_hash=None`` explicitly. The
        SHA-256 actually used is always written to the manifest.
    force
        Rebuild even if a store already exists.
    progressbar
        Show a download progress bar (requires ``tqdm``).
    threads
        Number of DuckDB threads. Defaults to all cores.

    Returns
    -------
    pathlib.Path
        The store directory.

    Raises
    ------
    SourceHashMismatchError
        If the zip does not match ``known_hash``.
    SchemaError
        If the CSV inside the zip does not have the expected columns.
    """
    from netviz_tools import __version__

    base = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    store = base / STORE_DIRNAME
    if (store / "manifest.json").exists() and not force:
        return store
    if zip_path is None:
        zpath = download(base, known_hash=known_hash, progressbar=progressbar)
        retrieved = "downloaded"
    else:
        zpath = Path(zip_path)
        retrieved = "local file"
    actual = _sha256(zpath)
    if known_hash is not None and actual != known_hash:
        raise SourceHashMismatchError(str(zpath), known_hash, actual)

    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=base, prefix=".build-") as tmp:
        tmpdir = Path(tmp)
        with zipfile.ZipFile(zpath) as zf:
            zf.extract(_registry.FAOSTAT_TRADE_CSV, tmpdir)
        csv = (tmpdir / _registry.FAOSTAT_TRADE_CSV).as_posix()
        con = _connect(threads)
        try:
            con.execute(
                f"CREATE VIEW raw AS SELECT * FROM read_csv({_sql_str(csv)}, header = true, "
                "types = {'Value': 'DOUBLE', 'Flag': 'VARCHAR'})"
            )
            cols = [r[0] for r in con.execute("DESCRIBE raw").fetchall()]
            missing = [c for c in _RAW_COLUMNS if c not in cols]
            if missing:
                raise SchemaError([f"FAOSTAT CSV is missing columns {missing}"])
            out = tmpdir / STORE_DIRNAME
            query = _STORE_SELECT.format(source="raw")
            con.execute(
                f"COPY ({query}) TO {_sql_str(out.as_posix())} "
                "(FORMAT parquet, PARTITION_BY (item_code), COMPRESSION zstd)"
            )
            out.mkdir(exist_ok=True)  # COPY writes nothing when there are no rows
            stats: list[tuple[Any, ...]] = []
            item_rows: list[tuple[Any, ...]] = []
            if any(out.glob("*/*.parquet")):
                pattern = _sql_str((out / "*" / "*.parquet").as_posix())
                source = f"read_parquet({pattern}, hive_partitioning = true)"
                stats = con.execute(
                    "SELECT measure, reported_by, count(*), min(year), max(year) "
                    f"FROM {source} GROUP BY ALL ORDER BY ALL"
                ).fetchall()
                item_rows = con.execute(
                    f"SELECT DISTINCT item_code, item FROM {source} ORDER BY item_code"
                ).fetchall()
        finally:
            con.close()
        mtime = dt.datetime.fromtimestamp(zpath.stat().st_mtime, tz=dt.UTC)
        manifest = {
            "dataset": "FAOSTAT Detailed Trade Matrix (TM)",
            "source_url": _registry.FAOSTAT_TRADE_URL,
            "source_sha256": actual,
            "pinned_sha256": _registry.FAOSTAT_TRADE_SHA256,
            "matches_pinned": actual == _registry.FAOSTAT_TRADE_SHA256,
            "source": retrieved,
            "retrieved_at": mtime.isoformat(timespec="seconds"),
            "built_at": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
            "netviz_tools_version": __version__,
            "licence": "CC BY 4.0",
            "rows": int(sum(r[2] for r in stats)),
            "rows_by_measure": [
                {
                    "measure": m,
                    "reported_by": rb,
                    "rows": int(n),
                    "first_year": int(y0),
                    "last_year": int(y1),
                }
                for m, rb, n, y0, y1 in stats
            ],
            "items": {str(code): name for code, name in item_rows},
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if store.exists():
            shutil.rmtree(store)
        out.replace(store)
    return store


def store_info(cache_dir: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Return the manifest of the local store.

    Parameters
    ----------
    cache_dir
        Cache directory. Defaults to :func:`default_cache_dir`.

    Returns
    -------
    dict
        The parsed ``manifest.json``.

    Raises
    ------
    StoreNotFoundError
        If the store has not been built.
    """
    manifest = store_path(cache_dir) / "manifest.json"
    if not manifest.exists():
        raise StoreNotFoundError(_missing_store_message(manifest.parent))
    data: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    return data


def _missing_store_message(store: Path) -> str:
    return (
        f"no FAOSTAT store at {store}. Build it once with "
        "netviz_tools.datasets.faostat.build_store() (downloads about 420 MB), "
        "or use load_sample() for the bundled sample."
    )


# ---------------------------------------------------------------------------
# Queries


def _query(
    source_sql: str,
    params: list[Any],
    codes: Sequence[int],
    years: Sequence[int] | None,
    reporter: Reporter,
    measure: Measure,
    self_loops: bool,
    details: bool,
) -> FlowFrame:
    if reporter not in get_args(Reporter):
        raise ValueError(f"reporter must be one of {list(get_args(Reporter))}, not {reporter!r}")
    if measure not in get_args(Measure):
        raise ValueError(f"measure must be one of {list(get_args(Measure))}, not {measure!r}")
    where = ["measure = ?", "list_contains(?, item_code)"]
    args: list[Any] = [*params, measure, list(codes)]
    if years is not None:
        where.append("list_contains(?, year)")
        args.append(list(years))
    if not self_loops:
        where.append("exporter_code <> importer_code")
    base = f"SELECT * FROM {source_sql} WHERE " + " AND ".join(where)
    if reporter == "combined":
        body = f"""
            WITH base AS ({base}),
            imp AS (SELECT * FROM base WHERE reported_by = 'importer'),
            exp AS (SELECT * FROM base WHERE reported_by = 'exporter')
            SELECT * FROM imp
            UNION ALL
            SELECT exp.* FROM exp
            ANTI JOIN imp USING (exporter_code, importer_code, item_code, year)
        """
    else:
        body = f"{base} AND reported_by = ?"
        args.append(reporter)
    extra = (
        ", item_code, exporter_code AS source_code, importer_code AS target_code, reported_by, flag"
        if details
        else ""
    )
    # The store keeps FAOSTAT names; the result uses the generic flow-frame names.
    sql = (
        "SELECT exporter AS source, importer AS target, CAST(year AS BIGINT) AS time, "
        f"item AS category, value AS weight, unit{extra} FROM ({body}) "
        "ORDER BY category, time, source, target"
    )
    con = _connect()
    try:
        df = con.execute(sql, args).df()
    finally:
        con.close()
    for col in ("source", "target", "category", "unit", "reported_by", "flag"):
        if col in df.columns:
            df[col] = df[col].astype(object)
    labels = dict(LABELS)
    if measure == "value":
        labels["weight"] = "Value"
    df.attrs["labels"] = labels
    return df


def load(
    items: str | int | Iterable[str | int],
    years: int | Iterable[int] | None = None,
    *,
    reporter: Reporter = "importer",
    measure: Measure = "quantity",
    self_loops: bool = False,
    details: bool = False,
    cache_dir: str | os.PathLike[str] | None = None,
) -> FlowFrame:
    """Load flows for some items and years from the local store.

    Parameters
    ----------
    items
        One or more items, each given as a FAOSTAT item code (``15``), a
        FAOSTAT name (``"Wheat"``, case-insensitive), or a 0.x slug
        (``"potatoes_frozen"``). See :func:`items` for the catalogue.
    years
        A year, an iterable of years (such as ``range(2015, 2023)``), or
        ``None`` for all years.
    reporter
        ``"importer"`` (default), ``"exporter"`` or ``"combined"``. See the
        module documentation.
    measure
        ``"quantity"`` (default) or ``"value"`` (``1000 USD``).
    self_loops
        Keep flows reported between a country and itself (re-imports).
    details
        Add ``item_code``, ``source_code`` and ``target_code`` (FAOSTAT area
        codes), ``reported_by`` (``"exporter"`` or ``"importer"``) and
        ``flag`` columns.
    cache_dir
        Cache directory that holds the store.

    Returns
    -------
    FlowFrame
        Columns ``source`` (exporter), ``target`` (importer), ``time``
        (year), ``category`` (item), ``weight`` (quantity or value) and
        ``unit``, sorted by category, time, source and target.
        ``flows.attrs["labels"]`` holds a copy of :data:`LABELS`, which plots
        use for axis and hover labels.

    Raises
    ------
    UnknownItemError
        If an item is not in the catalogue. The error suggests close matches.
    StoreNotFoundError
        If :func:`build_store` has not been run.
    """
    store = store_path(cache_dir)
    if not (store / "manifest.json").exists():
        raise StoreNotFoundError(_missing_store_message(store))
    extra = {int(k): str(v) for k, v in store_info(cache_dir).get("items", {}).items()}
    resolved = _resolve_items(items, extra)
    yrs = _resolve_years(years)
    files = [
        f.as_posix()
        for code, _ in resolved
        for f in sorted((store / f"item_code={code}").glob("*.parquet"))
    ]
    codes = [code for code, _ in resolved]
    if not files:
        # Known item without rows in this release: same query, no data.
        source = "(SELECT * FROM read_parquet(?) LIMIT 0)"
        return _query(source, [_sample_path()], codes, yrs, reporter, measure, self_loops, details)
    return _query(
        "read_parquet(?, hive_partitioning = true)",
        [files],
        codes,
        yrs,
        reporter,
        measure,
        self_loops,
        details,
    )


def _sample_path() -> str:
    return str(resources.files("netviz_tools.datasets").joinpath("sample", SAMPLE_FILE))


def load_sample(
    items: str | int | Iterable[str | int] | None = None,
    years: int | Iterable[int] | None = None,
    *,
    reporter: Reporter = "importer",
    self_loops: bool = False,
    details: bool = False,
) -> FlowFrame:
    """Load flows from the small sample that ships with the package.

    The sample holds quantity flows in tonnes (both reporting perspectives)
    for Wheat, Maize (corn) and Soya beans from 2010 to 2024, taken from the
    pinned FAOSTAT release (about 140,000 rows, 0.6 MB). It uses the same
    query as :func:`load`.

    Parameters
    ----------
    items
        Items to keep; defaults to all three sample items.
    years
        Years to keep; defaults to all.
    reporter, self_loops, details
        As for :func:`load`.

    Returns
    -------
    FlowFrame
        The same columns as :func:`load`: ``source`` (exporter), ``target``
        (importer), ``time`` (year), ``category`` (item), ``weight``
        (quantity) and ``unit``, sorted by category, time, source and target.
        ``flows.attrs["labels"]`` holds a copy of :data:`LABELS`, which plots
        use for axis and hover labels.

    Raises
    ------
    UnknownItemError
        If an item is not in the catalogue, or is in the catalogue but not in
        the sample.

    Examples
    --------
    >>> from netviz_tools.datasets import faostat
    >>> flows = faostat.load_sample(items="Wheat", years=2022)
    >>> flows.columns.tolist()
    ['source', 'target', 'time', 'category', 'weight', 'unit']
    >>> flows.attrs["labels"]["source"], flows.attrs["labels"]["target"]
    ('Exporter', 'Importer')
    """
    sample_items = _sample_items()
    if items is None:
        resolved = sample_items
    else:
        resolved = _resolve_items(items)
        absent = [name for code, name in resolved if (code, name) not in sample_items]
        if absent:
            raise UnknownItemError(absent[0], [name for _, name in sample_items])
    return _query(
        "read_parquet(?)",
        [_sample_path()],
        [code for code, _ in resolved],
        _resolve_years(years),
        reporter,
        "quantity",
        self_loops,
        details,
    )


@functools.cache
def _sample_items() -> list[tuple[int, str]]:
    con = _connect()
    try:
        rows = con.execute(
            "SELECT DISTINCT item_code, item FROM read_parquet(?) ORDER BY item", [_sample_path()]
        ).fetchall()
    finally:
        con.close()
    return [(int(c), str(n)) for c, n in rows]
