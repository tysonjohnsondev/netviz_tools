"""Clean messy edge and flow tables into a flow frame.

:func:`clean` is the first stage of the pipeline: it takes a table as it
arrives (a CSV export, a statistics office download, a hand-built sheet),
works out which column holds what, and returns a flow frame together with a
:class:`CleaningReport` that records every change: which columns were used,
how many rows each step removed and why, which names were rewritten, and how
well the two reports of each flow agree when both sides reported it.

The steps run in this order:

1. exact duplicates: rows identical in every raw column are dropped;
2. missing values: rows without a source, target (or reporter, partner),
   category or unit are dropped;
3. direction (reporter tables) or reported_by: each row becomes a flow from
   source to target;
4. weight, negative weight and time parsing;
5. unit conversions;
6. names: whitespace and Unicode normalization, ``aliases``, then merging of
   spellings that differ only in case, accents or punctuation;
7. drop nodes, zeros and self-loops;
8. duplicates: rows that share a key are summed or kept;
9. mirror flows: when both the source and the target reported a flow, one
   report is chosen or the two are combined.
"""

from __future__ import annotations

import difflib
import math
import re
import unicodedata
from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd

from netviz_tools._schema import FLOW_COLUMNS, FlowFrame, validate_flows
from netviz_tools.errors import SchemaError

__all__ = ["CleaningReport", "CleaningStep", "MirrorStats", "clean"]

Prefer: TypeAlias = Literal["target", "source", "mean", "max"]
"""How to resolve a flow reported by both sides."""

Direction: TypeAlias = Literal["export", "import"]

_SCHEMA_ROLES: Final = ("source", "target", "time", "category", "weight", "unit")
_COLUMN: Final[dict[str, str]] = dict(zip(_SCHEMA_ROLES, FLOW_COLUMNS, strict=True))
"""Role -> output column. The flow schema lists its columns in role order."""

BY_SOURCE: Final = "exporter"
"""``reported_by`` value of a flow reported by its source."""
BY_TARGET: Final = "importer"
"""``reported_by`` value of a flow reported by its target."""
BY_BOTH: Final = "both"
"""``reported_by`` value of a flow combined from both reports."""

_ROLES: Final = (
    "source",
    "target",
    "reporter",
    "partner",
    "direction",
    "reported_by",
    "time",
    "category",
    "weight",
    "unit",
)
_FILLABLE: Final = ("time", "category", "weight", "unit")
_EDGE_ROLES: Final = ("source", "target")
_REPORTER_ROLES: Final = ("reporter", "partner", "direction")
_TEXT_ROLES: Final = ("source", "target", "reporter", "partner", "category", "unit")
_CANDIDATES: Final[dict[str, tuple[str, ...]]] = {
    "source": (
        *("source", "exporter", "exporter country", "exporting country"),
        *("origin", "from", "src", "sender"),
    ),
    "target": (
        *("target", "importer", "importer country", "importing country"),
        *("destination", "dest", "to", "dst", "receiver"),
    ),
    "reporter": ("reporter", "reporter countries", "reporter country", "reporting country"),
    "partner": ("partner", "partner countries", "partner country"),
    "direction": ("direction", "element", "flow type", "trade flow"),
    "reported_by": ("reported by",),
    "time": ("time", "year", "period"),
    "category": ("category", "item", "commodity", "product"),
    "weight": ("weight", "quantity", "value", "amount", "flow", "volume", "count"),
    "unit": ("unit", "units"),
}
_SKIP_MARKERS: Final = ("code", "(m49)", "flag")
_THOUSANDS: Final = re.compile(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?")
_FUZZY_CUTOFF: Final = 0.9
_LIST_LIMIT: Final = 20

ObjArray: TypeAlias = npt.NDArray[np.object_]


# ---------------------------------------------------------------------------
# Report types


@dataclass(frozen=True)
class CleaningStep:
    """One step of :func:`clean` and its effect on the row count.

    Parameters
    ----------
    name
        Short name of the step, for example ``"zeros"``.
    rows_in
        Rows before the step.
    rows_out
        Rows after the step. Steps that aggregate (duplicates, mirror flows)
        reduce the count without discarding weight.
    detail
        What the step did, in plain words. May be empty.
    """

    name: str
    rows_in: int
    rows_out: int
    detail: str = ""


@dataclass(frozen=True)
class MirrorStats:
    """How often, and how closely, the two sides of a flow agree.

    A flow is identified by (source, target, time, category, unit). In a
    reporter-based trade table it can be reported by its source (the
    exporter), by its target (the importer), or by both.

    Parameters
    ----------
    both
        Flows reported by both sides.
    source_only
        Flows reported only by their source.
    target_only
        Flows reported only by their target.
    median_disagreement
        Median over the ``both`` flows of ``|s - t| / max(s, t)``, where ``s``
        and ``t`` are the two reported weights (0 when both are 0). A value of
        0.1 means that for the typical flow the smaller report is 10% below
        the larger one. ``None`` when no flow was reported by both sides.
    prefer
        The ``prefer`` setting used to resolve the ``both`` flows.
    """

    both: int
    source_only: int
    target_only: int
    median_disagreement: float | None
    prefer: Prefer


@dataclass(frozen=True, eq=False)
class CleaningReport:
    """Everything :func:`clean` did to a table.

    Parameters
    ----------
    columns
        Raw column name to role, for every column that was used.
    guessed
        Roles whose column was guessed from the header names. The other roles
        in ``columns`` were given by the caller.
    filled
        Roles filled with a constant (the ``fill`` argument).
    rows_in
        Rows in the raw table.
    rows_out
        Rows in the returned flow frame.
    steps
        Each step with its row counts, in the order they ran.
    renamed
        Original spelling to final name, for source, target and category
        names that changed (whitespace, ``aliases``, or merged variants).
    possible_aliases
        Pairs of distinct final names that look alike (``difflib`` ratio of
        at least 0.9) as ``(name_a, name_b, ratio)``. They are not merged;
        pass them in ``aliases`` if they are the same thing.
    mirror
        Agreement between source and target reports, or ``None`` when the
        table does not say who reported each flow.
    units
        For each category in the output, the number of rows per unit.
    dropped
        The discarded raw rows, with their original index and columns plus a
        ``reason`` column. Rows that were summed or combined into another row
        are not listed here, because their weight is kept.
    """

    columns: dict[str, str]
    guessed: tuple[str, ...]
    filled: dict[str, str | int | float]
    rows_in: int
    rows_out: int
    steps: tuple[CleaningStep, ...]
    renamed: dict[str, str]
    possible_aliases: tuple[tuple[str, str, float], ...]
    mirror: MirrorStats | None
    units: dict[str, dict[str, int]]
    dropped: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        """Return the steps as a table.

        Returns
        -------
        pandas.DataFrame
            Columns ``step``, ``rows_in``, ``rows_out``, ``removed`` and
            ``detail``, one row per step.
        """
        return pd.DataFrame(
            {
                "step": [s.name for s in self.steps],
                "rows_in": [s.rows_in for s in self.steps],
                "rows_out": [s.rows_out for s in self.steps],
                "removed": [s.rows_in - s.rows_out for s in self.steps],
                "detail": [s.detail for s in self.steps],
            }
        )

    def summary(self) -> str:
        """Describe the cleaning in plain English, one line per fact.

        Returns
        -------
        str
            A multi-line summary: the row counts, the columns used and the
            constants filled in, one line per step, then mirror statistics,
            renames, possible aliases and categories with more than one unit.
        """
        lines = [
            f"Cleaned {self.rows_in:,} raw rows into {self.rows_out:,} flows.",
            "Columns: "
            + ", ".join(
                f"{col!r} -> {role} ({'guessed' if role in self.guessed else 'given'})"
                for col, role in self.columns.items()
            ),
        ]
        if self.filled:
            lines.append("Filled: " + ", ".join(f"{r} = {v!r}" for r, v in self.filled.items()))
        for s in self.steps:
            line = f"- {s.name}: {s.rows_in:,} -> {s.rows_out:,} rows"
            lines.append(line + (f" ({s.detail})" if s.detail else ""))
        if self.mirror is not None:
            lines.append(_mirror_line(self.mirror))
        if self.renamed:
            lines.append(f"Renamed {len(self.renamed):,} spellings:")
            lines += _capped([f"  {a!r} -> {b!r}" for a, b in self.renamed.items()])
        if self.possible_aliases:
            lines.append(
                f"{len(self.possible_aliases):,} pairs of names look alike but were not merged "
                "(pass aliases= to merge them):"
            )
            lines += _capped([f"  {a!r} ~ {b!r} ({r:.2f})" for a, b, r in self.possible_aliases])
        mixed = {cat: u for cat, u in self.units.items() if len(u) > 1}
        if mixed:
            lines.append("Categories with more than one unit (build_graph will not mix them):")
            lines += _capped(
                [
                    f"  {cat!r}: " + ", ".join(f"{u} ({n:,} rows)" for u, n in units.items())
                    for cat, units in mixed.items()
                ]
            )
        return "\n".join(lines)


def _capped(lines: list[str]) -> list[str]:
    if len(lines) <= _LIST_LIMIT:
        return lines
    return [*lines[:_LIST_LIMIT], f"  ... and {len(lines) - _LIST_LIMIT:,} more"]


def _mirror_line(m: MirrorStats) -> str:
    policy = {
        "target": "used the target's report where both exist",
        "source": "used the source's report where both exist",
        "mean": "used the mean of the two reports where both exist",
        "max": "used the larger of the two reports where both exist",
    }[m.prefer]
    line = (
        f"Mirror flows: {m.both:,} reported by both sides, {m.source_only:,} by the "
        f"source only, {m.target_only:,} by the target only"
    )
    if m.median_disagreement is not None:
        line += f"; median disagreement {m.median_disagreement:.1%}"
    return f"{line}; {policy}."


# ---------------------------------------------------------------------------
# Small helpers


def _header_key(name: object) -> str:
    return re.sub(r"[\s_]+", " ", str(name).casefold().strip())


def _norm_text(value: object) -> str | None:
    """NFC, strip and collapse whitespace; ``None`` if nothing is left.

    Callers pass non-missing values only (see :func:`_map_unique`).
    """
    text = value if isinstance(value, str) else str(value)
    return " ".join(unicodedata.normalize("NFC", text).split()) or None


def _match_key(name: str) -> str:
    """Casefold, strip accents, keep only letters and digits."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    return "".join(ch for ch in decomposed if ch.isalnum() and not unicodedata.combining(ch))


def _map_unique(values: pd.Series[Any], func: Callable[[Any], object]) -> ObjArray:
    """Apply ``func`` once per distinct non-missing value; missing values map to ``None``."""
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    mapped = np.empty(len(uniques) + 1, dtype=object)
    mapped[:-1] = [func(u) for u in uniques]
    mapped[-1] = None
    out: ObjArray = mapped[codes]
    return out


def _replace(values: pd.Series[Any], mapping: Mapping[str, str]) -> ObjArray:
    """Replace values found in ``mapping``; others are kept."""
    return _map_unique(values, lambda v: mapping.get(v, v))


def _parse_number(value: object) -> tuple[float, str | None]:
    """Parse one non-missing weight or time cell. Returns (value, problem)."""
    if isinstance(value, bool | np.bool_):
        return math.nan, "unparseable"
    text: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return math.nan, "missing"
        if _THOUSANDS.fullmatch(text):
            text = text.replace(",", "")
    try:
        number = float(text)
    except (TypeError, ValueError):
        return math.nan, "unparseable"
    if math.isnan(number):
        return number, "missing"
    if math.isinf(number):
        return number, "infinite"
    return number, None


def _parse_column(values: pd.Series[Any]) -> tuple[npt.NDArray[np.float64], ObjArray]:
    """Parse a column of numbers. Returns float values and problems (``None`` if fine)."""
    if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
        numbers = values.to_numpy(dtype=float, na_value=np.nan)
        problems = np.full(len(numbers), None, dtype=object)
        problems[np.isnan(numbers)] = "missing"
        problems[np.isinf(numbers)] = "infinite"
        return numbers, problems
    parsed = _map_unique(values, _parse_number)
    numbers = np.array([math.nan if p is None else p[0] for p in parsed], dtype=float)
    problems = np.array(["missing" if p is None else p[1] for p in parsed], dtype=object)
    return numbers, problems


def _classify_direction(value: object) -> Direction | None:
    text = str(value).casefold()
    is_export, is_import = "export" in text, "import" in text
    if is_export == is_import:
        return None
    return "export" if is_export else "import"


def _near_matches(names: Iterable[str]) -> list[tuple[str, str, float]]:
    """Pairs of names whose casefolded ``difflib`` ratio is at least the cutoff.

    Candidates are pruned with a vectorized upper bound on the ratio (shared
    character counts, as in :meth:`difflib.SequenceMatcher.quick_ratio`) so a
    few thousand names take a fraction of a second.
    """
    c = _FUZZY_CUTOFF
    pool = sorted(set(names), key=lambda s: (len(s), s))
    folded = [s.casefold() for s in pool]
    lens = np.array([len(s) for s in folded])
    counts = np.zeros((len(pool), 64), dtype=np.int32)
    for row, s in enumerate(folded):
        for ch in s:
            counts[row, ord(ch) % 64] += 1
    out: list[tuple[str, str, float]] = []
    matcher = difflib.SequenceMatcher(autojunk=False)
    for i in range(len(pool)):
        hi = int(np.searchsorted(lens, lens[i] * (2 - c) / c, side="right"))
        js = np.arange(i + 1, hi)
        if not len(js):
            continue
        bound = 2 * np.minimum(counts[i], counts[js]).sum(axis=1) / (lens[i] + lens[js])
        matcher.set_seq2(folded[i])
        for j in js[bound >= c]:
            matcher.set_seq1(folded[j])
            ratio = matcher.ratio()
            if ratio >= c:
                a, b = sorted((pool[i], pool[j]))
                out.append((a, b, round(ratio, 3)))
    out.sort(key=lambda t: (-t[2], t[0], t[1]))
    return out


def _variant_merges(counts: pd.Series[int]) -> dict[str, str]:
    """Map spellings that share a match key to the most frequent one."""
    groups: dict[str, list[str]] = {}
    for name in counts.index:
        key = _match_key(name)
        if key:
            groups.setdefault(key, []).append(name)
    merges: dict[str, str] = {}
    for spellings in groups.values():
        if len(spellings) > 1:
            best = min(spellings, key=lambda s: (-int(counts[s]), s))
            merges.update({s: best for s in spellings if s != best})
    return merges


# ---------------------------------------------------------------------------
# Column mapping


def _resolve_columns(
    raw: pd.DataFrame, given: Mapping[str, Hashable], filled: Iterable[str]
) -> tuple[dict[Hashable, str], tuple[str, ...]]:
    """Return raw column -> role for the columns to use, and the guessed roles."""
    found = [str(c) for c in raw.columns]
    hint = (
        f"Columns found: {found}. Name the columns with keyword arguments, for example "
        "source='origin', target='dest', weight='n'."
    )
    if not raw.columns.is_unique:
        dupes = sorted({str(c) for c in raw.columns[raw.columns.duplicated()]})
        raise SchemaError([f"duplicate column names {dupes}"])
    filled = set(filled)
    problems = [f"{r}={c!r} is not a column of the table" for r, c in given.items() if c not in raw]
    cols = list(given.values())
    problems += sorted(
        {f"column {c!r} is given for more than one role" for c in cols if cols.count(c) > 1}
    )
    problems += [f"{r} is given both as a column and in fill" for r in given if r in filled]
    if given.keys() & set(_EDGE_ROLES) and given.keys() & set(_REPORTER_ROLES):
        problems.append("give source and target, or reporter, partner and direction, not both")
    if "reported_by" in given and given.keys() & set(_REPORTER_ROLES):
        problems.append("reported_by only applies to source/target tables")
    if problems:
        raise SchemaError([*problems, hint])

    taken = set(cols)
    matches: dict[str, list[Hashable]] = {r: [] for r in _ROLES}
    for col in raw.columns:
        key = _header_key(col)
        if col in taken or any(m in key for m in _SKIP_MARKERS):
            continue
        for role, names in _CANDIDATES.items():
            if key in names and role not in given and role not in filled:
                matches[role].append(col)

    def available(role: str) -> bool:
        return role in given or bool(matches[role])

    if given.keys() & {*_EDGE_ROLES, "reported_by"}:
        edge = True
    elif given.keys() & set(_REPORTER_ROLES):
        edge = False
    else:
        edge = all(available(r) for r in _EDGE_ROLES)
        if edge and all(available(r) for r in ("reporter", "partner")):
            raise SchemaError(
                ["the table has both source/target and reporter/partner columns. " + hint]
            )
        if not edge and not all(available(r) for r in ("reporter", "partner")):
            raise SchemaError(
                [
                    "could not tell which columns hold the source and target "
                    "(or the reporter, partner and direction). " + hint
                ]
            )
    unused = set(_REPORTER_ROLES) if edge else {*_EDGE_ROLES, "reported_by"}
    candidates = {r: m for r, m in matches.items() if m and r not in unused}
    ambiguous = [f"{r}: {[str(c) for c in m]}" for r, m in candidates.items() if len(m) > 1]
    if ambiguous:
        raise SchemaError(["more than one column could be " + "; ".join(ambiguous) + ". " + hint])
    mapping: dict[Hashable, str] = {c: r for r, c in given.items()}
    mapping.update({m[0]: r for r, m in candidates.items()})
    order = {r: i for i, r in enumerate(_ROLES)}
    mapping = dict(sorted(mapping.items(), key=lambda kv: order[kv[1]]))

    roles = set(mapping.values())
    required = _EDGE_ROLES if edge else _REPORTER_ROLES
    problems = [f"no column for {r!r}; pass {r}='column name'" for r in required if r not in roles]
    problems += [
        f"no column for {r!r}; pass {r}='column name' or fill={{{r!r}: ...}}"
        for r in _FILLABLE
        if r not in roles and r not in filled
    ]
    if problems:
        raise SchemaError([*problems, hint])
    guessed = tuple(r for r in _ROLES if r in candidates)
    return mapping, guessed


def _check_fill(fill: Mapping[str, object]) -> dict[str, str | int | float]:
    """Validate constants; text is normalized, time becomes int, weight float."""
    out: dict[str, str | int | float] = {}
    for role, value in fill.items():
        if role not in _FILLABLE:
            raise SchemaError([f"fill accepts the roles {list(_FILLABLE)}, not {role!r}"])
        num: float | None = None
        if isinstance(value, int | float | np.integer | np.floating) and not isinstance(
            value, bool
        ):
            num = float(value)
        text = _norm_text(value) if isinstance(value, str) else None
        if role == "time" and num is not None and num.is_integer():
            out[role] = int(num)
        elif role == "weight" and num is not None and 0 <= num < math.inf:
            out[role] = num
        elif role in ("category", "unit") and text is not None:
            out[role] = text
        else:
            raise ValueError(f"fill[{role!r}] is not a valid {role}: {value!r}")
    return out


# ---------------------------------------------------------------------------
# Row bookkeeping


class _Log:
    """Collects steps and the raw positions of dropped rows."""

    def __init__(self) -> None:
        self.steps: list[CleaningStep] = []
        self.rows: list[npt.NDArray[np.int64]] = []
        self.reasons: list[ObjArray] = []

    def note(self, name: str, rows_in: int, rows_out: int, detail: str = "") -> None:
        self.steps.append(CleaningStep(name, rows_in, rows_out, detail))

    def drop_rows(self, positions: npt.NDArray[np.int64], reasons: ObjArray) -> None:
        self.rows.append(positions)
        self.reasons.append(reasons)

    def filter(
        self, work: pd.DataFrame, name: str, reasons: ObjArray, detail: str = ""
    ) -> pd.DataFrame:
        """Drop the rows of ``work`` whose reason is not ``None`` and log the step."""
        bad = np.array([r is not None for r in reasons], dtype=bool)
        self.drop_rows(work["_row"].to_numpy()[bad], reasons[bad])
        out = work.loc[~bad].reset_index(drop=True)
        self.note(name, len(work), len(out), detail)
        return out

    def dropped(self, raw: pd.DataFrame) -> pd.DataFrame:
        positions = np.concatenate([np.zeros(0, dtype=np.int64), *self.rows])
        reasons = np.concatenate([np.zeros(0, dtype=object), *self.reasons])
        order = np.argsort(positions, kind="stable")
        out = raw.iloc[positions[order]].copy()
        out["reason"] = reasons[order]
        return out


def _count_detail(reasons: ObjArray) -> str:
    counts = pd.Series(reasons[[r is not None for r in reasons]], dtype=object).value_counts()
    return ", ".join(f"{n:,} {reason}" for reason, n in sorted(counts.items()))


def _where(mask: npt.NDArray[np.bool_], reason: str) -> ObjArray:
    out = np.full(len(mask), None, dtype=object)
    out[mask] = reason
    return out


# ---------------------------------------------------------------------------
# Main entry point


def clean(
    raw: pd.DataFrame,
    *,
    source: str | None = None,
    target: str | None = None,
    weight: str | None = None,
    time: str | None = None,
    category: str | None = None,
    unit: str | None = None,
    reporter: str | None = None,
    partner: str | None = None,
    direction: str | None = None,
    reported_by: str | None = None,
    fill: Mapping[str, object] | None = None,
    direction_values: Mapping[str, Direction] | None = None,
    prefer: Prefer = "target",
    aliases: Mapping[str, str] | None = None,
    unit_conversions: Mapping[str, tuple[str, float]] | None = None,
    drop_nodes: Iterable[str] = (),
    zeros: Literal["drop", "keep"] = "drop",
    self_loops: bool = False,
    duplicates: Literal["sum", "keep"] = "sum",
    labels: Mapping[str, str] | None = None,
) -> tuple[FlowFrame, CleaningReport]:
    """Turn a messy edge or trade table into a flow frame, and report every change.

    Parameters
    ----------
    raw
        The table as it arrived. It is not modified.
    source, target, weight, time, category, unit
        Names of the raw columns that hold each role. Any left as ``None``
        are guessed from the header names (case, spaces and underscores are
        ignored; columns whose names contain "code", "(m49)" or "flag" are
        skipped). For example ``origin``, ``exporter`` or ``from`` is taken
        as the source, and ``value``, ``quantity`` or ``count`` as the weight.
    reporter, partner, direction
        For trade tables where each row is one country's report: the
        reporting country, the partner country, and the column that says
        whether the row is an export or an import. Use these instead of
        ``source`` and ``target``.
    reported_by
        For source/target tables that already hold both reports of a flow:
        the column saying who reported it (values containing "export" or
        "import").
    fill
        Constants for roles the table has no column for, keyed by ``time``,
        ``category``, ``weight`` or ``unit``, for example
        ``fill={"time": 2020, "category": "migrants", "unit": "persons"}``.
    direction_values
        Raw direction value to ``"export"`` or ``"import"``. By default a
        value containing "export" (case-insensitive) is an export and one
        containing "import" is an import. Rows with other values are dropped.
    prefer
        For a flow reported by both sides: ``"target"`` keeps the target's
        (importer's) report, the same rule as ``faostat.load(...,
        reporter="combined")``; ``"source"`` keeps the source's report;
        ``"mean"`` and ``"max"`` combine the two (``reported_by`` becomes
        ``"both"``).
    aliases
        Name to replacement, applied to source, target and category names
        after whitespace normalization, for example ``{"Turkey": "Türkiye"}``.
    unit_conversions
        Unit to ``(new_unit, factor)``; matching weights are multiplied by
        ``factor``, for example ``{"1000 An": ("head", 1000.0)}``.
    drop_nodes
        Names of rows to drop wherever they appear as source or target, such
        as the aggregate ``"World"``.
    zeros
        ``"drop"`` (default) removes zero weights; ``"keep"`` keeps them.
    self_loops
        Keep flows from a node to itself.
    duplicates
        Rows that share (source, target, time, category, unit, reported_by)
        are summed (``"sum"``, default) or kept as separate rows (``"keep"``).
        Rows identical in every raw column are always dropped first.
    labels
        Display labels for charts, merged over the ones worked out from the
        raw table (see Returns).

    Returns
    -------
    flows : FlowFrame
        The flow-schema columns, then ``reported_by`` when the table says who
        reported each flow (``"exporter"``, ``"importer"`` or ``"both"``).
        Other raw columns are dropped; discarded rows keep them in
        ``report.dropped``. Names are strings, the time column is ``int64``
        and the weight column ``float64``. Rows are sorted by category, time,
        source, target and unit, with a fresh index. Several units may
        remain; see ``report.units``. ``flows.attrs["labels"]`` holds display
        labels for charts: the raw column names (so a column called
        ``"Persons"`` labels the weight axis "Persons"), or for reporter
        tables with export/import directions "Exporter", "Importer",
        "Exports" and "Imports".
    report : CleaningReport
        The columns used, per-step row counts, renames, look-alike names,
        mirror statistics and the dropped rows with reasons.

    Raises
    ------
    SchemaError
        If the columns cannot be worked out (none match, two match the same
        role, or a required role is missing), a named column does not exist,
        or a role is given both as a column and in ``fill``.
    ValueError
        If an option or a ``fill`` value is not valid.

    Notes
    -----
    Mirror flows are resolved after duplicates. With ``prefer="mean"`` or
    ``"max"`` each side's rows are summed before combining, even with
    ``duplicates="keep"``. ``prefer`` has no effect when the table does not
    say who reported each flow.

    Spellings are merged automatically only when they are identical after
    casefolding, removing accents and removing punctuation and spaces
    (``"Cote d'Ivoire"`` and ``"Côte d'Ivoire"``); the most frequent spelling
    wins, ties going to the first alphabetically. Near matches such as typos
    are only listed in ``report.possible_aliases``.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame({
    ...     "origin": ["Türkiye", "Türkiye", "Turkiye ", "Egypt"],
    ...     "destination": ["Egypt", "Egypt", "Iraq", "World"],
    ...     "year": ["2021", 2021, 2021.0, 2021],
    ...     "value": ["1,000", 250, "40", 5],
    ... })
    >>> flows, report = clean(raw, fill={"category": "Wheat", "unit": "t"},
    ...                       drop_nodes=["World"])
    >>> flows.iloc[:, [0, 1, 4]].values.tolist()  # source, target, weight
    [['Türkiye', 'Egypt', 1250.0], ['Türkiye', 'Iraq', 40.0]]
    >>> report.renamed
    {'Turkiye ': 'Türkiye'}
    """
    _check_choice("prefer", prefer, get_args(Prefer))
    _check_choice("zeros", zeros, ("drop", "keep"))
    _check_choice("duplicates", duplicates, ("sum", "keep"))
    for bad in set((direction_values or {}).values()) - {"export", "import"}:
        raise ValueError(f"direction_values must map to 'export' or 'import', not {bad!r}")
    conversions = _conversions(unit_conversions)
    drop_list = [drop_nodes] if isinstance(drop_nodes, str) else list(drop_nodes)
    constants = _check_fill(fill or {})
    named = {
        "source": source,
        "target": target,
        "weight": weight,
        "time": time,
        "category": category,
        "unit": unit,
        "reporter": reporter,
        "partner": partner,
        "direction": direction,
        "reported_by": reported_by,
    }
    given: dict[str, Hashable] = {r: c for r, c in named.items() if c is not None}
    mapping, guessed = _resolve_columns(raw, given, constants)

    log = _Log()
    work = pd.DataFrame({role: raw[col].to_numpy() for col, role in mapping.items()})
    work["_row"] = np.arange(len(raw), dtype=np.int64)
    work = work.assign(**constants)

    dup = raw.duplicated(keep="first").to_numpy(dtype=bool)
    work = log.filter(work, "exact duplicates", _where(dup, "exact duplicate of an earlier row"))

    # Names keep their raw spelling until the names step, so the report can
    # show what each one was renamed from.
    present = [r for r in _TEXT_ROLES if r in work.columns]
    normalized = {r: _map_unique(work[r], _norm_text) for r in present}
    reasons = np.full(len(work), None, dtype=object)
    for role in reversed(present):
        reasons[pd.isna(normalized[role])] = f"missing {role}"
    work["unit"] = normalized["unit"]
    work = log.filter(work, "missing values", reasons, _count_detail(reasons))

    if "direction" in work.columns:
        work = _orient(work, direction_values, log)
    elif "reported_by" in work.columns:
        sides, reasons, detail = _sides(work["reported_by"], _classify_direction, "reported_by")
        work["reported_by"] = np.where(sides == "export", BY_SOURCE, BY_TARGET)
        work = log.filter(work, "reported_by", reasons, detail)

    numbers, problems = _parse_column(work["weight"])
    work["weight"] = numbers
    reasons = np.array([None if p is None else f"{p} weight" for p in problems], dtype=object)
    work = log.filter(work, "weight", reasons, _count_detail(reasons))
    work = log.filter(
        work, "negative weight", _where(work["weight"].to_numpy() < 0, "negative weight")
    )
    work = _parse_times(work, log)

    if conversions:
        work = _convert_units(work, conversions, log)

    work, renamed, node_map, suggestions = _clean_names(work, aliases, log)

    if drop_list:
        targets = {node_map.get(n, n) for n in (_norm_text(d) for d in drop_list) if n}
        touched = work["source"].isin(targets) | work["target"].isin(targets)
        work = log.filter(
            work,
            "drop nodes",
            _where(touched.to_numpy(), "node listed in drop_nodes"),
            "dropped rows touching " + ", ".join(repr(t) for t in sorted(targets)),
        )

    is_zero = work["weight"].to_numpy() == 0
    if zeros == "drop":
        work = log.filter(work, "zeros", _where(is_zero, "zero weight"))
    else:
        log.note("zeros", len(work), len(work), f"kept {int(is_zero.sum()):,} zero rows")

    is_loop = (work["source"] == work["target"]).to_numpy()
    if self_loops:
        log.note("self-loops", len(work), len(work), f"kept {int(is_loop.sum()):,} self-loops")
    else:
        work = log.filter(work, "self-loops", _where(is_loop, "self-loop"))

    has_side = "reported_by" in work.columns
    key = ["source", "target", "time", "category", "unit"]
    work["_grp"] = work.groupby([*key, *(["reported_by"] if has_side else [])], sort=False).ngroup()
    rows_level = work
    work = _combine_duplicates(work, duplicates, log)

    mirror = None
    if has_side:
        work, mirror = _resolve_mirrors(work, rows_level, key, prefer, log)

    extra = ["reported_by"] if has_side else []
    order = ["category", "time", "source", "target", "unit", *extra, "_row"]
    work = work.sort_values(order, kind="mergesort").reset_index(drop=True)
    for role in ("source", "target", "category", "unit", *extra):
        work[role] = work[role].astype(object)
    work["time"] = work["time"].astype("int64")
    work["weight"] = work["weight"].astype("float64")
    flows = work[[*_SCHEMA_ROLES, *extra]].rename(columns=_COLUMN)
    validate_flows(flows)
    flows.attrs["labels"] = _display_labels(mapping, raw, labels)

    units: dict[str, dict[str, int]] = {}
    sizes = work.groupby(["category", "unit"], sort=True).size()
    for (cat, un), n in zip(sizes.index.tolist(), sizes.tolist(), strict=True):
        units.setdefault(str(cat), {})[str(un)] = int(n)
    report = CleaningReport(
        columns={str(c): r for c, r in mapping.items()},
        guessed=guessed,
        filled=constants,
        rows_in=len(raw),
        rows_out=len(flows),
        steps=tuple(log.steps),
        renamed=renamed,
        possible_aliases=tuple(suggestions),
        mirror=mirror,
        units=units,
        dropped=log.dropped(raw),
    )
    return flows, report


# ---------------------------------------------------------------------------
# Steps


def _display_labels(
    mapping: Mapping[Hashable, str], raw: pd.DataFrame, extra: Mapping[str, str] | None
) -> dict[str, str]:
    """Work out chart labels from the raw column names (see ``clean``)."""
    by_role = {role: str(col) for col, role in mapping.items()}
    out: dict[str, str] = {}
    if "direction" in by_role or "reported_by" in by_role:
        out.update({"source": "Exporter", "target": "Importer", "out": "Exports", "in": "Imports"})
    if "direction" in by_role:
        seen = " ".join(map(str, pd.unique(raw[by_role["direction"]].dropna()))).lower()
        if "quantity" in seen and "value" not in seen:
            out["weight"] = "Quantity"
        elif "value" in seen and "quantity" not in seen:
            out["weight"] = "Value"
    for role in ("source", "target", "time", "category", "weight"):
        if role in by_role and role not in out:
            name = " ".join(by_role[role].replace("_", " ").split())
            out[role] = name[:1].upper() + name[1:]
    out.update(extra or {})
    return out


def _check_choice(name: str, value: object, allowed: tuple[object, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {list(allowed)}, not {value!r}")


def _conversions(
    unit_conversions: Mapping[str, tuple[str, float]] | None,
) -> dict[str, tuple[str, float]]:
    out: dict[str, tuple[str, float]] = {}
    for old, (new, factor) in (unit_conversions or {}).items():
        old_n, new_n = _norm_text(old), _norm_text(new)
        if old_n is None or new_n is None:
            raise ValueError(f"unit_conversions has an empty unit: {old!r} -> {new!r}")
        number: object = factor
        if isinstance(number, bool) or not (
            isinstance(number, int | float) and math.isfinite(number) and number > 0
        ):
            raise ValueError(
                f"unit_conversions factor for {old!r} must be positive, not {factor!r}"
            )
        out[old_n] = (new_n, float(factor))
    return out


def _orient(
    work: pd.DataFrame, direction_values: Mapping[str, Direction] | None, log: _Log
) -> pd.DataFrame:
    """Turn reporter/partner/direction rows into source/target rows."""
    lookup = None
    if direction_values is not None:
        lookup = {str(k).strip(): v for k, v in direction_values.items()}

    def classify(value: object) -> Direction | None:
        if lookup is not None:
            return lookup.get(str(value).strip())
        return _classify_direction(value)

    sides, reasons, detail = _sides(work["direction"], classify, "direction")
    is_export = sides == "export"
    work = work.assign(
        source=np.where(is_export, work["reporter"], work["partner"]),
        target=np.where(is_export, work["partner"], work["reporter"]),
        reported_by=np.where(is_export, BY_SOURCE, BY_TARGET),
    ).drop(columns=["reporter", "partner", "direction"])
    return log.filter(work, "direction", reasons, detail)


def _sides(
    values: pd.Series[Any], classify: Callable[[Any], Direction | None], label: str
) -> tuple[ObjArray, ObjArray, str]:
    """Classify each value as export or import.

    Returns the side per row (``None`` if unclassified), the drop reason per
    row, and a description of how each distinct value was read.
    """
    codes, uniques = pd.factorize(values, use_na_sentinel=True)
    classes = [classify(u) for u in uniques]
    side_of = np.array([*classes, None], dtype=object)
    unrecognized = [
        None if c else f"unrecognized {label} {u!r}" for u, c in zip(uniques, classes, strict=True)
    ]
    reason_of = np.array([*unrecognized, f"missing {label}"], dtype=object)
    read = sorted((str(u), c or "dropped") for u, c in zip(uniques, classes, strict=True))
    detail = ", ".join(f"{u!r} -> {c}" for u, c in read[:_LIST_LIMIT])
    if len(read) > _LIST_LIMIT:
        detail += f", and {len(read) - _LIST_LIMIT:,} more values"
    sides: ObjArray = side_of[codes]
    reasons: ObjArray = reason_of[codes]
    return sides, reasons, detail


def _parse_times(work: pd.DataFrame, log: _Log) -> pd.DataFrame:
    times = work["time"]
    if pd.api.types.is_datetime64_any_dtype(times):
        numbers = times.dt.year.to_numpy(dtype=float, na_value=np.nan)
        problems = _where(np.isnan(numbers), "missing")
    else:
        numbers, problems = _parse_column(times)
    problems[np.isfinite(numbers) & (numbers != np.round(numbers))] = "invalid"
    problems[np.isinf(numbers)] = "invalid"
    reasons = np.array([None if p is None else f"{p} time" for p in problems], dtype=object)
    work["time"] = np.where(pd.isna(reasons), numbers, 0)
    return log.filter(work, "time", reasons, _count_detail(reasons))


def _convert_units(
    work: pd.DataFrame, conversions: Mapping[str, tuple[str, float]], log: _Log
) -> pd.DataFrame:
    parts = []
    weights = work["weight"].to_numpy(dtype=float).copy()
    units = work["unit"].to_numpy(dtype=object).copy()
    for old, (new, factor) in conversions.items():
        mask = work["unit"].to_numpy() == old
        if mask.any():
            weights[mask] *= factor
            units[mask] = new
            parts.append(f"{int(mask.sum()):,} rows {old!r} -> {new!r} (x{factor:g})")
    work = work.assign(weight=weights, unit=units)
    log.note("unit conversions", len(work), len(work), "; ".join(parts) or "no matching units")
    return work


def _clean_names(
    work: pd.DataFrame, aliases: Mapping[str, str] | None, log: _Log
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str], list[tuple[str, str, float]]]:
    """Normalize, alias and merge source, target and category names.

    Returns the frame, the renames (raw spelling -> final name), the map from
    normalized node name to final node name, and look-alike pairs.
    """
    alias_map: dict[str, str] = {}
    for old, new in (aliases or {}).items():
        old_n, new_n = _norm_text(old), _norm_text(new)
        if old_n is None or new_n is None:
            raise ValueError(f"aliases has an empty name: {old!r} -> {new!r}")
        alias_map[old_n] = new_n

    def normalize(value: object) -> str | None:
        text = _norm_text(value)
        return None if text is None else alias_map.get(text, text)

    original = {c: work[c].to_numpy(dtype=object) for c in ("source", "target", "category")}
    for col in original:
        work[col] = _map_unique(work[col], normalize)

    node_merge = _variant_merges(pd.concat([work["source"], work["target"]]).value_counts())
    cat_merge = _variant_merges(work["category"].value_counts())
    for col, merge in (("source", node_merge), ("target", node_merge), ("category", cat_merge)):
        work[col] = _replace(work[col], merge)

    renamed: dict[str, str] = {}
    by_alias = 0
    for col, before in original.items():
        pairs = pd.DataFrame({"a": before, "b": work[col].to_numpy()}).drop_duplicates()
        for a, b in zip(pairs["a"], pairs["b"], strict=True):
            if isinstance(a, str) and a != b and a not in renamed:
                renamed[a] = b
                by_alias += _norm_text(a) in alias_map
    renamed = dict(sorted(renamed.items()))
    node_map = {n: node_merge.get(t, t) for n, t in alias_map.items()}
    node_map.update(node_merge)
    suggestions = _near_matches(pd.unique(work[["source", "target"]].to_numpy().ravel()))
    suggestions += _near_matches(pd.unique(work["category"]))
    detail = (
        f"{len(renamed):,} spellings changed: {by_alias:,} by aliases, "
        f"{len(node_merge) + len(cat_merge):,} case, accent or punctuation variants merged"
    )
    log.note("names", len(work), len(work), detail)
    return work, renamed, node_map, suggestions


def _combine_duplicates(
    work: pd.DataFrame, duplicates: Literal["sum", "keep"], log: _Log
) -> pd.DataFrame:
    sizes = work.groupby("_grp")["_row"].transform("size").to_numpy()
    shared = int((sizes > 1).sum())
    if duplicates == "keep":
        log.note("duplicates", len(work), len(work), f"kept {shared:,} rows that share a key")
        return work
    out = work.drop_duplicates("_grp").copy()
    out["weight"] = out["_grp"].map(work.groupby("_grp")["weight"].sum())
    groups = int((work.groupby("_grp").size() > 1).sum())
    detail = f"summed {shared:,} rows that share a key into {groups:,}" if shared else ""
    log.note("duplicates", len(work), len(out), detail)
    return out.reset_index(drop=True)


def _resolve_mirrors(
    work: pd.DataFrame,
    rows_level: pd.DataFrame,
    key: list[str],
    prefer: Prefer,
    log: _Log,
) -> tuple[pd.DataFrame, MirrorStats]:
    """Choose or combine the source's and target's report of each flow."""
    totals = (
        work.groupby([*key, "reported_by"], sort=False)["weight"]
        .sum()
        .unstack("reported_by")
        .reindex(columns=[BY_SOURCE, BY_TARGET])
    )
    s, t = totals[BY_SOURCE], totals[BY_TARGET]
    both = (s.notna() & t.notna()).to_numpy()
    sb, tb = s[both].to_numpy(dtype=float), t[both].to_numpy(dtype=float)
    top = np.maximum(sb, tb)
    rel = np.divide(np.abs(sb - tb), top, out=np.zeros_like(top), where=top > 0)
    stats = MirrorStats(
        both=int(both.sum()),
        source_only=int((s.notna() & t.isna()).sum()),
        target_only=int((t.notna() & s.isna()).sum()),
        median_disagreement=float(np.median(rel)) if len(rel) else None,
        prefer=prefer,
    )
    in_both = pd.MultiIndex.from_frame(work[key]).isin(totals.index[both])
    if prefer in ("target", "source"):
        other = BY_SOURCE if prefer == "target" else BY_TARGET
        discard = in_both & (work["reported_by"] == other).to_numpy()
        lost = rows_level["_grp"].isin(work.loc[discard, "_grp"]).to_numpy()
        log.drop_rows(
            rows_level["_row"].to_numpy()[lost],
            _where(
                np.ones(int(lost.sum()), dtype=bool), f"mirror flow: the {prefer}'s report was used"
            ),
        )
        out = work.loc[~discard]
        detail = f"dropped {int(discard.sum()):,} {other} reports of flows both sides reported"
    else:
        combined = pd.DataFrame(
            {"weight": (sb + tb) / 2 if prefer == "mean" else top, "reported_by": BY_BOTH},
            index=totals.index[both],
        ).reset_index()
        first = work.loc[in_both].groupby(key, sort=False)["_row"].min()
        combined["_row"] = pd.MultiIndex.from_frame(combined[key]).map(first.to_dict()).to_numpy()
        out = pd.concat([work.loc[~in_both], combined], ignore_index=True)
        detail = f"combined {int(in_both.sum()):,} reports into {stats.both:,} flows ({prefer})"
    log.note("mirror flows", len(work), len(out), detail)
    return out.reset_index(drop=True), stats
