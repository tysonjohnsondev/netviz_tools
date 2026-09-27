"""Exception types raised by netviz_tools.

Every exception derives from :class:`NetvizError`, and also from the closest
built-in exception, so ``except ValueError`` keeps working for callers who do
not want to depend on this module.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = [
    "InsufficientDataError",
    "MixedSliceError",
    "MixedUnitError",
    "NetvizError",
    "SchemaError",
    "SourceHashMismatchError",
    "StoreNotFoundError",
    "UnknownCountryError",
    "UnknownItemError",
    "UnknownMetricError",
    "UnknownNameError",
]


class NetvizError(Exception):
    """Base class for all errors raised by netviz_tools."""


class SchemaError(NetvizError, ValueError):
    """A DataFrame does not match the expected flow schema.

    Parameters
    ----------
    problems
        One message per problem found. All problems are reported at once.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__("invalid flow data: " + "; ".join(self.problems))


class MixedSliceError(NetvizError, ValueError):
    """Flows span several years or items but no aggregation was requested."""


class MixedUnitError(NetvizError, ValueError):
    """Flows measured in different units would be combined."""


class UnknownMetricError(NetvizError, ValueError):
    """A metric or method name is not supported."""


class InsufficientDataError(NetvizError, ValueError):
    """There is not enough data for the requested computation."""


class UnknownNameError(NetvizError, LookupError):
    """A name could not be found. Close matches are offered as suggestions.

    Parameters
    ----------
    kind
        What was being looked up, for example ``"item"`` or ``"country"``.
    name
        The name that was not found.
    suggestions
        Close matches, best first. May be empty.
    """

    def __init__(self, kind: str, name: object, suggestions: Sequence[str] = ()) -> None:
        self.kind = kind
        self.name = name
        self.suggestions = tuple(suggestions)
        msg = f"unknown {kind} {name!r}"
        if self.suggestions:
            msg += ". Did you mean: " + ", ".join(repr(s) for s in self.suggestions) + "?"
        super().__init__(msg)

    def __str__(self) -> str:
        # LookupError subclasses such as KeyError quote their message; keep it plain.
        return str(self.args[0])


class UnknownItemError(UnknownNameError):
    """An item (commodity) name or code is not in the catalogue."""

    def __init__(self, name: object, suggestions: Sequence[str] = ()) -> None:
        super().__init__("item", name, suggestions)


class UnknownCountryError(UnknownNameError):
    """A country (node) name is not present in the flows."""

    def __init__(self, name: object, suggestions: Sequence[str] = ()) -> None:
        super().__init__("country", name, suggestions)


class StoreNotFoundError(NetvizError, FileNotFoundError):
    """The local Parquet store has not been built yet."""


class SourceHashMismatchError(NetvizError, ValueError):
    """A downloaded or local source file does not match the pinned SHA-256.

    Parameters
    ----------
    path
        The file that was checked.
    expected
        The pinned SHA-256.
    actual
        The SHA-256 of the file, if it was computed.
    """

    def __init__(self, path: str, expected: str, actual: str | None = None) -> None:
        self.path = path
        self.expected = expected
        self.actual = actual
        got = f" but the file has {actual}" if actual else ""
        super().__init__(
            f"{path}: expected SHA-256 {expected}{got}. FAO replaces the bulk file in place "
            "when it publishes an update, so this usually means a newer release is online. "
            "To build from the file you have, pass known_hash=None explicitly; the SHA-256 "
            "that was actually used is recorded in the store manifest."
        )
