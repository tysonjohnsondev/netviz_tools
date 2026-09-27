"""MkDocs hook: render Sphinx cross-reference roles left in docstrings.

The library's docstrings use roles such as ``:func:`build_graph``` and
``:class:`pandas.DataFrame```. mkdocstrings renders them literally, as
``:func:`` followed by a code span. This hook rewrites each one into a
mkdocs-autorefs reference marked ``optional``: it becomes a link when the target
is documented on this site, and plain code otherwise, with no build warning.
"""

from __future__ import annotations

import importlib
import inspect
import re
from functools import cache
from typing import Any

ROLE_RE = re.compile(
    r":(?:py:)?(?:func|class|mod|data|meth|attr|obj|exc|const):<code>(~?)([\w.]+)</code>"
)

# Modules whose public names can be referenced without a package prefix.
MODULES = (
    "netviz_tools",
    "netviz_tools._schema",
    "netviz_tools.graph",
    "netviz_tools.analysis",
    "netviz_tools.metrics",
    "netviz_tools.temporal",
    "netviz_tools.stats",
    "netviz_tools.errors",
    "netviz_tools.plot.network",
    "netviz_tools.plot.sankey",
    "netviz_tools.plot.geo",
    "netviz_tools.plot.charts",
    "netviz_tools.plot._style",
    "netviz_tools.datasets.faostat",
)


def _canonical(obj: Any, fallback: str) -> str:
    if inspect.ismodule(obj):
        return obj.__name__
    if inspect.isclass(obj) or inspect.isfunction(obj):
        return f"{obj.__module__}.{obj.__qualname__}"
    return fallback


@cache
def _bare_names() -> dict[str, str]:
    index: dict[str, str] = {}
    for modname in MODULES:
        mod = importlib.import_module(modname)
        for name in getattr(mod, "__all__", ()):
            obj = getattr(mod, name, None)
            index.setdefault(name, _canonical(obj, f"{modname}.{name}"))
    return index


@cache
def _resolve(target: str) -> str:
    if "." not in target:
        return _bare_names().get(target, target)
    parts = target.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for part in parts[i:]:
            obj = getattr(obj, part, None)
            if obj is None:
                return target
        return _canonical(obj, target)
    return target


def _replace(match: re.Match[str]) -> str:
    tilde, target = match.groups()
    shown = target.rsplit(".", 1)[-1] if tilde else target
    return f'<autoref identifier="{_resolve(target)}" optional><code>{shown}</code></autoref>'


def on_page_content(html: str, **kwargs: Any) -> str:
    """Rewrite Sphinx roles in the rendered page."""
    return ROLE_RE.sub(_replace, html)
