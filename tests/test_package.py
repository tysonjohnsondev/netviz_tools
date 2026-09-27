from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import netviz_tools as nv


def test_version() -> None:
    assert nv.__version__ == "1.0.0"


def test_public_api() -> None:
    for name in nv.__all__:
        assert hasattr(nv, name), name
    assert isinstance(nv.UnknownItemError("x"), nv.NetvizError)
    assert "compare_items" not in nv.__all__
    assert not hasattr(nv, "UnknownCountryError")


def test_unknown_name_errors_exported() -> None:
    from netviz_tools import errors

    assert nv.UnknownNameError is errors.UnknownNameError
    assert nv.UnknownNodeError is errors.UnknownNodeError
    assert {"UnknownNameError", "UnknownNodeError"} <= set(nv.__all__) & set(errors.__all__)
    node = nv.UnknownNodeError("Atlantis", ["Atlanta"])
    assert isinstance(node, nv.UnknownNameError)
    assert node.kind == "node"
    assert str(node) == "unknown node 'Atlantis'. Did you mean: 'Atlanta'?"
    assert isinstance(nv.UnknownItemError("x"), nv.UnknownNameError)
    with pytest.raises(nv.UnknownNameError, match=r"^unknown node 'x'$"):
        raise nv.UnknownNodeError("x")


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    """Importing must not create files, add logging handlers, or touch the network."""
    code = textwrap.dedent(
        """
        import logging, socket, sys

        def no_network(*args, **kwargs):
            raise RuntimeError("network access at import")

        socket.socket.connect = no_network
        before = list(logging.getLogger().handlers)
        import netviz_tools, netviz_tools.datasets.faostat, netviz_tools.plot
        assert logging.getLogger().handlers == before
        named = [n for n in logging.root.manager.loggerDict if n.startswith("netviz_tools")]
        assert all(not logging.getLogger(n).handlers for n in named)
        print("ok")
        """
    )
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "LOCALAPPDATA": str(tmp_path / "cache"),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    env.pop("NETVIZ_TOOLS_CACHE", None)
    for key in [k for k in env if k.startswith(("COV_CORE", "COVERAGE"))]:
        env.pop(key)  # keep coverage out of the child process
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
    assert sorted(p.name for p in tmp_path.iterdir()) == []
