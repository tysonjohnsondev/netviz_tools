"""Tests that download real data. Deselected by default; run with ``pytest -m network``."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

import netviz_tools as nv
from netviz_tools.datasets import _registry

pytestmark = pytest.mark.network


def test_pinned_release_is_still_online() -> None:
    """Fails when FAO has replaced the bulk file, which means the pin needs updating."""
    req = urllib.request.Request(_registry.FAOSTAT_TRADE_URL, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as resp:
        assert resp.status == 200
        assert int(resp.headers["Content-Length"]) == _registry.FAOSTAT_TRADE_SIZE


def test_full_build_and_load(tmp_path: Path) -> None:
    """Downloads about 420 MB and builds the full store (a few minutes)."""
    store = nv.datasets.faostat.build_store(tmp_path)
    info = nv.datasets.faostat.store_info(tmp_path)
    assert info["matches_pinned"] is True
    assert info["rows"] > 40_000_000
    assert store.is_dir()
    wheat = nv.datasets.faostat.load("Wheat", 2021, cache_dir=tmp_path)
    sample = nv.datasets.faostat.load_sample(items="Wheat", years=2021)
    assert wheat["quantity"].sum() == pytest.approx(sample["quantity"].sum())
