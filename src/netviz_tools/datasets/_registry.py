"""Pinned sources for the datasets module.

The FAOSTAT bulk file is replaced in place when FAO publishes an update, so the
SHA-256 below identifies the exact release this version of the library was
built and tested against. See ``netviz_tools.datasets.faostat.build_store`` for
how a mismatch is reported and how to opt in to a newer file.
"""

from __future__ import annotations

from typing import Final

FAOSTAT_TRADE_URL: Final = (
    "https://bulks-faostat.fao.org/production/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"
)
"""Bulk download of the FAOSTAT Detailed Trade Matrix (domain TM), normalized layout."""

FAOSTAT_TRADE_FILENAME: Final = "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip"

FAOSTAT_TRADE_CSV: Final = "Trade_DetailedTradeMatrix_E_All_Data_(Normalized).csv"
"""Name of the data file inside the zip."""

FAOSTAT_TRADE_SHA256: Final = "75b7ff8de04e7ae01f4b68150353cc0961fcbd5b0497c5f81338844345e348ec"
"""SHA-256 of the zip as retrieved on 2026-09-27 (inner files dated 2025-12-18)."""

FAOSTAT_TRADE_SIZE: Final = 420_650_070
"""Size of the pinned zip in bytes."""

FAOSTAT_TRADE_RELEASE: Final = "2025-12-18"
"""Timestamp of the data file inside the pinned zip."""

FAOSTAT_DATASET_PAGE: Final = "https://www.fao.org/faostat/en/#data/TM"
FAOSTAT_TERMS_URL: Final = "https://www.fao.org/contact-us/terms/db-terms-of-use/en/"
