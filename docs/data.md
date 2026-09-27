# Data and licences

This page documents where the bundled data and the FAOSTAT store come from, how they are transformed, and how to cite them.

## FAOSTAT Detailed Trade Matrix

The trade data come from the FAOSTAT Detailed Trade Matrix (domain TM), published by the Food and Agriculture Organization of the United Nations (FAO): bilateral trade in about 560 agricultural products between about 220 countries and territories, from 1986. The library uses the bulk "normalized" file.

| | |
| --- | --- |
| Dataset page | <https://www.fao.org/faostat/en/#data/TM> |
| Bulk file | <https://bulks-faostat.fao.org/production/Trade_DetailedTradeMatrix_E_All_Data_(Normalized).zip> |
| Pinned release | data file dated 2025-12-18 |
| Retrieved | 2026-09-27 |
| Size | 420,650,070 bytes |
| SHA-256 | `75b7ff8de04e7ae01f4b68150353cc0961fcbd5b0497c5f81338844345e348ec` |
| Licence | CC BY 4.0 |

The pinned values are in `netviz_tools.datasets._registry`.

### Licence

FAO publishes its statistical databases under the Creative Commons Attribution 4.0 International licence (CC BY 4.0), as set out in the [FAO database terms of use](https://www.fao.org/contact-us/terms/db-terms-of-use/en/). You may copy, redistribute and adapt the data, including commercially, provided you give attribution in the required form.

The terms also say that you may not represent that FAO participated in, sponsored, approved or endorsed the way you use the data, or use the data in a way that suggests FAO endorses a company, product or service. netviz-tools is an independent project and is not affiliated with or endorsed by FAO. FAO databases may also contain third-party data under different terms; check the dataset notes on the FAOSTAT site if you redistribute the data.

### Citing FAOSTAT

The terms of use require this citation format:

> FAO. [year]. [dataset]. Accessed on [date]. [URL]. Licence: CC-BY-4.0.

where the year is the year of the dataset's last update. For the release pinned in netviz-tools 1.0.0 (data file dated 2025-12-18, retrieved 2026-09-27):

> FAO. 2025. FAOSTAT: Detailed trade matrix. Accessed on 27 September 2026. https://www.fao.org/faostat/en/#data/TM. Licence: CC-BY-4.0.

`netviz_tools.datasets.faostat.CITATION` holds the template with `{year}` and `{accessed}` placeholders:

```python
from netviz_tools.datasets import faostat

faostat.CITATION.format(year=2025, accessed="27 September 2026")
```

If you build your own store from a newer file, cite that file's update year and your own access date; both are recorded in the store manifest (see below).

### The bundled sample

`faostat.load_sample()` reads a 0.6 MB Parquet file inside the package, cut from the pinned release by `scripts/make_sample.py`. It keeps quantity flows in tonnes for Wheat (item code 15), Maize (corn) (56) and Soya beans (236) from 2010 to 2024, reported by both importers and exporters. The file has 140,935 rows (72,012 importer-reported and 68,923 exporter-reported), of which 65 are self-loops that `load_sample` drops by default. It has the same columns as the store, so `load_sample` and `load` run the same query.

## How the store is built

`faostat.build_store()` converts the bulk CSV into a Parquet dataset partitioned by item code, using DuckDB. The rules:

- **Exporter is always the sender.** FAOSTAT rows are written from the reporter's point of view ("Import quantity" reported by Egypt from partner Russian Federation, "Export quantity" reported by the Russian Federation to partner Egypt). Every row is re-oriented so that `exporter` is the country the goods leave and `importer` the country they arrive in, whoever reported it. A `reported_by` column keeps the reporter's side (`"importer"` or `"exporter"`).
- **Units are normalized.** FAOSTAT's `An` (animals) becomes `head`; `1000 An` is multiplied by 1000 and becomes `head`; `No` becomes `number`. Tonnes stay `t` and trade values stay `1000 USD`. Quantity and value rows are kept apart in a `measure` column.
- **Zero values are dropped.** Rows with a value of zero or no value are not written.
- **Self-loops are kept in the store** (flows reported between a country and itself, such as re-imports) but `load()` and `load_sample()` drop them unless you pass `self_loops=True`.
- **Flags are kept.** The FAOSTAT observation flag is stored and returned by `load(..., details=True)`.

`load()` then selects the reporting perspective (`reporter="importer"`, `"exporter"` or `"combined"`), the measure, the items and the years. With `"combined"`, the importer's report is used where one exists and the exporter's otherwise, decided per exporter, importer, item and year.

### Why the reporter perspective matters

The two reports of the same flow often disagree, and some countries stop reporting. In the bundled sample, the Russian Federation's wheat exports look like this (million tonnes):

| Year | Importer-reported | Exporter-reported | Combined |
| --- | ---: | ---: | ---: |
| 2020 | 26.82 | 37.27 | 32.87 |
| 2021 | 22.39 | 27.26 | 26.62 |
| 2022 | 22.44 | no rows | 22.44 |
| 2023 | 33.96 | no rows | 33.96 |

The Russian Federation reports no wheat exports after 2021, so a graph built from exporter-reported data shows no Russian wheat exports from 2022 on. The importer-reported default keeps them. For wheat in 2021, 155 importing countries report against 99 exporting countries.

### Stale hashes and the manifest

FAO replaces the bulk file in place when it publishes an update: the URL stays the same and the content changes. The SHA-256 pinned in each netviz-tools release identifies the exact file it was built and tested against. When the file you download (or pass as `zip_path=`) does not match, `build_store()` and `download()` raise `SourceHashMismatchError`, which names the expected and actual hashes.

To build from the newer file anyway, opt in explicitly:

```python
faostat.build_store(known_hash=None)
```

Every build writes `manifest.json` next to the data. `faostat.store_info()` returns it. It records:

- `source_url` and whether the zip was downloaded or given as a local file;
- `source_sha256`, the SHA-256 actually used, with `pinned_sha256` and `matches_pinned` for comparison;
- `retrieved_at` (the zip's modification time) and `built_at`;
- `rows` and `rows_by_measure` (row counts and year ranges per measure and reporter);
- `items`, the item codes and names found in the file;
- `netviz_tools_version` and the licence.

So a store built from an unpinned file is still traceable to the exact bytes it came from.

## Country metadata

`faostat.countries()` returns one row per FAOSTAT area that appears in the trade matrix (221 areas), built by `scripts/build_metadata.py` from three sources:

| Source | Used for | Licence |
| --- | --- | --- |
| FAOSTAT Detailed Trade Matrix (pinned release above) | area codes, names, M49 codes; item catalogue | CC BY 4.0 |
| [UNSD Standard country or area codes for statistical use (M49)](https://unstats.un.org/unsd/methodology/m49/overview/) | ISO3 codes, region, sub-region; the `continent` column | see the UNSD site |
| [Natural Earth 1:10m Admin 0 Countries v5.1.2](https://github.com/nvkelso/natural-earth-vector/tree/v5.1.2) | label points (`lon`, `lat`) used as node positions in flow maps | public domain |

`continent` is the M49 region, except that the Americas are split into Northern America, Central America, the Caribbean and South America. The item catalogue (`faostat.items()`, 558 items) also comes from the FAOSTAT file.

### Manual overrides

Some FAOSTAT areas have no entry in the current M49 table or no admin-0 feature in Natural Earth: historical states, overseas departments, small islands. Each gap is filled by hand, and every override is listed in `datasets/_meta/sources.json` with its reason. There are 31:

| M49 | Area | Field | Reason |
| --- | --- | --- | --- |
| 058 | Belgium-Luxembourg | region | Belgium-Luxembourg (to 1999) is not in current M49; region of Belgium. |
| 058 | Belgium-Luxembourg | lon/lat | Historical area; label point of Belgium. |
| 074 | Bouvet Island | lon/lat | Not a separate Natural Earth admin-0 feature; approximate location. |
| 128 | Canton and Enderbury Islands | region | Canton and Enderbury Islands (historical) is not in current M49; now Kiribati. |
| 128 | Canton and Enderbury Islands | lon/lat | Historical area; label point of Kiribati. |
| 200 | Czechoslovakia | region | Czechoslovakia (to 1992) is not in current M49; region of Czechia. |
| 200 | Czechoslovakia | lon/lat | Historical area; label point of Czechia. |
| 230 | Ethiopia PDR | region | Ethiopia PDR (to 1992) is not in current M49; region of Ethiopia. |
| 230 | Ethiopia PDR | lon/lat | Historical area; label point of Ethiopia. |
| 254 | French Guiana | lon/lat | Part of France in Natural Earth admin-0; approximate centroid. |
| 312 | Guadeloupe | lon/lat | Part of France in Natural Earth admin-0; approximate centroid. |
| 396 | Johnston Island | region | Johnston Island (historical code) is part of US Minor Outlying Islands. |
| 396 | Johnston Island | lon/lat | Not a separate Natural Earth feature; approximate atoll location. |
| 474 | Martinique | lon/lat | Part of France in Natural Earth admin-0; approximate centroid. |
| 488 | Midway Island | region | Midway Island (historical code) is part of US Minor Outlying Islands. |
| 488 | Midway Island | lon/lat | Not a separate Natural Earth feature; approximate atoll location. |
| 638 | Réunion | lon/lat | Part of France in Natural Earth admin-0; approximate centroid. |
| 891 | Serbia and Montenegro | region | Serbia and Montenegro (1992-2005) is not in current M49; region of Serbia. |
| 891 | Serbia and Montenegro | lon/lat | Historical area; label point of Serbia. |
| 736 | Sudan (former) | region | Sudan (former, to 2011) is not in current M49; region of Sudan. |
| 736 | Sudan (former) | lon/lat | Historical area; label point of Sudan. |
| 158 | China, Taiwan Province of | region | M49 158 is not listed by UNSD; FAOSTAT places it in Eastern Asia with China. |
| 158 | China, Taiwan Province of | iso3 | ISO 3166-1 alpha-3 code; M49 158 is not listed in the UNSD table. |
| 772 | Tokelau | lon/lat | Not a separate Natural Earth admin-0 feature; approximate location. |
| 810 | USSR | region | USSR (to 1991) is not in current M49; region of the Russian Federation. |
| 810 | USSR | lon/lat | Historical area; label point of the Russian Federation. |
| 872 | Wake Island | region | Wake Island (historical code) is part of US Minor Outlying Islands. |
| 872 | Wake Island | lon/lat | Not a separate Natural Earth feature; approximate atoll location. |
| 890 | Yugoslav SFR | region | Yugoslav SFR (to 1991) is not in current M49; region of Serbia. |
| 890 | Yugoslav SFR | lon/lat | Historical area; label point of Serbia. |
| 744 | Svalbard and Jan Mayen Islands | lon/lat | Part of Norway in Natural Earth admin-0; approximate centroid. |

"Region of X" means the area borrows the M49 region and sub-region of X. "Label point of X" means it borrows X's Natural Earth label point. "Approximate" locations are hand-entered coordinates.
