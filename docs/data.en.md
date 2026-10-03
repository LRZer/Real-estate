# Dataset card

[中文](data.zh.md) · [Methods](methods.en.md) · [Results](results.en.md)

## Source and coverage

The source is the monthly “Sales Prices of Residential Buildings in 70 Medium and Large-sized Cities” release from the National Bureau of Statistics of China (NBS). The cleaned panel contains **57 consecutive months, 70 cities and 3,990 city-month records, May 2021–January 2026**, covering new and second-hand homes. No company clients, leads, transactions or marketing data are used.

| File | Contents |
| --- | --- |
| [Cleaned panel](../research/data/official_nbs_70city.csv) | Complete input data for reproducing the experiments |
| [Source index](../research/source_index.csv) | 57 official monthly URLs |
| [Release calendar](../research/results/research/release_calendar.csv) | Statistical month, publication date and earliest forecast origin |
| [Downloader/parser](../research/download_official_data.py) | Page acquisition and table parsing |
| [Data manifest](../research/data_manifest.json) | Cutoff, retrieval date and SHA256 |

Each city-month retains its source URL. Downloaded HTML pages are excluded from the repository; cleaned data, source links and the parser are included.

## Schema and units

| Column | Type | Definition / unit |
| --- | --- | --- |
| `period` | YYYY-MM | Statistical month, distinct from publication date |
| `city` | Chinese string | Official city name |
| `new_mom_pct` | Float | New-home monthly index minus 100; unit % |
| `second_mom_pct` | Float | Second-hand monthly index minus 100; unit % |
| `source_url` | URL | Official page for this statistical month |

An index of 99.7, with previous month=100, is stored as -0.3: a 0.3% monthly decline. The model forecasts this movement for the next statistical month. Error is measured in **percentage points (pp)**: predicting -0.2% against -0.5% gives an absolute error of 0.3 pp. This target does not identify an individual property's sale price or price per square metre.

![Complete input history](figures/en/01_market.png)

The plotted 70-city mean is an equal-weight statistic computed for this study. It is neither transaction-weighted nor an official national price index.

## Information availability

The information cutoff is **28 February 2026**. January 2026 is the last actual month; its official [release date is 13 February 2026](https://www.stats.gov.cn/sj/zxfb/202602/t20260213_1962617.html). A February forecast is therefore formed after 13 February, not on 1 February.

All historical origins follow this rule: to forecast m, inputs end at released month m-1. February 2026 actuals are excluded from training, selection and scoring.

Pages were retrieved on **28 September 2026**, and development took place in September–October 2026. The study reconstructs the earlier information set; it cannot establish whether public pages were revised after their first publication. Retrieval or completion must not be represented as taking place before March 2026.

## Quality checks and supervised samples

| Check / sample | Result |
| --- | ---: |
| Unique city-month keys | 3,990; no duplicates |
| Cities in every month | 70 |
| Consecutive raw months | 57 |
| Missing numeric values | 0 |
| Input releases after the cutoff | 0 |
| Known targets after seven-month warm-up | 50 months, Dec 2021–Jan 2026 |
| 2024 rolling validation | 840 rows |
| Jan 2025–Jan 2026 retrospective evaluation | 910 rows |
| February 2026 as-of output | 70 rows; no actual labels |

The first six raw months supply warm-up history; no missing observations are fabricated. Feature/window definitions are in [Methods](methods.en.md). Models are refitted using labels available at each origin; city-month observations are not randomly split.

## Limitations and use

- City-month aggregates do not distinguish developments, layouts, channels or customers.
- Published indices are rounded; flat observations include zeros at the reporting precision.
- January 2026 has base/category-weight changes. Results separately report 2025-only and January 2026.
- Cities within a month are dependent; 910 rows are not 910 independent temporal observations.
- Rights and applicable data-use terms remain with the original publisher. The repository does not grant a new license to official data.

Cleaned CSV SHA256:

```text
2C8F96499B410FFA693F17FF11708509E9A3D37C57804BE0E7CD96D7EF324714
```

The [repository verifier](../scripts/verify_repository.py) rechecks hashes, panel completeness, coverage, source alignment and release dates.
