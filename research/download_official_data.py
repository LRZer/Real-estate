"""Collect pre-March-2026 price-index tables from original NBS releases.

The bundled URL index is used only to locate NBS pages. Every value in the
output is parsed directly from the original National Bureau of Statistics HTML.
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "official_pages"
START_PERIOD = "2021-05"
END_PERIOD = "2026-01"
LAST_ALLOWED_RELEASE_MONTH = "202602"


def source_index() -> pd.DataFrame:
    index = pd.read_csv(ROOT / "source_index.csv")
    index = index[index["period"].between(START_PERIOD, END_PERIOD)]
    index = index[["period", "source_url"]].drop_duplicates()
    if index["period"].duplicated().any():
        raise RuntimeError("Multiple source URLs for one period")
    expected = pd.period_range(START_PERIOD, END_PERIOD, freq="M").astype(str)
    if set(index["period"]) != set(expected):
        raise RuntimeError("The source URL index does not cover every month")
    for row in index.itertuples(index=False):
        match = re.search(r"/(20\d{4})/t(20\d{6})_", row.source_url)
        if not match or match.group(1) > LAST_ALLOWED_RELEASE_MONTH:
            raise RuntimeError(f"Source page is after the cutoff or has an unknown date: {row.source_url}")
        if not row.source_url.startswith("https://www.stats.gov.cn/"):
            raise RuntimeError(f"Unexpected publisher: {row.source_url}")
    return index.sort_values("period").reset_index(drop=True)


def get_page(period: str, url: str) -> str:
    cache_path = CACHE / f"{period}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    last_error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=35)
            response.raise_for_status()
            html = response.content.decode("utf-8", errors="replace")
            cache_path.write_text(html, encoding="utf-8")
            return html
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Could not download {period}: {url}") from last_error


def parse_table(table: pd.DataFrame) -> dict[str, float]:
    if table.shape[1] not in (6, 8):
        raise ValueError(f"Unexpected table width: {table.shape}")
    right = 3 if table.shape[1] == 6 else 4
    values: dict[str, float] = {}
    for _, row in table.iloc[2:].iterrows():
        for city_col in (0, right):
            city = re.sub(r"\s+", "", str(row.iloc[city_col]))
            if city in ("", "nan"):
                continue
            change = float(row.iloc[city_col + 1]) - 100.0
            if city in values:
                raise ValueError(f"Duplicate city: {city}")
            values[city] = round(change, 3)
    if len(values) != 70:
        raise ValueError(f"Expected 70 cities; found {len(values)}")
    return values


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    links = source_index()
    records = []
    for row in links.itertuples(index=False):
        html = get_page(row.period, row.source_url)
        tables = pd.read_html(io.StringIO(html))
        if len(tables) < 2:
            raise ValueError(f"Missing new/second-hand tables in {row.source_url}")
        new = parse_table(tables[0])
        second = parse_table(tables[1])
        if new.keys() != second.keys():
            raise ValueError(f"City sets differ in {row.period}")
        for city in new:
            records.append({
                "period": row.period,
                "city": city,
                "new_mom_pct": new[city],
                "second_mom_pct": second[city],
                "source_url": row.source_url,
            })
        print(f"{row.period}: {len(new)} cities", flush=True)
    result = pd.DataFrame(records).sort_values(["period", "city"])
    result.to_csv(DATA / "official_nbs_70city.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(result):,} official city-month observations", flush=True)


if __name__ == "__main__":
    main()
