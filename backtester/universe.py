"""Point-in-time S&P 500 sector universes.

`SECTOR_UNIVERSES` is the canonical sector -> symbol map used across the
research notebook (Tech, Financials, Energy, Utilities). Pairs are only ever
tested *within* a sector — sectors are never pooled.

`get_point_in_time_universe(sector, as_of_date)` restricts a sector's symbol
list to the names that were actually S&P 500 constituents as of a given date,
using the publicly available historical-constituents file from
https://github.com/fja05680/sp500.

What this does and does not fix
-------------------------------
This is a *partial* survivorship-bias mitigation. It drops names that were not
yet in the index on the as-of date (so we do not trade a 2019 universe using a
2024 membership list). It does **not** resurrect companies that were delisted
or removed before today and therefore never made it into our hand-curated
sector lists — those are invisible to us. A full fix requires a point-in-time
constituent *and sector* database such as Compustat / CRSP. We are honest about
that limitation rather than pretending the bias is gone.

If the historical file cannot be downloaded or parsed, the loader logs a
warning and callers fall back to the current (hand-curated) sector list.
"""
import logging
import os
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical 4-sector universe. Tech and Financials are identical to the lists
# used by the original pairs strategy (strategies/pairs.py::SECTORS); Energy and
# Utilities are added for the multi-sector study.
SECTOR_UNIVERSES: Dict[str, List[str]] = {
    "tech": [
        "AAPL", "MSFT", "NVDA", "GOOGL", "META",
        "AVGO", "ORCL", "AMD", "QCOM", "TXN", "INTC", "CRM",
    ],
    "financials": [
        "JPM", "BAC", "WFC", "GS", "MS",
        "C", "BLK", "SCHW", "AXP", "USB", "PNC", "TFC", "COF",
    ],
    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG",
        "PXD", "MPC", "VLO", "PSX", "HAL",
    ],
    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP",
        "EXC", "SRE", "XEL", "ED", "WEC",
    ],
}

# Public historical-constituents file (snapshots of the full S&P 500 membership
# at each change date). Columns: date, tickers (comma-separated).
SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

_CACHE_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sp500_constituents.csv")
)


def download_sp500_constituents(
    url: str = SP500_HISTORY_URL,
    dest: str = _CACHE_PATH,
    force: bool = False,
) -> Optional[str]:
    """Download the historical-constituents CSV to `dest`. Returns the path on
    success, or None on failure (logged as a warning). No-op if already cached
    and `force` is False. `requests` is imported lazily so the module is usable
    offline as long as the cache exists."""
    if os.path.exists(dest) and not force:
        return dest
    try:
        import requests  # lazy import: only needed when actually downloading

        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            fh.write(resp.text)
        logger.info("downloaded S&P 500 constituents to %s", dest)
        return dest
    except Exception as exc:  # network error, HTTP error, write error, ...
        logger.warning("could not download S&P 500 constituents: %s", exc)
        return None


def load_sp500_constituents(
    path: str = _CACHE_PATH,
    auto_download: bool = True,
) -> Optional[pd.DataFrame]:
    """Load the historical-constituents file as a DataFrame sorted by date with
    columns ['date', 'tickers']. Downloads it first if missing and
    `auto_download` is True. Returns None (and logs a warning) if the file is
    unavailable or unparseable, so callers can fall back gracefully."""
    if not os.path.exists(path) and auto_download:
        download_sp500_constituents(dest=path)
    if not os.path.exists(path):
        logger.warning("S&P 500 constituents file unavailable at %s", path)
        return None
    try:
        df = pd.read_csv(path)
        # The fja05680 file uses 'date' and 'tickers'; be tolerant of casing.
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date")
        tick_col = cols.get("tickers") or cols.get("ticker")
        if date_col is None or tick_col is None:
            logger.warning(
                "constituents file has unexpected columns %s", list(df.columns)
            )
            return None
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]),
            "tickers": df[tick_col].astype(str),
        })
        return out.sort_values("date").reset_index(drop=True)
    except Exception as exc:
        logger.warning("could not parse constituents file %s: %s", path, exc)
        return None


def constituents_as_of(history: pd.DataFrame, as_of_date) -> set:
    """Return the set of constituent tickers in effect on `as_of_date`: the
    most recent snapshot at or before that date. If the date precedes the first
    snapshot, the earliest snapshot is used (best available)."""
    as_of = pd.Timestamp(as_of_date)
    eligible = history[history["date"] <= as_of]
    row = (eligible.iloc[-1] if not eligible.empty else history.iloc[0])
    return {
        t.strip().upper()
        for t in str(row["tickers"]).split(",")
        if t.strip()
    }


def get_point_in_time_universe(
    sector: str,
    as_of_date,
    history: Optional[pd.DataFrame] = None,
    base_universe: Optional[Dict[str, List[str]]] = None,
) -> List[str]:
    """Return the sector's symbols that were S&P 500 constituents as of
    `as_of_date`.

    Falls back to the full (current) sector list — with a logged warning — when
    the historical file is unavailable, or when the intersection would be empty
    (e.g. the snapshot date is malformed) so a backtest never silently ends up
    with an empty universe.
    """
    base = base_universe if base_universe is not None else SECTOR_UNIVERSES
    if sector not in base:
        raise KeyError(f"unknown sector {sector!r}; known sectors: {list(base)}")
    symbols = list(base[sector])

    if history is None:
        history = load_sp500_constituents()
    if history is None:
        logger.warning(
            "historical constituents unavailable; falling back to current "
            "universe for sector %r",
            sector,
        )
        return symbols

    members = constituents_as_of(history, as_of_date)
    pit = [s for s in symbols if s.upper() in members]
    if not pit:
        logger.warning(
            "no point-in-time members for sector %r as of %s; falling back to "
            "current universe",
            sector, as_of_date,
        )
        return symbols
    return pit
